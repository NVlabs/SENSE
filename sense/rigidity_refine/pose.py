"""
Copyright (C) 2019 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""
"""
The Least-Square method to minimize re-projection errors

@author: Zhaoyang Lv
@Date: March, 2019
"""

import torch
import torch.nn as nn
import torch.nn.functional as ft
import numpy as np

import sense.rigidity_refine.geometry as geometry

def flow2pose_least_square(invD0_np, invD1_np, fw_flow_np, K_mat, mask):
    """
    :input
    :param the inverse depth of the reference image
    :param the inverse depth of the target image
    :param forward optical flow
    :param the camera intrinsics
    :param the background segmentation mask
    --------
    :return 
    :param estimated transform 
    """
    H, W = invD0_np.shape
    # Hs, Ws = 128, 256
    # fx_s, fy_s = float(Ws) / W, float(Hs) / H

    d0 = torch.from_numpy(invD0_np).cuda().view(1,1,H,W)
    d1 = torch.from_numpy(invD1_np).cuda().view(1,1,H,W)
    fw_flow = torch.from_numpy(fw_flow_np.transpose(2,0,1)).cuda().view(1,2,H,W)
    K   =torch.FloatTensor([K_mat[0,0], K_mat[1,1], K_mat[0,2], K_mat[1,2]]).cuda().view(1,4)
    weight = torch.from_numpy(mask.astype(int)).view(1,1,H,W).type_as(d0)

    R0 = torch.eye(3,dtype=torch.float).expand(1,3,3).type_as(d0)
    t0 = torch.zeros(1,3,1,dtype=torch.float).type_as(d1)
    poseI = [R0, t0]

    # d0 = ft.interpolate(d0, size=(Hs,Ws), mode='nearest')
    # d1 = ft.interpolate(d1, size=(Hs,Ws), mode='nearest')
    # fw_flow = ft.interpolate(fw_flow, size=(Hs,Ws), mode='nearest')
    # fw_flow[:,0] *= fx_s
    # fw_flow[:,1] *= fy_s
    # weight = ft.interpolate(weight, size=(Hs,Ws), mode='nearest')
    # K[:,0] *= fx_s
    # K[:,1] *= fy_s
    # K[:,2] *= fx_s
    # K[:,3] *= fy_s
    
    import time 
    start = time.time()
    pose, d_u, d_v, inv_z, d1to0 = least_square(poseI, d0, d1, fw_flow, K, weight)
    # print('refine takes {:}'.format(time.time() - start))

    # d_u = ft.interpolate(d_u, size=(H,W), mode='bilinear') / fx_s
    # d_v = ft.interpolate(d_v, size=(H,W), mode='bilinear') / fy_s

    flow_est = torch.cat((d_u, d_v), dim=1)
    flow_background_refined = flow_est[0].cpu().numpy().transpose(1,2,0)

    inv_depth_warped = inv_z.view(H,W).cpu().numpy()

    R, t = pose

    pose_np = [R[0].cpu().numpy(), t[0].cpu().numpy()]

    return pose_np, flow_background_refined, inv_depth_warped

def least_square(pose, d0, d1, flow0to1, K, weight, timers=None):
    """
    :param pose, the initial pose
        (extrinsic of the target frame w.r.t. the referenc frame)
    :param the template image inverse depth
    :param the target image inverse depth
    """
    B, _, H, W = d0.shape
    assert(B == 1) # the current implementation cannot work with multiple batches

    px, py, pu, pv = geometry.generate_xy_grid(B,H,W,K)

    fu = flow0to1[:,0].view(B,-1,1)
    fv = flow0to1[:,1].view(B,-1,1)

    # check forward-backward geometry consitency
    u1 = flow0to1[:,0].view(B,1,H,W) + pu
    v1 = flow0to1[:,1].view(B,1,H,W) + pv
    d1to0 = geometry.warp_features(d1, u1, v1)
    depth0 = 1.0/(1e-8+d0.clamp(1.25e-2, 10))
    depth1 = 1.0/(1e-8+d1.clamp(1.25e-2, 10))
    depth_weight = torch.exp(-10*(depth0 - depth1).abs())
    
    # do not process all information, which makes the process faster
    inlier = (weight > 0) * (d0 > 1e-2) * (depth_weight > 0.25)
    inlier = inlier.view(B,-1,1) 
    total_num = inlier.sum().float()

    W_mat = (depth_weight).view(B,-1,1)
    W_u = W_mat[inlier].view(B,-1,1)
    W_v = W_mat[inlier].view(B,-1,1)

    Jx_p, Jy_p = compute_jacobian_warping(d0, K, px, py)

    Jx, Jy = [], []
    for idx in range(6):
        Jx.append(Jx_p[...,idx].unsqueeze(-1)[inlier])
        Jy.append(Jy_p[...,idx].unsqueeze(-1)[inlier])
    Jx = torch.stack(Jx, dim=1).unsqueeze(0)
    Jy = torch.stack(Jy, dim=1).unsqueeze(0)

    ############################################################
    ### Reweighted Least-Square using Gauss-Newton update
    last_error = 1e8
    max_iterations = 50
    initial_error = 1e8
    for idx in range(max_iterations):
        # warped (u, v, inverse z)
        if timers: timers.tic('Warping')
        u_w, v_w, inv_z = geometry.batch_warp_inverse_depth(px, py, d0, pose, K)
        d_u, d_v = u_w - pu, v_w - pv 
        if timers: timers.toc('Warping')

        if timers: timers.tic('construct Ax = b')
        r_u = fu - d_u.view(B,-1,1) 
        r_v = fv - d_v.view(B,-1,1)

        r_u = r_u[inlier].view(B,-1,1)
        r_v = r_v[inlier].view(B,-1,1)

        Jtx = torch.transpose(Jx,1,2)
        Jty = torch.transpose(Jy,1,2)
        # Apply the weight in u,v directions, \alpha is the threshold  
        # Note: it accumulates the weight matrix for every iteration. 
        # It somehow downweights the flow scale by itself. Working well.
        W_u *= weight_Huber(r_u, alpha=10)
        W_v *= weight_Huber(r_v, alpha=4)

        wJx = W_u * Jx
        wJy = W_v * Jy
        # J^{T}WJ
        JtWJ = torch.bmm(Jtx , wJx) + torch.bmm(Jty, wJy)
        JtWJ/= total_num
        # J^{T}Wr
        Rhs = torch.bmm(Jtx, W_u * r_u) + torch.bmm(Jty, W_v* r_v) 
        Rhs/= total_num
        if timers: timers.tic('construct Ax = b')

        Rhs_mean = Rhs.abs().mean()
        if idx == 0: initial_error = Rhs_mean
        error_reduce = last_error - Rhs_mean
        # print(Rhs_mean)

        # terminate criterion
        if error_reduce > 1e-6 * initial_error : 
            last_error = Rhs_mean
        else: 
            break

        invH = invert_Hessian(JtWJ) 

        if timers: timers.tic('solve x=A^{-1}b')
        xi = torch.bmm(invH, Rhs)   
        d_R = geometry.batch_twist2Mat(xi[:, :3].view(-1,3))
        d_t = xi[:, 3:]
        R, t = pose
        pose = geometry.batch_Rt_compose(R, t, d_R, d_t) 
        if timers: timers.toc('solve x=A^{-1}b')

    d1to0 = geometry.warp_features(d1, u_w, v_w)

    return pose, d_u, d_v, inv_z, d1to0

def compute_jacobian_warping(p_invdepth, K, px, py):
    """ Compute the Jacobian matrix of the warped (x,y) w.r.t. the inverse depth
    (linearized at origin)
    :param p_invdepth the input inverse depth
    :param the intrinsic calibration
    :param the pixel x map
    :param the pixel y map
     ------------
    :return the warping jacobian in x, y direction
    """
    B, C, H, W = p_invdepth.size()
    assert(C == 1)

    x = px.view(B, -1, 1)
    y = py.view(B, -1, 1)
    invd = p_invdepth.view(B, -1, 1)

    xy = x * y
    O = torch.zeros((B, H*W, 1)).type_as(p_invdepth)

    # This is cascaded Jacobian functions of the warping function
    # Refer to the supplementary materials for math documentation
    dx_dp = torch.cat((-xy,     1+x**2, -y, invd, O, -invd*x), dim=2)
    dy_dp = torch.cat((-1-y**2, xy,     x, O, invd, -invd*y), dim=2)

    fx, fy, cx, cy = torch.split(K, 1, dim=1)

    return dx_dp*fx.view(B,1,1), dy_dp*fy.view(B,1,1)

def invert_Hessian(H):
    """ Generate (H+damp)^{-1}, with predicted damping values
    :param approximate Hessian matrix JtWJ
    -----------
    :return the inverse of Hessian
    """
    # GPU is much slower for matrix inverse when the size is small (compare to CPU)
    # works (50x faster) than inversing the dense matrix in GPU

    # Note: the pytorch 0.40 does not support batch inverse process. 
    # Since we assume it is only batch 1 process. It should be fine.
    # You can remove the hacky solution if upgrade to pytorch 1.0
    if H.is_cuda:
        invH = torch.inverse(H[0].cpu()).cuda().unsqueeze(0)
    else:
        invH = torch.inverse(H)
    return invH

def weight_Huber(x, alpha = 6):
    """ weight function of Huber loss:
    refer to P. 24 w(x) at
    https://members.loria.fr/moberger/Enseignement/Master2/Documents/ZhangIVC-97-01.pdf

    Note this current implementation is not differentiable.
    """
    abs_x = torch.abs(x)
    linear_mask = abs_x > alpha
    w = torch.ones(x.shape).type_as(x)

    if linear_mask.sum().item() > 0: 
        w[linear_mask] = alpha / abs_x[linear_mask]
    return w