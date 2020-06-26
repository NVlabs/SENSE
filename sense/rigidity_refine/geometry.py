from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

"""
Copyright (C) 2019 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""
"""
Defines all the geometric transformation operations
"""

import torch
import torch.nn as nn
from torch import sin, cos, atan2, acos

_NEXT_AXIS = [1, 2, 0, 1]

# map axes strings to/from tuples of inner axis, parity, repetition, frame
_AXES2TUPLE = {
    'sxyz': (0, 0, 0, 0), 'sxyx': (0, 0, 1, 0), 'sxzy': (0, 1, 0, 0),
    'sxzx': (0, 1, 1, 0), 'syzx': (1, 0, 0, 0), 'syzy': (1, 0, 1, 0),
    'syxz': (1, 1, 0, 0), 'syxy': (1, 1, 1, 0), 'szxy': (2, 0, 0, 0),
    'szxz': (2, 0, 1, 0), 'szyx': (2, 1, 0, 0), 'szyz': (2, 1, 1, 0),
    'rzyx': (0, 0, 0, 1), 'rxyx': (0, 0, 1, 1), 'ryzx': (0, 1, 0, 1),
    'rxzx': (0, 1, 1, 1), 'rxzy': (1, 0, 0, 1), 'ryzy': (1, 0, 1, 1),
    'rzxy': (1, 1, 0, 1), 'ryxy': (1, 1, 1, 1), 'ryxz': (2, 0, 0, 1),
    'rzxz': (2, 0, 1, 1), 'rxyz': (2, 1, 0, 1), 'rzyz': (2, 1, 1, 1)}

_TUPLE2AXES = dict((v, k) for k, v in _AXES2TUPLE.items())

def meshgrid(H, W, B=None, is_cuda=False):
    """
    :param height
    :param width
    :param batch size
    :param initialize a cuda tensor if true
    """
    u = torch.arange(0, W)
    v = torch.arange(0, H)

    if is_cuda:
        u, v = u.cuda(), v.cuda()

    u = u.repeat(H, 1).view(1,H,W)
    v = v.repeat(W, 1).t_().view(1,H,W)

    if B is not None:
        u, v = u.repeat(B,1,1,1), v.repeat(B,1,1,1)
    return u, v

def generate_xy_grid(B, H, W, K):
    """ 
    :param batch size
    :param height
    :param width
    :param camera intrinsic array [fx,fy,cx,cy] 
    """
    fx, fy, cx, cy = K.split(1,dim=1)
    uv_grid = meshgrid(H, W, B)
    u_grid, v_grid = [uv.type_as(cx) for uv in uv_grid]
    px = ((u_grid.view(B,-1) - cx) / fx).view(B,1,H,W)
    py = ((v_grid.view(B,-1) - cy) / fy).view(B,1,H,W)
    return px, py, u_grid, v_grid

def batch_inverse_Rt(R, t):
    """ The inverse of the R, t: [R' | -R't] 
    :param rotation 
    :param translation
    ----------
    (tested)
    """
    R_t = R.transpose(1,2)
    t_inv = -torch.bmm(R_t, t.contiguous().view(-1, 3, 1))

    return R_t, t_inv.view(-1,3)

def batch_Rt_compose(d_R, d_t, R0, t0):
    """ Compose operator of R, t: [d_R*R | d_R*t + d_t] 
    :param rotation incremental
    :param translation incremental
    :param initial rotation
    :param initial translation
    ----------
    (tested)
    """
    R1 = d_R.bmm(R0)
    t1 = d_R.bmm(t0.view(-1,3,1)) + d_t.view(-1,3,1)
    return R1, t1.view(-1,3)

def batch_Rt_between(R0, t0, R1, t1): 
    """ Between operator of R, t, transform of [R0, t0] to [R1, t1]
    :param rotation of origin 
    :param translation of origin
    :param rotation of target 
    :param translation of target
    ----------
    (tested)
    """
    R0t = R0.transpose(1,2)
    dR = R1.bmm(R0t)
    dt = t1.view(-1,3) - dR.bmm(t0.view(-1,3,1)).view(-1,3)
    return dR, dt

def batch_skew(w):
    """ Generate skew-symmetric matrix
    :param skew symmetric matrix entry 
    ---------
    return the skew-symmetric matrix
    (tested)
    """
    B, D = w.size()
    assert(D == 3)
    o = torch.zeros(B).type_as(w)
    w0, w1, w2 = w[:, 0], w[:, 1], w[:, 2]
    return torch.stack((o, -w2, w1, w2, o, -w0, -w1, w0, o), 1).view(B, 3, 3)

def batch_twist2Mat(twist):
    """ Calculate the rotation matrix using Rodrigues' Rotation Formula
    http://electroncastle.com/wp/?p=39 or Ethan Eade's lie group note:
    http://ethaneade.com/lie.pdf
    
    :param twist parameter
    ------------
    (tested with cv2.Rodrigues implementation)
    """
    B = twist.size()[0]
    theta = twist.norm(p=2, dim=1).view(B, 1)
    w_so3 = twist / theta.expand(B, 3)
    W = batch_skew(w_so3)
    return torch.eye(3).repeat(B,1,1).type_as(W) \
        + W*sin(theta.view(B,1,1)) \
        + W.bmm(W)*(1-cos(theta).view(B,1,1))

def batch_mat2angle(R):
    """ Calcuate the angle from SO3 Rotation
    :param Rotation matrix
    --------
    (tested)
    """
    R1 = [torch.trace(R[i]) for i in range(R.size()[0])]
    R_trace = torch.stack(R1)
    # clamp if the angle is too large (break small angle assumption)
    angle = acos( ((R_trace - 1)/2).clamp(-1,1))
    return angle

def batch_mat2twist(R):
    """ Calculate the twist vector from SO3 Rotation
    Ethan Eade's lie group note: http://ethaneade.com/lie.pdf
    :param Rotation matrix
    --------
    (tested)
    """
    B = R.size()[0]
    theta = batch_mat2angle(R)
    # refers to gtsam implementation 
    # https://bitbucket.org/gtborg/gtsam/src/c84496632fd6e062e69cb1b9d7b77ac8d9390fbd/gtsam/geometry/SO3.cpp#lines-170
    r11,r12,r13,r21,r22,r23,r31,r32,r33 = torch.split(R.view(B,-1),1,dim=1)
    res = torch.cat([r32-r23, r13-r31, r21-r12],dim=1)  
    return (0.5*theta/sin(theta)).view(B,1) * res

def batch_warp_inverse_depth(p_x, p_y, p_invD, pose, K):
    """ Compute the warping grid w.r.t. the SE3 transform given the inverse depth
    :param p_x the x coordinate map
    :param p_y the y coordinate map
    :param p_invD the inverse depth
    :param pose the 3D transform in SE3
    :param K the intrinsics
    """
    [R, t] = pose
    B, _, H, W = p_x.shape

    I = torch.ones((B,1,H,W)).type_as(p_invD)
    x_y_1 = torch.cat((p_x, p_y, I), dim=1)

    warped = torch.bmm(R, x_y_1.view(B,3,H*W)) + \
        t.view(B,3,1).expand(B,3,H*W) * p_invD.view(B, 1, H*W).expand(B,3,H*W)

    x_, y_, s_ = torch.split(warped, 1, dim=1)
    fx, fy, cx, cy = torch.split(K, 1, dim=1)

    u_ = (x_ / s_).view(B,-1) * fx + cx
    v_ = (y_ / s_).view(B,-1) * fy + cy

    inv_z_ = p_invD / s_.view(B,1,H,W)

    return u_.view(B,1,H,W), v_.view(B,1,H,W), inv_z_

def batch_warp_affine(pu, pv, affine):
    # A = affine[:,:,:2]
    # t = affine[:,:, 2]
    B,_,H,W = pu.shape
    ones = torch.ones(pu.shape).type_as(pu)
    uv = torch.cat((pu, pv, ones), dim=1)
    uv = torch.bmm(affine, uv.view(B,3,-1)) #+ t.view(B,2,1)
    return uv[:,0].view(B,1,H,W), uv[:,1].view(B,1,H,W)

def check_occ(inv_z_buffer, inv_z_ref, u, v, thres=1e-1):
    """ z-buffering check of occlusion 
    :param inverse depth of target frame
    :param inverse depth of reference frame
    """
    B, _, H, W = inv_z_buffer.shape

    inv_z_warped = warp_features(inv_z_ref, u, v)
    inlier = (inv_z_buffer > inv_z_warped - thres)

    inviews = inlier & (u > 0) & (u < W) & \
        (v > 0) & (v < H)

    return 1-inviews

def warp_features(F, u, v):
    """ Warp the feature map (F) w.r.t. the grid (u, v)
    """
    B, C, H, W = F.shape

    u_norm = u / ((W-1)/2) - 1
    v_norm = v / ((H-1)/2) - 1
    uv_grid = torch.cat((u_norm.view(B,H,W,1), v_norm.view(B,H,W,1)), dim=3)
    F_warped = nn.functional.grid_sample(F, uv_grid,
        mode='bilinear', padding_mode='border')
    return F_warped

def warp_feature_from_flow(F, flow):
    """ Warp the feature map from the flow map
    """
    B, C, H, W = flow.shape
    uv = meshgrid(H, W, B)
    uv = [x.type_as(flow) for x in uv]
    return warp_features(F, 
        uv[0]+flow[:,0,:,:].view(B,1,H,W), 
        uv[1]+flow[:,1,:,:].view(B,1,H,W))

def warp_feature_from_disparity(F, disp):
    """ Warp the feature map from the target disparity map

        As all the convention, the disparity warping is an inverse warping

        @todo: it is still not clear yet 

    :input
    :param the feature map in reference image BxCxHxW
    :param the disparity map Bx1xHxW
    ----------
    :param warped feature
    """
    B, C, H, W = F.shape
    uv = meshgrid(H, W, B)
    uv = [x.type_as(disp) for x in uv]
    u_warped = uv[0] + disp
    F_w = warp_features(F, u_warped, uv[1])
    # for all invalid disparity region, set it to 0
    F_w[(disp==0).expand(B,C,H,W)] = 0

    in_view = (u_warped >= 0) & (u_warped < W)

    return F_w, in_view

def compute_warped_residual(pose, x0, x1, K, invD0, invD1,  
    px, py, obj_mask = None):
    """ Compute the residual error of the image from  
    w.r.t. the reference feature map.
    refer to equation (12) in the paper

    :param the forward warping pose from the reference camera to the target frame.
        Note that warping from the target frame to the reference frame is the inverse of this operation.
    :param the template feature image
    :param the image feature image
    :param the intrinsic calibration Bx4
    :param the reference inverse depth
    :param the target inverse depth
    :param the pixel x map
    :param the pixel y map
    :param object mask
    -----------
    :return the residual (of reference image), and occlusion information
    """
    u_warped, v_warped, inv_z_warped = batch_warp_inverse_depth(
        px, py, invD0, pose, K)
    
    x1_1to0 = warp_features(x1, u_warped, v_warped)

    occ = check_occ(inv_z_warped, invD1, u_warped, v_warped)

    residuals = x1_1to0 - x0

    B, C, H, W = x0.shape
    if obj_mask is not None:
        # determine whether the object is in-view
        occ = occ & (obj_mask.view(B,1,H,W) < 1)

    residuals[occ.expand(B,C,H,W)] = 0

    return residuals, occ

def batch_transform_xyz(xyz_tensor, R, t, get_Jacobian=True):
    """ Transform the point cloud w.r.t. the transformation matrix
    :param xyz_tensor: B * 3 * H * W
    :param R: rotation matrix B * 3 * 3
    :param t: translation vector B * 3
    """
    B, C, H, W = xyz_tensor.size()
    t_tensor = t.contiguous().view(B,3,1).repeat(1,1,H*W)
    p_tensor = xyz_tensor.contiguous().view(B, C, H*W)
    # the transformation process is simply:
    # p' = t + R*p
    xyz_t_tensor = torch.baddbmm(t_tensor, R, p_tensor)

    if get_Jacobian:
        # return both the transformed tensor and its Jacobian matrix
        J_r = R.bmm(batch_skew_symmetric_matrix(-1*p_tensor.permute(0,2,1)))
        J_t = -1 * torch.eye(3).view(1,3,3).expand(B,3,3)
        J = torch.cat((J_r, J_t), 1)
        return xyz_t_tensor.view(B, C, H, W), J
    else:
        return xyz_t_tensor.view(B, C, H, W)

def flow_from_rigid_transform(depth, extrinsic, intrinsic):
    """
    Get the optical flow induced by rigid transform [R,t] and depth
    """
    [R, t] = extrinsic
    [fx, fy, cx, cy] = intrinsic

def batch_project(xyz_tensor, K):
    """ Project a point cloud into pixels (u,v) given intrinsic K
    [u';v';w] = [K][x;y;z]
    u = u' / w; v = v' / w

    :param the xyz points 
    :param calibration is a torch array composed of [fx, fy, cx, cy]
    -------
    :return u, v grid tensor in image coordinate
    (tested through inverse project)
    """
    B, _, H, W = xyz_tensor.size()
    batch_K = K.expand(H, W, B, 4).permute(2,3,0,1)

    x, y, z = torch.split(xyz_tensor, 1, dim=1)
    fx, fy, cx, cy = torch.split(batch_K, 1, dim=1)

    u = fx*x / z + cx
    v = fy*y / z + cy
    return torch.cat((u,v), dim=1)

def batch_inverse_project(depth, K):
    """ Inverse project pixels (u,v) to a point cloud given intrinsic 
    :param depth dim B*H*W
    :param calibration is torch array composed of [fx, fy, cx, cy]
    :param color (optional) dim B*3*H*W
    -------
    :return xyz tensor (batch of point cloud)
    (tested through projection)
    """
    if depth.dim() == 3:
        B, H, W = depth.size()
    else: 
        B, _, H, W = depth.size()

    x, y = generate_xy_grid(B,H,W,K)
    z = depth.view(B,1,H,W)
    return torch.cat((x*z, y*z, z), dim=1)

def batch_euler2mat(ai, aj, ak, axes='sxyz'):
    """ A torch implementation euler2mat from transform3d:
    https://github.com/matthew-brett/transforms3d/blob/master/transforms3d/euler.py
    :param ai : First rotation angle (according to `axes`).
    :param aj : Second rotation angle (according to `axes`).
    :param ak : Third rotation angle (according to `axes`).
    :param axes : Axis specification; one of 24 axis sequences as string or encoded tuple - e.g. ``sxyz`` (the default).
    -------
    :return rotation matrix, array-like shape (B, 3, 3)

    Tested w.r.t. transforms3d.euler module
    """
    B = ai.size()[0]

    try:
        firstaxis, parity, repetition, frame = _AXES2TUPLE[axes.lower()]
    except (AttributeError, KeyError):
        _TUPLE2AXES[axes]  # validation
        firstaxis, parity, repetition, frame = axes

    i = firstaxis
    j = _NEXT_AXIS[i+parity]
    k = _NEXT_AXIS[i-parity+1]
    order = [i, j, k]

    if frame:
        ai, ak = ak, ai
    if parity:
        ai, aj, ak = -ai, -aj, -ak

    si, sj, sk = sin(ai), sin(aj), sin(ak)
    ci, cj, ck = cos(ai), cos(aj), cos(ak)
    cc, cs = ci*ck, ci*sk
    sc, ss = si*ck, si*sk

    # M = torch.zeros(B, 3, 3).cuda()
    if repetition:
        c_i = [cj, sj*si, sj*ci]
        c_j = [sj*sk, -cj*ss+cc, -cj*cs-sc]
        c_k = [-sj*ck, cj*sc+cs, cj*cc-ss]
    else:
        c_i = [cj*ck, sj*sc-cs, sj*cc+ss]
        c_j = [cj*sk, sj*ss+cc, sj*cs-sc]
        c_k = [-sj, cj*si, cj*ci]

    def permute(X): # sort X w.r.t. the axis indices
        return [ x for (y, x) in sorted(zip(order, X)) ]

    c_i = permute(c_i)
    c_j = permute(c_j)
    c_k = permute(c_k)

    r =[torch.stack(c_i, 1),
        torch.stack(c_j, 1),
        torch.stack(c_k, 1)]
    r = permute(r)

    return torch.stack(r, 1)

def batch_mat2euler(M, axes='sxyz'): 
    """ A torch implementation euler2mat from transform3d:
    https://github.com/matthew-brett/transforms3d/blob/master/transforms3d/euler.py
    :param array-like shape (3, 3) or (4, 4). Rotation matrix or affine.
    :param  Axis specification; one of 24 axis sequences as string or encoded tuple - e.g. ``sxyz`` (the default).
    --------
    :returns 
    :param ai : First rotation angle (according to `axes`).
    :param aj : Second rotation angle (according to `axes`).
    :param ak : Third rotation angle (according to `axes`).
    """
    try:
        firstaxis, parity, repetition, frame = _AXES2TUPLE[axes.lower()]
    except (AttributeError, KeyError):
        _TUPLE2AXES[axes]  # validation
        firstaxis, parity, repetition, frame = axes

    i = firstaxis
    j = _NEXT_AXIS[i+parity]
    k = _NEXT_AXIS[i-parity+1]

    if repetition:
        sy = torch.sqrt(M[:, i, j]**2 + M[:, i, k]**2)
        # A lazy way to cope with batch data. Can be more efficient
        mask = ~(sy > 1e-8) 
        ax = atan2( M[:, i, j],  M[:, i, k])
        ay = atan2( sy,          M[:, i, i])
        az = atan2( M[:, j, i], -M[:, k, i])
        if mask.sum() > 0:
            ax[mask] = atan2(-M[:, j, k][mask], M[:, j, j][mask])
            ay[mask] = atan2( sy[mask],         M[:, i, i][mask])
            az[mask] = 0.0
    else:
        cy = torch.sqrt(M[:, i, i]**2 + M[:, j, i]**2)
        mask = ~(cy > 1e-8)
        ax = atan2( M[:, k, j],  M[:, k, k])
        ay = atan2(-M[:, k, i],  cy)
        az = atan2( M[:, j, i],  M[:, i, i])
        if mask.sum() > 0:
            ax[mask] = atan2(-M[:, j, k][mask],  M[:, j, j][mask])
            ay[mask] = atan2(-M[:, k, i][mask],  cy[mask])
            az[mask] = 0.0

    if parity:
        ax, ay, az = -ax, -ay, -az
    if frame:
        ax, az = az, ax
    return ax, ay, az
