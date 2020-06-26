"""
Copyright (C) 2019 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""
import gzip, os
import scipy.io as sio
import os.path as osp
import matplotlib.pyplot as plt

from scipy.misc import imread, imsave

from .io_utils import *
from .KITTI_simple_loader import KITTI_sceneflow
from .refine_utils import *
from .pose import flow2pose_least_square

from skimage.morphology import dilation, square, erosion
from scipy import ndimage


from PIL import Image
from argparse import ArgumentParser

from tqdm import tqdm

def check_directory(filename):
    target_dir = osp.dirname(filename)
    if not osp.isdir(target_dir):
        os.makedirs(target_dir)

def depth2xyz(K, depth):
    H, W = depth.shape

    inv_K = np.linalg.inv(K)

    rows = np.linspace(0, H-1, H)
    cols = np.linspace(0, W-1, W)
    u_mat, v_mat = np.meshgrid(cols, rows)

    fx, fy, cx, cy = K[0,0], K[1,1], K[0,2], K[1,2]

    x = (u_mat - cx) * depth / fx
    y = (v_mat - cy) * depth / fy

    pcd = np.stack((x,y,depth), axis=2)

    # uvd = np.stack((depth*u_mat, depth*v_mat, depth), axis=0)
    # vertex_map = np.tensordot(inv_K, uvd, axes=1)
    #return vertex_map.transpose((1,2,0))
    return pcd

def flow_warp(disp, flow):
    H, W = disp.shape
    [u_grid, v_grid] = np.meshgrid(np.arange(0,W),np.arange(0,H))

    u_f = flow[:,:,0]
    v_f = flow[:,:,1]

    u1 = u_grid + u_f
    v1 = v_grid + v_f

    return ndimage.map_coordinates(disp, [v1, u1])

def colorize_mask(mask):
    # mask: numpy array of the mask

    palette = [128, 64, 128, 244, 35, 232, 70, 70, 70, 102, 102, 156, 190, 153, 153, 153, 153, 153, 250, 170, 30,
            220, 220, 0, 107, 142, 35, 152, 251, 152, 70, 130, 180, 220, 20, 60, 255, 0, 0, 0, 0, 142, 0, 0, 70,
            0, 60, 100, 0, 80, 100, 0, 0, 230, 119, 11, 32]

    new_mask = Image.fromarray(mask.astype(np.uint8)).convert('P')
    new_mask.putpalette(palette)
    return new_mask

def create_sceneflow(V0, V1, flow, T, mask): 
    """  
    This may be slow due to not efficient forward warping
    """

    V0t = V0.transpose()
    V1t = V1.transpose()

    R, t = T
    
    V1to0t = np.matmul(R, V1t) + t.reshape(3,-1)

    H, W = flow.shape[:2]
    V1to0 = V1to0t.transpose().reshape(H,W,3)
    V1 = V1.transpose().reshape(H,W,3)
    V0 = V0.reshape(H,W,3)

    stride = 10

    sceneflow = np.zeros((H,W,3), dtype=float)
    for v0 in range(0, H, stride): 
        for u0 in range(0, W, stride):
            if not mask[v0, u0]: 
                continue

            du, dv = flow[v0,u0,0], flow[v0,u0,1]
            u1, v1 = int(round(u0 + du)), int(round(v0 + dv))
            if (u1 < 0 or u1 >= W or v1 < 0 or v1 >= H): 
                continue            

            sceneflow[v0,u0] = V1to0[v1, u1] - V0[v0, u0]
        
    return sceneflow

def warp_disp_refine_rigid(disp0, disp1, flow, seg, K0, K1):
    baseline = 0.54
    fB = baseline * K0[0,0]

    outlier_mask = [10, # sky
        11, # pedestrians
        12, # human
        13, # vehicles
        14, # van
        18, # bicycle
    ]

    # generate inlier labels 
    inliers = np.ones(disp0.shape).astype(bool)
    for label in outlier_mask: 
        inliers *= (seg != label)

    # erode the background region to ensure the foreground is preserved as much as possible
    inliers = erosion(inliers, square(20))

    ### use flow consistency loss to refine the rigid flow
    pose, flow_background_refined, inv_depth_warped = flow2pose_least_square(
        disp0/fB, disp1/fB, flow, K0, inliers)

    # stitch the second frame optical flow 
    du = flow[:,:,0].copy()
    dv = flow[:,:,1].copy()
    du[inliers] = flow_background_refined[:,:,0][inliers]
    dv[inliers] = flow_background_refined[:,:,1][inliers]
    flow_composite = np.stack((du,dv), axis=2)

    disp1to0_flow_warp = flow_warp(disp1, flow_composite)

    fg_mask = ~inliers
    disp1_warped_rigid = inv_depth_warped * fB

    disp1_warped_rigid[fg_mask] = disp1to0_flow_warp[fg_mask]

    return flow_composite, disp1to0_flow_warp, disp1_warped_rigid

def main(args):
    disp1_fg_rigid = 0
    disp1_bg_rigid = 0
    disp1_fg_raw_warp = 0
    disp1_bg_raw_warp = 0

    refined_bg = 0
    refined_fg = 0
    raw_bg = 0
    raw_fg = 0
    count = 0

    bad_frames = [105]

    generate_train = False

    kitti_loader = KITTI_sceneflow(args.kitti_dir, generate_train, None)

    outlier_mask = [10, # sky
        11, # pedestrians
        12, # human
        13, # vehicles
        14, # van
        18, # bicycle
    ]

    generate_train = args.do_train
        
    for idx, batch in tqdm(enumerate(kitti_loader)): 
        count += 1

        ###################################
        # load ground truth information 
        if generate_train:
            img0_L, img1_L, disp_T, disp_I, flow_T, obj_mask, K0, K1 = batch
        else: 
            img0_L, img1_L, K0, K1 = batch

        ###################################
        # load estimated information

        #disp0_gen_path = folder + '/000165_10_baseline.mat'
        #disp0_gen_path = folder + '/000136_10_segDistill.mat'
        #disp0_gen_path = folder + '/000165_10_softOcc.mat'

        disp0_gen_path = args.res_dir + '/disp_0/{:06}_{:02}.mat'.format(idx, 10)
        disp1_gen_path = args.res_dir + '/disp_1_raw/{:06}_{:02}.mat'.format(idx, 10)
        seg_gen_path = args.res_dir + '/seg/{:06}_{:02}.png'.format(idx, 10)
        forward_flow_gen_path = args.res_dir + '/flow_raw/{:06}_{:02}.flo'.format(idx, 10)

        disp0 = read_disp_gen(disp0_gen_path)
        disp1 = read_disp_gen(disp1_gen_path)
        # disp_occ = read_occ_gen(occ_gen_path)    
        flow = read_flow_gen(forward_flow_gen_path)
        # bw_flow = read_flow_gen(backward_flow_gen_path)
        seg  = read_seg_gen(seg_gen_path)
        
        seg_image = colorize_mask(seg)

        fw_flow = flow.copy()

        baseline = 0.54
        fB = baseline * K0[0,0]

        # generate inlier labels 
        inliers = np.ones(disp0.shape).astype(bool)
        for label in outlier_mask: 
            inliers *= (seg != label)

        # erode the background region to ensure the foreground is preserved as much as possible
        inliers = erosion(inliers, square(20))

        ### use flow consistency loss to refine the rigid flow
        pose, flow_background_refined, inv_depth_warped = flow2pose_least_square(
            disp0/fB, disp1/fB, flow, K0, inliers)

        # print(pose)

        # stitch the second frame optical flow 
        du = flow[:,:,0].copy()
        dv = flow[:,:,1].copy()
        du[inliers] = flow_background_refined[:,:,0][inliers]
        dv[inliers] = flow_background_refined[:,:,1][inliers]
        flow_composite = np.stack((du,dv), axis=2)

        disp1to0_flow_warp = flow_warp(disp1, flow_composite)

        fg_mask = ~inliers
        disp1_warped_rigid = inv_depth_warped * fB

        disp1_warped_rigid[fg_mask] = disp1to0_flow_warp[fg_mask]

        # # visualize scene flow 
        # from open3d_utils import depth2pointcloud, np2pointcloud, construct_sceneflow

        # depth0 = (fB / disp0).clip(0, 80)
        # depth1 = (fB / disp1).clip(0, 80)
        # sky = (seg == 10)
        # depth0[sky] = 80
        # depth1[sky] = 80
        # K = [K0[0,0], K0[1,1], K0[0,2], K0[1,2]]
        # pcd0 = depth2pointcloud(depth0, K)
        # pcd1 = depth2pointcloud(depth1, K)

        # motion_mask = fg_mask.copy()
        # motion_mask[sky] = False
        # sceneflow = create_sceneflow(pcd0, pcd1, flow_composite, pose, motion_mask)

        # pcd3d0 = np2pointcloud(pcd0, img0_L.reshape((-1,3))/255.0)

        # pcd_flow = construct_sceneflow(pcd0, sceneflow)

        # import open3d 
        # open3d.draw_geometries([pcd3d0] + [pcd_flow] )

        # plt.imshow(disp1_warped_rigid)
        # plt.show()

        ### save the refined flow result 

        refined_flow_path = forward_flow_gen_path.replace('flow_raw', 'flow_rigid')
        check_directory(refined_flow_path)
        write_flow(flow_composite, refined_flow_path)

        vis_flow = flow_visualize(flow_composite)
        refined_flow_vis_path = seg_gen_path.replace('seg', 'flow_rigid_vis')
        check_directory(refined_flow_vis_path)
        imsave(refined_flow_vis_path, vis_flow)

        refined_disparity_path = disp1_gen_path.replace('disp_1_raw', 'disp_1_rigid')
        check_directory(refined_disparity_path)

        write_disp(disp1_warped_rigid, refined_disparity_path)

        ############################################################################
        # evaluation 

        if generate_train:

            print('current results of frame {:}'.format(count) )

            disp_T_metrics = eval_disp(disp0, disp_T, obj_mask)
            disp_T_fg_percent = disp_T_metrics['percent_fg']
            disp_T_bg_percent = disp_T_metrics['percent_bg']
            print('fg error of the first frame disp {0:.3f}%'.format(disp_T_fg_percent*100))  
            print('bg error of the first frame disp {0:.3f}%'.format(disp_T_bg_percent*100))

            disp_I_metrics = eval_disp(disp1_warped_rigid, disp_I, obj_mask)
            disp_I_fg_percent = disp_I_metrics['percent_fg']
            disp_I_bg_percent = disp_I_metrics['percent_bg']
            print('fg error of the second frame (rigid) {0:.3f}%').format(disp_I_fg_percent*100)   
            print('bg error of the second frame (rigid) {0:.3f}%').format(disp_I_bg_percent*100)   

            disp_I_metrics_all = eval_disp(disp1to0_flow_warp, disp_I, obj_mask)
            disp_I_fg_percent_all = disp_I_metrics_all['percent_fg']
            disp_I_bg_percent_all = disp_I_metrics_all['percent_bg']
            print('fg error of the second frame (raw warp) {0:.3f}%').format(disp_I_fg_percent_all*100)   
            print('bg error of the second frame (raw warp) {0:.3f}%').format(disp_I_bg_percent_all*100)   

            disp1_fg_rigid += disp_I_fg_percent
            disp1_bg_rigid += disp_I_bg_percent
            print('fg error of accumulated second frame (rigid) {0:.3f}%').format(disp1_fg_rigid*100 / count)
            print('bg error of accumulated second frame (rigid) {0:.3f}%').format(disp1_bg_rigid*100 / count)

            disp1_fg_raw_warp += disp_I_fg_percent_all
            disp1_bg_raw_warp += disp_I_bg_percent_all
            print('fg error of accumulated second frame (raw warp) {0:.3f}%').format(disp1_fg_raw_warp*100 / count)
            print('bg error of accumulated second frame (raw warp) {0:.3f}%').format(disp1_bg_raw_warp*100 / count)

            metrics_before = eval_flow(fw_flow, flow_T, obj_mask)
            raw_bg_percent = metrics_before['percent_bg']
            raw_fg_percent = metrics_before['percent_fg']
            raw_bg += raw_bg_percent
            raw_fg += raw_fg_percent
            print('fg error of raw flow {0:.3f}%'.format(raw_fg_percent*100))  
            print('bg error of raw flow {0:.3f}%'.format(raw_bg_percent*100))
            print('fg error of raw acc flow {0:.3f}%'.format(raw_fg / count*100))
            print('bg error of raw acc flow {0:.3f}%'.format(raw_bg / count*100))

            metrics_refined = eval_flow(flow_composite, flow_T, obj_mask)
            this_bg_percent = metrics_refined['percent_bg']
            this_fg_percent = metrics_refined['percent_fg']
            refined_bg += this_bg_percent
            refined_fg += this_fg_percent
            print('fg error of this frame {0:.3f}%'.format(this_fg_percent*100))  
            print('bg error of this frame {0:.3f}%'.format(this_bg_percent*100))
            print('fg error of accumulation {0:.3f}%'.format(refined_fg*100 / count))
            print('bg error of accumulation {0:.3f}%'.format(refined_bg*100 / count))

            if this_bg_percent > 0.20: 
                import pdb; pdb.set_trace()

    # print(bad_frames)

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--refine-type', choices=['nn', 'rigid'])
    parser.add_argument('--kitti-dir', type=str)
    parser.add_argument('--res-dir', type=str)
    parser.add_argument('--do-train', action='store_true')
    args = parser.parse_args()

    main(args)