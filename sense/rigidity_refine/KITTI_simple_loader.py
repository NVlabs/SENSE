"""
Copyright (C) 2019 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""

import os, copy, random, pickle
import numpy as np 
import os.path as osp

import torch.utils.data as data
import cv2

from scipy.misc import imread



class KITTI_sceneflow(data.Dataset):

    def __init__(self, base_dir, is_train=True, data_transform=None):
        
        self.transforms = data_transform

        if is_train: 
            self.is_train = True        
            # load kitti train 200 images as ground truth 
            with open(osp.join(base_dir, 'training/train.pkl'), 'rb') as pkl_file:
                files = pickle.load(pkl_file)

                self.rgb_L_files = [[osp.join(base_dir, x[0]), 
                    osp.join(base_dir, x[1])] for x in files['rgb-L'] ]
                self.rgb_R_files = [[osp.join(base_dir, x[0]), 
                    osp.join(base_dir, x[1])] for x in files['rgb-R'] ]
                self.disp_occ_0_files = [osp.join(base_dir, x) for x in files['disp_occ_0'] ]
                self.disp_occ_1_files = [osp.join(base_dir, x) for x in files['disp_occ_1'] ] 
                self.disp_noc_0_files = [osp.join(base_dir, x) for x in files['disp_noc_0'] ] 
                self.disp_noc_1_files = [osp.join(base_dir, x) for x in files['disp_noc_1'] ] 
                self.flow_occ_files = [osp.join(base_dir, x) for x in files['flow_occ'] ] 
                self.flow_noc_files = [osp.join(base_dir, x) for x in files['flow_noc'] ] 
                self.obj_mask       = [osp.join(base_dir, x) for x in files['obj_mask'] ] 
        else: 
            self.is_train = False        

            self.rgb_L_files = []
            self.cam2cam_files = []
            for idx in range(200):
                rgb0_path = osp.join(base_dir, 'image_2/{:06}_10.png'.format(idx))
                rgb1_path = osp.join(base_dir, 'image_2/{:06}_11.png'.format(idx))
                cam2cam_path=osp.join(base_dir,'calib_cam_to_cam/{:06}.txt'.format(idx))

                self.rgb_L_files.append([rgb0_path, rgb1_path])
                self.cam2cam_files.append(cam2cam_path)
                
        self.ids = 200
            
    def __getitem__(self, idx):

        if self.is_train:
            img0_L = self._load_rgb_tensor(self.rgb_L_files[idx][0])
            img1_L = self._load_rgb_tensor(self.rgb_L_files[idx][1])
            #img0_R = self._load_rgb_tensor(self.rgb_R_files[idx][0])
            #img1_R = self._load_rgb_tensor(self.rgb_R_files[idx][1])

            disp_T = self._load_disp_tensor(self.disp_occ_0_files[idx])
            disp_I = self._load_disp_tensor(self.disp_occ_1_files[idx])
            flow_T = self._load_flow_tensor(self.flow_occ_files[idx])
            obj_mask = self._load_object_mask_tensor(self.obj_mask[idx])
            cam2cam_path = self.rgb_L_files[idx][0][:-7].replace('image_2', 'calib_cam_to_cam') + '.txt'

            K2, K3 = self._load_camera_tensor(cam2cam_path)

            return img0_L, img1_L, disp_T, disp_I, flow_T, obj_mask, K2, K3
        else: 
            img0_L = self._load_rgb_tensor(self.rgb_L_files[idx][0])
            img1_L = self._load_rgb_tensor(self.rgb_L_files[idx][1])

            cam2cam_path = self.cam2cam_files[idx]

            K2, K3 = self._load_camera_tensor(cam2cam_path)

            return img0_L, img1_L, K2, K3

    def __len__(self):
        return self.ids

    def _load_rgb_tensor(self, path):
        image = imread(path)
        return  image #image.astype(np.float32) / 256.0

    def _load_disp_tensor(self, path):
        disparity = imread(path)
        disparity = disparity.astype(np.float32) / 256.0
        return disparity # invalid regions are marked as 0 in original map

    def _load_flow_tensor(self, path):
        bgr = cv2.imread(path, cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)

        H, W, C = bgr.shape
        assert bgr.dtype == np.uint16 and C == 3

        bgr = bgr.transpose(2,0,1)
        invalid = (bgr[0] == 0)
        u, v = bgr[2], bgr[1]

        def uint16toflow(x):
            return (x.astype(np.float32)-2**15) / 64.0

        u = uint16toflow(u)
        v = uint16toflow(v)
        flow = np.stack((u,v), axis=2)
 
        return flow.astype(np.float32)

    def _load_object_mask_tensor(self, path):
        mask = imread(path)
        return mask
    
    def _load_camera_tensor(self, path):

        filedata = read_calib_file(path)

        # Create 3x4 projection matrices
        P_rect_20 = np.reshape(filedata['P_rect_02'], (3, 4))
        P_rect_30 = np.reshape(filedata['P_rect_03'], (3, 4))
        K2 = P_rect_20[0:3, 0:3]
        K3 = P_rect_30[0:3, 0:3]

        # fx = P_rect_20[0,0]
        # fy = P_rect_20[1,1]
        # cx = P_rect_20[0,2]
        # cy = P_rect_20[1,2]

        # K = np.array([fx,fy,cx,cy], dtype=np.float32)
        return K2, K3

def read_calib_file(filepath):
    """Read in a calibration file and parse into a dictionary."""
    data = {}

    with open(filepath, 'r') as f:
        for line in f.readlines():
            key, value = line.split(':', 1)
            # The only non-float values in these files are dates, which
            # we don't care about anyway
            try:
                data[key] = np.array([float(x) for x in value.split()])
            except ValueError:
                pass

    return data