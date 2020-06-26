"""
Copyright (C) 2019 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""
import torch
import torch.utils.data as data

import os
import os.path
import glob
import numpy as np
import pdb
import time
import sys

from PIL import Image
import scipy.io as sio

from .dataset_utils import imread

def read_kitti_disp(disp_path):
    disp = Image.open(disp_path)
    disp = np.ascontiguousarray(disp, dtype=np.float32) / 256
    return disp[:, :, np.newaxis]

class WarpDispRefineKITTI2015(data.Dataset):
    def __init__(self, path_list, transform=None, co_transform=None):
        super(WarpDispRefineKITTI2015, self).__init__()

        self.path_list = path_list

        self.disp_loader = read_kitti_disp
        self.transform = transform
        self.co_transform = co_transform

        # np.random.seed(0)
        # np.random.shuffle(self.path_list)

    def __len__(self):
        return len(self.path_list)

    def __getitem__(self, index):
        im_path, flow0_occ_path, first_disp_path, ref_disp_path, gt_ref_disp_path = self.path_list[index]
        
        im = imread(im_path)
        flow_occ = imread(flow0_occ_path, 0)
        if first_disp_path.endswith('.png'):
            first_disp = self.disp_loader(first_disp_path)
        elif first_disp_path.endswith('.mat'):
            first_disp_data = sio.loadmat(first_disp_path)
            first_disp = first_disp_data['disp'][:, :, np.newaxis]
        else:
            raise Exception('Not supported format for disparity: {}'.format(first_disp_path[-4:]))

        if ref_disp_path.endswith('.png'):
            ref_disp = self.disp_loader(ref_disp_path)
        elif ref_disp_path.endswith('.mat'):
            ref_disp_data = sio.loadmat(ref_disp_path)
            ref_disp = ref_disp_data['disp'][:, :, np.newaxis]
        else:
            raise Exception('Not supported format for disparity: {}'.format(first_disp_path[-4:]))

        gt_ref_disp = self.disp_loader(gt_ref_disp_path)

        if self.transform is not None:
            im = self.transform(im)

        data = [im, flow_occ, first_disp, ref_disp, gt_ref_disp]
        if self.co_transform is not None:
            data, _ = self.co_transform(data, None)

        im, flow_occ, first_disp, ref_disp, gt_ref_disp = data

        im = torch.from_numpy(im.transpose(2, 0, 1)).float()
        flow_occ = torch.from_numpy(flow_occ[np.newaxis, :, :]).float()
        first_disp = torch.from_numpy(first_disp.transpose(2, 0, 1)).float() / 20
        ref_disp = torch.from_numpy(ref_disp.transpose(2, 0, 1)).float() / 20
        gt_ref_disp = torch.from_numpy(gt_ref_disp.transpose(2, 0, 1)).float() / 20

        return im, flow_occ, first_disp, ref_disp, gt_ref_disp