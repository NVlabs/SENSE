"""
Copyright (C) 2019 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""
import os
import numpy as np
import re

from PIL import Image
import cv2

import time
import sys

from joblib import Parallel, delayed

import sense.datasets.sintel_io as sintel_io

def imread(im_path, flag=1):
    im = cv2.imread(im_path, flag)
    im = im.astype(np.float32) / 255.0
    return im

def load_pfm(filename):
    '''
    Load a PFM file into a Numpy array. Note that it will have
    a shape of H x W, not W x H. Returns a tuple containing the
    loaded image and the scale factor from the file.
    '''
    color = None
    width = None
    height = None
    scale = None
    endian = None

    with open(filename, 'rb') as f:
        header = f.readline().rstrip()
        if header == b'PF':
            color = True    
        elif header == b'Pf':
            color = False
        else:
            print('header: ', header)
            raise Exception('Not a PFM file: {}.'.format(filename))

        dim_match = re.match(r'^(\d+)\s(\d+)\s$', f.readline().decode('utf-8'))
        if dim_match:
            width, height = map(int, dim_match.groups())
        else:
            raise Exception('Malformed PFM header.')

        scale = float(f.readline().rstrip().decode('utf-8'))
        if scale < 0: # little-endian
            endian = '<'
            scale = -scale
        else:
            endian = '>' # big-endian

        data = np.fromfile(f, endian + 'f')
        shape = (height, width, 3) if color else (height, width)

    assert scale==1, 'pfm scale is not 1'
    data = np.flipud(np.reshape(data, shape)).copy()
    return  data, scale

def load_flo(path):
    with open(path, 'rb') as f:
        magic = np.fromfile(f, np.float32, count=1)
        assert(202021.25 == magic),'Magic number incorrect. Invalid .flo file'
        h = np.fromfile(f, np.int32, count=1)[0]
        w = np.fromfile(f, np.int32, count=1)[0]
        data = np.fromfile(f, np.float32, count=2*w*h)
    # Reshape data into 3D array (columns, rows, bands)
    data2D = np.resize(data, (w, h, 2))

    return data2D

def read_flow_pfm(path):
    flow, _ = load_pfm(path)
    return flow[:, :, :2]

def read_disp_pfm(path):
    disp, _ = load_pfm(path)
    return np.abs(disp[:, :, np.newaxis])

def sintel_flow_loader(root, path_imgs, path_flo):
    imgs = [os.path.join(root,path) for path in path_imgs]
    flo = os.path.join(root,path_flo)
    im_all = [imread(img) for img in imgs]
    return [im[:, :, :3] for im in im_all], load_flo(flo) 

def optical_flow_loader(root, path_imgs, target_paths):
    imgs = [os.path.join(root,path) for path in path_imgs]
    path_flo, path_flo_occ = target_paths
    flo = os.path.join(root,path_flo)
    # start = time.time()
    im_all = [imread(img) for img in imgs]
    im_all = [im[:, :, :3] for im in im_all]
    # im_load_time = time.time() - start
    # start = time.time()
    if flo.endswith('.pfm'):
        flow = read_flow_pfm(flo)
    elif flo.endswith('.flo'):
        flow = load_flo(flo)
    else:
        raise Exception('Unsupported flow format: {}.'.format(flo))
    # fl_load_time = time.time() - start
    # print('{:>.15f}\t'
    #       '{:>.15f}'.format(
    #         im_load_time,
    #         fl_load_time
    #         ))
    # sys.stdout.flush()
    if path_flo_occ is None:
        imh, imw, _ = im_all[0].shape
        flow_occ = -np.ones((imh, imw), dtype=np.int64)
    else:
        path_flo_occ = os.path.join(root, path_flo_occ)
        flow_occ = cv2.imread(path_flo_occ, 0)
        # flow_occ = (flow_occ > 128).astype(np.float32)
        flow_occ = flow_occ.astype(np.float32) / 255.0
    return im_all, [flow, flow_occ]

def sceneflow_disp_loader(root, path_imgs, path_targets):
    imgs = [os.path.join(root,path) for path in path_imgs]
    im_all = [imread(img) for img in imgs]

    path_disp, path_disp_occ = path_targets
    path_disp = os.path.join(root, path_disp)
    disp = read_disp_pfm(path_disp) 
    if path_disp_occ is None:
        imh, imw, _ = im_all[0].shape
        disp_occ = -np.ones((imh, imw), dtype=np.int64)
    else:
        path_disp_occ = os.path.join(root, path_disp_occ)
        disp_occ = cv2.imread(path_disp_occ, 0)
        # disp_occ = (disp_occ > 128).astype(np.float32)
        disp_occ = disp_occ.astype(np.float32) / 255.0
    seg_im = np.array([255])
    return [im[:, :, :3] for im in im_all], [disp, disp_occ, seg_im]

def sintel_disp_loader(root, path_imgs, target_paths):
    imgs = [os.path.join(root,path) for path in path_imgs]
    path_disp, path_disp_occ = target_paths
    disp = os.path.join(root,path_disp)
    # start = time.time()
    im_all = [imread(img) for img in imgs]
    im_all = [im[:, :, :3] for im in im_all]
    # im_load_time = time.time() - start
    # start = time.time()
    disp = sintel_io.disparity_read(path_disp)[:, :, np.newaxis]
    # fl_load_time = time.time() - start
    # print('{:>.15f}\t'
    #       '{:>.15f}'.format(
    #         im_load_time,
    #         fl_load_time
    #         ))
    # sys.stdout.flush()
    if path_disp_occ is None:
        imh, imw, _ = im_all[0].shape
        disp_occ = -np.ones((imh, imw), dtype=np.int64)
    else:
        path_disp_occ = os.path.join(root, path_disp_occ)
        disp_occ = cv2.imread(path_disp_occ, 0)
        # disp_occ = (disp_occ > 128).astype(np.float32)
        disp_occ = disp_occ.astype(np.float32) / 255.0
    return im_all, [disp, disp_occ]

def sintel_disp_seg_loader(root, path_imgs, target_paths):
    imgs = [os.path.join(root,path) for path in path_imgs]
    if len(target_paths) == 3:
        path_disp, path_disp_occ, seg_gt_path = target_paths
    elif len(target_paths) == 2:
        path_disp, path_disp_occ = target_paths
        seg_gt_path = None
    disp = os.path.join(root,path_disp)
    # start = time.time()
    im_all = [imread(img) for img in imgs]
    im_all = [im[:, :, :3] for im in im_all]
    # im_load_time = time.time() - start
    # start = time.time()
    disp = sintel_io.disparity_read(path_disp)[:, :, np.newaxis]
    # fl_load_time = time.time() - start
    # print('{:>.15f}\t'
    #       '{:>.15f}'.format(
    #         im_load_time,
    #         fl_load_time
    #         ))
    # sys.stdout.flush()
    if path_disp_occ is None:
        imh, imw, _ = im_all[0].shape
        disp_occ = -np.ones((imh, imw), dtype=np.int64)
    else:
        path_disp_occ = os.path.join(root, path_disp_occ)
        disp_occ = cv2.imread(path_disp_occ, 0)
        # disp_occ = (disp_occ > 128).astype(np.float32)
        disp_occ = disp_occ.astype(np.float32) / 255.0
    if seg_gt_path is not None:
        seg_im = cv2.imread(seg_gt_path, 0)
    else:
        seg_im = np.array([255])
    return [im[:, :, :3] for im in im_all], [disp, disp_occ, seg_im]

def kitti_disp_loader(root, im_paths, target_paths):
    disp_path, disp_occ_path = target_paths
    disp = Image.open(disp_path)
    disp = np.ascontiguousarray(disp, dtype=np.float32) / 256
    if disp_occ_path is None:
        disp_occ = -np.ones(disp.shape)
    else:
        disp_occ = cv2.imread(disp_occ_path, 0)
        # disp_occ = (disp_occ > 128).astype(np.float32)
        disp_occ = disp_occ.astype(np.float32) / 255.0
    im_all = [imread(n) for n in im_paths]
    return [im[:, :, :3] for im in im_all], [disp[:, :, np.newaxis], disp_occ]

def kitti_disp_seg_loader(root, im_paths, target_paths):
    if len(target_paths) == 3:
        disp_path, disp_occ_path, seg_gt_path = target_paths
    elif len(target_paths) == 2:
        disp_path, disp_occ_path = target_paths
        seg_gt_path = None
    disp = Image.open(disp_path)
    disp = np.ascontiguousarray(disp, dtype=np.float32) / 256
    if disp_occ_path is None:
        disp_occ = -np.ones(disp.shape)
    else:
        disp_occ = cv2.imread(disp_occ_path, 0)
        # disp_occ = (disp_occ > 128).astype(np.float32)
        disp_occ = disp_occ.astype(np.float32) / 255.0
    im_all = [imread(n) for n in im_paths]
    if seg_gt_path is not None:
        seg_im = cv2.imread(seg_gt_path, 0)
    else:
        seg_im = np.array([255])
    return [im[:, :, :3] for im in im_all], [disp[:, :, np.newaxis], disp_occ, seg_im]

def read_vkitti_flow(flow_path):
    "Convert from .png to (h, w, 2) (flow_x, flow_y) float32 array"
    # read png to bgr in 16 bit unsigned short
    bgr = cv2.imread(flow_path, cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
    h, w, _c = bgr.shape
    assert bgr.dtype == np.uint16 and _c == 3
    # b == invalid flow flag == 0 for sky or other invalid flow
    invalid = bgr[..., 0] == 0
    # g,r == flow_y,x normalized by height,width and scaled to [0;2**16 - 1]
    out_flow = 2.0 / (2**16 - 1.0) * bgr[..., ::-1].astype('f4') - 1
    out_flow[..., 0] *= w - 1
    out_flow[..., 1] *= h - 1
    out_flow[..., 2] = 1 - invalid.astype('f4')  # or another value (e.g., np.nan)
    return out_flow

def read_kitti_flow_raw(flow_path):
    I = cv2.imread(flow_path, -1).astype(np.float64)

    F_u = (I[:,:,2]-2**15) / 64.0
    F_v = (I[:,:,1]-2**15) / 64.0
    F_valid = np.minimum(I[:,:,0], 1)
    idxes = np.where(F_valid == 0)
    F_u[idxes] = 0
    F_v[idxes] = 0
    F = np.zeros((F_u.shape[0], F_u.shape[1], 3), np.double)
    F[:,:,0] = F_u
    F[:,:,1] = F_v
    F[:,:,2] = F_valid
    return F

def read_kitti_flow(flow_path):
    I = cv2.imread(flow_path, -1).astype(np.float64)

    F_u = (I[:,:,2]-2**15) / 64.0
    F_v = (I[:,:,1]-2**15) / 64.0
    F_valid = np.minimum(I[:,:,0], 1)
    idxes = np.where(F_valid == 0)
    F_u[idxes] = np.nan
    F_v[idxes] = np.nan
    F = np.zeros((F_u.shape[0], F_u.shape[1], 2), np.double)
    F[:,:,0] = F_u
    F[:,:,1] = F_v
    # F[:,:,2] = F_valid
    return F

def kitti_flow_loader(root, im_paths, target_paths):
    flow_path, flow_occ_path = target_paths
    flow = read_kitti_flow(flow_path)
    flow = np.ascontiguousarray(flow, dtype=np.float32)
    if flow_occ_path is None:
        flow_occ = -np.ones(flow.shape[:2])
    else:
        flow_occ = cv2.imread(flow_occ_path, 0)
        # flow_occ = (flow_occ > 128).astype(np.float32)
        flow_occ = flow_occ.astype(np.float32) / 255.0
    im_all = [imread(n) for n in im_paths]
    return [im[:, :, :3] for im in im_all], [flow, flow_occ]

def pose_loader(root, im_paths, target):
    im_all = [imread(n) for n in im_paths]
    return im_all, target

def pose_crop_im_loader(root, im_paths, target):
    im_all = [imread(n)[:780, :, :] for n in im_paths]
    return im_all, target

def is_good_flow(flow_path, thresh=500):
    if flow_path.endswith('.pfm'):
        flow, scale = load_pfm(flow_path)
    elif flow_path.endswith('.flo'):
        flow = load_flo(flow_path)
    else:
        raise Exception('Unsupported flow format: {}'.format(flow_path[-4:]))
    if np.any(np.isnan(flow)) or np.any(np.isinf(flow)):
        return False
    max_flow = np.max(np.abs(flow))
    if max_flow > thresh:
        return False
    return True

def remove_flow_outliers(data, flow_thresh):
    input_paths, target_paths = data
    flags = Parallel(n_jobs=32)(delayed(is_good_flow)(tp[0], flow_thresh) for tp in target_paths)
    ret_input_paths = []
    ret_target_paths = []
    for i, f in enumerate(flags):
        if f:
            ret_input_paths.append(input_paths[i])
            ret_target_paths.append(target_paths[i])
    return ret_input_paths, ret_target_paths

def remove_disp_outliers(data, disp_thresh):
    input_paths, disp_paths = data
    # from joblib import Parallel, delayed
    # flags = Parallel(n_jobs=32)(delayed(is_good_disparity)(fp, disp_thresh) for fp in disp_paths)
    # ret_left_im_paths = []
    # ret_right_im_paths = []
    # ret_disp_paths = []
    # for i, f in enumerate(flags):
    #     if f:
    #         ret_left_im_paths.append(left_im_paths[i])
    #         ret_right_im_paths.append(right_im_paths[i])
    #         ret_disp_paths.append(disp_paths[i])
    ret_input_paths = input_paths
    ret_disp_paths = disp_paths
    return ret_input_paths, ret_disp_paths

def remove_flow_disp_outliers(data, flow_thresh):
    input_paths, target_paths = data
    flags = Parallel(n_jobs=32)(delayed(is_good_flow)(fp[0], flow_thresh) for fp in target_paths)
    ret_input_paths = []
    ret_target_paths = []
    for idx, f in enumerate(flags):
        if f:
            ret_input_paths.append(input_paths[idx])
            ret_target_paths.append(target_paths[idx])
    return ret_input_paths, ret_target_paths