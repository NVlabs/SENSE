"""
Copyright (C) 2019 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""
import numpy as np
import matplotlib.colors as color
from scipy.misc import imread, imsave
import scipy.io as sio

TAG_CHAR = 'PIEH'

def read_flow(filename):
    """
    read optical flow in Middlebury .flo file format
    :param filename:
    :return:
    """
    f = open(filename, 'rb')
    magic = np.fromfile(f, np.float32, count=1)
    data2d = None
    if 202021.25 != magic:
        print('Magic number incorrect. Invalid .flo file')
    else:
        w = np.fromfile(f, np.int32, count=1)[0]
        h = np.fromfile(f, np.int32, count=1)[0]
        data2d = np.fromfile(f, np.float32, count=2 * w * h)
        # reshape data into 3D array (columns, rows, channels)
        data2d = np.reshape(data2d, (h, w, 2))
    f.close()
    return data2d

def write_flow(flow, filename):
    """
    write optical flow in Middlebury .flo format
    :param flow: optical flow map
    :param filename: optical flow file path to be saved
    :return: None
    """
    f = open(filename, 'wb')
    magic = np.array([202021.25], dtype=np.float32)
    (height, width, c) = flow.shape
    assert c == 2, c
    w = np.array([width], dtype=np.int32)
    h = np.array([height], dtype=np.int32)
    # empty_map = np.zeros((height, width), dtype=np.float32)
    # data = np.dstack((flow, empty_map))
    data = flow
    magic.tofile(f)
    w.tofile(f)
    h.tofile(f)
    data.tofile(f)
    f.close()

def flow_visualize(flow, max_range = 1e3):
    du = flow[:,:,0]
    dv = flow[:,:,1]
    [h,w] = du.shape
    max_flow = min(max_range, np.max(np.sqrt(du * du + dv * dv)))
    img = np.ones((h, w, 3), dtype=np.float64)
    # angle layer
    img[:, :, 0] = (np.arctan2(dv, du) / (2 * np.pi) + 1) % 1.0
    # magnitude layer, normalized to 1
    img[:, :, 1] = np.sqrt(du * du + dv * dv) / (max_flow + 1e-8)
    # phase layer
    #img[:, :, 2] = valid
    # convert to rgb
    img = color.hsv_to_rgb(img)
    # remove invalid point
    img[:, :, 0] = img[:, :, 0]
    img[:, :, 1] = img[:, :, 1]
    img[:, :, 2] = img[:, :, 2]
    return img

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

def read_camera_data(filepath):
    filedata = read_calib_file(filepath)

    # Create 3x4 projection matrices
    P_rect_20 = np.reshape(filedata['P_rect_02'], (3, 4))
    P_rect_30 = np.reshape(filedata['P_rect_03'], (3, 4))
    K2 = P_rect_20[0:3, 0:3]
    K3 = P_rect_30[0:3, 0:3]
    return K2, K3

def unzip_gz(path):
    if not osp.exists(path):
        with gzip.open(path+'.gz', 'rb') as f_in: 
            with open(path, 'wb') as f_out:
                shutil.copyfileobj(f_in, f_out)

def read_disp_gen(path):
    """ read the generated disparity image
    """
    # unzip_gz(path)
    disparity = sio.loadmat(path)['disp']
    return disparity

def write_disp(disp, path):
    sio.savemat(path, {'disp': disp})

def read_flow_gen(path):
    """ read the generated flow image
    """
    # unzip_gz(path)
    return read_flow(path)

def read_seg_gen(path):
    """ read the generated segmentation image
    """
    # unzip_gz(path)
    return imread(path)

def read_occ_gen(path):
    """ read the generated occlusion image 
    """
    # unzip_gz(path)
    return imread(path) / 255.0
