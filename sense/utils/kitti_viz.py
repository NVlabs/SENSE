"""
Copyright (C) 2019 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""
import numpy as np
import cv2

def disp_map(disp):
    color_map = np.array([
        [0, 0, 0, 114],
        [0, 0, 1, 185],
        [1, 0, 0, 114],
        [1, 0, 1, 174],
        [0, 1, 0, 114],
        [0, 1, 1, 185],
        [1, 1, 0, 114],
        [1, 1, 1, 0]
    ])

    bins = color_map[:-1, 3]
    cbins = np.cumsum(bins)
    bins = bins / cbins[-1]
    cbins = cbins[:-1] / cbins[-1]

    I = disp.reshape(1, -1)
    ind = np.tile(I, (6, 1)) > np.tile(cbins[:, np.newaxis], (1, I.size))
    ind = np.minimum(np.sum(ind, axis=0), 6).astype(int)
    bins = 1 / bins
    cbins = np.hstack(([0], cbins))
    I = (I - cbins[ind]) * bins[ind]
    I = I.transpose()
    I = color_map[ind, :3] * np.tile(1-I, (1, 3)) + color_map[ind+1, :3] * np.tile(I, (1, 3))
    I = np.reshape(I, (disp.shape[0], disp.shape[1], 3))
    I = np.clip(I, 0, 1)
    return I

def disp_to_color(disp, max_disp=None):
    if max_disp is None:
        max_disp = np.amax(disp)

    im = disp_map(disp / max_disp)
    # BGR order, consistent with OpenCV
    im = im[:, :, ::-1]
    return (im * 255).astype(np.uint8)
    
def viz_flow(u,v,logscale=True,scaledown=6,output=False):
    """
    topleft is zero, u is horiz, v is vertical
    red is 3 o'clock, yellow is 6, light blue is 9, blue/purple is 12
    """
    colorwheel = makecolorwheel()
    ncols = colorwheel.shape[0]

    radius = np.sqrt(u**2 + v**2)
    if output:
        print("Maximum flow magnitude: %04f" % np.max(radius))
    if logscale:
        radius = np.log(radius + 1)
        if output:
            print("Maximum flow magnitude (after log): %0.4f" % np.max(radius))
    radius = radius / scaledown    
    if output:
        print("Maximum flow magnitude (after scaledown): %0.4f" % np.max(radius))
    rot = np.arctan2(-v, -u) / np.pi

    fk = (rot+1)/2 * (ncols-1)  # -1~1 maped to 0~ncols
    k0 = fk.astype(np.uint8)       # 0, 1, 2, ..., ncols

    k1 = k0+1
    k1[k1 == ncols] = 0

    f = fk - k0

    ncolors = colorwheel.shape[1]
    img = np.zeros(u.shape+(ncolors,))
    for i in range(ncolors):
        tmp = colorwheel[:,i]
        col0 = tmp[k0]
        col1 = tmp[k1]
        col = (1-f)*col0 + f*col1
       
        idx = radius <= 1
        # increase saturation with radius
        col[idx] = 1 - radius[idx]*(1-col[idx])
        # out of range    
        col[~idx] *= 0.75
        img[:,:,i] = np.floor(255*col).astype(np.uint8)
    
    return img.astype(np.uint8)
    
def flow_to_color(flow):
    return viz_flow(flow[:, :, 0], flow[:, :, 1])
    
def makecolorwheel():
    # Create a colorwheel for visualization
    RY = 15
    YG = 6
    GC = 4
    CB = 11
    BM = 13
    MR = 6
    
    ncols = RY + YG + GC + CB + BM + MR
    
    colorwheel = np.zeros((ncols,3))
    
    col = 0
    # RY
    colorwheel[0:RY,0] = 1
    colorwheel[0:RY,1] = np.arange(0,1,1./RY)
    col += RY
    
    # YG
    colorwheel[col:col+YG,0] = np.arange(1,0,-1./YG)
    colorwheel[col:col+YG,1] = 1
    col += YG
    
    # GC
    colorwheel[col:col+GC,1] = 1
    colorwheel[col:col+GC,2] = np.arange(0,1,1./GC)
    col += GC
    
    # CB
    colorwheel[col:col+CB,1] = np.arange(1,0,-1./CB)
    colorwheel[col:col+CB,2] = 1
    col += CB
    
    # BM
    colorwheel[col:col+BM,2] = 1
    colorwheel[col:col+BM,0] = np.arange(0,1,1./BM)
    col += BM
    
    # MR
    colorwheel[col:col+MR,2] = np.arange(1,0,-1./MR)
    colorwheel[col:col+MR,0] = 1

    return colorwheel 

def flow_write(flow_path, flow):
    imh, imw, imc = flow.shape
    assert imc == 2, 'Incorrect input channel: {}. Must be 2.'.format(imc)
    flow_im = np.ones((imh, imw, 3))
    flow_im[:, :, 0] = np.maximum(np.minimum(flow[:, :, 0] * 64 + 2 ** 15, 2 ** 16 - 1), 0)
    flow_im[:, :, 1] = np.maximum(np.minimum(flow[:, :, 1] * 64 + 2 ** 15, 2 ** 16 - 1), 0)
    cv2.imwrite(flow_path, flow_im[:, :, ::-1].astype(np.uint16))

def disp_write(disp_path, disp):
    disp_im = (disp * 256).astype(np.uint16)
    cv2.imwrite(disp_path, disp_im)