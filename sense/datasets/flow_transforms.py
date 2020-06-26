"""
Copyright (C) 2019 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""
from __future__ import division
import torch
import math
import random
from PIL import Image, ImageOps
import numpy as np
import numbers
import types
import scipy.ndimage as ndimage
import pdb
import cv2


'''Set of tranform random routines that takes both input and target as arguments,
in order to have random but coherent transformations.
inputs are PIL Image pairs and targets are ndarrays'''

class Compose(object):
    """ Composes several co_transforms together.
    For example:
    >>> co_transforms.Compose([
    >>>     co_transforms.CenterCrop(10),
    >>>     co_transforms.ToTensor(),
    >>>  ])
    """
    def __init__(self, co_transforms):
        self.co_transforms = co_transforms

    def __call__(self, input, target):
        for t in self.co_transforms:
            input,target = t(input,target)
        return input,target

class ArrayNormalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, im_array):
        for i in range(len(im_array)):
            for c in range(2):
                im_array[i][:, :, c] = (im_array[i][:, :, c] - self.mean[c]) / self.std[c]
        return im_array

class Normalize(object):
    """Given mean: (R, G, B) and std: (R, G, B),
    will normalize each channel of the torch.*Tensor, i.e.
    channel = (channel - mean) / std
    """

    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, flow_data):
        flow, flow_occ = flow_data
        # image = (image - self.mean) / self.std
        assert flow.shape[0] == 2 or flow.shape[0] == 3, flow.shape
        for c in range(2):
            flow[c,:,:] = (flow[c,:,:] - self.mean[c]) / self.std[c]
        return [flow, flow_occ]

class ArrayToTensor(object):
    """Converts a numpy.ndarray (H x W x C) to a torch.FloatTensor of shape (C x H x W)."""
    def __call__(self, array):
        if isinstance(array, np.ndarray):
            if len(array.shape) == 3:
                array = np.transpose(array, (2, 0, 1))
            # handle numpy array
            tensor = torch.from_numpy(array)
            # put it from HWC to CHW format
            return tensor.float()

        
        if type(array) is list:
            array_tensor = []
            for i in range(len(array)):
                assert isinstance(array[i], np.ndarray), type(array[i])
                if len(array[i].shape) == 3:
                    array[i] = np.transpose(array[i], (2, 0, 1))
                array_tensor.append(torch.from_numpy(array[i]).float())
            return array_tensor

        raise Exception('Unsupported array type: {}'.format(type(array)))  

class ArrayToTensorBatch(object):
    """Converts a numpy.ndarray (H x W x C) to a torch.FloatTensor of shape (C x H x W)."""
    def __call__(self, array):
        assert(isinstance(array, np.ndarray))
        array = np.transpose(array, (0, 3, 1, 2))
        # handle numpy array
        tensor = torch.from_numpy(array)
        # put it from HWC to CHW format
        return tensor.float()


class ArrayToTensor_all(object):
    """Converts a numpy.ndarray (H x W x C) to a torch.FloatTensor of shape (C x H x W)."""
    def __call__(self, inputs):
        for i, _ in enumerate(inputs):
            inputs[i] = np.transpose(inputs[i], (2, 0, 1))
            inputs[i] = torch.from_numpy(inputs[i])
            inputs[i].float()

        return inputs




class Lambda(object):
    """Applies a lambda as a transform"""
    def __init__(self, lambd):
        assert type(lambd) is types.LambdaType
        self.lambd = lambd

    def __call__(self, input,target):
        return self.lambd(input,target)

class CenterCrop(object):
    """Crops the given inputs and target arrays at the center to have a region of
    the given size. size can be a tuple (target_height, target_width)
    or an integer, in which case the target will be of a square shape (size, size)
    Careful, img1 and img2 may not be the same size
    """
    def __init__(self, size):
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            self.size = size

    def __call__(self, inputs, targets=None):
        h1, w1, _ = inputs[0].shape
        # h2, w2, _ = inputs[1].shape
        th, tw = self.size
        x1 = int(round((w1 - tw) / 2.))
        y1 = int(round((h1 - th) / 2.))
        # print x1, y1, h1, w1, th, tw
        # x2 = int(round((w2 - tw) / 2.))
        # y2 = int(round((h2 - th) / 2.))
        for i, _ in enumerate(inputs):
            inputs[i] = inputs[i][y1 : y1 + th, x1 : x1 + tw]

        if targets is None:
            return inputs, None

        if type(targets) is list:
            for i, tgt in enumerate(targets):
                if len(tgt.shape) >= 2:
                    targets[i] = targets[i][y1 : y1 + th, x1 : x1 + tw]
                else:
                    assert i == len(targets) - 1, 'Something is seriously wrong in random crop.'
        else:
            targets = targets[y1 : y1 + th, x1 : x1 + tw]
        return inputs,targets

class CenterCropByFactor(object):
    """Crops the given inputs and target arrays at the center to have a region of
    the given size. size can be a tuple (target_height, target_width)
    or an integer, in which case the target will be of a square shape (size, size)
    Careful, img1 and img2 may not be the same size
    """
    def __init__(self, factor):
        self.factor = factor

    def __call__(self, inputs, targets):
        h1, w1, _ = inputs[0].shape
        # h2, w2, _ = inputs[1].shape
        th, tw = int(self.factor*h1), int(self.factor*w1)
        x1 = int(round((w1 - tw) / 2.))
        y1 = int(round((h1 - th) / 2.))
        # print x1, y1, h1, w1, th, tw
        # x2 = int(round((w2 - tw) / 2.))
        # y2 = int(round((h2 - th) / 2.))
        scaled_inputs = np.zeros((inputs.shape[0], th, tw, inputs.shape[3]))

        for i, _ in enumerate(inputs):
            scaled_inputs[i] = inputs[i][y1 : y1 + th, x1 : x1 + tw]

        if type(targets) is list:
            for i, _ in enumerate(targets):
                targets[i] = targets[i][y1 : y1 + th, x1 : x1 + tw]
            return scaled_inputs, targets
        else:
            scaled_targets = np.zeros((th, tw, targets.shape[2]))
            scaled_targets = targets[y1 : y1 + th, x1 : x1 + tw]
            return scaled_inputs, scaled_targets

class RandomCrop(object):
    """Crops the given PIL.Image at a random location to have a region of
    the given size. size can be a tuple (target_height, target_width)
    or an integer, in which case the target will be of a square shape (size, size)
    """
    def __init__(self, size):
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            self.size = size

    def __call__(self, inputs, targets=None):
        h, w, _ = inputs[0].shape
        th, tw = self.size
        if w == tw and h == th:
            return inputs,targets

        x1 = random.randint(0, w - tw)
        y1 = random.randint(0, h - th)

        for i, _ in enumerate(inputs):
            if len(inputs[i].shape) >= 2:
                inputs[i] = inputs[i][y1 : y1 + th, x1 : x1 + tw]

        if targets is None:
            return inputs, None

        if type(targets) is list:
            for i, tgt in enumerate(targets):
                if len(tgt.shape) >= 2:
                    targets[i] = targets[i][y1 : y1 + th, x1 : x1 + tw]
                else:
                    assert i == len(targets) - 1, 'Something is seriously wrong in random crop.'
        else:
            targets = targets[y1 : y1 + th, x1 : x1 + tw]

        return inputs, targets

# modified by Huaizu Jiang
# class RandomHorizontalFlip(object):
#     """Randomly horizontally flips the given PIL.Image with a probability of 0.5
#     """
#     def __call__(self, inputs):
#         if random.random() < 0.5:
#             inputs[0] = np.fliplr(inputs[0]).copy()
#             inputs[1] = np.fliplr(inputs[1]).copy()
#         return inputs

class RandomVerticalFlip(object):
    """Randomly horizontally flips the given PIL.Image with a probability of 0.5
    """
    def __call__(self, inputs, target=None):
        prob = random.random()
        if prob < 0.5:
            for i, _ in enumerate(inputs):
                if len(inputs[i].shape) >= 3:
                    inputs[i] = np.flipud(inputs[i]).copy()

        if target is None:
            return inputs, None

        assert type(target) is list, 'horizontal flip should be used for disparity gt only.'
        assert len(target) == 3 or len(target) == 2, \
            'horizontal flip should be used for disparity gt only. {}'.format(len(target))
        if prob < 0.5:
            for i in range(len(target)):
                if len(target[i].shape) >= 2:
                        target[i] = np.flipud(target[i]).copy()
                else:
                    assert i == len(target) - 1, 'Something is seriously wrong in horizontal flip.'
        return inputs, target

class ResizeSeg(object):
    def __init__(self, resize_ratio=None):
        self.resize_ratio = resize_ratio

    def __call__(self, inputs, targets):
        if self.resize_ratio is not None:
            assert len(targets) == 3, 'Something is seriously wrong in resizing segmentation gt.'
            targets[-1] = cv2.resize(targets[-1], None, None,
                fx=self.resize_ratio, fy=self.resize_ratio, interpolation=cv2.INTER_NEAREST)
        return inputs, targets

# added by Huaizu Jiang
class Resize(object):
    def __init__(self, resize_ratio=None):
        self.resize_ratio = resize_ratio

    def __call__(self, inputs, targets=None):
        if self.resize_ratio is not None:
            resize_ratio = self.resize_ratio
            for i, _ in enumerate(inputs):
                inputs[i] = cv2.resize(inputs[i], None, None,
                    fx=resize_ratio, fy=resize_ratio)

        if targets is None:
            return inputs

        uvt, uv_occ = targets
        if self.resize_ratio is not None:
            # assert len(uvt.shape) == 2
            uvt = cv2.resize(uvt, None, None, fx=resize_ratio, fy=resize_ratio) * resize_ratio
            uv_occ = cv2.resize(uv_occ, None, None,
                fx=resize_ratio, fy=resize_ratio, interpolation=cv2.INTER_NEAREST)
        return inputs, [uvt, uv_occ]



class RandomGammaImg(object):
    def __init__(self, val_range, p=-1):
        self.val_range = val_range
        self.p = p

    def __call__(self, inputs):
        if np.random.random() > self.p:
            value = random.uniform(self.val_range[0], self.val_range[1])
            # print "gamma: ", value
            if type(inputs) is list:
                for i, _ in enumerate(inputs):
                    inputs[i] = pow(inputs[i], value)
            else:
                inputs = pow(inputs, value)
        return inputs


class RandomBrightnessImg(object):
    def __init__(self, val_sigma, p=-1):
        self.val_sigma = val_sigma
        self.p = p

    def __call__(self, inputs):
        if np.random.random() > self.p:
            value = np.random.normal(0, self.val_sigma, 1)
            # print "brightness: ", value
            if type(inputs) is list:
                for i, _ in enumerate(inputs):
                    inputs[i] = np.clip(0.0 + inputs[i] + value[0], 0.0, 1.0)
            else:
                inputs = np.clip(0.0 + inputs + value[0], 0.0, 1.0)
        return inputs

class RandomContrastImg(object):
    def __init__(self, val_range, p=-1):
        self.val_range = val_range
        self.p = p

    def __call__(self, inputs):
        if np.random.random() > self.p:
            value = random.uniform(self.val_range[0], self.val_range[1])
            # print "contrast: ", value
            if type(inputs) is list:
                for i, _ in enumerate(inputs):
                    for c in range(0, 3):
                        cur_mean = inputs[i][:, :, c].mean()
                        inputs[i][:, :, c] = np.clip(1.0*(inputs[i][:, :, c] - cur_mean)*value+cur_mean, 0.0, 1.0)
            else:
                for c in range(0, 3):
                    cur_mean = inputs[:, :, c].mean()
                    inputs[:, :, c] = np.clip(1.0*(inputs[:, :, c] - cur_mean)*value+cur_mean, 0.0, 1.0)
        return inputs

class RandomGaussianNoiseImg(object):
    def __init__(self, val_sigma, p=-1):
        self.val_sigma = val_sigma
        self.p = p

    def __call__(self, inputs):
        if np.random.random() > self.p:
            value = random.uniform(0, self.val_sigma)
            if type(inputs) is list:
                for i, _ in enumerate(inputs):
                    noise = np.random.normal(0, value, inputs[i].shape)
                    inputs[i] = np.clip(0.0 + inputs[i] + noise, 0.0, 1.0)
            else:
                noise = np.random.normal(0, value, inputs.shape)
                inputs = np.clip(0.0 + inputs + noise, 0.0, 1.0)
        return inputs


class SubstractVal(object):
    def __init__(self, val):
        self.val = val

    def __call__(self, inputs):
        if type(inputs) is list:
            for i, _ in enumerate(inputs):
                inputs[i] = inputs[i] - self.val

        else:
            inputs = inputs - self.val
        return inputs

class SubstractVal_caffe(object):
    def __init__(self):
        pass


    def __call__(self, im):
        if type(im) is list:
            for _i, _ in enumerate(im):
                im[_i] = im[_i][:, :, ::-1].copy()
                im[_i][:, :, 0] = im[_i][:, :, 0] - 0.412
                im[_i][:, :, 1] = im[_i][:, :, 1] - 0.435
                im[_i][:, :, 2] = im[_i][:, :, 2] - 0.456

        else:
            im = im[:, :, ::-1].copy()
            im[:, :, 0] = im[:, :, 0] - 0.412
            im[:, :, 1] = im[:, :, 1] - 0.435
            im[:, :, 2] = im[:, :, 2] - 0.456
        return im


class colorChannelRev(object):
    def __init__(self):
        pass


    def __call__(self, im):
        if type(im) is list:
            for _i, _ in enumerate(im):
                im[_i] = im[_i][:, :, ::-1].copy()

        else:
            im = im[:, :, ::-1].copy()

        return im

class FlowDataAugmentation(object):
    def __init__(self, crop_size):
        self.crop_size = crop_size

    def __call__(self, im_all, uv_data):
        im_all_aug, uv_data_aug = augment_data(im_all, uv_data, self.crop_size)
        return im_all_aug, uv_data_aug

hor_flip_threshold = 0.5
ver_flip_threshold = 0.5

def augment_data(im_data, uv_data, crop_size = (320, 448), basis_x=None, basis_y=None):

    im_data_aug, uv_data_aug = spatial_transform(im_data, uv_data, crop_size, basis_x, basis_y)

    # #scale to [0,1] img2numpy: im/255 -0.5
    # im1 = im1 + 0.5
    # im2 = im2 + 0.5

    # # im1, params = transform_chromatic_eigen(im1)
    # # im2, params = transform_chromatic_eigen(im2, params) # same parameters for 1 & 2
    # # # small color perturbation to im2
    # # im2 = transform_chromatic(im2)

    # # nstd = np.random.uniform(0,0.04) # low, high, size
    # # im1 = im1 + np.random.normal(0., nstd, im1.shape)
    # # im2 = im2 + np.random.normal(0., nstd, im2.shape)

    # # recompute mean and subtract
    # im1 = im1 - 0.5
    # im2 = im2 - 0.5

    uvt, uv_occ = uv_data_aug

    if np.random.uniform() > hor_flip_threshold:
        # randomly flip horizontally
        # channel-height-width
        for i in range(len(im_data_aug)):
            if len(im_data_aug[i].shape) == 3:
                im_data_aug[i] = im_data_aug[i][:, ::-1, :]
        uvt = uvt[:, ::-1, :]
        uvt[:, :, 0] = -uvt[:, :, 0]
        uv_occ = uv_occ[:, ::-1]
    if np.random.uniform() > ver_flip_threshold:
        # randomly flip vertically
        for i in range(len(im_data_aug)):
            if len(im_data_aug[i].shape) == 3:
                im_data_aug[i] = im_data_aug[i][::-1, :, :]
        uvt = uvt[::-1, :, :]
        uvt[:, :, 1] = -uvt[:, :, 1]
        uv_occ = uv_occ[::-1, :]

    for i in range(len(im_data_aug)):
        im_data_aug[i] = np.array(im_data_aug[i], dtype=np.float32)
    uvt = np.array(uvt,dtype='float32')
    uv_occ = np.array(uv_occ, dtype='float32')
    return im_data_aug, [uvt, uv_occ]

def identity_matrix():
    return np.array([1,0, 0, 0,1,0])
def translation_matrix(t):
    return np.array([1,0,t[0], 0,1,t[1]])
def rotation_matrix(angle):
    return np.array([np.cos(angle), np.sin(angle), 0, -np.sin(angle), np.cos(angle), 0])
def zoom_matrix(zoom):  
    return np.array([1/zoom[0],0, 0, 0,1/zoom[1],0])

def left_multiply(R, L):
    # effect: apply transMat1 first and then apply transMat2
    # [L0 L1 L2;   [R0 R1 R2;
    #  L3 L4 L5; *  R3 R4 R5;
    #  0  0   1]    0  0   1] 
    return np.array([L[0]*R[0]+L[1]*R[3], L[0]*R[1]+L[1]*R[4], L[0]*R[2]+L[1]*R[5]+L[2], L[3]*R[0]+L[4]*R[3], L[3]*R[1]+L[4]*R[4], L[3]*R[2]+L[4]*R[5]+L[5] ])

def generate_good_coefficients(im1, crop_size, max_iters=50):
    height, width, channels = im1.shape
    des_height = crop_size[0]
    des_width  = crop_size[1]

    origin = np.array([0.5*width, 0.5*height])
    des_origin = np.array([0.5*des_width, 0.5*des_height])


    for iter in range(max_iters):
        # geometric transformation applied to both images

        # original version used for pre-training
        t     = np.random.uniform(-0.2,0.2, 2)
        t[0]  = t[0] * width
        t[1]  = t[1] * height

        angle = np.random.uniform(-0.2,0.2)
        zoom  = np.tile(np.exp(np.random.uniform(0,0.4)), (2,1)) #np.exp(np.random.uniform(0,0.4,2))
        squeeze  = np.exp(np.random.uniform(-0.15,0.15))
        zoom[0]  = zoom[0]*squeeze
        zoom[1]  = zoom[1]/squeeze

        # geometric transformation applied additional to the second image
        t2 = np.random.normal(0,0.03, 2)
        t2[0]  = t2[0] * width
        t2[1]  = t2[1] * height             
        angle2 = np.random.normal(0,0.03)
        zoom2  = np.exp(np.random.normal(0,0.03,2))
        

        # aug_prob = 0.5     

        # # reproducible version of Caffe version on KITTI
        # t     = np.random.uniform(-0.4, 0.4, 2)
        # t[0]  = t[0] * width
        # t[1]  = t[1] * height
        # if np.random.random() > aug_prob:
        #     t[0] = 0
        #     t[1] = 0

        # if np.random.random() > aug_prob:
        #     angle = 0
        # else:
        #     angle = np.random.uniform(-0.4, 0.4)

        # zoom  = np.tile(np.exp(np.random.uniform(-0.4, 0.4)), (2,1)) #np.exp(np.random.uniform(0,0.4,2))
        # if np.random.random() > aug_prob:
        #     zoom[0] = 1
        #     zoom[1] = 1

        # squeeze  = np.exp(np.random.uniform(-0.3, 0.3))
        # if np.random.random() > aug_prob:
        #     squeeze = 1

        # zoom[0]  = zoom[0]*squeeze
        # zoom[1]  = zoom[1]/squeeze

        # # geometric transformation applied additional to the second image
        # t2 = np.random.normal(0,0.03, 2)
        # t2[0]  = t2[0] * width
        # t2[1]  = t2[1] * height             
        # angle2 = np.random.normal(0,0.03)
        # zoom2  = np.exp(np.random.normal(0,0.03,2))     

        # construct trans matrice 
        # move origin, translate, rotate, zoom, move origin back
        transMat = left_multiply(translation_matrix(-origin),  translation_matrix(t))
        transMat = left_multiply(transMat,  rotation_matrix(angle))
        transMat = left_multiply(transMat,  zoom_matrix(zoom))
        transMat = left_multiply(transMat,  translation_matrix(des_origin))

        transMatInv = left_multiply(translation_matrix(-des_origin),  zoom_matrix(1./zoom))
        transMatInv = left_multiply(transMatInv,  rotation_matrix(-angle))
        transMatInv = left_multiply(transMatInv,  translation_matrix(-t))
        transMatInv = left_multiply(transMatInv,  translation_matrix(origin))

        transMat2 = left_multiply(transMat,  translation_matrix(-des_origin))
        transMat2 = left_multiply(transMat2,  translation_matrix(t2))
        transMat2 = left_multiply(transMat2,  rotation_matrix(angle2))
        transMat2 = left_multiply(transMat2,  zoom_matrix(zoom2))
        transMat2 = left_multiply(transMat2,  translation_matrix(des_origin))

        transMatInv2 = left_multiply(translation_matrix(-des_origin),  zoom_matrix(1./zoom2))
        transMatInv2 = left_multiply(transMatInv2,  rotation_matrix(-angle2))
        transMatInv2 = left_multiply(transMatInv2,  translation_matrix(-t2))
        transMatInv2 = left_multiply(transMatInv2,  translation_matrix(des_origin))
        transMatInv2 = left_multiply(transMatInv2,  transMatInv)
        success = True

        xpos = np.array([0, des_width])
        ypos = np.array([0, des_height])

        # whether trans 1 maps the four corners out of image boundary 
        xpos2 = xpos*transMatInv[0] + ypos*transMatInv[1] + transMatInv[2]
        ypos2 = xpos*transMatInv[3] + ypos*transMatInv[4] + transMatInv[5]
        if xpos2.min()< 0 or xpos2.max()>width-1 or ypos2.min()<0 or ypos2.max()> height-1:
            success = False

        xpos2 = xpos*transMatInv2[0] + ypos*transMatInv2[1] + transMatInv2[2]
        ypos2 = xpos*transMatInv2[3] + ypos*transMatInv2[4] + transMatInv2[5]
        if xpos2.min()< 0 or xpos2.max()>width-1 or ypos2.min()<0 or ypos2.max()> height-1:
            success = False

        if success:
            # print 'iter', iter, 'succeeds.'
            return transMat, transMatInv, transMat2, transMatInv2

    return transMat, transMatInv, transMat2, transMatInv2

def transform_flow_bilinear(uvt, transMatInv, transMat2, basis_x, basis_y):
    height, width, _ = uvt.shape

    # Compute new flow fields
    # Apply inverse transformation: corresponding postions in old image 1 for every pixel in new image1
    xpos = basis_x*transMatInv[0] + basis_y * transMatInv[1] + transMatInv[2]
    ypos = basis_x*transMatInv[3] + basis_y * transMatInv[4] + transMatInv[5] # image 1's axis
    
    # Apply Flow field of from old image 1 to old image 2
    xposi = np.minimum(np.maximum(xpos, 0.0), width-1.05).astype(int)
    yposi = np.minimum(np.maximum(ypos, 0.0), height-1.05).astype(int)
    xpos2 = xpos + uvt[yposi, xposi, 0]
    ypos2 = ypos + uvt[yposi, xposi, 1] # image 1's axis

    # Apply transformation of image 2
    xpos3 = xpos2*transMat2[0] + ypos2 * transMat2[1] + transMat2[2]
    ypos3 = xpos2*transMat2[3] + ypos2 * transMat2[4] + transMat2[5]

    des_height = basis_x.shape[0]
    des_width = basis_x.shape[1]
    uvdt = np.zeros((des_height, des_width, 2))
    uvdt[:, :, 0] = xpos3 - basis_x
    uvdt[:, :, 1] = ypos3 - basis_y

    return uvdt   

def transform_flow_nearest(uvt, transMatInv, transMat2, basis_x, basis_y):
    height, width, _ = uvt.shape

    # setp 1: apply inverse transformation of Image 1
    xpos1 = basis_x*transMatInv[0] + basis_y * transMatInv[1] + transMatInv[2]
    ypos1 = basis_x*transMatInv[3] + basis_y * transMatInv[4] + transMatInv[5]
    # xpos1 = min(max(int(xpos1 + 0.5), 0), width - 1)
    # ypos1 = min(max(int(ypos1 + 0.5), 0), height - 1)
    xpos1 = np.minimum(np.maximum((xpos1 + 0.5), 0), width-1).astype(int)
    ypos1 = np.minimum(np.maximum((ypos1 + 0.5), 0), height-1).astype(int)

    # step 2: apply flow field
    xpos2 = xpos1 + uvt[ypos1, xpos1, 0]
    ypos2 = ypos1 + uvt[ypos1, xpos1, 1] # image 1's axis

    # step 3: apply transformation of image 2
    xpos3 = xpos2*transMat2[0] + ypos2 * transMat2[1] + transMat2[2]
    ypos3 = xpos2*transMat2[3] + ypos2 * transMat2[4] + transMat2[5]

    # step 4: difference between the new and old positions gives the flow
    des_height = basis_x.shape[0]
    des_width = basis_x.shape[1]
    uvdt = np.zeros((des_height, des_width, 2))
    uvdt[:, :, 0] = xpos3 - basis_x
    uvdt[:, :, 1] = ypos3 - basis_y

    # float xpos1, ypos1, xpos2, ypos2, xpos3, ypos3;
    # // Step 1: Apply inverse tranformation of Image 1
    # xpos1 = x * transMat1->t0 + y * transMat1->t2 + transMat1->t4;
    # ypos1 = x * transMat1->t1 + y * transMat1->t3 + transMat1->t5;

    # // Step 2: Apply flow field
    # int srcIdxOffx = width*(height*(2*n+0) + 
    #                  (int)(ypos1+(Dtype)0.5)) + 
    #                  (int)(xpos1+(Dtype)0.5);
    # int srcIdxOffy = width*(height*(2*n+1) + 
    #                  (int)(ypos1+(Dtype)0.5)) + 
    #                  (int)(xpos1+(Dtype)0.5);
    
    # xpos2 = xpos1 + src_data[min(srcIdxOffx,src_count)];
    # ypos2 = ypos1 + src_data[min(srcIdxOffy,src_count)];
    
    # // Step 3: Apply tranformation of Image 2
    # xpos3 = xpos2 * transMat2->t0 + ypos2 * transMat2->t2 + transMat2->t4;
    # ypos3 = xpos2 * transMat2->t1 + ypos2 * transMat2->t3 + transMat2->t5;
    
    # // Step 4: Difference between the new and old positions gives the flow
    # dest_data[dest_width*(dest_height*(2*n+0) + (int)y) + (int)x] = xpos3 - x;
    # dest_data[dest_width*(dest_height*(2*n+1) + (int)y) + (int)x] = ypos3 - y;
    return uvdt


def spatial_transform(im_data, uv_data, crop_size = (256, 256), basis_x=None, basis_y=None): 
    # TODO: provide basis_x and basis_y 
    assert len(im_data) == 2 or len(im_data) == 6, \
        'Incorrect number of input: {}'.format(len(im_data))
    use_semi_supervised_data = len(im_data) == 6
    if use_semi_supervised_data:
        im1, im2, ss_im1, ss_im2, im1_seg, im2_seg = im_data
    else:
        im1, im2 = im_data
        ss_im1 = ss_im2 = im1_seg = im2_seg = None
    uvt, uv_occ = uv_data
    height, width, channels = im1.shape

    if use_semi_supervised_data:
        process_seg = len(im1_seg.shape) == 3
        if process_seg:
            _, _, seg_channels = im1_seg.shape

    des_height = crop_size[0]
    des_width  = crop_size[1]
    origin = np.array([0.5*width, 0.5*height])
    des_origin = np.array([0.5*des_width, 0.5*des_height])

    transMat, transMatInv, transMat2, transMatInv2 = generate_good_coefficients(im1, crop_size, 50) 
    
    if basis_y is None:
        ys = np.arange(0, des_height)
        basis_y  = np.transpose(np.tile(ys, (des_width, 1)))
    if basis_x is None:     
        xs = np.arange(0, des_width)
        basis_x = np.tile(xs, (des_height, 1))

    xpos = basis_x*transMatInv[0] + basis_y * transMatInv[1] + transMatInv[2]
    ypos = basis_x*transMatInv[3] + basis_y * transMatInv[4] + transMatInv[5]
    xpos = np.minimum(np.maximum(xpos, 0.0), width-1.05)
    ypos = np.minimum(np.maximum(ypos, 0.0), height-1.05)

    x0 = np.floor(xpos).astype(int)
    x1 = np.ceil(xpos).astype(int)
    y0 = np.floor(ypos).astype(int)
    y1 = np.ceil(ypos).astype(int)      
    xdist = xpos  - x0
    ydist = ypos  - y0    
    imd1 = np.zeros((des_height, des_width, channels))
    for i in range(channels):   
        imd1[:,:, i] = (1.-xdist)*(1.-ydist) * im1[y0, x0, i]+ (1.-xdist)*ydist*im1[y1, x0, i] + xdist*(1.- ydist) * im1[y0, x1, i] + xdist*ydist*im1[y1, x1, i]

    if use_semi_supervised_data:
        ss_imd1 = np.zeros((des_height, des_width, channels))
        for i in range(channels):   
            ss_imd1[:,:, i] = (1.-xdist)*(1.-ydist) * ss_im1[y0, x0, i]+ (1.-xdist)*ydist*ss_im1[y1, x0, i] + xdist*(1.- ydist) * ss_im1[y0, x1, i] + xdist*ydist*ss_im1[y1, x1, i]

        if process_seg:
            imd1_seg = np.zeros((des_height, des_width, seg_channels))
            for i in range(seg_channels):
                imd1_seg[:,:, i] = (1.-xdist)*(1.-ydist) * im1_seg[y0, x0, i]+ (1.-xdist)*ydist*im1_seg[y1, x0, i] + xdist*(1.- ydist) * im1_seg[y0, x1, i] + xdist*ydist*im1_seg[y1, x1, i]
        else:
            imd1_seg = im1_seg

    if uv_occ.max() >= 0:
        uv_occ_aug = np.zeros((des_height, des_width))
        uv_occ_aug = (1.-xdist)*(1.-ydist) * uv_occ[y0, x0]+ (1.-xdist)*ydist*uv_occ[y1, x0] + xdist*(1.- ydist) * uv_occ[y0, x1] + xdist*ydist*uv_occ[y1, x1]     
        # uv_occ_aug = (uv_occ_aug > 0.45).astype(np.float32)
    else:
        uv_occ_aug = -np.ones((des_height, des_width))

    # # process occlusion valid mask
    # if uvt.shape[0] > 2:
    #     assert uvt.shape[0] == 3, 'Invalid flow shape {}'.format(uvt.shape)
    #     uvt_mask = uvt[2]
    #     uvdt_mask = np.zeros((des_height, des_width))
    #     uvdt_mask = (1.-xdist)*(1.-ydist) * uvt_mask[y0, x0]+ (1.-xdist)*ydist*uvt_mask[y1, x0] + xdist*(1.- ydist) * uvt_mask[y0, x1] + xdist*ydist*uvt_mask[y1, x1]
    #     # uvdt_mask = (uvdt_mask > 0.75).astype(np.float32)

    xpos = basis_x*transMatInv2[0] + basis_y * transMatInv2[1] + transMatInv2[2]
    ypos = basis_x*transMatInv2[3] + basis_y * transMatInv2[4] + transMatInv2[5]
    xpos = np.minimum(np.maximum(xpos, 0.0), width-1.05)
    ypos = np.minimum(np.maximum(ypos, 0.0), height-1.05)

    x0 = np.floor(xpos).astype(int)
    x1 = np.ceil(xpos).astype(int)
    y0 = np.floor(ypos).astype(int)
    y1 = np.ceil(ypos).astype(int)      
    xdist = xpos  - x0
    ydist = ypos  - y0    
    imd2 = np.zeros((des_height, des_width, channels))
    for i in range(channels):           
        imd2[:,:,i] = (1.-xdist)*(1.-ydist) * im2[y0, x0, i]+ (1.-xdist)*ydist*im2[y1, x0, i] + xdist*(1.- ydist) * im2[y0, x1, i] + xdist*ydist*im2[y1, x1, i]

    if use_semi_supervised_data:
        ss_imd2 = np.zeros((des_height, des_width, channels))
        for i in range(channels):           
            ss_imd2[:,:,i] = (1.-xdist)*(1.-ydist) * ss_im2[y0, x0, i]+ (1.-xdist)*ydist*ss_im2[y1, x0, i] + xdist*(1.- ydist) * ss_im2[y0, x1, i] + xdist*ydist*ss_im2[y1, x1, i]

        if process_seg:
            imd2_seg = np.zeros((des_height, des_width, seg_channels))
            for i in range(seg_channels):           
                imd2_seg[:,:,i] = (1.-xdist)*(1.-ydist) * im2_seg[y0, x0, i]+ (1.-xdist)*ydist*im2_seg[y1, x0, i] + xdist*(1.- ydist) * im2_seg[y0, x1, i] + xdist*ydist*im2_seg[y1, x1, i]
        else:
            imd2_seg = im2_seg

    # if uvt.shape[0] > 2:
    #     u = uvt[0].copy()
    #     v = uvt[1].copy()
    #     mask = uvt[2].copy()
    #     invalid_idxes = np.where(mask < 1)
    #     u[invalid_idxes] = np.nan
    #     v[invalid_idxes] = np.nan
    #     uvt = np.stack((u, v))

    uvdt = transform_flow_bilinear(uvt, transMatInv, transMat2, basis_x, basis_y)

    imd1 = np.array(imd1)
    imd2 = np.array(imd2)
    if use_semi_supervised_data:
        ss_imd1 = np.array(ss_imd1)
        ss_imd2 = np.array(ss_imd2)
        imd1_seg = np.array(imd1_seg)
        imd2_seg = np.array(imd2_seg)
        im_data_aug = [imd1, imd2, ss_imd1, ss_imd2, imd1_seg, imd2_seg]
    else:
        im_data_aug = [imd1, imd2]

    uv_data_aug = [np.array(uvdt), np.array(uv_occ_aug)]

    return im_data_aug, uv_data_aug