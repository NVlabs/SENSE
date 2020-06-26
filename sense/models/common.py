"""
Copyright (C) 2019 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""
from __future__ import print_function

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from sense.lib.nn import SynchronizedBatchNorm2d

def make_bn_layer(bn_type, plane):
    if bn_type == 'plain':
        return nn.BatchNorm2d(plane)
    elif bn_type == 'syncbn':
        return SynchronizedBatchNorm2d(plane)
    elif bn_type == 'encoding':
        raise NotImplementedError
        import encoding
        import encoding.nn
        return encoding.nn.BatchNorm2d(plane)
    else:
        raise Exception('Not supported BN type: {}.'.format(bn_type))

def convbn(in_planes, out_planes, kernel_size=3, 
        stride=1, padding=1, dilation=1, bias=True, bn_type='syncbn', no_relu=False
    ):
    layers = []
    layers.append(
        nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, 
                        padding=padding, dilation=dilation, bias=bias)
    )   
    layers.append(make_bn_layer(bn_type, out_planes))
    if not no_relu:
        layers.append(nn.ReLU())
    return nn.Sequential(*layers)

def deconv(in_planes, out_planes, kernel_size=4, stride=2, padding=1):
    return nn.ConvTranspose2d(in_planes, out_planes, kernel_size, stride, padding, bias=False)

def predict_flow(in_planes):
    return nn.Conv2d(in_planes,2,kernel_size=3,stride=1,padding=1,bias=True)

def predict_depth(in_planes):
    return nn.Sequential(
        nn.Conv2d(in_planes,1,kernel_size=3,stride=1,padding=1,bias=False) ,
        nn.ReLU()
    )

def predict_class(in_planes, num_classes=192, bias=True):
    return nn.Conv2d(in_planes,num_classes,kernel_size=3,stride=1,padding=1,bias=bias)

class disparityregression(nn.Module):
    def __init__(self, maxdisp):
        super(disparityregression, self).__init__()
        self.disp = torch.Tensor(np.reshape(np.array(range(maxdisp)),[1,maxdisp,1,1])).cuda()

    def forward(self, x):
        disp = self.disp.repeat(x.size()[0],1,x.size()[2],x.size()[3])
        out = torch.sum(x*disp, 1, keepdim=True)
        return out

def weight_init(mdl):
    for m in mdl.modules():
        if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
            nn.init.kaiming_normal_(m.weight.data, mode='fan_in')
            if m.bias is not None:
                m.bias.data.zero_()
        if isinstance(m, SynchronizedBatchNorm2d):
            m.weight.data.fill_(1)
            m.bias.data.zero_()
        if isinstance(m, nn.BatchNorm2d):
            m.weight.data.fill_(1)
            m.bias.data.zero_()

class Hourglass(nn.Module):
    def __init__(self, in_planes, do_flow=False, no_output=False, bn_type='plain'):
        super(Hourglass, self).__init__()
        self.no_output = no_output
        # in 1/2, out: 1/4
        self.conv1 = convbn(
            in_planes, 
            in_planes * 2, 
            kernel_size=3, 
            stride=2, 
            padding=1,
            bn_type=bn_type,
        )
        # in: 1/4, out: 1/8
        self.conv2 = convbn(
            in_planes * 2,
            in_planes * 2,
            kernel_size=3,
            stride=2,
            padding=1,
            bn_type=bn_type
        )
        # in: 1/8, out : 1/8
        self.conv3 = convbn(
            in_planes * 2,
            in_planes * 2,
            kernel_size=3,
            stride=1,
            padding=1,
            bn_type=bn_type
        )
        # in: 1/8, out: 1/4
        self.conv4 = nn.Sequential(
            nn.ConvTranspose2d(in_planes * 2, in_planes * 2, 3, padding=1, output_padding=1, stride=2,bias=False),
            make_bn_layer(bn_type, in_planes * 2),
            nn.ReLU()
        )
        self.conv5 = nn.Sequential(
            nn.ConvTranspose2d(in_planes * 2, in_planes, 3, padding=1, output_padding=1, stride=2,bias=False),
            make_bn_layer(bn_type, in_planes),
            nn.ReLU()
        )
        if not no_output:
            if do_flow:
                self.output = predict_flow(in_planes)
            else:
                self.output = predict_class(in_planes, 1, bias=False)

        weight_init(self)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        if not self.no_output:
            y = self.output(x)
        else:
            y = None
        return y, x

# pyramid pooling
# code borrowed from UPerNet
# https://github.com/CSAILVision/semantic-segmentation-pytorch/blob/master/models/models.py
class PPM(nn.Module):
    def __init__(self, encoder_planes, pool_scales=(1, 2, 3, 6), bn_type='plain',
            ppm_last_conv_planes=256, ppm_inter_conv_planes=128
        ):
        super(PPM, self).__init__()
        # Parymid Pooling Module (PPM)
        self.ppm_pooling = []
        self.ppm_conv = []

        self.ppm_last_conv_planes = ppm_last_conv_planes

        for scale in pool_scales:
            self.ppm_pooling.append(nn.AdaptiveAvgPool2d(scale))
            self.ppm_conv.append(
                nn.Sequential(
                    nn.Conv2d(encoder_planes[-1], ppm_inter_conv_planes, kernel_size=1, bias=False),
                    make_bn_layer(bn_type, ppm_inter_conv_planes),
                    nn.ReLU(inplace=True)
                )
            )
        self.ppm_pooling = nn.ModuleList(self.ppm_pooling)
        self.ppm_conv = nn.ModuleList(self.ppm_conv)
        self.ppm_last_conv = convbn(
            encoder_planes[-1] + len(pool_scales)*128, 
            self.ppm_last_conv_planes, 
            bias=False,
            bn_type=bn_type
        )

        weight_init(self)

    def forward(self, conv5):
        input_size = conv5.size()
        ppm_out = [conv5]
        for pool_scale, pool_conv in zip(self.ppm_pooling, self.ppm_conv):
            ppm_out.append(pool_conv(F.upsample(
                pool_scale(conv5),
                (input_size[2], input_size[3]),
                mode='bilinear', align_corners=False
            )))
        ppm_out = torch.cat(ppm_out, 1)
        f = self.ppm_last_conv(ppm_out)
        return f

def make_refinement_module(module_type, in_planes, 
                           do_flow=False, no_output=False, bn_type='plain'):
    # module type: lightweight or hourglass
    if module_type == 'none':
        return None
    elif module_type == 'lightweight':
        # return nn.Sequential(
        #             nn.Conv2d(in_planes, in_planes * 2, 3, padding=1, stride=1),
        #             make_bn_layer(bn_type, in_planes * 2),
        #             nn.ReLU(),
        #             nn.Conv2d(in_planes * 2, in_planes, 3, padding=1, stride=1),
        #             make_bn_layer(bn_type, in_planes),
        #             nn.ReLU(),
        #             nn.Conv2d(in_planes, out_plane, 3, padding=1, stride=1, bias=False),
        #             nn.ReLU()
        #         )
        raise NotImplementedError
    elif module_type == 'hourglass':
        return Hourglass(in_planes, do_flow, no_output, bn_type)
    else:
        raise Exception('{} refinement module is not supported.'.format(module_type))

def flow_warp(x, flo):
    """
    warp an image/tensor (im2) back to im1, according to the optical flow
    x: [B, C, H, W] (im2)
    flo: [B, 2, H, W] flow
    """
    B, C, H, W = x.size()
    # mesh grid 
    xx = torch.arange(0, W).view(1,-1).repeat(H,1)
    yy = torch.arange(0, H).view(-1,1).repeat(1,W)
    xx = xx.view(1,1,H,W).repeat(B,1,1,1)
    yy = yy.view(1,1,H,W).repeat(B,1,1,1)
    grid = torch.cat((xx,yy),1).float()

    if x.is_cuda:
        grid = grid.cuda()
    vgrid = grid + flo

    # scale grid to [-1,1] 
    vgrid[:,0,:,:] = 2.0*vgrid[:,0,:,:]/max(W-1,1)-1.0
    vgrid[:,1,:,:] = 2.0*vgrid[:,1,:,:]/max(H-1,1)-1.0

    vgrid = vgrid.permute(0,2,3,1)        
    output = nn.functional.grid_sample(x, vgrid)
    mask = torch.ones(x.size()).cuda()
    mask = nn.functional.grid_sample(mask, vgrid)

    # if W==128:
        # np.save('mask.npy', mask.cpu().data.numpy())
        # np.save('warp.npy', output.cpu().data.numpy())
    
    mask[mask.data < 0.9999] = 0
    mask[mask.data > 0] = 1
    
    return output*mask

def disp_warp(rim, disp):
    """
    warp stereo image (right image) with disparity

    rim: [B, C, H, W] image/tensor

    disp: [B, 1, H, W] (left) disparity

    for disparity (left), we have

        left_image(x,y) = right_image(x-d(x,y),y)

    """
    B, C, H, W = rim.size()
    # mesh grid 
    xx = torch.arange(0, W).view(1,-1).repeat(H,1)
    yy = torch.arange(0, H).view(-1,1).repeat(1,W)
    xx = xx.view(1,1,H,W).repeat(B,1,1,1)
    yy = yy.view(1,1,H,W).repeat(B,1,1,1)
    grid = torch.cat((xx,yy),1).float()

    if rim.is_cuda:
        grid = grid.cuda()
    vgrid = grid

    # scale grid to [-1,1] 
    vgrid[:,0,:,:] = 2.0*(vgrid[:,0,:,:]-disp[:,0,:,:])/max(W-1,1)-1.0
    vgrid[:,1,:,:] = 2.0*vgrid[:,1,:,:]/max(H-1,1)-1.0

    return nn.functional.grid_sample(rim, vgrid.permute(0,2,3,1))