"""
Copyright (C) 2019 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

# from lib.correlation_package.modules.corr import Correlation, Correlation1d
from sense.lib.correlation_package.correlation import Correlation
import numpy as np

def conv(in_planes, out_planes, kernel_size=3, stride=1, padding=1, dilation=1):   
    return nn.Sequential(
        nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, 
            padding=padding, dilation=dilation, bias=True
        ),
        nn.LeakyReLU(0.1)
    )

def deconv(in_planes, out_planes, kernel_size=4, stride=2, padding=1):
    return nn.ConvTranspose2d(in_planes, out_planes, kernel_size, stride, padding, bias=True)

def predict_flow(in_planes):
    return nn.Conv2d(in_planes,2,kernel_size=3,stride=1,padding=1,bias=True)

def predict_depth(in_planes):
    return nn.Conv2d(in_planes,1,kernel_size=3,stride=1,padding=1,bias=True) 

def weight_init(mdl):
    for m in mdl.modules():
        if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
            nn.init.kaiming_normal(m.weight.data, mode='fan_in')
            if m.bias is not None:
                m.bias.data.zero_()

class PWC6LEncoder(nn.Module):
    def __init__(self):
        super(PWC6LEncoder, self).__init__()
        self.conv1a  = conv(3,   16, kernel_size=3, stride=2)
        self.conv1b  = conv(16,  16, kernel_size=3, stride=1)
        self.conv2a  = conv(16,  32, kernel_size=3, stride=2)
        self.conv2b  = conv(32,  32, kernel_size=3, stride=1)
        self.conv3a  = conv(32,  64, kernel_size=3, stride=2)
        self.conv3b  = conv(64,  64, kernel_size=3, stride=1)
        self.conv4a  = conv(64,  96, kernel_size=3, stride=2)
        self.conv4b  = conv(96,  96, kernel_size=3, stride=1)
        self.conv5a  = conv(96, 128, kernel_size=3, stride=2)
        self.conv5b  = conv(128,128, kernel_size=3, stride=1)
        self.conv6a  = conv(128,196, kernel_size=3, stride=2)
        self.conv6b  = conv(196,196, kernel_size=3, stride=1)

        weight_init(self)

    def forward(self, x):
        c1 = self.conv1b(self.conv1a(x))
        c2 = self.conv2b(self.conv2a(c1))
        c3 = self.conv3b(self.conv3a(c2))
        c4 = self.conv4b(self.conv4a(c3))        
        c5 = self.conv5b(self.conv5a(c4))
        c6 = self.conv6b(self.conv6a(c5))
        return [c1, c2, c3, c4, c5, c6]

class PWC6LFlowDecoder(nn.Module):
    def __init__(self, md=4):
        super(PWC6LFlowDecoder, self).__init__()
        self.corr = Correlation(
            pad_size=md, 
            kernel_size=1, 
            max_displacement=md, 
            stride1=1, 
            stride2=1, 
            corr_multiply=1
        )
        self.leakyRELU = nn.LeakyReLU(0.1)
        
        nd = (2*md+1)**2
        dd = np.cumsum([128,128,96,64,32]).tolist()

        od = nd
        self.conv6_0 = conv(od,      128, kernel_size=3, stride=1)
        self.conv6_1 = conv(od+dd[0],128, kernel_size=3, stride=1)
        self.conv6_2 = conv(od+dd[1],96,  kernel_size=3, stride=1)
        self.conv6_3 = conv(od+dd[2],64,  kernel_size=3, stride=1)
        self.conv6_4 = conv(od+dd[3],32,  kernel_size=3, stride=1)        
        self.predict_flow6 = predict_flow(od+dd[4])
        self.deconv6 = deconv(2, 2, kernel_size=4, stride=2, padding=1) 
        self.upfeat6 = deconv(od+dd[4], 2, kernel_size=4, stride=2, padding=1) 
        
        od = nd+128+4
        self.conv5_0 = conv(od,      128, kernel_size=3, stride=1)
        self.conv5_1 = conv(od+dd[0],128, kernel_size=3, stride=1)
        self.conv5_2 = conv(od+dd[1],96,  kernel_size=3, stride=1)
        self.conv5_3 = conv(od+dd[2],64,  kernel_size=3, stride=1)
        self.conv5_4 = conv(od+dd[3],32,  kernel_size=3, stride=1)
        self.predict_flow5 = predict_flow(od+dd[4]) 
        self.deconv5 = deconv(2, 2, kernel_size=4, stride=2, padding=1) 
        self.upfeat5 = deconv(od+dd[4], 2, kernel_size=4, stride=2, padding=1) 
        
        od = nd+96+4
        self.conv4_0 = conv(od,      128, kernel_size=3, stride=1)
        self.conv4_1 = conv(od+dd[0],128, kernel_size=3, stride=1)
        self.conv4_2 = conv(od+dd[1],96,  kernel_size=3, stride=1)
        self.conv4_3 = conv(od+dd[2],64,  kernel_size=3, stride=1)
        self.conv4_4 = conv(od+dd[3],32,  kernel_size=3, stride=1)
        self.predict_flow4 = predict_flow(od+dd[4]) 
        self.deconv4 = deconv(2, 2, kernel_size=4, stride=2, padding=1) 
        self.upfeat4 = deconv(od+dd[4], 2, kernel_size=4, stride=2, padding=1) 
        
        od = nd+64+4
        self.conv3_0 = conv(od,      128, kernel_size=3, stride=1)
        self.conv3_1 = conv(od+dd[0],128, kernel_size=3, stride=1)
        self.conv3_2 = conv(od+dd[1],96,  kernel_size=3, stride=1)
        self.conv3_3 = conv(od+dd[2],64,  kernel_size=3, stride=1)
        self.conv3_4 = conv(od+dd[3],32,  kernel_size=3, stride=1)
        self.predict_flow3 = predict_flow(od+dd[4]) 
        self.deconv3 = deconv(2, 2, kernel_size=4, stride=2, padding=1) 
        self.upfeat3 = deconv(od+dd[4], 2, kernel_size=4, stride=2, padding=1) 
        
        od = nd+32+4
        self.conv2_0 = conv(od,      128, kernel_size=3, stride=1)
        self.conv2_1 = conv(od+dd[0],128, kernel_size=3, stride=1)
        self.conv2_2 = conv(od+dd[1],96,  kernel_size=3, stride=1)
        self.conv2_3 = conv(od+dd[2],64,  kernel_size=3, stride=1)
        self.conv2_4 = conv(od+dd[3],32,  kernel_size=3, stride=1)
        self.predict_flow2 = predict_flow(od+dd[4]) 
        self.deconv2 = deconv(2, 2, kernel_size=4, stride=2, padding=1) 
        
        self.dc_conv1 = conv(od+dd[4], 128, kernel_size=3, stride=1, padding=1,  dilation=1)
        self.dc_conv2 = conv(128,      128, kernel_size=3, stride=1, padding=2,  dilation=2)
        self.dc_conv3 = conv(128,      128, kernel_size=3, stride=1, padding=4,  dilation=4)
        self.dc_conv4 = conv(128,      96,  kernel_size=3, stride=1, padding=8,  dilation=8)
        self.dc_conv5 = conv(96,       64,  kernel_size=3, stride=1, padding=16, dilation=16)
        self.dc_conv6 = conv(64,       32,  kernel_size=3, stride=1, padding=1,  dilation=1)
        self.dc_conv7 = predict_flow(32)

        weight_init(self)

    def warp(self, x, flo):
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
        vgrid = Variable(grid) + flo

        # scale grid to [-1,1] 
        vgrid[:,0,:,:] = 2.0*vgrid[:,0,:,:]/max(W-1,1)-1.0
        vgrid[:,1,:,:] = 2.0*vgrid[:,1,:,:]/max(H-1,1)-1.0

        vgrid = vgrid.permute(0,2,3,1)        
        output = nn.functional.grid_sample(x, vgrid)
        mask = torch.autograd.Variable(torch.ones(x.size())).cuda()
        mask = nn.functional.grid_sample(mask, vgrid)

        # if W==128:
            # np.save('mask.npy', mask.cpu().data.numpy())
            # np.save('warp.npy', output.cpu().data.numpy())
        
        mask[mask.data < 0.9999] = 0
        mask[mask.data > 0] = 1
        
        return output*mask

    def forward(self, x1, x2):
        c11, c12, c13, c14, c15, c16 = x1
        c21, c22, c23, c24, c25, c26 = x2

        corr6 = self.corr(c16, c26) 
        corr6 = self.leakyRELU(corr6)   


        x = torch.cat((self.conv6_0(corr6), corr6),1)
        x = torch.cat((self.conv6_1(x), x),1)
        x = torch.cat((self.conv6_2(x), x),1)
        x = torch.cat((self.conv6_3(x), x),1)
        x = torch.cat((self.conv6_4(x), x),1)
        flow6 = self.predict_flow6(x)
        up_flow6 = self.deconv6(flow6)
        up_feat6 = self.upfeat6(x)

        # np.save('concat_6.npy', x.cpu().data.numpy())

        
        warp5 = self.warp(c25, up_flow6*0.625)
        corr5 = self.corr(c15, warp5) 
        corr5 = self.leakyRELU(corr5)
        x = torch.cat((corr5, c15, up_flow6, up_feat6), 1)
        x = torch.cat((self.conv5_0(x), x),1)
        x = torch.cat((self.conv5_1(x), x),1)
        x = torch.cat((self.conv5_2(x), x),1)
        x = torch.cat((self.conv5_3(x), x),1)
        x = torch.cat((self.conv5_4(x), x),1)
        flow5 = self.predict_flow5(x)
        up_flow5 = self.deconv5(flow5)
        up_feat5 = self.upfeat5(x)

        # np.save('concat_5.npy', x.cpu().data.numpy())
        
        warp4 = self.warp(c24, up_flow5*1.25)
        corr4 = self.corr(c14, warp4)  
        corr4 = self.leakyRELU(corr4)
        x = torch.cat((corr4, c14, up_flow5, up_feat5), 1)
        x = torch.cat((self.conv4_0(x), x),1)
        x = torch.cat((self.conv4_1(x), x),1)
        x = torch.cat((self.conv4_2(x), x),1)
        x = torch.cat((self.conv4_3(x), x),1)
        x = torch.cat((self.conv4_4(x), x),1)
        flow4 = self.predict_flow4(x)
        up_flow4 = self.deconv4(flow4)
        up_feat4 = self.upfeat4(x)

        # np.save('concat_4.npy', x.cpu().data.numpy())
        # np.save('up_flow4.npy', up_flow4.cpu().data.numpy())
        # np.save('up_feat4.npy', up_feat4.cpu().data.numpy())
        

        warp3 = self.warp(c23, up_flow4*2.5)
        corr3 = self.corr(c13, warp3) 
        corr3 = self.leakyRELU(corr3)
        
        # np.save('warp3.npy', warp3.cpu().data.numpy())
        # np.save('corr3.npy', corr3.cpu().data.numpy())

        x = torch.cat((corr3, c13, up_flow4, up_feat4), 1)
        x = torch.cat((self.conv3_0(x), x),1)
        x = torch.cat((self.conv3_1(x), x),1)
        x = torch.cat((self.conv3_2(x), x),1)
        x = torch.cat((self.conv3_3(x), x),1)
        x = torch.cat((self.conv3_4(x), x),1)
        flow3 = self.predict_flow3(x)
        up_flow3 = self.deconv3(flow3)
        up_feat3 = self.upfeat3(x)

        # np.save('concat_3.npy', x.cpu().data.numpy())
        
        warp2 = self.warp(c22, up_flow3*5.0) 
        corr2 = self.corr(c12, warp2)
        corr2 = self.leakyRELU(corr2)
        x = torch.cat((corr2, c12, up_flow3, up_feat3), 1)
        x = torch.cat((self.conv2_0(x), x),1)
        x = torch.cat((self.conv2_1(x), x),1)
        x = torch.cat((self.conv2_2(x), x),1)
        x = torch.cat((self.conv2_3(x), x),1)
        x = torch.cat((self.conv2_4(x), x),1)
        flow2 = self.predict_flow2(x)
 
        # np.save('concat_2.npy', x.cpu().data.numpy())

        x = self.dc_conv4(self.dc_conv3(self.dc_conv2(self.dc_conv1(x))))
        flow2 += self.dc_conv7(self.dc_conv6(self.dc_conv5(x)))

        return (flow2, flow3, flow4, flow5, flow6)

class PWC6LDispDecoder(nn.Module):
    def __init__(self, md=4):
        super(PWC6LDispDecoder, self).__init__()

        self.disp_corr = Correlation1d(
            pad_size=md, 
            kernel_size=1, 
            max_displacement=md, 
            stride1=1, 
            stride2=1, 
            corr_multiply=1
        )

        nd = 2*md+1
        dd = np.cumsum([128,128,96,64,32]).tolist()

        od = nd
        self.disp_conv6_0 = conv(od,      128, kernel_size=3, stride=1)
        self.disp_conv6_1 = conv(od+dd[0],128, kernel_size=3, stride=1)
        self.disp_conv6_2 = conv(od+dd[1],96,  kernel_size=3, stride=1)
        self.disp_conv6_3 = conv(od+dd[2],64,  kernel_size=3, stride=1)
        self.disp_conv6_4 = conv(od+dd[3],32,  kernel_size=3, stride=1)        
        self.predict_depth6 = predict_depth(od+dd[4]) 
        self.disp_deconv6 = deconv(1, 1, kernel_size=4, stride=2, padding=1) 
        self.disp_upfeat6 = deconv(od+dd[4], 1, kernel_size=4, stride=2, padding=1) 
        
        od = nd+128+2
        self.disp_conv5_0 = conv(od,      128, kernel_size=3, stride=1)
        self.disp_conv5_1 = conv(od+dd[0],128, kernel_size=3, stride=1)
        self.disp_conv5_2 = conv(od+dd[1],96,  kernel_size=3, stride=1)
        self.disp_conv5_3 = conv(od+dd[2],64,  kernel_size=3, stride=1)
        self.disp_conv5_4 = conv(od+dd[3],32,  kernel_size=3, stride=1)
        self.predict_depth5 = predict_depth(od+dd[4]) 
        self.disp_deconv5 = deconv(1, 1, kernel_size=4, stride=2, padding=1) 
        self.disp_upfeat5 = deconv(od+dd[4], 1, kernel_size=4, stride=2, padding=1) 
        
        od = nd+96+2
        self.disp_conv4_0 = conv(od,      128, kernel_size=3, stride=1)
        self.disp_conv4_1 = conv(od+dd[0],128, kernel_size=3, stride=1)
        self.disp_conv4_2 = conv(od+dd[1],96,  kernel_size=3, stride=1)
        self.disp_conv4_3 = conv(od+dd[2],64,  kernel_size=3, stride=1)
        self.disp_conv4_4 = conv(od+dd[3],32,  kernel_size=3, stride=1)
        self.predict_depth4 = predict_depth(od+dd[4]) 
        self.disp_deconv4 = deconv(1, 1, kernel_size=4, stride=2, padding=1) 
        self.disp_upfeat4 = deconv(od+dd[4], 1, kernel_size=4, stride=2, padding=1) 
        
        od = nd+64+2
        self.disp_conv3_0 = conv(od,      128, kernel_size=3, stride=1)
        self.disp_conv3_1 = conv(od+dd[0],128, kernel_size=3, stride=1)
        self.disp_conv3_2 = conv(od+dd[1],96,  kernel_size=3, stride=1)
        self.disp_conv3_3 = conv(od+dd[2],64,  kernel_size=3, stride=1)
        self.disp_conv3_4 = conv(od+dd[3],32,  kernel_size=3, stride=1)
        self.predict_depth3 = predict_depth(od+dd[4]) 
        self.disp_deconv3 = deconv(1, 1, kernel_size=4, stride=2, padding=1) 
        self.disp_upfeat3 = deconv(od+dd[4], 1, kernel_size=4, stride=2, padding=1) 
        
        od = nd+32+2
        self.disp_conv2_0 = conv(od,      128, kernel_size=3, stride=1)
        self.disp_conv2_1 = conv(od+dd[0],128, kernel_size=3, stride=1)
        self.disp_conv2_2 = conv(od+dd[1],96,  kernel_size=3, stride=1)
        self.disp_conv2_3 = conv(od+dd[2],64,  kernel_size=3, stride=1)
        self.disp_conv2_4 = conv(od+dd[3],32,  kernel_size=3, stride=1)
        self.predict_depth2 = predict_depth(od+dd[4])
        self.disp_deconv2 = deconv(1, 1, kernel_size=4, stride=2, padding=1) 
        
        self.disp_dc_conv1 = conv(od+dd[4], 128, kernel_size=3, stride=1, padding=1,  dilation=1)
        self.disp_dc_conv2 = conv(128,      128, kernel_size=3, stride=1, padding=2,  dilation=2)
        self.disp_dc_conv3 = conv(128,      128, kernel_size=3, stride=1, padding=4,  dilation=4)
        self.disp_dc_conv4 = conv(128,      96,  kernel_size=3, stride=1, padding=8,  dilation=8)
        self.disp_dc_conv5 = conv(96,       64,  kernel_size=3, stride=1, padding=16, dilation=16)
        self.disp_dc_conv6 = conv(64,       32,  kernel_size=3, stride=1, padding=1,  dilation=1)
        self.disp_dc_conv7 = predict_depth(32)

        weight_init(self)

    def warp_R2L(self, rim, disp):
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

    def forward(self, x1, x2):
        c11, c12, c13, c14, c15, c16 = x1
        c21, c22, c23, c24, c25, c26 = x2

        # disparity decoder
        disp_corr6 = self.disp_corr(c16, c26) 
        disp_corr6 = F.leaky_relu(disp_corr6, negative_slope=0.1)        
        x = torch.cat((disp_corr6, self.disp_conv6_0(disp_corr6)),1)
        x = torch.cat((x, self.disp_conv6_1(x)),1)
        x = torch.cat((x, self.disp_conv6_2(x)),1)
        x = torch.cat((x, self.disp_conv6_3(x)),1)
        x = torch.cat((x, self.disp_conv6_4(x)),1)
        depth6 = self.predict_depth6(x)
        up_depth6 = self.disp_deconv6(depth6)
        up_disp_feat6 = self.disp_upfeat6(x)
        
        disp_warp5 = self.warp_R2L(c25, up_depth6 / 32)
        disp_corr5 = self.disp_corr(c15, disp_warp5) 
        disp_corr5 = F.leaky_relu(disp_corr5, negative_slope=0.1)
        x = torch.cat((disp_corr5, c15, up_depth6, up_disp_feat6), 1)
        x = torch.cat((x, self.disp_conv5_0(x)),1)
        x = torch.cat((x, self.disp_conv5_1(x)),1)
        x = torch.cat((x, self.disp_conv5_2(x)),1)
        x = torch.cat((x, self.disp_conv5_3(x)),1)
        x = torch.cat((x, self.disp_conv5_4(x)),1)
        depth5 = self.predict_depth5(x)
        up_depth5 = self.disp_deconv5(depth5)
        up_disp_feat5 = self.disp_upfeat5(x)
        
        disp_warp4 = self.warp_R2L(c24, up_depth5 / 16)
        disp_corr4 = self.disp_corr(c14, disp_warp4)  
        disp_corr4 = F.leaky_relu(disp_corr4, negative_slope=0.1)
        x = torch.cat((disp_corr4, c14, up_depth5, up_disp_feat5), 1)
        x = torch.cat((x, self.disp_conv4_0(x)),1)
        x = torch.cat((x, self.disp_conv4_1(x)),1)
        x = torch.cat((x, self.disp_conv4_2(x)),1)
        x = torch.cat((x, self.disp_conv4_3(x)),1)
        x = torch.cat((x, self.disp_conv4_4(x)),1)
        depth4 = self.predict_depth4(x)
        up_depth4 = self.disp_deconv4(depth4)
        up_flow_feat4 = self.disp_upfeat4(x)

        disp_warp3 = self.warp_R2L(c23, up_depth4 / 8)
        disp_corr3 = self.disp_corr(c13, disp_warp3) 
        disp_corr3 = F.leaky_relu(disp_corr3, negative_slope=0.1)
        x = torch.cat((disp_corr3, c13, up_depth4, up_flow_feat4), 1)
        x = torch.cat((x, self.disp_conv3_0(x)),1)
        x = torch.cat((x, self.disp_conv3_1(x)),1)
        x = torch.cat((x, self.disp_conv3_2(x)),1)
        x = torch.cat((x, self.disp_conv3_3(x)),1)
        x = torch.cat((x, self.disp_conv3_4(x)),1)
        depth3 = self.predict_depth3(x)
        up_depth3 = self.disp_deconv3(depth3)
        up_disp_feat3 = self.disp_upfeat3(x)
        
        disp_warp2 = self.warp_R2L(c22, up_depth3 / 4) 
        disp_corr2 = self.disp_corr(c12, disp_warp2)
        disp_corr2 = F.leaky_relu(disp_corr2, negative_slope=0.1)
        x = torch.cat((disp_corr2, c12, up_depth3, up_disp_feat3), 1)
        x = torch.cat((x, self.disp_conv2_0(x)),1)
        x = torch.cat((x, self.disp_conv2_1(x)),1)
        x = torch.cat((x, self.disp_conv2_2(x)),1)
        x = torch.cat((x, self.disp_conv2_3(x)),1)
        x = torch.cat((x, self.disp_conv2_4(x)),1)
        depth2 = self.predict_depth2(x)
 
        x = self.disp_dc_conv4(self.disp_dc_conv3(self.disp_dc_conv2(self.disp_dc_conv1(x))))
        depth2 += self.disp_dc_conv7(self.disp_dc_conv6(self.disp_dc_conv5(x)))

        depth2 = F.upsample(depth2, scale_factor=4, mode='bilinear', align_corners=False)
        depth3 = F.upsample(depth3, scale_factor=4, mode='bilinear', align_corners=False)
        depth4 = F.upsample(depth4, scale_factor=4, mode='bilinear', align_corners=False)
        depth5 = F.upsample(depth5, scale_factor=4, mode='bilinear', align_corners=False)
        depth6 = F.upsample(depth6, scale_factor=4, mode='bilinear', align_corners=False)

        depth2 = torch.squeeze(depth2, 1)
        depth3 = torch.squeeze(depth3, 1)
        depth4 = torch.squeeze(depth4, 1)
        depth5 = torch.squeeze(depth5, 1)
        depth6 = torch.squeeze(depth6, 1)

        return (depth2, depth3, depth4, depth5, depth6)