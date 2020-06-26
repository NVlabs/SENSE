"""
Copyright (C) 2019 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""
import torch
import torch.nn as nn

from sense.lib.nn import SynchronizedBatchNorm2d

# borrowed from https://github.com/CSAILVision/semantic-segmentation-pytorch/blob/master/models/models.py
def conv3x3(in_planes, out_planes, stride=1, has_bias=False):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=has_bias)


def conv3x3_bn_relu(in_planes, out_planes, stride=1):
    return nn.Sequential(
            conv3x3(in_planes, out_planes, stride),
            SynchronizedBatchNorm2d(out_planes),
            nn.ReLU(inplace=True),
            )

# lightweight upernet
class UPerNetLight(nn.Module):
    def __init__(self, num_class=150, fc_dim=4096,
                 fpn_inplanes=(256,512,1024,2048), fpn_dim=256,
                 ):
        super(UPerNetLight, self).__init__()

        # # PPM Module
        # self.ppm_pooling = []
        # self.ppm_conv = []

        # for scale in pool_scales:
        #     self.ppm_pooling.append(nn.AdaptiveAvgPool2d(scale))
        #     self.ppm_conv.append(nn.Sequential(
        #         nn.Conv2d(fc_dim, ppm_last_conv_dim, kernel_size=1, bias=False),
        #         SynchronizedBatchNorm2d(ppm_last_conv_dim),
        #         nn.ReLU(inplace=True)
        #     ))
        # self.ppm_pooling = nn.ModuleList(self.ppm_pooling)
        # self.ppm_conv = nn.ModuleList(self.ppm_conv)
        # self.ppm_last_conv = conv3x3_bn_relu(fc_dim + len(pool_scales)*ppm_last_conv_dim, fpn_dim, 1)

        self.trans_conv = conv3x3_bn_relu(fc_dim, fpn_dim, 1)

        # FPN Module
        self.fpn_in = []
        for fpn_inplane in fpn_inplanes[:-1]: # skip the top layer
            self.fpn_in.append(nn.Sequential(
                nn.Conv2d(fpn_inplane, fpn_dim, kernel_size=1, bias=False),
                SynchronizedBatchNorm2d(fpn_dim),
                nn.ReLU(inplace=True)
            ))
        self.fpn_in = nn.ModuleList(self.fpn_in)

        self.fpn_out = []
        for i in range(len(fpn_inplanes) - 1): # skip the top layer
            self.fpn_out.append(nn.Sequential(
                conv3x3_bn_relu(fpn_dim, fpn_dim, 1),
            ))
        self.fpn_out = nn.ModuleList(self.fpn_out)

        self.conv_last = nn.Sequential(
            conv3x3_bn_relu(len(fpn_inplanes) * fpn_dim, fpn_dim, 1),
            nn.Conv2d(fpn_dim, num_class, kernel_size=1)
        )

    def forward(self, conv_out, segSize=None, T=1, use_softmax=False):
        conv5 = conv_out[-1]

        # input_size = conv5.size()
        # ppm_out = [conv5]
        # for pool_scale, pool_conv in zip(self.ppm_pooling, self.ppm_conv):
        #     ppm_out.append(pool_conv(nn.functional.upsample(
        #         pool_scale(conv5),
        #         (input_size[2], input_size[3]),
        #         mode='bilinear', align_corners=False)))
        # ppm_out = torch.cat(ppm_out, 1)
        # f = self.ppm_last_conv(ppm_out)

        f = self.trans_conv(conv5)

        fpn_feature_list = [f]
        for i in reversed(range(len(conv_out) - 1)):
            conv_x = conv_out[i]
            conv_x = self.fpn_in[i](conv_x) # lateral branch

            f = nn.functional.upsample(
                f, size=conv_x.size()[2:], mode='bilinear', align_corners=False) # top-down branch
            f = conv_x + f

            fpn_feature_list.append(self.fpn_out[i](f))

        fpn_feature_list.reverse() # [P2 - P5]
        output_size = fpn_feature_list[0].size()[2:]
        fusion_list = [fpn_feature_list[0]]
        for i in range(1, len(fpn_feature_list)):
            fusion_list.append(nn.functional.upsample(
                fpn_feature_list[i],
                output_size,
                mode='bilinear', align_corners=False))
        fusion_out = torch.cat(fusion_list, 1)
        x = self.conv_last(fusion_out)

        if use_softmax:  # is True during inference
            x = nn.functional.upsample(
                x, size=segSize, mode='bilinear', align_corners=False)
            x = nn.functional.softmax(x / T, dim=1)
            return x

        if segSize is not None:
            x = nn.functional.upsample(
                x, size=segSize, mode='bilinear', align_corners=False)
        x = nn.functional.log_softmax(x / T, dim=1)

        return x

# upernet
class UPerNet(nn.Module):
    def __init__(self, num_class=150, fc_dim=4096,
                 pool_scales=(1, 2, 3, 6),
                 ppm_last_conv_dim=512,
                 fpn_inplanes=(256,512,1024,2048), fpn_dim=256
                 ):
        super(UPerNet, self).__init__()

        # PPM Module
        self.ppm_pooling = []
        self.ppm_conv = []

        for scale in pool_scales:
            self.ppm_pooling.append(nn.AdaptiveAvgPool2d(scale))
            self.ppm_conv.append(nn.Sequential(
                nn.Conv2d(fc_dim, ppm_last_conv_dim, kernel_size=1, bias=False),
                SynchronizedBatchNorm2d(ppm_last_conv_dim),
                nn.ReLU(inplace=True)
            ))
        self.ppm_pooling = nn.ModuleList(self.ppm_pooling)
        self.ppm_conv = nn.ModuleList(self.ppm_conv)
        self.ppm_last_conv = conv3x3_bn_relu(fc_dim + len(pool_scales)*ppm_last_conv_dim, fpn_dim, 1)

        # FPN Module
        self.fpn_in = []
        for fpn_inplane in fpn_inplanes[:-1]: # skip the top layer
            self.fpn_in.append(nn.Sequential(
                nn.Conv2d(fpn_inplane, fpn_dim, kernel_size=1, bias=False),
                SynchronizedBatchNorm2d(fpn_dim),
                nn.ReLU(inplace=True)
            ))
        self.fpn_in = nn.ModuleList(self.fpn_in)

        self.fpn_out = []
        for i in range(len(fpn_inplanes) - 1): # skip the top layer
            self.fpn_out.append(nn.Sequential(
                conv3x3_bn_relu(fpn_dim, fpn_dim, 1),
            ))
        self.fpn_out = nn.ModuleList(self.fpn_out)

        self.conv_last = nn.Sequential(
            conv3x3_bn_relu(len(fpn_inplanes) * fpn_dim, fpn_dim, 1),
            nn.Conv2d(fpn_dim, num_class, kernel_size=1)
        )

    def forward(self, conv_out, segSize=None, T=1, use_softmax=False):
        conv5 = conv_out[-1]

        input_size = conv5.size()
        ppm_out = [conv5]
        for pool_scale, pool_conv in zip(self.ppm_pooling, self.ppm_conv):
            ppm_out.append(pool_conv(nn.functional.upsample(
                pool_scale(conv5),
                (input_size[2], input_size[3]),
                mode='bilinear', align_corners=False)))
        ppm_out = torch.cat(ppm_out, 1)
        f = self.ppm_last_conv(ppm_out)

        fpn_feature_list = [f]
        for i in reversed(range(len(conv_out) - 1)):
            conv_x = conv_out[i]
            conv_x = self.fpn_in[i](conv_x) # lateral branch

            f = nn.functional.upsample(
                f, size=conv_x.size()[2:], mode='bilinear', align_corners=False) # top-down branch
            f = conv_x + f

            fpn_feature_list.append(self.fpn_out[i](f))

        fpn_feature_list.reverse() # [P2 - P5]
        output_size = fpn_feature_list[0].size()[2:]
        fusion_list = [fpn_feature_list[0]]
        for i in range(1, len(fpn_feature_list)):
            fusion_list.append(nn.functional.upsample(
                fpn_feature_list[i],
                output_size,
                mode='bilinear', align_corners=False))
        fusion_out = torch.cat(fusion_list, 1)
        x = self.conv_last(fusion_out)

        if use_softmax:  # is True during inference
            x = nn.functional.upsample(
                x, size=segSize, mode='bilinear', align_corners=False)
            x = nn.functional.softmax(x / T, dim=1)
            return x

        if segSize is not None:
            x = nn.functional.upsample(
                x, size=segSize, mode='bilinear', align_corners=False)
        x = nn.functional.log_softmax(x / T, dim=1)

        return x

class Resnet(nn.Module):
    def __init__(self, orig_resnet):
        super(Resnet, self).__init__()

        # take pretrained resnet, except AvgPool and FC
        self.conv1 = orig_resnet.conv1
        self.bn1 = orig_resnet.bn1
        self.relu1 = orig_resnet.relu1
        self.conv2 = orig_resnet.conv2
        self.bn2 = orig_resnet.bn2
        self.relu2 = orig_resnet.relu2
        self.conv3 = orig_resnet.conv3
        self.bn3 = orig_resnet.bn3
        self.relu3 = orig_resnet.relu3
        self.maxpool = orig_resnet.maxpool
        self.layer1 = orig_resnet.layer1
        self.layer2 = orig_resnet.layer2
        self.layer3 = orig_resnet.layer3
        self.layer4 = orig_resnet.layer4

    def forward(self, x, return_feature_maps=False):
        conv_out = []

        x = self.relu1(self.bn1(self.conv1(x)))
        x = self.relu2(self.bn2(self.conv2(x)))
        x = self.relu3(self.bn3(self.conv3(x)))
        x = self.maxpool(x)

        x = self.layer1(x); conv_out.append(x);
        x = self.layer2(x); conv_out.append(x);
        x = self.layer3(x); conv_out.append(x);
        x = self.layer4(x); conv_out.append(x);

        if return_feature_maps:
            return conv_out
        return [x]

class SegmentationModule(nn.Module):
    def __init__(self, net_enc, net_dec, deep_sup_scale=None):
        super(SegmentationModule, self).__init__()
        self.encoder = net_enc
        self.decoder = net_dec
        self.deep_sup_scale = deep_sup_scale

    def forward(self, img_data, segSize=None, T=1, use_softmax=False):
        if segSize is None: # training
            if self.deep_sup_scale is not None: # use deep supervision technique
                (pred, pred_deepsup) = self.decoder(self.encoder(img_data, return_feature_maps=True), T=T, use_softmax=use_softmax)
            else:
                pred = self.decoder(self.encoder(img_data, return_feature_maps=True), T=T, use_softmax=use_softmax)
                pred_deepsup = None
            return pred, pred_deepsup
        else: # inference
            pred = self.decoder(self.encoder(img_data, return_feature_maps=True), segSize=segSize, T=T, use_softmax=use_softmax)
            return pred

def make_resnet101_upernet(args):
    assert args.seg_teacher_encoder_weights is not None, 'Teacher encoder weights file is not given.'
    assert args.seg_teacher_decoder_weights is not None, 'Teacher decoder weights file is not given.'

    # building the model
    orig_resnet = resnet.__dict__['resnet101'](pretrained=False)
    net_encoder = Resnet(orig_resnet)
    net_encoder.load_state_dict(
        torch.load(args.seg_teacher_encoder_weights, map_location=lambda storage, loc: storage), strict=False)

    net_decoder = UPerNet(
        fc_dim=2048,
        num_class=args.num_seg_class,
        fpn_dim=512)
    net_decoder.load_state_dict(
        torch.load(args.seg_teacher_decoder_weights, map_location=lambda storage, loc: storage), strict=False)

    segmentation_module = SegmentationModule(
        net_encoder, net_decoder)
    return segmentation_module