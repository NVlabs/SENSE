"""
Copyright (C) 2019 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""

from __future__ import print_function
from __future__ import division

import numpy as np
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Module

class UNet(nn.Module):
    def __init__(self):
        super(UNet, self).__init__()

        # --------------------- encoder --------------------
        # conv1
        self.encoder_layer1 = self.make_encoder_layer(6, 32, 7, 3)
        self.encoder_layer2 = self.make_encoder_layer(32, 64, 5, 2)
        self.encoder_layer3 = self.make_encoder_layer(64, 128)
        self.encoder_layer4 = self.make_encoder_layer(128, 256)
        self.encoder_layer5 = self.make_encoder_layer(256, 512)

        self.bottleneck = self.make_encoder_layer(512, 512)

        self.decoder_layer5 = self.make_decoder_layer(512, 512)
        self.decoder_layer4 = self.make_decoder_layer(1024, 256)
        self.decoder_layer3 = self.make_decoder_layer(512, 128)
        self.decoder_layer2 = self.make_decoder_layer(256, 64)
        self.decoder_layer1 = self.make_decoder_layer(128, 32)

        self.refine_layer = nn.Sequential(
            nn.Conv2d(64, 32, 3, padding=1),
            nn.ReLU(inplace=True)
        )

        self.out_conv5 = nn.Conv2d(512, 1, 3, padding=1)
        self.out_conv4 = nn.Conv2d(256, 1, 3, padding=1)
        self.out_conv3 = nn.Conv2d(128, 1, 3, padding=1)
        self.out_conv2 = nn.Conv2d(64, 1, 3, padding=1)
        self.out_conv1 = nn.Conv2d(32, 1, 3, padding=1)

        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                nn.init.kaiming_normal_(m.weight.data, mode='fan_in')
                if m.bias is not None:
                    m.bias.data.zero_()
            if isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def make_encoder_layer(self, in_chn, out_chn, kernel_size=3, padding=1):
        layer = nn.Sequential(
            nn.Conv2d(in_chn, out_chn, kernel_size, padding=padding),
            nn.BatchNorm2d(out_chn),
            nn.ReLU(inplace=True)
        )
        return layer

    def make_decoder_layer(self, in_chn, out_chn):
        layer = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(in_chn, out_chn, 3, padding=1),
            nn.BatchNorm2d(out_chn),
            nn.ReLU(inplace=True)
        )
        return layer

    def check_nan_in_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight.data, mode='fan_in')
                if m.bias is not None:
                    m.bias.data.zero_()
            if isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, x):
        # self.check_nan_in_weights()

        encoder_out1 = self.encoder_layer1(x)
        encoder_out1_pool = F.avg_pool2d(encoder_out1, 2, stride=2)

        encoder_out2 = self.encoder_layer2(encoder_out1_pool)
        encoder_out2_pool = F.avg_pool2d(encoder_out2, 2, stride=2)

        encoder_out3 = self.encoder_layer3(encoder_out2_pool)
        encoder_out3_pool = F.avg_pool2d(encoder_out3, 2, stride=2)

        encoder_out4 = self.encoder_layer4(encoder_out3_pool)
        encoder_out4_pool = F.avg_pool2d(encoder_out4, 2, stride=2)

        encoder_out5 = self.encoder_layer5(encoder_out4_pool)
        encoder_out5_pool = F.avg_pool2d(encoder_out5, 2, stride=2)

        bottleneck_out = self.bottleneck(encoder_out5_pool)

        decoder_out5 = self.decoder_layer5(bottleneck_out)
        out5 = self.out_conv5(decoder_out5)
        decoder_out5 = torch.cat((encoder_out5, decoder_out5), dim=1)

        decoder_out4 = self.decoder_layer4(decoder_out5)
        out4 = self.out_conv4(decoder_out4)
        decoder_out4 = torch.cat((encoder_out4, decoder_out4), dim=1)

        decoder_out3 = self.decoder_layer3(decoder_out4)
        out3 = self.out_conv3(decoder_out3)
        decoder_out3 = torch.cat((encoder_out3, decoder_out3), dim=1)

        decoder_out2 = self.decoder_layer2(decoder_out3)
        out2 = self.out_conv2(decoder_out2)
        decoder_out2 = torch.cat((encoder_out2, decoder_out2), dim=1)

        decoder_out1 = self.decoder_layer1(decoder_out2)
        decoder_out1 = torch.cat((encoder_out1, decoder_out1), dim=1)

        motion_rep = self.refine_layer(decoder_out1)

        out1 = self.out_conv1(motion_rep)

        return (out5, out4, out3, out2, out1)
        # return out1

class LightWeightUNet(nn.Module):
    def __init__(self):
        super(LightWeightUNet, self).__init__()

        # --------------------- encoder --------------------
        # conv1
        self.encoder_layer1 = self.make_encoder_layer(6, 32, 7, 3)
        self.encoder_layer2 = self.make_encoder_layer(32, 64, 5, 2)
        self.encoder_layer3 = self.make_encoder_layer(64, 128)

        self.bottleneck = self.make_encoder_layer(128, 128)

        self.decoder_layer3 = self.make_decoder_layer(128, 128)
        self.decoder_layer2 = self.make_decoder_layer(256, 64)
        self.decoder_layer1 = self.make_decoder_layer(128, 32)

        self.refine_layer = nn.Sequential(
            nn.Conv2d(64, 32, 3, padding=1),
            nn.ReLU(inplace=True)
        )

        self.out_conv3 = nn.Conv2d(128, 1, 3, padding=1)
        self.out_conv2 = nn.Conv2d(64, 1, 3, padding=1)
        self.out_conv1 = nn.Conv2d(32, 1, 3, padding=1)

        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                nn.init.kaiming_normal_(m.weight.data, mode='fan_in')
                if m.bias is not None:
                    m.bias.data.zero_()
            if isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def make_encoder_layer(self, in_chn, out_chn, kernel_size=3, padding=1):
        layer = nn.Sequential(
            nn.Conv2d(in_chn, out_chn, kernel_size, padding=padding),
            nn.BatchNorm2d(out_chn),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_chn, out_chn, kernel_size, padding=padding),
            nn.BatchNorm2d(out_chn),
            nn.ReLU(inplace=True)
        )
        return layer

    def make_decoder_layer(self, in_chn, out_chn):
        layer = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(in_chn, out_chn, 3, padding=1),
            nn.BatchNorm2d(out_chn),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_chn, out_chn, 3, padding=1),
            nn.BatchNorm2d(out_chn),
            nn.ReLU(inplace=True)
        )
        return layer

    def check_nan_in_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight.data, mode='fan_in')
                if m.bias is not None:
                    m.bias.data.zero_()
            if isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, x):
        # self.check_nan_in_weights()

        encoder_out1 = self.encoder_layer1(x)
        encoder_out1_pool = F.avg_pool2d(encoder_out1, 2, stride=2)

        encoder_out2 = self.encoder_layer2(encoder_out1_pool)
        encoder_out2_pool = F.avg_pool2d(encoder_out2, 2, stride=2)

        encoder_out3 = self.encoder_layer3(encoder_out2_pool)
        encoder_out3_pool = F.avg_pool2d(encoder_out3, 2, stride=2)

        bottleneck_out = self.bottleneck(encoder_out3_pool)

        decoder_out3 = self.decoder_layer3(bottleneck_out)
        out3 = self.out_conv3(decoder_out3)
        decoder_out3 = torch.cat((encoder_out3, decoder_out3), dim=1)

        decoder_out2 = self.decoder_layer2(decoder_out3)
        out2 = self.out_conv2(decoder_out2)
        decoder_out2 = torch.cat((encoder_out2, decoder_out2), dim=1)

        decoder_out1 = self.decoder_layer1(decoder_out2)
        decoder_out1 = torch.cat((encoder_out1, decoder_out1), dim=1)

        motion_rep = self.refine_layer(decoder_out1)

        out1 = self.out_conv1(motion_rep)

        return (out3, out2, out1)
        # return out1