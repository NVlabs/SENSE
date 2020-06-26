# BSD 3-Clause License

# Copyright (c) 2019, MIT CSAIL Computer Vision
# All rights reserved.

# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:

# * Redistributions of source code must retain the above copyright notice, this
#   list of conditions and the following disclaimer.

# * Redistributions in binary form must reproduce the above copyright notice,
#   this list of conditions and the following disclaimer in the documentation
#   and/or other materials provided with the distribution.

# * Neither the name of the copyright holder nor the names of its
#   contributors may be used to endorse or promote products derived from
#   this software without specific prior written permission.

# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

# -*- coding: utf-8 -*-
# File   : test_sync_batchnorm.py
# Author : Jiayuan Mao
# Email  : maojiayuan@gmail.com
# Date   : 27/01/2018
# 
# This file is part of Synchronized-BatchNorm-PyTorch.

import unittest

import torch
import torch.nn as nn
from torch.autograd import Variable

from sync_batchnorm import SynchronizedBatchNorm1d, SynchronizedBatchNorm2d, DataParallelWithCallback
from sync_batchnorm.unittest import TorchTestCase


def handy_var(a, unbias=True):
    n = a.size(0)
    asum = a.sum(dim=0)
    as_sum = (a ** 2).sum(dim=0)  # a square sum
    sumvar = as_sum - asum * asum / n
    if unbias:
        return sumvar / (n - 1)
    else:
        return sumvar / n


def _find_bn(module):
    for m in module.modules():
        if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, SynchronizedBatchNorm1d, SynchronizedBatchNorm2d)):
            return m


class SyncTestCase(TorchTestCase):
    def _syncParameters(self, bn1, bn2):
        bn1.reset_parameters()
        bn2.reset_parameters()
        if bn1.affine and bn2.affine:
            bn2.weight.data.copy_(bn1.weight.data)
            bn2.bias.data.copy_(bn1.bias.data)

    def _checkBatchNormResult(self, bn1, bn2, input, is_train, cuda=False):
        """Check the forward and backward for the customized batch normalization."""
        bn1.train(mode=is_train)
        bn2.train(mode=is_train)

        if cuda:
            input = input.cuda()

        self._syncParameters(_find_bn(bn1), _find_bn(bn2))

        input1 = Variable(input, requires_grad=True)
        output1 = bn1(input1)
        output1.sum().backward()
        input2 = Variable(input, requires_grad=True)
        output2 = bn2(input2)
        output2.sum().backward()

        self.assertTensorClose(input1.data, input2.data)
        self.assertTensorClose(output1.data, output2.data)
        self.assertTensorClose(input1.grad, input2.grad)
        self.assertTensorClose(_find_bn(bn1).running_mean, _find_bn(bn2).running_mean)
        self.assertTensorClose(_find_bn(bn1).running_var, _find_bn(bn2).running_var)

    def testSyncBatchNormNormalTrain(self):
        bn = nn.BatchNorm1d(10)
        sync_bn = SynchronizedBatchNorm1d(10)

        self._checkBatchNormResult(bn, sync_bn, torch.rand(16, 10), True)

    def testSyncBatchNormNormalEval(self):
        bn = nn.BatchNorm1d(10)
        sync_bn = SynchronizedBatchNorm1d(10)

        self._checkBatchNormResult(bn, sync_bn, torch.rand(16, 10), False)

    def testSyncBatchNormSyncTrain(self):
        bn = nn.BatchNorm1d(10, eps=1e-5, affine=False)
        sync_bn = SynchronizedBatchNorm1d(10, eps=1e-5, affine=False)
        sync_bn = DataParallelWithCallback(sync_bn, device_ids=[0, 1])

        bn.cuda()
        sync_bn.cuda()

        self._checkBatchNormResult(bn, sync_bn, torch.rand(16, 10), True, cuda=True)

    def testSyncBatchNormSyncEval(self):
        bn = nn.BatchNorm1d(10, eps=1e-5, affine=False)
        sync_bn = SynchronizedBatchNorm1d(10, eps=1e-5, affine=False)
        sync_bn = DataParallelWithCallback(sync_bn, device_ids=[0, 1])

        bn.cuda()
        sync_bn.cuda()

        self._checkBatchNormResult(bn, sync_bn, torch.rand(16, 10), False, cuda=True)

    def testSyncBatchNorm2DSyncTrain(self):
        bn = nn.BatchNorm2d(10)
        sync_bn = SynchronizedBatchNorm2d(10)
        sync_bn = DataParallelWithCallback(sync_bn, device_ids=[0, 1])

        bn.cuda()
        sync_bn.cuda()

        self._checkBatchNormResult(bn, sync_bn, torch.rand(16, 10, 16, 16), True, cuda=True)


if __name__ == '__main__':
    unittest.main()
