"""
Copyright (C) 2019 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""
from __future__ import print_function

import torch
import torch.nn as nn
import torchvision
from sense.lib.nn import SynchronizedBatchNorm2d

class SceneNet(nn.Module):
	def __init__(self, net_enc, flow_dec=None, disp_dec=None, 
			seg_dec=None, bn_type='plain',
			disp_with_ppm=False, flow_with_ppm=False
		):
		super(SceneNet, self).__init__()
		self.encoder = net_enc
		self.flow_decoder = flow_dec
		self.disp_decoder = disp_dec
		self.seg_decoder = seg_dec
		assert flow_dec is not None \
			   or disp_dec is not None \
			   or seg_dec is not None, \
			   'at least one of the decoders should not be None'

		self.bn_type = bn_type
		self.disp_with_ppm = disp_with_ppm
		self.flow_with_ppm = flow_with_ppm

		self.weight_init()

	def weight_init(self):
		if self.bn_type == 'encoding':
			raise NotImplementedError
		for m in self.modules():
			if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
				nn.init.kaiming_normal_(m.weight.data, mode='fan_in')
				if m.bias is not None:
					m.bias.data.zero_()
			if isinstance(m, SynchronizedBatchNorm2d):
				if self.bn_type == 'syncbn':
					m.weight.data.fill_(1)
					m.bias.data.zero_()
				else:
					raise Exception('There should be no SynchronizedBatchNorm2d layers.')
			if isinstance(m, nn.BatchNorm2d):
				if self.bn_type == 'plain':
					m.weight.data.fill_(1)
					m.bias.data.zero_()
				else:
					raise Exception('There should be no nn.BatchNorm2d layers.')		

	def forward(self, cur_im, nxt_im=None, left_im=None, right_im=None, 
			reuse_first_im=True, seg_use_softmax=False, seg_size=None,
			do_seg_every_im=False, crop_imh_for_seg=-1, crop_imw_for_seg=-1
		):
		cur_x = self.encoder(cur_im)
		if nxt_im is not None:
			nxt_x = self.encoder(nxt_im)
		if left_im is not None:
			left_x = self.encoder(left_im)
		if not reuse_first_im and right_im is not None:
			right_x = self.encoder(right_im)

		if self.flow_decoder is not None and nxt_im is not None:
			if self.flow_with_ppm:
				cur_x_ = cur_x[:4] + [cur_x[-1]]
				nxt_x_ = nxt_x[:4] + [nxt_x[-1]]
			else:
				cur_x_ = cur_x[:5]
				nxt_x_ = nxt_x[:5]
			flow_multi_scale = self.flow_decoder(cur_x_, nxt_x_)
		else:
			flow_multi_scale = None

		if self.disp_decoder is not None and left_im is not None:
			if reuse_first_im:
				if self.disp_with_ppm:
					cur_x_ = cur_x[:4] + [cur_x[-1]]
					left_x_ = left_x[:4] + [left_x[-1]]
				else:
					cur_x_ = cur_x[:5]
					left_x_ = left_x[:5]
				disp_multi_scale = self.disp_decoder(cur_x_, left_x_)
			else:
				if self.disp_with_ppm:
					left_x_ = left_x[:4] + [left_x[-1]]
					right_x_ = right_x[:4] + [right_x[-1]]
				else:
					left_x_ = left_x[:5]
					right_x_ = right_x[:5]
				disp_multi_scale = self.disp_decoder(left_x_, right_x_)
		else:
			disp_multi_scale = None

		# if seg_size is None:
		# 	seg_size = cur_im.shape[2:]
		if do_seg_every_im:
			assert len(cur_x) == 6, 'Something is seriously wrong with the encoder'
			cur_x_ = cur_x[1:4] + [cur_x[-1]]
			cur_seg = self.seg_decoder(cur_x_, use_softmax=seg_use_softmax, segSize=seg_size)

			if nxt_im is not None:
				assert len(nxt_x) == 6, 'Something is seriously wrong with the encoder'
				nxt_x_ = nxt_x[1:4] + [nxt_x[-1]]
				nxt_seg = self.seg_decoder(nxt_x_, use_softmax=seg_use_softmax, segSize=seg_size)
			else:
				nxt_seg = None

			if left_im is not None:
				assert len(left_x) == 6, 'Something is seriously wrong with the encoder'
				left_x_ = left_x[1:4] + [left_x[-1]]
				left_seg = self.seg_decoder(left_x_, use_softmax=seg_use_softmax, segSize=seg_size)
			else:
				left_seg = None

			if right_im is not None:
				assert len(right_x) == 6, 'Something is seriously wrong with the encoder'
				right_x_ = right_x[1:4] + [right_x[-1]]
				right_seg = self.seg_decoder(right_x_, use_softmax=seg_use_softmax, segSize=seg_size)
			else:
				right_seg = None

			seg = (cur_seg, nxt_seg, left_seg, right_seg)
		else:
			if self.seg_decoder is not None:
				if reuse_first_im:
					assert len(cur_x) == 6, 'Something is seriously wrong with the encoder'
					cur_x_ = cur_x[1:4] + [cur_x[-1]]
					seg = self.seg_decoder(cur_x_, use_softmax=seg_use_softmax, segSize=seg_size)
				else:
					assert len(left_x) == 6, 'Something is seriously wrong with the encoder'
					left_x_ = left_x[1:4] + [left_x[-1]]
					seg = self.seg_decoder(left_x_, use_softmax=seg_use_softmax, segSize=seg_size)
			else:
				seg = None
			seg = (seg,)

		return flow_multi_scale, disp_multi_scale, seg
