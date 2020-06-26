"""
Copyright (C) 2019 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""
from __future__ import print_function
import argparse
import os
import random
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
from torch.autograd import Variable
import torch.nn.functional as F
import skimage
import skimage.io
import skimage.transform
import numpy as np
import time
import math
import cv2
import scipy.misc as smisc
from PIL import Image
from tqdm import tqdm

from sense.datasets.dataset_catlog import make_flow_disp_data
from sense.datasets.dataset_utils import imread, read_disp_pfm
import sense.datasets.sintel_io as sintel_io

import sense.models.model_utils as model_utils
from sense.models.self_supervised_loss import make_self_supervised_loss
from sense.models.common import disp_warp, flow_warp

import sense.utils.flowlib as flowlib
from sense.utils.arguments import parse_args
import sense.utils.kitti_viz as kitti_viz

def save_disp_results(save_dir, im1_path, im2_path, pred_disp):
	seq_dir, im1_name = os.path.split(im1_path)
	_, im2_name = os.path.split(im2_path)

	im1 = cv2.imread(im1_path)
	im2 = cv2.imread(im2_path)
	cv2.imwrite(os.path.join(save_dir, im1_name[:-4] + '_left.png'), im1)
	cv2.imwrite(os.path.join(save_dir, im1_name[:-4] + '_right.png'), im2)

	# save prediction
	cv2.imwrite(os.path.join(save_dir, im1_name[:-4] + '_disp.png'), pred_disp.astype(np.uint8))

def prepare_input_image(im_path, crop_imh=320, crop_imw=768, stride=32):
	im = imread(im_path)

	# padding
	orig_imh, orig_imw, _ = im.shape
	if orig_imh % stride != 0:
		num_feat_pix = orig_imh // stride
		new_imh = (num_feat_pix + 1) * stride
		im = np.pad(im, ((0, new_imh - orig_imh), (0, 0), (0, 0)), 'constant', constant_values=(0, 0))
	else:
		new_imh = orig_imh

	if orig_imw % stride != 0:
		num_feat_pix = orig_imw // stride
		new_imw = (num_feat_pix + 1) * stride
		im = np.pad(im, ((0, 0), (0, new_imw - orig_imw), (0, 0)), 'constant', constant_values=(0, 0))
	else:
		new_imw = orig_imw

	im = torch.from_numpy(im.transpose(2, 0, 1)).float().unsqueeze(0).cuda()
	return im, (orig_imh, orig_imw)

def check_disp_prediction(model, left_im_path, right_im_path, args):
	left_im, (imh, imw) = prepare_input_image(left_im_path)
	right_im, _ = prepare_input_image(right_im_path)

	im_dir, im_name = os.path.split(left_im_path)
	data_dir, seq_name = os.path.split(im_dir)
	_, split = os.path.split(data_dir)
	disp_path = os.path.join(args.save_dir, 'disp', split, seq_name, im_name)
	disp_occ_path = os.path.join(args.save_dir, 'disp_occ', split, seq_name, im_name)
	make_dir_p(disp_path)
	make_dir_p(disp_occ_path)

	with torch.no_grad():
		_, disp_pred, _ = model(left_im, None, right_im)

	disp = disp_pred[0][0].cpu().numpy().squeeze()

	if not args.save_occ_only:
		sintel_io.disparity_write(disp_path, disp[:imh, :imw])
		cv2.imwrite(disp_path[:-4] + '_raw.png', kitti_viz.disp_to_color(disp[:imh, :imw]))

	if args.disp_occ_thresh > 0:
		disp_occ = F.softmax(disp_pred[1][0], dim=1)[:, 1].cpu().numpy().squeeze()
		disp_occ_soft = (disp_occ * 255).astype(np.float32)
		if args.soft_occ_gt:
			cv2.imwrite(disp_occ_path, disp_occ_soft[:imh, :imw])
		else:
			disp_occ = disp_occ > args.disp_occ_thresh
			disp_occ = (disp_occ * 255).astype(np.uint8)
			cv2.imwrite(disp_occ_path, disp_occ[:imh, :imw])
		cv2.imwrite(disp_occ_path[:-4] + '_soft.png', disp_occ_soft[:imh, :imw])

def make_dir_p(im_path):
	im_dir, _ = os.path.split(im_path)
	if not os.path.exists(im_dir):
		os.makedirs(im_dir)

def check_flow_prediction(model, cur_im_path, nxt_im_path, args):
	cur_im, (imh, imw) = prepare_input_image(cur_im_path)
	nxt_im, _ = prepare_input_image(nxt_im_path)

	im_dir, im_name = os.path.split(cur_im_path)
	data_dir, seq_name = os.path.split(im_dir)
	_, split = os.path.split(data_dir)
	flow_path = os.path.join(args.save_dir, 'flow', split, seq_name, im_name[:-4] + '.flo')
	flow_im_path = os.path.join(args.save_dir, 'flow', split, seq_name, im_name[:-4] + '.png')
	flow_occ_path = os.path.join(args.save_dir, 'flow_occ', split, seq_name, im_name)
	make_dir_p(flow_path)
	make_dir_p(flow_occ_path)

	with torch.no_grad():
		flow_pred, _, _ = model(cur_im, nxt_im)

	if not args.upsample_flow_output:
		flow = F.upsample(flow_pred[0][0] * args.div_flow, scale_factor=4, mode='bilinear')
		flow_occ = F.upsample(flow_pred[1][0], scale_factor=4, mode='bilinear')
	else:
		flow = flow_pred[0][0] * args.div_flow
		flow_occ = flow_pred[1][0]
	flow = flow.cpu().numpy().squeeze().transpose(1, 2, 0)
	if not args.save_occ_only:
		flowlib.write_flow(flow[:imh, :imw], flow_path)
		flow_im = kitti_viz.viz_flow(flow[:imh, :imw, 0], flow[:imh, :imw, 1])
		cv2.imwrite(flow_path[:-4] + '.png', flow_im[:, :, ::-1])

	if args.flow_occ_thresh > 0:
		flow_occ = F.softmax(flow_occ, dim=1)[:, 1].cpu().numpy().squeeze()
		flow_occ_soft = (flow_occ * 255).astype(np.uint8)
		if args.soft_occ_gt:
			cv2.imwrite(flow_occ_path, flow_occ_soft[:imh, :imw])
		else:
			flow_occ = flow_occ > args.flow_occ_thresh
			flow_occ = (flow_occ * 255).astype(np.uint8)
			cv2.imwrite(flow_occ_path, flow_occ[:imh, :imw])
		cv2.imwrite(flow_occ_path[:-4] + '_soft.png', flow_occ_soft[:imh, :imw])		

def process_single_data(model, data, args):
	cur_im_path, nxt_im_path, left_im_path, right_im_path = data[0]
	flow_path, flow_occ_path, disp_path, disp_occ_path = data[1]

	# compute disparity
	check_disp_prediction(model, left_im_path, right_im_path, args)

	# compute optical flow
	check_flow_prediction(model, cur_im_path, nxt_im_path, args)

def main(args):
	if args.save_dir is not None and args.save_dir != 'None':
		if not os.path.exists(args.save_dir):
			os.makedirs(args.save_dir)
			
	# load model
	do_flow = True
	do_disp = False
	if args.joint_model:
		do_disp = True
	model = model_utils.make_model(args, do_flow, do_disp, do_seg=args.do_seg)
	print('Number of model parameters: {}'.format(sum([p.data.nelement() for p in model.parameters()])))

	dataset_info = args.dataset + '_{}'.format(args.split)
	if args.pass_opt is not None:
		dataset_info += '_{}'.format(args.pass_opt)

	ckpt = torch.load(args.loadmodel)
	state_dict = ckpt['state_dict']
	model.load_state_dict(model_utils.patch_model_state_dict(state_dict))
	print('==> Successfully loaded a model {}.'.format(args.loadmodel))

	model.eval()
	cudnn.benchmark = True

	train_data, test_data = make_flow_disp_data(args.dataset)
	if args.split.startswith('train'):      # train or training
		data = train_data
	else:
		data = test_data
	# print('There are {} data items to be processed.'.format(len(data)))

	for i in tqdm(range(len(data))):
		process_single_data(model, data[i], args)

if __name__ == '__main__':
	parser = parse_args()
	parser.add_argument('--joint-model', action='store_true')
	parser.add_argument(
		'--split', choices=['training', 'test', 'val'],
		default='training'
	)
	parser.add_argument(
		'--pass-opt', choices=[None, 'clean', 'final'],
		default=None
	)
	parser.add_argument('--stride', type=int, default=32)
	parser.add_argument('--flow-occ-thresh', type=float, default=0.45)
	parser.add_argument('--disp-occ-thresh', type=float, default=0.55)
	parser.add_argument('--save-occ-only', action='store_true')
	args = parser.parse_args()
	main(args)
