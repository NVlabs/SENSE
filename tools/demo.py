"""
Copyright (C) 2019 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""

import sys
sys.path.insert(0, '.')

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
import argparse

from sense.models.unet import UNet
import sense.models.model_utils as model_utils
from sense.datasets.dataset_utils import imread, read_disp_pfm
from sense.datasets.cityscapes_dataset import CITYSCAPE_PALETTE
from sense.rigidity_refine.io_utils import read_camera_data
from sense.rigidity_refine.rigidity_refine import warp_disp_refine_rigid
import sense.utils.kitti_viz as kitti_viz
from sense.utils.arguments import parse_args

def image_to_tensor(image):
	# padding
	stride = args.stride
	orig_imh, orig_imw, _ = image.shape
	if orig_imh % stride != 0:
		num_feat_pix = orig_imh // stride
		new_imh = (num_feat_pix + 1) * stride
		image = np.pad(image, ((0, new_imh - orig_imh), (0, 0), (0, 0)), 'constant', constant_values=(0, 0))
	else:
		new_imh = orig_imh

	if orig_imw % stride != 0:
		num_feat_pix = orig_imw // stride
		new_imw = (num_feat_pix + 1) * stride
		image = np.pad(image, ((0, 0), (0, new_imw - orig_imw), (0, 0)), 'constant', constant_values=(0, 0))
	else:
		new_imw = orig_imw

	# to tensor
	image = image.transpose(2, 0, 1)
	image = torch.from_numpy(image).float().unsqueeze(0).cuda()
	return image, (orig_imh, orig_imw)

def run_holistic_scene_model(cur_left_im, cur_right_im, 
	nxt_left_im, nxt_right_im, holistic_scene_model):
	cur_left_im, (orig_imh, orig_imw) = image_to_tensor(cur_left_im)
	cur_right_im, _ = image_to_tensor(cur_right_im)
	nxt_left_im, _ = image_to_tensor(nxt_left_im)
	nxt_right_im, _ = image_to_tensor(nxt_right_im)

	with torch.no_grad():
		# current time step
		print('cur_left_im: {:.3f}, {:.3f}'.format(cur_left_im.min().item(), cur_left_im.max().item()))
		print('nxt_left_im: {:.3f}, {:.3f}'.format(nxt_left_im.min().item(), nxt_left_im.max().item()))
		flow_pred, disp_pred0, seg_pred = holistic_scene_model(
			cur_left_im, 
			nxt_left_im, 
			cur_right_im,
			seg_size=cur_left_im.shape[2:]
		)
		flow = flow_pred[0][0] * args.div_flow
		flow_occ = flow_pred[1][0]
		flow_occ = F.softmax(flow_occ, dim=1)[:, 1].unsqueeze(1)
		disp0 = disp_pred0[0][0]
		seg = seg_pred[0]
		_, seg = torch.max(seg, 1)
		# next time step
		_, disp_pred1, _ = holistic_scene_model(nxt_left_im, None, nxt_right_im)
		disp1 = disp_pred1[0][0]

	# convert to CPU
	flow = flow.cpu().numpy().squeeze()[:, :orig_imh, :orig_imw].transpose(1, 2, 0)
	flow_occ = flow_occ.cpu().numpy().squeeze()[:orig_imh, :orig_imw]
	disp0 = disp0.cpu().numpy().squeeze()[:orig_imh, :orig_imw]
	disp1 = disp1.cpu().numpy().squeeze()[:orig_imh, :orig_imw]
	seg = seg.cpu().numpy().squeeze()[:orig_imh, :orig_imw]
	return flow, flow_occ, disp0, disp1, seg

def run_warped_disparity_refinement(cur_left_im, flow_raw, flow_occ,
		disp0, disp1_unwarped, seg, camera_data, warp_disp_refine_model):
	# rigidity-based refinement first
	print('    ==> 1. Disparity-based refinement first.')
	K0, K1 = camera_data
	flow_rigid, disp1_raw, disp1_rigid = warp_disp_refine_rigid(disp0, disp1_unwarped, flow_raw, seg, K0, K1)
	disp1_rigid_orig = disp1_rigid.copy()

	# network-based refinement then
	print('    ==> 2. Network-based refinement.')
	disp0 = disp0[:, :, np.newaxis]
	disp1_rigid = disp1_rigid[:, :, np.newaxis]
	cur_left_im, (orig_imh, orig_imw) = image_to_tensor(cur_left_im)
	flow_occ = flow_occ[:, :, np.newaxis]
	flow_occ, _ = image_to_tensor(flow_occ)
	disp0, _ = image_to_tensor(disp0)
	disp1_rigid, _ = image_to_tensor(disp1_rigid)
	with torch.no_grad():
		input_data = torch.cat(
			(cur_left_im, flow_occ, disp0 / 20, disp1_rigid / 20),
			dim=1
		)
		warp_disp_residual = warp_disp_refine_model(input_data)
		disp1_nn = (disp1_rigid / 20 + warp_disp_residual[-1]) * 20
	disp1_nn = disp1_nn.cpu().numpy().squeeze()
	disp1_nn = disp1_nn[:orig_imh, :orig_imw]
	disp1_nn[disp1_nn < 0] = 0
	return flow_rigid, disp1_raw, disp1_rigid_orig, disp1_nn

def main(args):
	# holistic scene model		
	print('==> Making a holistic scene model.')
	holistic_scene_model = model_utils.make_model(args, do_flow=True, do_disp=True, do_seg=True)
	# print('Number of model parameters: {}'.format(sum([p.data.nelement() for p in model.parameters()])))
	holistic_scene_model_path = 'data/pretrained_models/kitti2012+kitti2015_new_lr_schedule_lr_disrupt+semi_loss_v3.pth'
	ckpt = torch.load(holistic_scene_model_path)
	state_dict = model_utils.patch_model_state_dict(ckpt['state_dict'])
	holistic_scene_model.load_state_dict(state_dict)
	holistic_scene_model.eval()

	# warped disparity refinement model for scene flow estimation
	print('==> Making a warped disparity refinment model for scene flow estimation.')
	warp_disp_ref_model = UNet()
	warp_disp_ref_model = nn.DataParallel(warp_disp_ref_model).cuda()
	warp_disp_ref_model_path = 'data/pretrained_models/kitti2015_warp_disp_refine_1500.pth'
	ckpt = torch.load(warp_disp_ref_model_path)
	state_dict = ckpt['state_dict']
	warp_disp_ref_model.load_state_dict(state_dict)
	warp_disp_ref_model.eval()

	cudnn.benchmark = True

	# input data
	cur_left_im = imread('data/image_2/000010_10.png')
	cur_right_im = imread('data/image_3/000010_10.png')
	nxt_left_im = imread('data/image_2/000010_11.png')
	nxt_right_im = imread('data/image_3/000010_11.png')
	camera_data = read_camera_data('data/calib_cam_to_cam/000010.txt')

	# optical flow, stereo disparity, and semantic segmentation estimation
	print('==> Running the holistic scene model for optical flow, stereo disparity, and semantic segmentation.')
	flow_raw, flow_occ, disp0, disp1_unwarped, seg = run_holistic_scene_model(
		cur_left_im, cur_right_im,
		nxt_left_im, nxt_right_im,
		holistic_scene_model
	)

	# run refinement for warped disparity (disp1) for scene flow estimation
	print('==> Running the warped disparity refinement model.')
	flow_rigid, disp1_raw, disp1_rigid, disp1_nn = run_warped_disparity_refinement(
		cur_left_im,
		flow_raw, flow_occ,
		disp0, disp1_unwarped,
		seg,
		camera_data,
		warp_disp_ref_model
	)

	# save results
	print('==> Saving results.')
	os.makedirs('data/results', exist_ok=True)
	smisc.imsave('data/results/000010_flow_raw.png', kitti_viz.flow_to_color(flow_raw))
	smisc.imsave('data/results/000010_flow_rigid.png', kitti_viz.flow_to_color(flow_rigid))
	smisc.imsave('data/results/000010_disp0.png', kitti_viz.disp_to_color(disp0))
	smisc.imsave('data/results/000010_disp1_raw.png', kitti_viz.disp_to_color(disp1_raw))
	smisc.imsave('data/results/000010_disp1_rigid.png', kitti_viz.disp_to_color(disp1_rigid))
	smisc.imsave('data/results/000010_disp1_nn.png', kitti_viz.disp_to_color(disp1_nn))

	seg_im = CITYSCAPE_PALETTE[seg]
	smisc.imsave('data/results/000010_seg.png', seg_im.astype('uint8'))

if __name__ == '__main__':
	parser = argparse.ArgumentParser('SENSE demo for scene flow estimation')
	args = parser.parse_args()

	# stereo disparity
	args.enc_arch = 'psm'
	args.dec_arch = 'pwcdc'
	args.disp_refinement = 'hourglass'
	args.no_ppm = False
	args.do_class = False
	# optical flow
	args.flow_dec_arch = 'pwcdc'
	args.flow_refinement = 'none'
	args.flow_no_ppm = True
	args.upsample_flow_output = True
	args.div_flow = 20
	# semantic segmentation
	args.num_seg_class = 19
	# other options
	args.bn_type = 'syncbn'
	args.corr_radius = 4
	args.no_occ = False
	args.cat_occ = False
	args.stride = 32

	main(args)