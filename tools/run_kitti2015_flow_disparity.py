"""
Copyright (C) 2019 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""
import os
import glob
import numpy as np
import argparse
import scipy.misc as smisc
import cv2
import scipy.io as sio
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F

from sense.models.unet import UNet
import sense.utils.flowlib as flowlib
import sense.models.model_utils as model_utils
from sense.models.common import flow_warp
import sense.utils.kitti_viz as kitti_viz
from sense.utils.arguments import parse_args
from sense.datasets.dataset_catlog import make_flow_data
from sense.datasets.dataset_utils import imread, read_kitti_flow_raw, read_vkitti_flow
from sense.datasets.cityscapes_dataset import CITYSCAPE_PALETTE

from tqdm import tqdm

def make_kitti2015_paths(kitti_dir, split):
	kitti_dir = os.path.join(kitti_dir, split)
	path_list = []
	for idx in range(200):
		cur_left_im_path = os.path.join(kitti_dir, 'image_2/{:06d}_10.png'.format(idx))
		nxt_left_im_path = os.path.join(kitti_dir, 'image_2/{:06d}_11.png'.format(idx))
		cur_right_im_path = os.path.join(kitti_dir, 'image_3/{:06d}_10.png'.format(idx))
		nxt_right_im_path = os.path.join(kitti_dir, 'image_3/{:06d}_11.png'.format(idx))
		assert os.path.exists(cur_left_im_path)
		assert os.path.exists(nxt_left_im_path)
		assert os.path.exists(cur_right_im_path)
		assert os.path.exists(nxt_right_im_path)
		path_list.append([
			cur_left_im_path, nxt_left_im_path,
			cur_right_im_path, nxt_right_im_path
		])
	return path_list

def flow_error_map(F_gt, F_est):
	F_gt_du  = np.squeeze(F_gt[:,:,0])
	F_gt_dv  = np.squeeze(F_gt[:,:,1])
	F_gt_val = np.squeeze(F_gt[:,:,2])

	F_est_du = np.squeeze(F_est[:,:,0])
	F_est_dv = np.squeeze(F_est[:,:,1])

	E_du = F_gt_du - F_est_du
	E_dv = F_gt_dv - F_est_dv
	E    = np.sqrt(E_du * E_du + E_dv * E_dv)
	idxes = np.where(F_gt_val == 0)
	E[idxes] = 0
	return E, F_gt_val

def flow_error(F_gt, F_est, tau=3):
	[E, F_val] = flow_error_map (F_gt,F_est);
	idxes = np.where(E > tau)
	idxes2 = np.where(F_val > 0)
	percent = len(idxes[0]) / len(idxes2[0])
	epe = np.mean(E[idxes2])
	return epe, percent

def save_flow_results(save_dir, im1_path, im2_path, pred_flow, gt_flow=None, flow_occ=None):
	seq_dir, im1_name = os.path.split(im1_path)
	_, seq_name = os.path.split(seq_dir)
	_, im2_name = os.path.split(im2_path)

	save_dir = os.path.join(save_dir, seq_name)
	if not os.path.exists(save_dir):
		os.makedirs(save_dir)

	im1 = cv2.imread(im1_path)
	im2 = cv2.imread(im2_path)
	cv2.imwrite(os.path.join(save_dir, im1_name), im1)
	cv2.imwrite(os.path.join(save_dir, im2_name), im2)

	# save prediction
	# flow_im = flowlib.flow_to_image(pred_flow)
	flow_im = kitti_viz.viz_flow(pred_flow[:, :, 0], pred_flow[:, :, 1])
	cv2.imwrite(os.path.join(save_dir, im1_name[:-4] + '_flow.png'), flow_im[:, :, ::-1])

	if gt_flow is not None:
		gt_flow_im = flowlib.flow_to_image(gt_flow)
		cv2.imwrite(os.path.join(save_dir, im1_name[:-4] + '_gt.png'), gt_flow_im[:, :, ::-1])

	if flow_occ is not None:
		flow_occ = (flow_occ * 255).astype(np.uint8)
		cv2.imwrite(os.path.join(save_dir, im1_name[:-4] + '_flow_occ.png'), flow_occ)

	# # ground-truth flow
	# flow_dir = seq_dir.replace('image_2', 'viz_flow_occ_dilate_1')
	# assert flow_dir != seq_dir, seq_dir
	# gt_im = cv2.imread(os.path.join(flow_dir, im1_name))
	# cv2.imwrite(os.path.join(save_dir, im1_name[:-4] + '_flow_gt.png'), gt_im)

def save_raw_results(save_dir, im1_path, im2_path, pred_flow, flow_occ=None):
	seq_dir, im1_name = os.path.split(im1_path)
	_, seq_name = os.path.split(seq_dir)

	save_dir = os.path.join(save_dir, seq_name)
	if not os.path.exists(save_dir):
		os.makedirs(save_dir)

	flowlib.write_flow(pred_flow, os.path.join(save_dir, im1_name[:-4] + '.flo'))

	im1 = cv2.imread(im1_path)
	cv2.imwrite(os.path.join(save_dir, im1_name), im1)

	flow_im = flowlib.flow_to_image(pred_flow)
	cv2.imwrite(os.path.join(save_dir, im1_name[:-4] + '_flow.png'), flow_im[:, :, ::-1])

	if flow_occ is not None:
		flow_occ = (flow_occ * 255).astype(np.uint8)
		cv2.imwrite(os.path.join(save_dir, im1_name[:-4] + '_flow_occ.png'), flow_occ)

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

def save_results(flow, flow_occ, disp_0, disp_1, seg_im, 
	save_dir, cur_left_im_path, orig_imh, orig_imw):
	flow = flow.cpu().numpy().squeeze()[:, :orig_imh, :orig_imw].transpose(1, 2, 0)
	flow_occ = flow_occ.cpu().numpy().squeeze()[:orig_imh, :orig_imw]

	cur_left_im_name = os.path.basename(cur_left_im_path)
	flowlib.write_flow(
		flow, 
		os.path.join(save_dir, 'flow_raw', cur_left_im_name[:-4] + '.flo')
	)
	flow_occ = (flow_occ * 255).astype(np.uint8)
	cv2.imwrite(
		os.path.join(save_dir, 'flow_occ', cur_left_im_name), 
		flow_occ
	)

	disp_0 = disp_0.cpu().numpy().squeeze()[:orig_imh, :orig_imw]
	sio.savemat(
		os.path.join(save_dir, 'disp_0', cur_left_im_name[:-4] + '.mat'),
		{'disp': disp_0}
	)

	# png files are needed for KITTI submission
	kitti_viz.disp_write(
		os.path.join(save_dir, 'disp_0', cur_left_im_name),
		disp_0
	)

	disp_1 = disp_1.cpu().numpy().squeeze()[:orig_imh, :orig_imw]
	sio.savemat(
		os.path.join(save_dir, 'disp_1_unwarped', cur_left_im_name[:-4] + '.mat'),
		{'disp': disp_1}
	)

	seg_im = seg_im.cpu().numpy().squeeze()[:orig_imh, :orig_imw]
	seg_im = Image.fromarray(seg_im.astype(np.uint8))
	seg_im.save(os.path.join(save_dir, 'seg', cur_left_im_name))

def process_single_data(model, im_paths, args):
	cur_left_im_path, nxt_left_im_path, cur_right_im_path, nxt_right_im_path = im_paths

	# compute optical flow
	cur_left_im, (orig_imh, orig_imw) = image_to_tensor(imread(cur_left_im_path))
	nxt_left_im, _ = image_to_tensor(imread(nxt_left_im_path))
	cur_right_im, _ = image_to_tensor(imread(cur_right_im_path))
	nxt_right_im, _ = image_to_tensor(imread(nxt_right_im_path))

	with torch.no_grad():
		flow_pred, disp_pred_0, seg_pred = model(
			cur_left_im, 
			nxt_left_im, 
			cur_right_im,
			seg_size=cur_left_im.shape[2:]
		)
	if not args.upsample_flow_output:
		flow = F.upsample(flow_pred[0][0] * args.div_flow, scale_factor=4, mode='bilinear')
		flow_occ = F.upsample(flow_pred[1][0], scale_factor=4, mode='bilinear')
	else:
		flow = flow_pred[0][0] * args.div_flow
		flow_occ = flow_pred[1][0]
	flow_occ = F.softmax(flow_occ, dim=1)[:, 1].unsqueeze(1)
	disp_0 = disp_pred_0[0][0]

	seg = seg_pred[0]
	_, seg_im = torch.max(seg, 1)

	with torch.no_grad():
		_, disp_pred_1, _ = model(nxt_left_im, None, nxt_right_im)
	disp_1 = disp_pred_1[0][0]

	save_results(
		flow, flow_occ, disp_0, disp_1, seg_im,
		args.save_dir, cur_left_im_path, orig_imh, orig_imw
	)

def main(args):
	if args.save_dir is not None and args.save_dir != 'None':
		if not os.path.exists(args.save_dir):
			os.makedirs(args.save_dir)
		else:
			os.system('rm -rf {}/*'.format(args.save_dir))
		os.makedirs(os.path.join(args.save_dir, 'flow_raw'))
		os.makedirs(os.path.join(args.save_dir, 'flow_occ'))
		os.makedirs(os.path.join(args.save_dir, 'disp_0'))
		os.makedirs(os.path.join(args.save_dir, 'disp_1_unwarped'))
		os.makedirs(os.path.join(args.save_dir, 'seg'))
			
	# make flow, disp, and seg model
	print('==> Making a holistic backbone model')
	do_flow = True
	do_disp = True
	do_seg = True
	model = model_utils.make_model(
		args, 
		do_flow, 
		do_disp, 
		do_seg=do_seg
	)

	ckpt = torch.load(args.loadmodel)
	state_dict = model_utils.patch_model_state_dict(ckpt['state_dict'])
	model.load_state_dict(state_dict)
	model.eval()
	print('==> Successfully loaded a model {}.'.format(args.loadmodel))

	path_list = make_kitti2015_paths(args.kitti_dir, args.split)

	args.stride = 32
	for i in tqdm(range(len(path_list))):
		process_single_data(model, path_list[i], args)

if __name__ == '__main__':
	parser = parse_args()
	parser.add_argument('--kitti-dir', type=str)
	parser.add_argument('--joint-model', action='store_true')
	parser.add_argument(
		'--split', 
		choices=['training', 'testing'],
		default='training'
	)
	args = parser.parse_args()
	main(args)
	