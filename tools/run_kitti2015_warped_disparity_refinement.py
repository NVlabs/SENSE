"""
Copyright (C) 2019 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""
import argparse
import os, sys
import random
import skimage
import skimage.io
import skimage.transform
import numpy as np
import time
import math
import glob
from tqdm import tqdm
import cv2
from PIL import Image
from distutils.version import LooseVersion
import scipy.io as sio

import torch
import torch.nn as nn

from sense.models.unet import UNet, LightWeightUNet
from sense.rigidity_refine.io_utils import read_camera_data, read_flow_gen, read_seg_gen, read_disp_gen, write_flow
from sense.rigidity_refine.rigidity_refine import warp_disp_refine_rigid
import sense.utils.kitti_viz as kitti_viz

def make_kitti2015_paths(kitti_dir, res_dir, split):
	im_dir = os.path.join(kitti_dir, split, 'image_2')
	im_paths = glob.glob(os.path.join(im_dir, '*_10.png'))

	def make_file_list():
		path_list = []
		for p in im_paths:
			_, im_name = os.path.split(p)
			flow_path = os.path.join(res_dir, 'flow_raw', im_name[:-4] + '.flo')
			flow0_occ_path = os.path.join(res_dir, 'flow_occ', im_name)
			seg_path = os.path.join(res_dir, 'seg', im_name)
			disp0_path = os.path.join(res_dir, 'disp_0', im_name[:-4] + '.mat')
			disp1_path = os.path.join(res_dir, 'disp_1_unwarped', im_name[:-4] + '.mat')
			cam2cam_path=os.path.join(kitti_dir, split, 'calib_cam_to_cam', im_name[:-7] + '.txt')
			assert os.path.exists(p), p
			assert os.path.exists(flow0_occ_path), flow0_occ_path
			assert os.path.exists(disp0_path), disp0_path
			assert os.path.exists(disp1_path), disp1_path
			path_list.append([p, flow_path, flow0_occ_path, seg_path, disp0_path, disp1_path, cam2cam_path])
		return path_list

	data = make_file_list()
	return data

def read_kitti_disp(disp_path):
	# disp = Image.open(disp_path)
	# disp = np.ascontiguousarray(disp, dtype=np.float32) / 256
	# return disp[:, :, np.newaxis]
	data = sio.loadmat(disp_path)
	return data['disp'][:, :, np.newaxis]

def prepare_data(im1, flow_occ, disp0, disp1):
	# padding
	stride = 32
	orig_imh = im1.shape[0]
	orig_imw = im1.shape[1]
	if orig_imh % stride != 0:
		num_feat_pix = orig_imh // stride
		new_imh = (num_feat_pix + 1) * stride
	else:
		new_imh = orig_imh

	if orig_imw % stride != 0:
		num_feat_pix = orig_imw // stride
		new_imw = (num_feat_pix + 1) * stride
	else:
		new_imw = orig_imw

	# to tensor
	im1 = torch.from_numpy(im1.transpose(2, 0, 1)).float().unsqueeze(0).cuda()
	flow_occ = torch.from_numpy(flow_occ[np.newaxis, :, :]).float().unsqueeze(0).cuda()
	disp0 = torch.from_numpy(disp0.transpose(2, 0, 1)).float().unsqueeze(0).cuda() / 20
	disp1 = torch.from_numpy(disp1.transpose(2, 0, 1)).float().unsqueeze(0).cuda() / 20

	pad = torch.nn.ZeroPad2d([0, new_imw - orig_imw, 0, new_imh - orig_imh])
	im1 = pad(im1)
	flow_occ = pad(flow_occ)
	disp0 = pad(disp0)
	disp1 = pad(disp1)

	return im1, flow_occ, disp0, disp1, (orig_imh, orig_imw)

def prepare_data_resize(data, norm_imh=352, norm_imw=1216):
	im_path, flow0_occ_path, disp0_path, disp1_path = data

	# resizing
	im1 = cv2.imread(im_path).astype(np.float32) / 255
	flow_occ = cv2.imread(flow0_occ_path, 0).astype(np.float32) / 255
	disp0 = read_kitti_disp(disp0_path)
	disp1 = read_kitti_disp(disp1_path)
	orig_imh, orig_imw, _ = im1.shape

	im1 = cv2.resize(im1, (norm_imw, norm_imh), None, interpolation=cv2.INTER_NEAREST)
	flow_occ = cv2.resize(flow_occ, (norm_imw, norm_imh), None, interpolation=cv2.INTER_NEAREST)
	disp0 = cv2.resize(disp0, (norm_imw, norm_imh), None, interpolation=cv2.INTER_NEAREST)
	disp1 = cv2.resize(disp1, (norm_imw, norm_imh), None, interpolation=cv2.INTER_NEAREST)

	# to tensor
	im1 = torch.from_numpy(im1.transpose(2, 0, 1)).float().unsqueeze(0).cuda()
	flow_occ = torch.from_numpy(flow_occ[np.newaxis, :, :]).float().unsqueeze(0).cuda()
	disp0 = torch.from_numpy(disp0[np.newaxis, :, :]).float().unsqueeze(0).cuda() / 20
	disp1 = torch.from_numpy(disp1[np.newaxis, :, :]).float().unsqueeze(0).cuda() / 20

	return im1, flow_occ, disp0, disp1, (orig_imh, orig_imw)

def warp_disp_refine_nn(im1, flow_occ, disp0, disp1_rigid, model):
	disp0 = disp0[:, :, np.newaxis]
	disp1_rigid = disp1_rigid[:, :, np.newaxis]
	im1, flow_occ, disp0, disp1_rigid, (orig_imh, orig_imw) = prepare_data(im1, flow_occ, disp0, disp1_rigid)
	with torch.no_grad():
		input_data = torch.cat((im1, flow_occ, disp0, disp1_rigid), dim=1)
		pred = model(input_data)
		disp1_nn = (disp1_rigid + pred[-1]) * 20
	disp1_nn = disp1_nn.cpu().numpy().squeeze()
	disp1_nn = disp1_nn[:orig_imh, :orig_imw]
	disp1_nn[disp1_nn < 0] = 0
	return disp1_nn

def main(args):
	if args.loadmodel is not None:
		model = UNet()
		model = nn.DataParallel(model).cuda()
		model.eval()
		print('#parameters in warp disp model: {}'.format(sum([p.data.nelement() for p in model.parameters()])))

		ckpt = torch.load(args.loadmodel)
		state_dict = ckpt['state_dict']
		# if LooseVersion(torch.__version__) >= LooseVersion('0.4.0'):
		# 	keys = list(state_dict.keys())
		# 	for k in keys:
		# 		if k.find('num_batches_tracked') >= 0:
		# 			state_dict.pop(k)
		model.load_state_dict(state_dict)
		print('==> A pre-trained checkpoint has been loaded.')
	else:
		model = None

	kitti_dir = '/home/hzjiang/workspace/Data/KITTI_scene_flow'
	all_im_paths = make_kitti2015_paths(kitti_dir, args.res_dir, args.split)
	print('{} samples found for {}.'.format(len(all_im_paths), args.split))

	total_err = 0
	total_err_pct = 0
	total_time = 0
	for i in tqdm(range(len(all_im_paths))):
		im_paths_list = all_im_paths[i]
		im_path, flow_path, flow0_occ_path, seg_path, disp0_path, disp1_path, calib_path = im_paths_list

		im1 = cv2.imread(im_path).astype(np.float32) / 255
		flow = read_flow_gen(flow_path)
		flow_occ = cv2.imread(flow0_occ_path, 0).astype(np.float32) / 255
		seg = read_seg_gen(seg_path)
		disp0 = read_disp_gen(disp0_path)
		disp1 = read_disp_gen(disp1_path)
		K0, K1 = read_camera_data(calib_path)
	
		flow_rigid, _, disp1_rigid = warp_disp_refine_rigid(disp0, disp1, flow, seg, K0, K1)
		if model is not None:
			disp1_nn = warp_disp_refine_nn(im1, flow_occ, disp0, disp1_rigid, model)

		_, im_name = os.path.split(all_im_paths[i][0])
		# skimage.io.imsave(
		# 	os.path.join(args.res_dir, 'disp_1_nn', im_name),
		# 	(np.abs(disp) * 256).astype('uint16')
		# )
		write_flow(
			flow_rigid,
			os.path.join(args.res_dir, 'flow_rigid', im_name[:-4] + '.flo')
		)
		sio.savemat(
			os.path.join(args.res_dir, 'disp_1_rigid', im_name[:-4] + '.mat'),
			{'disp': disp1_rigid}
		)
		if model is not None:
			sio.savemat(
				os.path.join(args.res_dir, 'disp_1_nn', im_name[:-4] + '.mat'),
				{'disp': disp1_nn}
			)

		# png files are needed for KITTI submission
		kitti_viz.flow_write(
			os.path.join(args.res_dir, 'flow_rigid', im_name),
			flow_rigid
		)
		if model is not None:
			kitti_viz.disp_write(
				os.path.join(args.res_dir, 'disp_1_nn', im_name),
				disp1_nn
			)

if __name__ == '__main__':
	import argparse
	parser = argparse.ArgumentParser()
	parser.add_argument('--loadmodel', default=None)
	parser.add_argument('--res-dir', required=True)
	parser.add_argument('--split', default='testing')
	args = parser.parse_args()

	if args.loadmodel is not None:
		os.makedirs(
			os.path.join(args.res_dir, 'disp_1_nn'),
			exist_ok=True
		)
	os.makedirs(
		os.path.join(args.res_dir, 'flow_rigid'),
		exist_ok=True
	)
	os.makedirs(
		os.path.join(args.res_dir, 'disp_1_rigid'),
		exist_ok=True
	)

	main(args)


