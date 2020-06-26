"""
Copyright (C) 2019 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""
from __future__ import print_function

import argparse
import os, sys
import random
import skimage
import skimage.io
import skimage.transform
import numpy as np
import time
import math

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torch.nn.functional as F
import torch.optim.lr_scheduler as lr_scheduler
import torchvision.transforms as transforms

import sense.models.model_utils as model_utils
from sense.models.self_supervised_loss import make_self_supervised_loss
import sense.datasets.flow_transforms as flow_transforms
import sense.datasets.dataset_catlog as dataset_catlog
from sense.datasets.listdataset import ListDataset
from sense.datasets.flow_disp_listdataset import SemiSupervisedFlowDispListDataset
from sense.datasets.dataset_utils import *
from sense.utils.arguments import parse_args

CITYSCAPE_PALETTE = np.asarray([
	[128, 64, 128],
	[244, 35, 232],
	[70, 70, 70],
	[102, 102, 156],
	[190, 153, 153],
	[153, 153, 153],
	[250, 170, 30],
	[220, 220, 0],
	[107, 142, 35],
	[152, 251, 152],
	[70, 130, 180],
	[220, 20, 60],
	[255, 0, 0],
	[0, 0, 142],
	[0, 0, 70],
	[0, 60, 100],
	[0, 80, 100],
	[0, 0, 230],
	[119, 11, 32],
	[0, 0, 0]], dtype=np.uint8)

def save_seg_teacher_result(im_tensor, seg_tensor):
	im = im_tensor.cpu()[0].numpy().transpose(1, 2, 0) * 255

	_, seg = torch.max(seg_tensor, 1)
	seg = seg.cpu().numpy()[0]
	seg_im = Image.fromarray(CITYSCAPE_PALETTE[seg.squeeze()])

	idx = np.random.randint(10000)
	cv2.imwrite('{:5d}_im.png'.format(idx), im.astype(np.uint8))
	seg_im.save('{:5d}_seg.png'.format(idx))

def make_data_loader(args):
	input_transform = transforms.Compose([
		flow_transforms.RandomGammaImg((0.7,1.5), p=0.5),
		flow_transforms.RandomBrightnessImg(0.2, p=0.5),
		flow_transforms.RandomContrastImg((0.8, 1.2), p=0.5),
		flow_transforms.RandomGaussianNoiseImg(0.02, p=0.5),            
	])

	# light color jittering parameters, empricially found helpful
	ss_input_transform = transforms.Compose([
		flow_transforms.RandomGammaImg((0.9,1.2), p=0.5),
		flow_transforms.RandomBrightnessImg(0.1, p=0.5),
		flow_transforms.RandomContrastImg((0.9, 1.1), p=0.5),
		flow_transforms.RandomGaussianNoiseImg(0.01, p=0.5),            
	])
	
	input_transform_toTensor = transforms.Compose([ 
		flow_transforms.ArrayToTensor()
	])

	disp_target_transform = transforms.Compose([
		flow_transforms.ArrayToTensor()
	])
	flow_target_transform = transforms.Compose([
		flow_transforms.ArrayToTensor(),
		flow_transforms.Normalize(mean=[0,0],std=[args.div_flow,args.div_flow])
	])

	height_new = args.disp_crop_imh
	width_new = args.disp_crop_imw
	seg_resize_ratio = None
	if args.seg_root_dir is not None:
		seg_resize_ratio = 0.25
	disp_co_transform=flow_transforms.Compose([
		flow_transforms.RandomCrop((height_new,width_new)),
		flow_transforms.RandomVerticalFlip(),
		flow_transforms.ResizeSeg(seg_resize_ratio),
	])

	disp_co_transform_test=flow_transforms.Compose([
		flow_transforms.CenterCrop((352, 1216))
	])

	height_new = args.flow_crop_imh
	width_new = args.flow_crop_imw
	flow_co_transform=flow_transforms.Compose([
		flow_transforms.Resize(args.flow_dim_ratio),
		flow_transforms.FlowDataAugmentation(crop_size=(height_new, width_new))
	])

	flow_co_transform_test=flow_transforms.Compose([
		flow_transforms.CenterCrop((352, 1216))
	])
		
	train_data, test_data = dataset_catlog.make_flow_disp_data(
		args.dataset, 
		pseudo_gt_dir=args.pseudo_gt_dir
	)

	if args.seg_root_dir is not None:
		train_data = dataset_catlog.patch_with_seg_gt(
			train_data, 
			args.seg_root_dir, 
			args.dataset, 
			'train'
		)
		if len(test_data) > 0:
			test_data = dataset_catlog.patch_with_seg_gt(
				test_data, 
				args.seg_root_dir, 
				args.dataset, 
				'val'
			)

	train_data = dataset_catlog.patched_with_saved_seg_logits(
		train_data, 
		args.saved_seg_res_dir, 
		args.dataset, 
		'train'
	)
	test_data = dataset_catlog.patched_with_saved_seg_logits(
		test_data, 
		args.saved_seg_res_dir, 
		args.dataset, 
		'val'
	)

	# train_data = train_data[:20000]
	# flow_test_data = flow_test_data[:11]
	# disp_test_data = disp_test_data[:11]
	print('{} samples found for joint training.'.format(len(train_data)))
	print('{} samples found for joint testing.'.format(len(test_data)))

	train_set = SemiSupervisedFlowDispListDataset(
		'', 
		train_data,
		flow_loader=kitti_flow_loader,
		disp_loader=kitti_disp_seg_loader, 
		transform=input_transform, 
		flow_target_transform=flow_target_transform, 
		disp_target_transform=disp_target_transform, 
		flow_co_transform=flow_co_transform,
		disp_co_transform=disp_co_transform, 
		flow_co_transform_test=None, 
		disp_co_transform_test=None, 
		transform_additional=input_transform_toTensor,
		ss_data_transform=ss_input_transform
	)
	test_set = SemiSupervisedFlowDispListDataset(
		'', 
		test_data,
		flow_loader=kitti_flow_loader,
		disp_loader=kitti_disp_seg_loader, 
		transform=input_transform, 
		flow_target_transform=flow_target_transform, 
		disp_target_transform=disp_target_transform, 
		flow_co_transform=None,
		disp_co_transform=None, 
		flow_co_transform_test=flow_co_transform_test, 
		disp_co_transform_test=disp_co_transform_test, 
		transform_additional=input_transform_toTensor,
		ss_data_transform=ss_input_transform
	)

	train_loader = torch.utils.data.DataLoader(
		train_set,
		batch_size=args.batch_size,
		shuffle=True,
		num_workers=args.workers, 
		drop_last=True,
		pin_memory=True,
		worker_init_fn = lambda _: np.random.seed(int(torch.initial_seed()%(2**32 -1)))
	)
	test_loader = torch.utils.data.DataLoader(
		test_set,
		batch_size=2,
		shuffle=False,
		num_workers=args.workers, 
		drop_last=True,
		pin_memory=True,
		worker_init_fn = lambda _: np.random.seed(int(torch.initial_seed()%(2**32 -1)))
	)

	# sanity check to ensure everything goes well
	# import cv2
	# from PIL import Image
	# import tools.flowlib as flowlib
	# import tools.kitti_viz as kitti_viz

	# def save_single_im(im_path, raw_im):
	# 	raw_im = raw_im.numpy().transpose(1, 2, 0) * 255
	# 	cv2.imwrite(im_path, raw_im.astype(np.uint8))

	# def save_seg_im(im_path, seg):
	# 	if len(seg.shape) > 2:
	# 		seg = np.argmax(seg.numpy(), axis=0)
	# 	else:
	# 		seg = seg.numpy()
	# 		seg[seg == 255] = 0
	# 	im = Image.fromarray(CITYSCAPE_PALETTE[seg.astype(int)])
	# 	im.save(im_path)

	# print(len(train_set))
	# for i, batch_data in enumerate(train_set):
	# 	if i > 3:
	# 		break		
	# 	flow_ims, flow_gt, disp_ims, disp_gt = batch_data
	# 	cur_im, nxt_im, ss_cur_im, ss_nxt_im, cur_seg, nxt_seg = flow_ims
	# 	flow, flow_occ = flow_gt
	# 	left_im, right_im, ss_left_im, ss_right_im, left_seg, right_seg = disp_ims
	# 	disp, disp_occ, left_seg_gt = disp_gt
		
	# 	save_single_im('{:03d}_a_cur_im.png'.format(i), cur_im)
	# 	save_single_im('{:03d}_b_nxt_im.png'.format(i), nxt_im)
	# 	save_single_im('{:03d}_c_ss_cur_im.png'.format(i), ss_cur_im)
	# 	save_single_im('{:03d}_d_ss_nxt_im.png'.format(i), ss_nxt_im)
	# 	if len(cur_seg.size()) == 3:
	# 		save_seg_im('{:03d}_e_cur_seg.png'.format(i), cur_seg)
	# 		save_seg_im('{:03d}_f_nxt_seg.png'.format(i), nxt_seg)
	# 	else:
	# 		print(cur_seg.size(), nxt_seg.size())

	# 	# g, h
	# 	flow_im = flowlib.flow_to_image(flow.numpy().transpose(1, 2, 0))
	# 	cv2.imwrite('{:03d}_g_flow.png'.format(i), flow_im[:, :, ::-1])
	# 	flow_occ_im = (flow_occ.numpy() * 255).astype(np.uint8)
	# 	cv2.imwrite('{:03d}_h_flow_occ.png'.format(i), flow_occ_im)

	# 	save_single_im('{:03d}_i_left_im.png'.format(i), left_im)
	# 	save_single_im('{:03d}_j_right_im.png'.format(i), right_im)
	# 	save_single_im('{:03d}_k_ss_left_im.png'.format(i), ss_left_im)
	# 	save_single_im('{:03d}_l_ss_right_im.png'.format(i), ss_right_im)
	# 	if len(left_seg.size()) == 3:
	# 		save_seg_im('{:03d}_m_left_seg.png'.format(i), left_seg)
	# 		save_seg_im('{:03d}_n_right_seg.png'.format(i), right_seg)
	# 	else:
	# 		print(left_seg.size(), right_seg.size())

	# 	disp_im = kitti_viz.disp_to_color(disp.numpy().squeeze())
	# 	cv2.imwrite('{:03d}_o_disp.png'.format(i), disp_im[:,:,::-1])
	# 	disp_occ_im = (disp_occ.numpy() * 255).astype(np.uint8)
	# 	cv2.imwrite('{:03d}_p_disp_occ.png'.format(i), disp_occ_im)
	# 	save_seg_im('{:03d}_q_seg_gt.png'.format(i), left_seg_gt)

	return train_loader, test_loader

def train(model, optimizer, data, criteria, args):
	model.train()

	start = time.time()

	disp_loss_weight = args.disp_loss_weight
	hard_seg_loss_weight = args.hard_seg_loss_weight
	soft_seg_loss_weight = args.soft_seg_loss_weight
	work_mode = args.cmd

	# ss data is used for self supervision
	cur_im, nxt_im, ss_cur_im, ss_nxt_im, cur_seg_logits, nxt_seg_logits = data[0] 
	flow, flow_occ, flow_annot_mask = data[1]
	left_im, right_im, ss_left_im, ss_right_im, left_seg_logits, right_seg_logits = data[2]
	disp, disp_occ, seg_im, disp_annot_mask = data[3]

	left_im, right_im = left_im.cuda(), right_im.cuda()
	disp_true = disp.cuda()

	cur_im, nxt_im = cur_im.cuda(), nxt_im.cuda()
	flow_gt = flow.cuda()

	if args.soft_occ_gt:
		flow_occ = flow_occ.cuda()
		disp_occ = disp_occ.cuda()
	else:
		flow_occ = flow_occ.long().cuda()
		disp_occ = disp_occ.long().cuda()

	if args.disp_semantic_consist_wt > 0 or \
		args.disp_photo_consist_wt > 0 or \
		args.disp_smoothness_wt > 0 or \
		args.disp_ssim_wt > 0:
		ss_left_im = ss_left_im.cuda()
		ss_right_im = ss_right_im.cuda()
		flow_annot_mask = flow_annot_mask.float().cuda()
	if args.flow_semantic_consist_wt > 0 or \
		args.flow_photo_consist_wt > 0 or \
		args.flow_smoothness_wt > 0 or \
		args.flow_ssim_wt > 0:
		ss_cur_im = ss_cur_im.cuda()
		ss_nxt_im = ss_nxt_im.cuda()
		disp_annot_mask = disp_annot_mask.float().cuda()

	if args.disp_semantic_consist_wt > 0 or args.do_seg_distill:
		left_seg_logits = left_seg_logits.cuda()
		right_seg_logits = right_seg_logits.cuda()
	if args.flow_semantic_consist_wt > 0 or args.do_seg_distill:
		cur_seg_logits = cur_seg_logits.cuda()
		nxt_seg_logits = nxt_seg_logits.cuda()

	if args.disp_semantic_consist_wt > 0:
		with torch.no_grad():
			left_seg = F.softmax(left_seg_logits, dim=1)
			right_seg = F.softmax(right_seg_logits, dim=1)
	else:
		left_seg = None
		right_seg = None

	if args.flow_semantic_consist_wt > 0:
		with torch.no_grad():
			cur_seg = F.softmax(cur_seg_logits, dim=1)
			nxt_seg = F.softmax(nxt_seg_logits, dim=1)
	else:
		cur_seg = None
		nxt_seg = None

	# prepare disparity and optical flow mask
	if work_mode == 'pre-train':
		raise Exception('Pre-trained mode is not applicable.')
	else:
		disp_mask = (disp_true > 0)
		flow_mask = flow_gt[:, 0, :, :] != flow_gt[:, 0, :, :]
		flow_mask = flow_mask.unsqueeze(1)
		flow_mask = 1 - flow_mask.expand_as(flow_gt).float()
		flow_mask = flow_mask.detach()
	disp_mask.detach_()
	flow_mask = flow_mask.detach()

	(disp_crit, disp_occ_crit, flow_crit, flow_occ_crit, hard_seg_crit, soft_seg_crit, self_spvsd_crit) = criteria

	optimizer.zero_grad()

	flow_pred, disp_pred, seg_pred = model(
		cur_im, nxt_im, 
		left_im, right_im, 
		reuse_first_im=False, 
		do_seg_every_im=args.do_seg_distill
	)
	flow_loss, flow_looses = flow_crit(flow_pred[0], flow_gt, flow_mask)
	disp_loss, disp_looses = disp_crit(disp_pred[0], disp_true, disp_mask)

	multi_scale_flow_occ = []
	for i in range(len(flow_pred[1])):
		multi_scale_flow_occ.append(
			F.softmax(flow_pred[1][i], dim=1)[:, 1, :, :].unsqueeze(1)
		)

	multi_scale_disp_occ = []
	for i in range(len(disp_pred[1])):
		multi_scale_disp_occ.append(
			F.softmax(disp_pred[1][i], dim=1)[:, 1, :, :].unsqueeze(1)
		)

	if flow_occ_crit is not None:
		if args.soft_occ_gt:
			assert flow_occ.max().item() <= 1.000001, 'flow occluion mask is not soft!'
			flow_occ = flow_occ.unsqueeze(1)
			flow_occ_mask = (flow_occ >= 0)
			flow_occ_mask.detach_()
			if args.mask_semi_loss:
				flow_occ_mask = flow_occ_mask.float() + (1 - flow_mask[:, 0, :, :].unsqueeze(1))
				flow_occ_mask = flow_occ_mask == 2
			flow_occ_loss, flow_occ_losses = flow_occ_crit(
				tuple(multi_scale_flow_occ), flow_occ, flow_occ_mask
			)
		else:
			flow_occ_loss, flow_occ_losses = flow_occ_crit(flow_pred[1], flow_occ)
	else:
		flow_occ_loss = torch.FloatTensor([0]).cuda()

	if disp_occ_crit is not None:
		if args.soft_occ_gt:
			assert disp_occ.max().item() <= 1.000001, 'disp occluion mask is not soft!'
			disp_occ = disp_occ.unsqueeze(1)
			disp_occ_mask = (disp_occ >= 0)
			disp_occ_mask.detach_()
			if args.mask_semi_loss:
				disp_occ_mask = disp_occ_mask.float() + (1 - disp_mask.float())
				disp_occ_mask = disp_occ_mask == 2
			disp_occ_loss, disp_occ_losses = disp_occ_crit(
				tuple(multi_scale_disp_occ), disp_occ, disp_occ_mask
			)

			# # debug
			# for idx in range(disp_occ_mask.size(0)):
			# 	docc = disp_occ_mask[idx].cpu().float().numpy().squeeze()
			# 	cv2.imwrite('docc_{:03d}.png'.format(idx), (docc * 255).astype(np.uint8))
		else:	
			disp_occ_loss, disp_occ_losses = disp_occ_crit(disp_pred[1], disp_occ)
	else:
		disp_occ_loss = torch.FloatTensor([0]).cuda()

	# hard segmentation loss
	if args.do_seg:
		if args.seg_root_dir is not None:
			# hard segmentation label
			seg_gt = seg_im.long().cuda()
		else:
			# soft segmentation label
			# seg_gt = F.softmax(left_seg.cuda() / args.seg_distill_T, dim=1)
			raise NotImplementedError
		# print(type(seg_pred), seg_pred[0].size(), seg_gt.size())
		hard_seg_loss = hard_seg_crit(seg_pred[0], seg_gt)
	else:
		hard_seg_loss = torch.FloatTensor([0]).cuda()

	# soft segmentation loss
	if args.do_seg_distill:
		cur_seg_pred, nxt_seg_pred, left_seg_pred, right_seg_pred = seg_pred
		cnt = 1
		soft_seg_loss = 0

		# hard coded parameter...
		seg_ds_f = 4
		cur_seg_gt = F.softmax(cur_seg_logits / args.seg_distill_T, dim=1)
		# print(cur_seg_pred.size(), cur_seg_gt.size())
		soft_seg_loss += soft_seg_crit(
			cur_seg_pred, 
			cur_seg_gt[:, :, ::seg_ds_f, ::seg_ds_f]
		)

		if nxt_seg_pred is not None:
			nxt_seg_gt = F.softmax(nxt_seg_logits / args.seg_distill_T, dim=1)
			soft_seg_loss += soft_seg_crit(
				nxt_seg_pred, 
				nxt_seg_gt[:, :, ::seg_ds_f, ::seg_ds_f]
			)
			cnt += 1

		if left_seg_pred is not None:
			left_seg_gt = F.softmax(left_seg_logits / args.seg_distill_T, dim=1)
			soft_seg_loss += soft_seg_crit(
				left_seg_pred, 
				left_seg_gt[:, :, ::seg_ds_f, ::seg_ds_f]
			)
			cnt += 1

		if right_seg_pred is not None:
			right_seg_gt = F.softmax(right_seg_logits / args.seg_distill_T, dim=1)
			soft_seg_loss += soft_seg_crit(
				right_seg_pred, 
				right_seg_gt[:, :, ::seg_ds_f, ::seg_ds_f]
			)
			cnt += 1

		soft_seg_loss = soft_seg_loss / cnt
	else:
		soft_seg_loss = torch.FloatTensor([0]).cuda()		

	# self-supervised loss
	scaled_flow_pred = []
	for i in range(len(flow_pred[0])):
		scaled_flow_pred.append(flow_pred[0][i] * args.div_flow)

	if args.mask_semi_loss:
		# disp_semi_loss_mask = (1 - disp_mask).float()
		# flow_semi_loss_mask = (1 - flow_mask).float()
		# flow_semi_loss_mask = flow_semi_loss_mask[:, 0, :, :].unsqueeze(1)
		disp_semi_loss_mask = 1 - disp_annot_mask
		flow_semi_loss_mask = 1 - flow_annot_mask
	else:
		disp_semi_loss_mask = None
		flow_semi_loss_mask = None

	self_spvsd_loss, ss_losses = self_spvsd_crit(
		cur_im=ss_cur_im, 
		nxt_im=ss_nxt_im,
		cur_seg=cur_seg,
		nxt_seg=nxt_seg, 
		multi_scale_flow=tuple(scaled_flow_pred),
		multi_scale_flow_occ=tuple(multi_scale_flow_occ),
		flow_semi_loss_mask=flow_semi_loss_mask,
		left_im=ss_left_im,
		right_im=ss_right_im,
		left_seg=left_seg,
		right_seg=right_seg,
		multi_scale_disp=disp_pred[0],
		multi_scale_disp_occ=tuple(multi_scale_disp_occ),
		disp_semi_loss_mask=disp_semi_loss_mask
	)

	loss = flow_loss + args.occ_loss_wt * flow_occ_loss + \
		   disp_loss_weight * (disp_loss + args.occ_loss_wt * disp_occ_loss) + \
		   hard_seg_loss_weight * hard_seg_loss + \
		   soft_seg_loss_weight * soft_seg_loss + \
		   self_spvsd_loss

	loss.backward()
	optimizer.step()

	loss = loss.item()
	flow_loss = flow_loss.item()
	flow_occ_loss = flow_occ_loss.item() * args.occ_loss_wt
	disp_loss = disp_loss.item() * disp_loss_weight
	disp_occ_loss = disp_occ_loss.item() * disp_loss_weight * args.occ_loss_wt
	hard_seg_loss = hard_seg_loss.item() * hard_seg_loss_weight
	soft_seg_loss = soft_seg_loss.item() * soft_seg_loss_weight
	self_spvsd_loss = self_spvsd_loss.item()

	return loss, flow_loss, flow_occ_loss, disp_loss, disp_occ_loss, hard_seg_loss, soft_seg_loss, self_spvsd_loss, ss_losses

def adjust_learning_rate(optimizer, epoch, iter_per_epoch):
	"""Sets the learning rate to the initial LR decayed by 2 after 300K iterations, 400K and 500K"""
	# if epoch == 200000/iter_per_epoch or epoch == 400000/iter_per_epoch or epoch == 600000/iter_per_epoch:
	# if epoch == 400000/iter_per_epoch or epoch == 600000/iter_per_epoch or epoch == 800000/iter_per_epoch or epoch == 1000000/iter_per_epoch or epoch == 1200000/iter_per_epoch or epoch == 1400000/iter_per_epoch or epoch == 1600000/iter_per_epoch:
	#     for param_group in optimizer.param_groups:
	#         param_group['lr'] = param_group['lr']/2
	#         lr = param_group['lr']

	lr_steps = args.lr_steps
	lr_gamma = args.lr_gamma

	num_steps = 0
	while num_steps < len(lr_steps) and epoch >= lr_steps[num_steps]:
		num_steps += 1

	lr = args.lr * (lr_gamma ** num_steps)
	for param_group in optimizer.param_groups:
		param_group['lr'] = lr

	return lr

def save_checkpoint(model, optimizer, epoch, global_step, args):
	#SAVE
	save_dir = model_utils.make_joint_checkpoint_name(args, epoch)
	save_dir = os.path.join(args.savemodel, save_dir)
	if not os.path.exists(save_dir):
		os.makedirs(save_dir)

	model_path = os.path.join(save_dir, 'model_{:04d}.pth'.format(epoch))

	if epoch % args.save_freq == 0:
		torch.save({
			'state_dict': model.state_dict(),
			'optimizer': optimizer.state_dict(),
			'epoch': epoch
			}, model_path)
		print('<=== checkpoint has been saved to {}.'.format(model_path))

def main(args):
	train_loader, test_loader = make_data_loader(args)

	torch.manual_seed(args.seed)
	torch.cuda.manual_seed(args.seed)
	np.random.seed(args.seed)
	random.seed(args.seed)

	model = model_utils.make_model(
		args, 
		do_flow=not args.no_flow,
		do_disp=not args.no_disp,
		do_seg=(args.do_seg or args.do_seg_distill)
	)
	print('Number of model parameters: {}'.format(
		sum([p.data.nelement() for p in model.parameters()]))
	)

	optimizer = optim.Adam(model.parameters(), 
		lr=args.lr, 
		betas=(0.9, 0.999),
		eps=1e-08, 
		weight_decay=0.0004
	)

	if args.loadmodel is not None:
		ckpt = torch.load(args.loadmodel)
		state_dict = ckpt['state_dict']
		missing_keys, unexpected_keys = model.load_state_dict(model_utils.patch_model_state_dict(state_dict))
		assert not unexpected_keys, 'Got unexpected keys: {}'.format(unexpected_keys)
		if missing_keys:
			for mk in missing_keys:
				assert mk.find('seg_decoder') >= 0, 'Only segmentation decoder can be initialized randomly.'
		print('==> A pre-trained model has been loaded.')
	start_epoch = 1

	if args.auto_resume:
		# search for the latest saved checkpoint
		epoch_found = -1
		for epoch in range(args.epochs+1, 1, -1):
			ckpt_dir = model_utils.make_joint_checkpoint_name(args, epoch)
			ckpt_dir = os.path.join(args.savemodel, ckpt_dir)
			ckpt_path = os.path.join(ckpt_dir, 'model_{:04d}.pth'.format(epoch))
			if os.path.exists(ckpt_path):
				epoch_found = epoch
				break
		if epoch_found > 0:
			ckpt = torch.load(ckpt_path)
			assert ckpt['epoch'] == epoch_found, [ckpt['epoch'], epoch_found]
			start_epoch = ckpt['epoch'] + 1
			optimizer.load_state_dict(ckpt['optimizer'])
			model.load_state_dict(ckpt['state_dict'])
			print('==> Automatically resumed training from {}.'.format(ckpt_path))
	else:
		if args.resume is not None:
			ckpt = torch.load(args.resume)
			start_epoch = ckpt['epoch'] + 1
			optimizer.load_state_dict(ckpt['optimizer'])
			model.load_state_dict(ckpt['state_dict'])
			print('==> Manually resumed training from {}.'.format(args.resume))
	
	cudnn.benchmark = True

	(flow_crit, flow_occ_crit), flow_down_scales, flow_weights = model_utils.make_flow_criteria(args)
	(disp_crit, disp_occ_crit), disp_down_scales, disp_weights = model_utils.make_disp_criteria(args)

	hard_seg_crit = model_utils.make_seg_criterion(args, hard_lab=True)
	soft_seg_crit = model_utils.make_seg_criterion(args, hard_lab=False)
	args.hard_seg_loss_weight *= float(disp_weights[0])
	args.soft_seg_loss_weight *= float(disp_weights[0])

	self_supervised_crit = make_self_supervised_loss(
		args,
		disp_downscales=disp_down_scales,
		disp_pyramid_weights=disp_weights,
		flow_downscales=flow_down_scales,
		flow_pyramid_weights=flow_weights
	).cuda()
	criteria = (
		disp_crit, disp_occ_crit, 
		flow_crit, flow_occ_crit, 
		hard_seg_crit, 
		soft_seg_crit, 
		self_supervised_crit
	)

	min_loss=100000000000000000
	min_epo=0
	min_err_pct = 10000
	start_full_time = time.time()

	train_print_format = '{}\t{:d}\t{:d}\t{:d}\t{:d}\t{:.3f}\t{:.3f}\t{:.3f}'\
						 '\t{:.3f}\t{:.3f}\t{:.3f}\t{:.3f}\t{:.3f}\t{:.3f}\t{:.3f}\t{:.6f}'
	test_print_format = '{}\t{:d}\t{:d}\t{:.3f}\t{:.2f}\t{:.3f}\t{:.2f}\t{:.2f}\t{:.2f}'\
						'\t{:.3f}\t{:.3f}\t{:.3f}\t{:.3f}\t{:.3f}\t{:.6f}'

	global_step = 0
	for epoch in range(start_epoch, args.epochs+1):
		total_train_loss = 0
		total_err = 0
		total_test_err_pct = 0
		total_disp_occ_acc = 0
		total_epe = 0
		total_flow_occ_acc = 0
		total_seg_acc = 0
		lr = adjust_learning_rate(optimizer, epoch, len(train_loader))
			 
		## training ##
		start_time = time.time() 
		for batch_idx, batch_data in enumerate(train_loader):
			end = time.time()
			# (cur_im, nxt_im), (flow, flow_occ), (left_im, right_im), (disp, disp_occ, seg_im) = data
			# if args.seg_root_dir is None:
			# 	seg_im = None
			train_res = train(model, optimizer, batch_data, criteria, args)
			loss, flow_loss, flow_occ_loss, disp_loss, disp_occ_loss, seg_loss, seg_distill_loss, ss_loss, ss_losses = train_res
			global_step += 1
			if (batch_idx + 1) % args.print_freq == 0:
				print(train_print_format.format(
					'Train', global_step, epoch, batch_idx, len(train_loader),
					loss, 
					flow_loss, flow_occ_loss, 
					disp_loss, disp_occ_loss,
					seg_loss, seg_distill_loss, ss_loss,
					end - start_time, time.time() - start_time, lr
				))
				for k, v in ss_losses.items():
					print('{: <10}\t{:.3f}'.format(k, v))
				sys.stdout.flush()
			start_time = time.time()
			total_train_loss += loss

		# should have had a validation set
		
		save_checkpoint(model, optimizer, epoch, global_step, args)
	print('Elapsed time = %.2f HR' %((time.time() - start_full_time)/3600))

if __name__ == '__main__':
	parser = parse_args()
	parser.add_argument('--mask-semi-loss', action='store_true',
		help='whether to use mask (on unannotated pixels only) for semi-supervised loss.')
	args = parser.parse_args()

	args.upsample_flow_output = True
	if args.do_seg:
		assert args.seg_root_dir is not None or segs.seg_teacher_encoder_weights is not None, \
			'Either gt or pre-trained seg model should be provided.'

	if args.do_seg_distill:
		assert args.saved_seg_res_dir is not None, 'Saved seg logits must be provided.'

	# whether to compute self-supervised loss
	args.do_ss_loss = args.disp_photo_consist_wt > 0 or \
		args.disp_semantic_consist_wt > 0 or \
		args.flow_photo_consist_wt > 0 or \
		args.flow_semantic_consist_wt > 0 or \
		args.disp_temporal_consist_wt > 0 or \
		args.flow_disp_consist_wt > 0 or \
		args.flow_smoothness_wt > 0 or \
		args.disp_smoothness_wt > 0 or \
		args.flow_ssim_wt > 0 or \
		args.disp_ssim_wt > 0

	print('Use following parameters:')
	for k, v in vars(args).items():
		print('{}\t{}'.format(k, v))
	print('=======================================\n')

	if not os.path.exists(args.savemodel):
		os.makedirs(args.savemodel)
	main(args)