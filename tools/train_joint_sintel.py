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
from sense.datasets.dataset_catlog import make_flow_disp_data, make_flow_data, make_disp_data
from sense.datasets.listdataset import ListDataset
from sense.datasets.flow_disp_listdataset import FlowDispListDataset
from sense.datasets.dataset_utils import *
from sense.utils.arguments import parse_args

def make_data_loader(args):
	transform_list = [
		flow_transforms.RandomGammaImg((0.7,1.5)),
		flow_transforms.RandomBrightnessImg(0.2),
		flow_transforms.RandomContrastImg((0.8, 1.2))
	]
	if not args.no_gaussian_noise:
		transform_list.append(flow_transforms.RandomGaussianNoiseImg(0.02))
	input_transform = transforms.Compose(transform_list)
	
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
	disp_co_transform=flow_transforms.Compose([
		flow_transforms.RandomCrop((height_new,width_new)),
		flow_transforms.RandomVerticalFlip()
	])
	disp_co_transform_test=flow_transforms.Compose([
		flow_transforms.CenterCrop((416, 1024))
	])

	height_new = args.flow_crop_imh
	width_new = args.flow_crop_imw
	flow_co_transform=flow_transforms.Compose([
  		flow_transforms.Resize(args.flow_dim_ratio),
  		flow_transforms.FlowDataAugmentation(crop_size=(height_new, width_new))
	])
	flow_co_transform_test=flow_transforms.Compose([
		flow_transforms.CenterCrop((416, 1024))
	])
		
	train_data, _ = make_flow_disp_data(args.dataset)
	test_data, _ = make_flow_disp_data('sintel', pass_opt='clean')

	# train_data = train_data[:20000]
	# flow_test_data = flow_test_data[:11]
	# disp_test_data = disp_test_data[:11]
	print('{} samples found for joint training.'.format(len(train_data)))
	print('{} samples found for flow testing.'.format(len(test_data)))
	print('{} samples found for disparity testing.'.format(len(test_data)))

	train_set = FlowDispListDataset(
		'', 
		train_data,
		flow_loader=optical_flow_loader,
		disp_loader=sintel_disp_seg_loader, 
		transform=input_transform, 
		flow_target_transform=flow_target_transform, 
		disp_target_transform=disp_target_transform, 
		flow_co_transform=flow_co_transform,
		disp_co_transform=disp_co_transform, 
		flow_co_transform_test=None, 
		disp_co_transform_test=None, 
		transform_additional=input_transform_toTensor
	)
	test_set = FlowDispListDataset(
		'', 
		test_data,
		flow_loader=optical_flow_loader,
		disp_loader=sintel_disp_seg_loader, 
		transform=input_transform, 
		flow_target_transform=flow_target_transform, 
		disp_target_transform=disp_target_transform, 
		flow_co_transform=None,
		disp_co_transform=None, 
		flow_co_transform_test=flow_co_transform_test, 
		disp_co_transform_test=disp_co_transform_test, 
		transform_additional=input_transform_toTensor,
	)

	train_loader = torch.utils.data.DataLoader(
		train_set,
		batch_size=args.batch_size,
		shuffle=True,
		num_workers=args.workers, 
		drop_last=True,
		pin_memory=True
	)
	test_loader = torch.utils.data.DataLoader(
		test_set,
		batch_size=args.batch_size,
		shuffle=False,
		num_workers=args.workers, 
		drop_last=True,
		pin_memory=True
	)

	return train_loader, test_loader

def train(model, optimizer, data, criteria, args):
	model.train()

	start = time.time()

	disp_loss_weight = args.disp_loss_weight
	work_mode = args.cmd

	cur_im, nxt_im = data[0] 
	flow, flow_occ = data[1]
	left_im, right_im = data[2]
	disp, disp_occ, _ = data[3]

	left_im, right_im = left_im.cuda(), right_im.cuda()
	disp_true, disp_occ = disp.cuda(), disp_occ.long().cuda()

	cur_im, nxt_im = cur_im.cuda(), nxt_im.cuda()
	flow_gt, flow_occ = flow.cuda(), flow_occ.long().cuda()

	disp_mask = (disp_true < args.maxdisp)
	disp_mask.detach_()

	disp_crit, disp_occ_crit, flow_crit, flow_occ_crit = criteria

	optimizer.zero_grad()

	flow_pred, disp_pred, _ = model(
		cur_im, nxt_im, 
		left_im, right_im, 
		reuse_first_im=False
	)
	flow_loss, flow_looses = flow_crit(flow_pred[0], flow_gt)
	disp_loss, disp_looses = disp_crit(disp_pred[0], disp_true, disp_mask)

	if flow_occ_crit is not None:
		flow_occ_loss, flow_occ_losses = flow_occ_crit(flow_pred[1], flow_occ)
	else:
		flow_occ_loss = torch.FloatTensor([0]).cuda()
	if disp_occ_crit is not None:
		disp_occ_loss, disp_occ_losses = disp_occ_crit(disp_pred[1], disp_occ)
	else:
		disp_occ_loss = torch.FloatTensor([0]).cuda()

	loss = flow_loss + flow_occ_loss + \
		   disp_loss_weight * (disp_loss + disp_occ_loss)

	loss.backward()
	optimizer.step()

	loss = loss.item()
	flow_loss = flow_loss.item()
	flow_occ_loss = flow_occ_loss.item()
	disp_loss = disp_loss.item() * disp_loss_weight
	disp_occ_loss = disp_occ_loss.item() * disp_loss_weight

	return loss, flow_loss, flow_occ_loss, disp_loss, disp_occ_loss

def adjust_learning_rate(optimizer, epoch, iter_per_epoch):
	"""Sets the learning rate to the initial LR decayed by 2 after 300K iterations, 400K and 500K"""
	# if epoch == 200000/iter_per_epoch or epoch == 400000/iter_per_epoch or epoch == 600000/iter_per_epoch:
	# if epoch == 400000/iter_per_epoch or epoch == 600000/iter_per_epoch or epoch == 800000/iter_per_epoch or epoch == 1000000/iter_per_epoch or epoch == 1200000/iter_per_epoch or epoch == 1400000/iter_per_epoch or epoch == 1600000/iter_per_epoch:
	#     for param_group in optimizer.param_groups:
	#         param_group['lr'] = param_group['lr']/2
	#         lr = param_group['lr']

	lr_step = 100
	# if args.cmd == 'pre-train':
	# 	lr_step = 70

	lr_gamma = 0.5

	lr = args.lr * (lr_gamma ** (epoch // lr_step))
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

	model = model_utils.make_model(args, do_seg=args.do_seg)
	print('Number of model parameters: {}'.format(sum([p.data.nelement() for p in model.parameters()])))

	optimizer = optim.Adam(model.parameters(), 
		lr=args.lr, 
		betas=(0.9, 0.999),
		eps=1e-08, 
		weight_decay=0.0004
	)

	if args.loadmodel is not None:
		ckpt = torch.load(args.loadmodel)
		state_dict = ckpt['state_dict']
		model.load_state_dict(model_utils.patch_model_state_dict(state_dict))
		print('==> A pre-trained checkpoint has been loaded {}'.format(args.loadmodel))
	start_epoch = 1

	if args.auto_resume:
		# search for the latest saved checkpoint
		epoch_found = -1
		for epoch in range(args.epochs+1, 1, -1):
			ckpt_path = model_utils.make_joint_checkpoint_name(args, epoch)
			ckpt_path = os.path.join(args.savemodel, ckpt_path)
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
	
	cudnn.benchmark = True

	(flow_crit, flow_occ_crit), flow_down_scales, flow_weights = model_utils.make_flow_criteria(args)
	(disp_crit, disp_occ_crit), disp_down_scales, disp_weights = model_utils.make_disp_criteria(args)

	criteria = (disp_crit, disp_occ_crit, flow_crit, flow_occ_crit)

	min_loss=100000000000000000
	min_epo=0
	min_err_pct = 10000
	start_full_time = time.time()

	train_print_format = '{}\t{:d}\t{:d}\t{:d}\t{:d}\t{:.3f}\t{:.3f}\t{:.3f}'\
						 '\t{:.3f}\t{:.3f}\t{:.3f}\t{:.3f}\t\t{:.6f}'
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
			train_res = train(model, optimizer, batch_data, criteria, args)
			loss, flow_loss, flow_occ_loss, disp_loss, disp_occ_loss = train_res
			global_step += 1
			if (batch_idx + 1) % args.print_freq == 0:
				print(train_print_format.format(
					'Train', global_step, epoch, batch_idx, len(train_loader),
					loss, 
					flow_loss, flow_occ_loss, 
					disp_loss, disp_occ_loss,
					end - start_time, time.time() - start_time, lr
				))
				sys.stdout.flush()
			start_time = time.time()
			total_train_loss += loss
		
		save_checkpoint(model, optimizer, epoch, global_step, args)
	print('Elapsed time = %.2f HR' %((time.time() - start_full_time)/3600))

if __name__ == '__main__':
	parser = parse_args()
	parser.add_argument('--no-gaussian-noise', action='store_true',
		help='whether to add Gaussian noise during training.')
	args = parser.parse_args()

	# whether to compute self-supervised loss
	# we don't need self-supervised loss on fully annotated data
	args.do_ss_loss = False

	print('Use following parameters:')
	for k, v in vars(args).items():
		print('{}\t{}'.format(k, v))
	print('=======================================\n')

	if not os.path.exists(args.savemodel):
		os.makedirs(args.savemodel)
	main(args)