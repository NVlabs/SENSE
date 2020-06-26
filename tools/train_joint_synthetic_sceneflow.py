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
import sense.datasets.flow_transforms as flow_transforms
import sense.datasets.dataset_catlog as dataset_catlog
from sense.datasets.listdataset import ListDataset
from sense.datasets.flow_disp_listdataset import FlowDispListDataset
from sense.datasets.dataset_utils import *
from sense.utils.arguments import parse_args

def make_data_loader(args):
	input_transform = transforms.Compose([
		flow_transforms.RandomGammaImg((0.7,1.5)),
		flow_transforms.RandomBrightnessImg(0.2),
		flow_transforms.RandomContrastImg((0.8, 1.2)),
		flow_transforms.RandomGaussianNoiseImg(0.02),            
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
	disp_co_transform=flow_transforms.Compose([
		flow_transforms.RandomCrop((height_new,width_new))
	])

	height_new = 352
	width_new = 1216
	disp_co_transform_test=flow_transforms.Compose([
		flow_transforms.CenterCrop((height_new, width_new))
	])

	height_new = args.flow_crop_imh
	width_new = args.flow_crop_imw
	flow_co_transform=flow_transforms.Compose([
		flow_transforms.Resize(args.flow_dim_ratio),
		flow_transforms.FlowDataAugmentation(crop_size=(height_new, width_new))
	])

	height_new = 384
	width_new = 1024
	flow_co_transform_test=flow_transforms.Compose([
		flow_transforms.CenterCrop((height_new, width_new))
	])

	train_data, _ = dataset_catlog.make_flow_disp_data(args.dataset)
	flow_test_data, _ = dataset_catlog.make_flow_data('sintel')
	_, disp_test_data = dataset_catlog.make_disp_data('sceneflow')

	# train_data = train_data[:200]
	# flow_test_data = flow_test_data[:11]
	# disp_test_data = disp_test_data[:11]
	print('{} samples found for joint training.'.format(len(train_data)))
	print('{} samples found for flow testing.'.format(len(flow_test_data)))
	print('{} samples found for disparity testing.'.format(len(disp_test_data)))

	train_set = FlowDispListDataset(
		'', 
		train_data,
		flow_loader=optical_flow_loader,
		disp_loader=sceneflow_disp_loader, 
		transform=input_transform, 
		flow_target_transform=flow_target_transform, 
		disp_target_transform=disp_target_transform, 
		flow_co_transform=flow_co_transform,
		disp_co_transform=disp_co_transform, 
		flow_co_transform_test=None, 
		disp_co_transform_test=None, 
		transform_additional=input_transform_toTensor
	)
	flow_test_set = ListDataset(
		'', 
		flow_test_data,
		optical_flow_loader, 
		transform=None, 
		target_transform=flow_target_transform, 
		co_transform=None, 
		co_transform_test=flow_co_transform_test, 
		transform_additional=input_transform_toTensor
	)
	disp_test_set = ListDataset(
		'',
		disp_test_data,
		sceneflow_disp_loader,
		transform=None,
		target_transform=disp_target_transform,
		co_transform=None,
		co_transform_test=disp_co_transform_test,
		transform_additional=input_transform_toTensor
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
	flow_test_loader = torch.utils.data.DataLoader(
		flow_test_set,
		batch_size=2,
		shuffle=False,
		num_workers=args.workers, 
		drop_last=True,
		pin_memory=True
	)
	disp_test_loader = torch.utils.data.DataLoader(
		disp_test_set,
		batch_size=2,
		shuffle=False,
		num_workers=args.workers, 
		drop_last=True,
		pin_memory=True
	)

	return train_loader, flow_test_loader, disp_test_loader

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
	disp_gt, disp_occ = disp.cuda(), disp_occ.long().cuda()

	cur_im, nxt_im = cur_im.cuda(), nxt_im.cuda()
	flow_gt, flow_occ = flow.cuda(), flow_occ.long().cuda()

	disp_mask = (disp_gt < args.maxdisp)
	disp_mask.detach_()

	disp_crit, disp_occ_crit, flow_crit, flow_occ_crit = criteria

	optimizer.zero_grad()

	flow_pred, disp_pred, _ = model(
		cur_im, nxt_im, 
		left_im, right_im, 
		reuse_first_im=False
	)
	flow_loss, flow_looses = flow_crit(flow_pred[0], flow_gt)
	disp_loss, disp_looses = disp_crit(disp_pred[0], disp_gt, disp_mask)

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

def test_disp(model, batch_data, criteria, work_mode):
	model.eval() 
	
	left_im, right_im = batch_data[0]
	disp_gt, disp_occ, _ = batch_data[1]
	left_im, right_im = left_im.cuda(), right_im.cuda()
	disp_gt = disp_gt.float().cuda()
	disp_occ = disp_occ.long().cuda()

	mask = disp_gt < args.maxdisp
	mask.detach_()

	disp_crit, disp_occ_crit, _, _ = criteria

	with torch.no_grad():
		_, output, _ = model(left_im, None, right_im)
		# supervised loss
		disp_loss, losses = disp_crit(output[0], disp_gt, mask)
		if disp_occ_crit is not None:
			disp_occ_loss, _ = disp_occ_crit(output[1], disp_occ)
		else:
			disp_occ_loss = torch.FloatTensor([0]).cuda()
		loss = disp_loss + disp_occ_loss
	pred_disp = torch.squeeze(output[0][0].detach().cpu())
	disp_gt = torch.squeeze(disp_gt.detach())

	if disp_occ_crit is not None:
		pred_occ = torch.squeeze(output[1][0].detach().cpu())
		pred_occ = torch.argmax(pred_occ, dim=1)
		gt_occ = torch.squeeze(disp_occ.detach().cpu())
		occ_acc = torch.mean((pred_occ == gt_occ).float())
	else:
		occ_acc = -1

	#computing 3-px error#
	pred_err = np.abs(pred_disp - disp_gt.cpu())
	if work_mode == 'pre-train':
		valid_condition = disp_gt < args.maxdisp
	else:
		valid_condition = disp_gt > 0
	valid_pix = np.where(valid_condition)
	if len(valid_pix[0]) == 0:
		err = 0
		err_pct = 0
	else:
		err_pix = np.where(np.logical_and(
			np.logical_and(valid_condition, pred_err > 3),
			pred_err / (np.abs(disp_gt) + 1e-10) > 0.05
		))
		err_pct = len(err_pix[0]) / (len(valid_pix[0]) + 1e-20)
		err = torch.sum(pred_err[valid_pix]) / len(valid_pix[0])
	torch.cuda.empty_cache()
	loss = loss.item()
	disp_loss = disp_loss.item()
	disp_occ_loss = disp_occ_loss.item()
	return err, err_pct, occ_acc, loss, disp_loss, disp_occ_loss

def test_flow(model, batch_data, criteria, work_mode, upsample_factor=4):
	model.eval() 

	cur_im, nxt_im = batch_data[0] 
	flow, flow_occ = batch_data[1]
	
	cur_im, nxt_im = cur_im.cuda(), nxt_im.cuda()
	flow_gt, flow_occ = flow.cuda(), flow_occ.long().cuda()

	_, _, flow_crit, flow_occ_crit = criteria

	with torch.no_grad():
		output, _, _ = model(cur_im, nxt_im)
		# supervised loss
		flow_loss, losses = flow_crit(output[0], flow_gt)
		if flow_occ_crit is not None:
			flow_occ_loss, _ = flow_occ_crit(output[1], flow_occ)
		else:
			flow_occ_loss = torch.FloatTensor([0]).cuda()
		loss = flow_loss + flow_occ_loss
		flow_pred = F.upsample(
			output[0][0],
			scale_factor=upsample_factor,
			mode='bilinear',
			align_corners=False
		)
		flow_pred = torch.squeeze(flow_pred.data.cpu())

	if flow_occ_crit is not None:
		pred_occ = F.upsample(
			output[1][0], 
			scale_factor=upsample_factor,
			mode='bilinear',
			align_corners=False
		)
		pred_occ = torch.squeeze(pred_occ.detach().cpu())
		pred_occ = torch.argmax(pred_occ, dim=1)
		gt_occ = torch.squeeze(flow_occ.detach().cpu())
		occ_acc = torch.mean((pred_occ == gt_occ).float())
	else:
		occ_acc = -1

	#computing EPE#
	pred_err = flow_pred - flow
	epe = torch.mean(torch.sqrt(torch.sum(pred_err ** 2, dim=1)))
	loss = loss.item()
	flow_loss = flow_loss.item()
	flow_occ_loss = flow_occ_loss.item()
	return epe, occ_acc, loss, flow_loss, flow_occ_loss

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
	train_loader, flow_test_loader, disp_test_loader = make_data_loader(args)

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
		model.load_state_dict(model_utils.patch_model_state_dict(state_dict))
		print('==> A pre-trained checkpoint has been loaded.')
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

	hard_seg_crit = None
	soft_seg_crit = None
	self_supervised_crit = None
	criteria = (
		disp_crit, disp_occ_crit, 
		flow_crit, flow_occ_crit
	)

	min_loss=100000000000000000
	min_epo=0
	min_err_pct = 10000
	start_full_time = time.time()

	train_print_format = '{}\t{:d}\t{:d}\t{:d}\t{:d}\t{:.3f}\t{:.3f}\t{:.3f}'\
		'\t{:.3f}\t{:.3f}\t{:.3f}\t{:.3f}\t{:.6f}'
	test_print_format = '{}\t{:d}\t{:d}\t{:.3f}\t{:.2f}\t{:.3f}\t{:.2f}\t{:.2f}\t{:.2f}'\
		'\t{:.3f}\t{:.3f}\t{:.3f}\t{:.3f}\t{:.6f}'

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

		# should have used the validation set to select the best model
		start_time = time.time()
		for batch_idx, batch_data in enumerate(flow_test_loader):
			loss_data = test_flow(
				model, 
				batch_data,
				criteria, 
				args.cmd, 
				flow_down_scales[0]
			)
			epe, flow_occ_acc, loss, flow_loss, flow_occ_loss = loss_data
			total_epe += epe
			total_flow_occ_acc += flow_occ_acc

		for batch_idx, batch_data in enumerate(disp_test_loader):
			loss_data = test_disp(
				model, 
				batch_data, 
				criteria, 
				args.cmd
			)
			err, err_pct, disp_occ_acc, loss, disp_loss, disp_occ_loss = loss_data
			total_err += err
			total_test_err_pct += err_pct
			total_disp_occ_acc += disp_occ_acc

		if total_test_err_pct/len(disp_test_loader) * 100 < min_err_pct:
			min_loss = total_err/len(disp_test_loader)
			min_epo = epoch
			min_err_pct = total_test_err_pct/len(disp_test_loader) * 100

		print(test_print_format.format(
			'Test', global_step, epoch,
			total_epe / len(flow_test_loader) * args.div_flow,
			total_flow_occ_acc / len(flow_test_loader) * 100,
			total_err/len(disp_test_loader), 
			total_test_err_pct/len(disp_test_loader) * 100,
			total_disp_occ_acc / len(disp_test_loader) * 100,
			flow_loss, flow_occ_loss,
			disp_loss * args.disp_loss_weight, 
			disp_occ_loss * args.disp_loss_weight,
			time.time() - start_time, lr
		))
		
		save_checkpoint(model, optimizer, epoch, global_step, args)
	print('Elapsed time = %.2f HR' %((time.time() - start_full_time)/3600))

if __name__ == '__main__':
	parser = parse_args()
	args = parser.parse_args()

	# args.upsample_flow_output = True
	# if args.do_seg:
	# 	assert args.seg_root_dir is not None or segs.seg_teacher_encoder_weights is not None, \
	# 		'Either gt or pre-trained seg model should be provided.'

	# if args.do_seg_distill:
	# 	assert args.saved_seg_res_dir is not None, 'Saved seg logits must be provided.'

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