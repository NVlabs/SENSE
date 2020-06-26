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
import glob
from distutils.version import LooseVersion

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
from torch.autograd import Variable
import torch.nn.functional as F
import torch.optim.lr_scheduler as lr_scheduler
import torchvision.transforms as transforms

from tensorboardX import SummaryWriter

import sense.models.model_utils as model_utils
import sense.datasets.flow_transforms as flow_transforms
import sense.datasets.disp_ref_transforms as disp_ref_transforms
from sense.datasets.dataset_catlog import make_warp_disp_refine_data
from sense.datasets.warp_disp_refine_kitti2015 import WarpDispRefineKITTI2015
import sense.datasets.dataset_utils as dataset_utils
from sense.utils.arguments import parse_args
from sense.models.loss import multiscaleloss

from sense.models.unet import UNet
from sense.models.common import flow_warp
from sense.lib.nn import DataParallelWithCallback

def make_kitti2015_paths(kitti_dir, kitti_cache_dir, split_id, use_rigid_refine):
	im_dir = os.path.join(kitti_dir, 'training/image_2')

	if split_id > 0:
		val_idxes_file = os.path.join(kitti_dir, 'val_idxes_split{}.txt'.format(split_id))
		assert os.path.exists(val_idxes_file), 'Val indexes file not found {}'.format(val_idxes_file)
		val_idxes = np.loadtxt(val_idxes_file, delimiter=',').astype(int).tolist()
	else:
		val_idxes = []

	val = ['%06d_10.png' % idx for idx in val_idxes]
	train = ['%06d_10.png' % idx for idx in range(200) if idx not in val_idxes]

	def make_file_list(im_names):
		im_paths = [os.path.join(im_dir, n) for n in im_names]
		path_list = []
		for p in im_paths:
			_, im_name = os.path.split(p)
			flow0_occ_path = os.path.join(kitti_cache_dir, 'flow_occ', im_name)
			first_disp_path = os.path.join(kitti_cache_dir, 'disp_0', im_name[:-4] + '.mat')
			if use_rigid_refine:
				ref_disp_path = os.path.join(kitti_cache_dir, 'disp_1_rigid', im_name[:-4] + '.mat')
			else:
				ref_disp_path = os.path.join(kitti_cache_dir, 'disp_1_raw', im_name[:-4] + '.mat')
			gt_ref_disp_path = os.path.join(kitti_dir, 'training/disp_occ_1', im_name)
			assert os.path.exists(p), p
			assert os.path.exists(flow0_occ_path), flow0_occ_path
			assert os.path.exists(first_disp_path), first_disp_path
			assert os.path.exists(ref_disp_path), ref_disp_path
			assert os.path.exists(gt_ref_disp_path), gt_ref_disp_path
			path_list.append([p, flow0_occ_path, first_disp_path, ref_disp_path, gt_ref_disp_path])
		return path_list

	train_data = make_file_list(train)
	test_data = make_file_list(val)
	return train_data, test_data

def make_data_loader(args):
	input_transform = transforms.Compose([
		flow_transforms.RandomGammaImg((0.7,1.5)),
		flow_transforms.RandomBrightnessImg(0.2),
		flow_transforms.RandomContrastImg((0.8, 1.2)),
		flow_transforms.RandomGaussianNoiseImg(0.02),            
	])

	height_new = args.disp_crop_imh
	width_new = args.disp_crop_imw
	co_transform=flow_transforms.Compose([
		flow_transforms.RandomCrop((height_new,width_new)),
		disp_ref_transforms.RandomHorizontalFlip(),
		disp_ref_transforms.RandomVerticalFlip()
		])

	co_transform_test=flow_transforms.Compose([
		flow_transforms.CenterCrop((352, 1216))
		])

	kitti_dir = '/home/hzjiang/workspace/Data/KITTI_scene_flow'
	train_data, test_data = make_kitti2015_paths(
		kitti_dir, 
		args.kitti_cache_dir, 
		args.split_id,
		args.use_rigid_refine
	)
	# train_data.extend(test_data)
	print('{} samples found for training.'.format(len(train_data)))
	print('{} samples found for testing.'.format(len(test_data)))

	print('======== training data ========')
	print(train_data[0])
	print(train_data[-1])
	print('===============================')

	train_set = WarpDispRefineKITTI2015(
		train_data,
		transform=input_transform, 
		co_transform=co_transform
	)
	test_set = WarpDispRefineKITTI2015(
		test_data,
		transform=None,
		co_transform=co_transform_test
	)

	train_loader = torch.utils.data.DataLoader(
		train_set,
		batch_size=args.batch_size,
		shuffle=True,
		num_workers=args.workers, 
		drop_last=True,
		pin_memory=True,
		worker_init_fn=lambda _: np.random.seed(int(torch.initial_seed() % (2 ** 32 - 1)))
	)
	test_loader = torch.utils.data.DataLoader(
		test_set,
		batch_size=args.batch_size,
		shuffle=False,
		num_workers=args.workers,
		drop_last=True,
		pin_memory=True,
		worker_init_fn=lambda _: np.random.seed(int(torch.initial_seed() % (2 ** 32 - 1)))
	)

	# # sanity check to ensure everything goes well
	# import tools.flowlib as flowlib
	# import tools.kitti_viz as kitti_viz
	# import cv2
	# for i, data in enumerate(train_set):
	# 	if i > 3:
	# 		break
	# 	im, flow0_occ, ref_disp, gt_ref_disp = data
		
	# 	im = im.numpy().transpose(1, 2, 0) * 255
	# 	cv2.imwrite('{:03d}_a_im.png'.format(i), im.astype(np.uint8))
	# 	# disp0 = kitti_viz.disp_to_color(disp0.numpy().squeeze())
	# 	# cv2.imwrite('{:03d}_b_disp0.png'.format(i), disp0[:, :, ::-1])
	# 	# disp1 = kitti_viz.disp_to_color(disp1.numpy().squeeze())
	# 	# cv2.imwrite('{:03d}_b_disp1.png'.format(i), disp1[:, :, ::-1])
	# 	# flow0 = flowlib.flow_to_image(flow0.numpy().transpose(1, 2, 0))
	# 	# cv2.imwrite('{:03d}_e_flow0.png'.format(i), flow0[:, :, ::-1])
	# 	flow0_occ = flow0_occ.numpy().squeeze() * 255
	# 	cv2.imwrite('{:03d}_f_flow0_occ.png'.format(i), flow0_occ.astype(np.uint8))
	# 	print(ref_disp.min().item(), ref_disp.max().item())
	# 	ref_disp = kitti_viz.disp_to_color(ref_disp.numpy().squeeze())
	# 	cv2.imwrite('{:03d}_g_ref_disp.png'.format(i), ref_disp[:, :, ::-1])
	# 	print(gt_ref_disp.min().item(), gt_ref_disp.max().item())
	# 	gt_ref_disp = kitti_viz.disp_to_color(gt_ref_disp.numpy().squeeze())
	# 	cv2.imwrite('{:03d}_h_gt_ref_disp.png'.format(i), gt_ref_disp[:, :, ::-1])

	return train_loader, test_loader

def found_nan(x):
	x_ = x.cpu().detach().numpy()
	return np.any(np.isnan(x_))

def train(model, crit, optimizer, data):
	model.train()

	im, flow0_occ, first_disp, ref_disp, gt_ref_disp = data

	im = im.cuda()
	flow0_occ = flow0_occ.cuda()
	first_disp = first_disp.cuda()
	ref_disp = ref_disp.cuda()
	gt_ref_disp = gt_ref_disp.cuda()

	optimizer.zero_grad()

	# print('im: ', im.min().item(), im.max().item())
	# # print('warp_disp1: ', warp_disp1.min().item(), warp_disp1.max().item())
	# print('flow0_occ: ', flow0_occ.min().item(), flow0_occ.max().item())
	# print('ref_disp: ', ref_disp.min().item(), ref_disp.max().item())

	wdisp_input = torch.cat((im, flow0_occ, first_disp, ref_disp), dim=1)

	# wdisp_x = wdisp_encoder(wdisp_input)
	# assert wdisp_x[-1] is None, 'PPM is not used in wdisp encoder.'
	# pred = wdisp_decoder(wdisp_x[:5])

	pred = model(wdisp_input)

	# # check nan
	# if found_nan(im):
	# 	print('Nan found in image.')
	# if found_nan(warp_disp1):
	# 	print('Nan found in warp_disp1.')
	# if found_nan(flow0_occ):
	# 	print('Nan found in flow0_occ.')
	# if found_nan(wdisp_input):
	# 	raise Exception('Nan found in input.')
	# if found_nan(disp_change):
	# 	raise Exception('Nan found in GT.')
	# # for i, p in enumerate(pred):
	# # 	if found_nan(p):
	# # 		raise Exception('Nan found in prediction {}.'.format(i))
	# if found_nan(pred):
	# 	print('Nan found in pred.')
	# 	print(pred.min().item(), pred.max().item(), pred.mean().item())
	# 	print(disp_change.min().item(), disp_change.max().item())
	# 	raise Exception

	gt_mask = gt_ref_disp > 0
	gt_mask = gt_mask.detach()
	with torch.no_grad():
		gt_ref_disp = gt_ref_disp - ref_disp

	# print('gt_ref_disp: ', gt_ref_disp.min().item(), gt_ref_disp.max().item())

	loss, losses = crit(pred, gt_ref_disp, gt_mask)
	# loss = F.smooth_l1_loss(pred, disp_change)
	# losses = None

	loss.backward()
	optimizer.step()

	return loss.item(), losses

def test_disp(model, crit, data, work_mode):
	model.eval()

	im, flow0_occ, first_disp, ref_disp, gt_ref_disp = data

	im = im.cuda()
	flow0_occ = flow0_occ.cuda()
	first_disp = first_disp.cuda()
	ref_disp = ref_disp.cuda()
	gt_ref_disp = gt_ref_disp.cuda()

	with torch.no_grad():
		wdisp_input = torch.cat((im, flow0_occ, first_disp, ref_disp), dim=1)

		# wdisp_x = wdisp_encoder(wdisp_input)
		# assert wdisp_x[-1] is None, 'PPM is not used in wdisp encoder.'

		# pred = wdisp_decoder(wdisp_x[:5])
		pred = model(wdisp_input)

		gt_mask = gt_ref_disp > 0
		gt_mask = gt_mask.detach()
		gt_ref = gt_ref_disp - ref_disp

		loss, _ = crit(pred, gt_ref, gt_mask)
		# loss = F.smooth_l1_loss(pred, disp_change)

	pred_disp = ref_disp + pred[-1]
	# pred_disp = torch.squeeze(pred[-1].detach().cpu())
	pred_disp = torch.squeeze(pred_disp).detach().cpu().numpy() * 20
	pred_disp = np.abs(pred_disp)
	disp_true = torch.squeeze(gt_ref_disp.detach().cpu()).numpy() * 20

	#computing 3-px error#
	# print('pred_disp: ', pred_disp.size())
	pred_err = np.abs(pred_disp - disp_true)
	# print('pred_err: ', pred_err.size())
	valid_condition = disp_true > 0
	valid_pix = np.where(valid_condition)
	if len(valid_pix[0]) == 0:
		err = 0
		err_pct = 0
	else:
		err_pix = np.where(np.logical_and(
						  np.logical_and(valid_condition, pred_err > 3),
						  pred_err / (np.abs(disp_true) + 1e-10) > 0.05
						  )
				  )
		err_pct = len(err_pix[0]) / (len(valid_pix[0]) + 1e-20)
		err = np.sum(pred_err[valid_pix]) / len(valid_pix[0])
	torch.cuda.empty_cache()
	loss = loss.item()
	return err, err_pct, loss

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
	savefilename = '{}_warp_disp_refine_{}.tar'.format(args.dataset, epoch)

	# wdisp_encoder, wdisp_decoder = model

	if epoch % args.save_freq == 0:
		torch.save({
			'state_dict': model.state_dict(),
			'optimizer': optimizer.state_dict(),
			'epoch': epoch
			}, os.path.join(args.savemodel, savefilename)
			)
		print('<=== checkpoint has been saved to {}.'.format(os.path.join(args.savemodel, savefilename)))

class WarpDispRefineModel(torch.nn.Module):
	def __init__(self, enc, dec):
		super(WarpDispRefineModel, self).__init__()
		self.enc = enc
		self.dec = dec

	def forward(self, x):
		wdisp_x = self.enc(x)
		assert wdisp_x[-1] is None, 'PPM is not used in wdisp encoder.'

		pred = self.dec(wdisp_x[:5])
		return pred

def main(args):
	train_loader, test_loader = make_data_loader(args)

	torch.manual_seed(args.seed)
	torch.cuda.manual_seed(args.seed)

	if args.resnet_arch is None:
		model = UNet()
	else:
		model = ResNetUNet(args.resnet_arch)
	# model = DataParallelWithCallback(model)
	model = nn.DataParallel(model).cuda()
	print('#parameters in warp disp model: {}'.format(sum([p.data.nelement() for p in model.parameters()])))

	optimizer = optim.Adam(model.parameters(), 
		lr=args.lr, 
		betas=(0.9, 0.999),
		eps=1e-08, 
		weight_decay=0.0004
	)

	if args.loadmodel is not None:
		state_dict = torch.load(args.loadmodel)['state_dict']
		# if LooseVersion(torch.__version__) >= LooseVersion('0.4.0'):
		# 	keys = list(state_dict.keys())
		# 	for k in keys:
		# 		if k.find('num_batches_tracked') >= 0:
		# 			state_dict.pop(k)
		model.load_state_dict(state_dict)
		print('==> A pre-trained checkpoint has been loaded: {}.'.format(args.loadmodel))
	start_epoch = 1

	if args.auto_resume:
		raise NotImplementedError
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

	crit = multiscaleloss(
		downsample_factors=(16, 8, 4, 2, 1),
		weights=(1, 1, 2, 4, 8), 
		loss='l1',
		size_average=True
		).cuda()

	start_full_time = time.time()

	train_print_format = '{}\t{:d}\t{:d}\t{:d}\t{:d}\t{:.3f}\t{:.3f}\t{:.3f}'\
						 '\t{:.6f}'
	test_print_format = '{}\t{:d}\t{:d}\t{:.3f}\t{:.2f}\t{:.3f}\t{:.2f}'\
						'\t{:.6f}'

	os.makedirs(os.path.join(args.savemodel, 'tensorboard'), exist_ok=True)
	writer = SummaryWriter(os.path.join(args.savemodel, 'tensorboard'))

	global_step = 0
	for epoch in range(start_epoch, args.epochs+1):
		total_err = 0
		total_test_err_pct = 0
		total_test_loss = 0
		lr = adjust_learning_rate(optimizer, epoch, len(train_loader))
			 
		## training ##
		start_time = time.time() 
		for batch_idx, data in enumerate(train_loader):
			end = time.time()
			loss, losses = train(
				model, crit, optimizer, data
			)
			global_step += 1
			writer.add_scalar('train/total_loss', loss * 20, global_step)
			if (batch_idx + 1) % args.print_freq == 0:
				print(train_print_format.format(
					'Train', global_step, epoch, batch_idx, len(train_loader),
					loss, 
					end - start_time, time.time() - start_time, lr
					))
				sys.stdout.flush()
			start_time = time.time()
		 
		## test ##
		start_time = time.time()

		for batch_idx, batch_data in enumerate(test_loader):
			err, err_pct, loss = test_disp(
				model, crit, batch_data, args.cmd
			)
			total_err += err
			total_test_err_pct += err_pct
			total_test_loss += loss

		writer.add_scalar('test/loss', total_test_loss / (len(test_loader) + 1e-30) * 20, epoch)
		writer.add_scalar('test/err', total_err / (len(test_loader) + 1e-30), epoch)
		writer.add_scalar('test/err_pct', total_test_err_pct / (len(test_loader) + 1e-30) * 100, epoch)
		print(test_print_format.format(
					'Test', global_step, epoch,
					total_err/(len(test_loader) + 1e-30), 
					total_test_err_pct/(len(test_loader) + 1e-30) * 100,
					total_test_loss / (len(test_loader) + 1e-30),
					time.time() - start_time, lr
					))
		sys.stdout.flush()
		
		save_checkpoint(model, optimizer, epoch, global_step, args)
	print('full time = %.2f HR' %((time.time() - start_full_time)/3600))

if __name__ == '__main__':
	parser = parse_args()
	parser.add_argument('--resnet-arch', default=None)
	parser.add_argument('--kitti-cache-dir', required=True)
	parser.add_argument('--split-id', default=1, type=int)
	parser.add_argument('--use-rigid-refine', action='store_true')
	args = parser.parse_args()

	cudnn.benchmark = True	

	for k, v in vars(args).items():
		print('{}\t{}'.format(k, v))

	if not os.path.exists(args.savemodel):
		os.makedirs(args.savemodel)
	main(args)