"""
Copyright (C) 2019 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""

import os
import glob
import datetime
import argparse
from distutils.version import LooseVersion

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from scipy.io import loadmat
from PIL import Image

import sense.models.model_utils as model_utils
from sense.utils.arguments import parse_args
from sense.datasets.dataset_utils import imread, read_kitti_flow_raw, read_vkitti_flow
from sense.datasets.cityscapes_dataset import SegList, CITYSCAPE_PALETTE, TRIPLET_PALETTE
import sense.datasets.flow_transforms as flow_transforms

from tqdm import tqdm

def make_kitti2015_paths(kitti_dir, split):
	im_dir = os.path.join(kitti_dir, split, 'image_2')
	im_paths = glob.glob(os.path.join(im_dir, '*_10.png'))
	return im_paths

def fast_hist(pred, label, n):
	k = (label >= 0) & (label < n)
	return np.bincount(
		n * label[k].astype(int) + pred[k], minlength=n ** 2).reshape(n, n)


def per_class_iu(hist):
	return np.diag(hist) / (hist.sum(1) + hist.sum(0) - np.diag(hist))

def save_input_images(filenames, output_dir):
	for ind in range(len(filenames)):
		im = Image.open(filenames[ind])
		_, fn = os.path.split(filenames[ind])
		fn = os.path.join(output_dir, fn[:-4] + '.png')
		out_dir = os.path.split(fn)[0]
		if not os.path.exists(out_dir):
			os.makedirs(out_dir)
		im.save(fn)

def save_output_images(prediction, filename, output_dir):
	"""
	Saves a given (B x C x H x W) into an image file.
	If given a mini-batch tensor, will save the tensor as a grid of images.
	"""
	# pdb.set_trace()
	im = Image.fromarray(prediction.astype(np.uint8))
	_, fn = os.path.split(filename)
	fn = os.path.join(output_dir, fn[:-4] + '.png')
	# print('save_output_images', fn)
	out_dir = os.path.split(fn)[0]
	if not os.path.exists(out_dir):
		os.makedirs(out_dir)
	im.save(fn)


def save_colorful_images(prediction, filename, output_dir, palettes):
	"""
	Saves a given (B x C x H x W) into an image file.
	If given a mini-batch tensor, will save the tensor as a grid of images.
	"""
	im = Image.fromarray(palettes[prediction.squeeze()])
	_, fn = os.path.split(filename)
	fn = os.path.join(output_dir, fn[:-4] + '.png')
	out_dir = os.path.split(fn)[0]
	if not os.path.exists(out_dir):
		os.makedirs(out_dir)
	im.save(fn)

def pixel_acc(pred, label):
	_, preds = torch.max(pred, dim=1)
	valid = (label >= 0).long()
	acc_sum = torch.sum(valid * (preds == label).long())
	pixel_sum = torch.sum(valid)
	acc = acc_sum.float() / (pixel_sum.float() + 1e-10)
	return acc

def test(all_im_paths, model, num_classes,
		 output_dir='pred', save_vis=False):
	model.eval()
	hist = np.zeros((num_classes, num_classes))
	pix_acc = 0
	for im_path in tqdm(all_im_paths):
		im = imread(im_path)
		im = torch.from_numpy(im.transpose(2, 0, 1)).float().unsqueeze(0).cuda()
		segSize = im.shape[2:]
		with torch.no_grad():
			# print('image: {}'.format(image.shape))
			_, _, seg_pred = model(cur_im=im, seg_use_softmax=True, seg_size=segSize)
			assert seg_pred is not None, 'segmentation is seriously wrong.'
			seg_pred = seg_pred[0]
		_, pred = torch.max(seg_pred, 1)
		pred = pred.cpu().data.numpy().squeeze()
		if save_vis:
			save_output_images(pred, im_path, os.path.join(output_dir, 'seg'))
			save_colorful_images(
				pred, im_path, os.path.join(output_dir, 'seg_color'),
				TRIPLET_PALETTE if num_classes == 3 else CITYSCAPE_PALETTE
			)

class ToTensor(object):
	"""Converts a PIL.Image or numpy.ndarray (H x W x C) in the range
	[0, 255] to a torch.FloatTensor of shape (C x H x W) in the range [0.0, 1.0].
	"""

	def __init__(self, convert_pix_range=True):
		self.convert_pix_range = convert_pix_range

	def __call__(self, pic, label=None):
		if isinstance(pic, np.ndarray):
			# handle numpy array
			img = torch.from_numpy(pic.transpose(2, 0, 1))
		else:
			# handle PIL Image
			img = torch.ByteTensor(torch.ByteStorage.from_buffer(pic.tobytes()))
			# PIL image mode: 1, L, P, I, F, RGB, YCbCr, RGBA, CMYK
			if pic.mode == 'YCbCr':
				nchannel = 3
			else:
				nchannel = len(pic.mode)
			img = img.view(pic.size[1], pic.size[0], nchannel)
			# put it from HWC to CHW format
			# yikes, this transpose takes 80% of the loading time/CPU
			img = img.transpose(0, 1).transpose(0, 2).contiguous()
		img = img.float()
		if self.convert_pix_range:
			img = img.div(255)
		if label is None:
			return img,
		else:
			return img, torch.LongTensor(np.array(label, dtype=np.int))

def main(args):
	if args.save_dir is not None and args.save_dir != 'None':
		if not os.path.exists(args.save_dir):
			os.makedirs(args.save_dir)
			
	# load model
	model = model_utils.make_model(
		args, 
		do_flow=not args.no_flow, 
		do_disp=not args.no_disp, 
		do_seg=True
	)

	ckpt = torch.load(args.loadmodel)
	model.load_state_dict(ckpt['state_dict'])
	print('==> Successfully loaded a model {}.'.format(args.loadmodel))

	model.eval()

	# if args.split == 'training':
	#     args.split = 'train'
	# if args.split == 'test' and args.dataset.startswith('kitti'):
	#     args.split = 'val'
	# phase = args.dataset + '_' + args.split
	# data_dir = args.datapath
	
	# # single-scale testing
	# dataset = SegList(data_dir, phase, 
	#     ToTensor(convert_pix_range=False), 
	#     list_dir=data_dir, out_name=True, im_format='cv2'
	# )
	# test_loader = torch.utils.data.DataLoader(
	#     dataset,
	#     batch_size=1, shuffle=False, num_workers=8,
	#     pin_memory=False
	# )

	kitti_dir = '/home/hzjiang/workspace/Data/KITTI_scene_flow'
	all_im_paths = make_kitti2015_paths(kitti_dir, args.split)
	print('{} samples found for {}.'.format(len(all_im_paths), args.split))

	cudnn.benchmark = True

	out_dir = args.save_dir
	test(
		all_im_paths, 
		model, 
		args.num_seg_class, 
		save_vis=args.save_dir is not None,
		output_dir=args.save_dir
	)

if __name__ == '__main__':
	assert LooseVersion(torch.__version__) >= LooseVersion('0.4.0'), \
		'PyTorch>=0.4.0 is required'

	parser = parse_args()
	parser.add_argument('--joint-model', action='store_true')
	parser.add_argument(
		'--split', choices=['training', 'testing', 'val'],
		default='training'
	)
	args = parser.parse_args()

	args.save_vis = args.save_dir is not None and args.save_dir != 'None'

	if args.save_vis:
		if not os.path.isdir(args.save_dir):
			os.makedirs(args.save_dir)

	main(args)
