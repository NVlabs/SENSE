"""
Copyright (C) 2019 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""
import numpy as np

import torch
import torch.nn

from .models import SceneNet
from .pwc6l import PWC6LEncoder, PWC6LFlowDecoder, PWC6LDispDecoder
from .pwc import PWCEncoder, PWCFlowDecoder, PWCDispDecoder
from .psmnet import PSMEncoder
from .upernet import UPerNetLight
from .loss import multiscaleloss

from sense.lib.nn import DataParallelWithCallback

def make_model(args, do_flow=True, do_disp=True, do_pose=False, do_seg=False):
	assert do_flow or do_disp, 'At least one decoder is required.'
	
	# encoder
	with_ppm = (not args.no_ppm) or (not args.flow_no_ppm)
	if args.enc_arch == 'pwc':
		enc = PWCEncoder(args.bn_type, with_ppm)
		num_channels = [16, 32, 64, 96, 128]
	elif args.enc_arch == 'pwc6l':
		# 6-layer PWC
		enc = PWC6LEncoder()
		num_channels = [16, 32, 64, 96, 128, 196]
	elif args.enc_arch == 'psm':
		enc = PSMEncoder(args.bn_type, with_ppm)
		num_channels = [32, 32, 64, 128, 128]
		num_channels = [32, 32, 64, 128, 128]
	else:
		raise Exception('Unsupported encoder architecture: {}'.format(args.enc_arch))

	# decoder
	flow_dec = None
	disp_dec = None
	if do_flow:
		if args.flow_dec_arch == 'pwcdc':
			flow_dec = PWCFlowDecoder(encoder_planes=num_channels,
									md=args.corr_radius,
									refinement_module=args.flow_refinement,
									bn_type=args.bn_type,
									pred_occ=not args.no_occ,
									cat_occ=args.cat_occ,
									upsample_output=args.upsample_flow_output)
		elif args.flow_dec_arch == 'pwcdc6l':
			assert args.enc_arch == 'pwc6l', '6-layer decoder only supports 6-layer encoder.'
			flow_dec = PWC6LFlowDecoder()
		else:
			raise Exception('Not supported optical flow decoder: {}'.format(args.flow_dec_arch))

	if do_disp:
		if args.dec_arch == 'pwcdc':
			disp_dec = PWCDispDecoder(encoder_planes=num_channels,
									md=args.corr_radius,
									do_class=args.do_class,
									refinement_module=args.disp_refinement,
									bn_type=args.bn_type,
									pred_occ=not args.no_occ,
									cat_occ=args.cat_occ
									)
		elif args.dec_arch == 'pwcdc6l':
			assert args.enc_arch == 'pwc6l', '6-layer decoder only supports 6-layer encoder.'
			disp_dec = PWC6LDispDecoder()
		elif args.dec_arch == 'pwcdc2':
			disp_dec = PWC2DispDecoder(encoder_planes=num_channels,
									md=args.corr_radius,
									do_class=args.do_class,
									refinement_module=args.disp_refinement,
									bn_type=args.bn_type,
									pred_occ=not args.no_occ,
									cat_occ=args.cat_occ
									)
	# elif args.dec_arch == 'pwcdc3':
	# 	flow_dec = PWC3DispDecoder(encoder_planes=num_channels,
	# 								md=args.corr_radius,
	# 								do_class=args.do_class,
	# 								with_ppm=not args.no_ppm,
	# 								refinement_module=args.disp_refinement
	# 								)
	# elif args.dec_arch == 'pwcx':
	# 	flow_dec = PWCXDispDecoder(encoder_planes=num_channels,
	# 							   md=args.corr_radius,
	# 							   do_class=args.do_class,
	# 							   with_ppm=not args.no_ppm,
	# 							   refinement_module=args.disp_refinement
	# 							   )
	# elif args.dec_arch == 'pwcx2':
	# 	flow_dec = PWCX2DispDecoder(encoder_planes=num_channels,
	# 							   md=args.corr_radius,
	# 							   do_class=args.do_class,
	# 							   with_ppm=not args.no_ppm,
	# 							   refinement_module=args.disp_refinement
	# 							   )
		elif args.dec_arch == 'pwcx3':
			disp_dec = PWCX3DispDecoder(encoder_planes=num_channels,
									   md=args.corr_radius,
									   do_class=args.do_class,
									   with_ppm=not args.no_ppm,
									   refinement_module=args.disp_refinement,
									   bn_type=args.bn_type
									   )
		elif args.dec_arch == 'pwcx5':
			disp_dec = PWCX5DispDecoder(encoder_planes=num_channels,
									   md=args.corr_radius,
									   do_class=args.do_class,
									   with_ppm=not args.no_ppm,
									   refinement_module=args.disp_refinement,
									   bn_type=args.bn_type
									   )
		else:
			raise Exception('Not supported decoder {}'.format(args.dec_arch))

	if do_pose:
		pose_dec = PoseDecoder(encoder_last_plane=num_channels[-1],
			md=args.corr_radius, bn_type=args.bn_type)
	else:
		pose_dec = None

	if do_seg:
		seg_dec = UPerNetLight(
			num_class=args.num_seg_class, 
			fc_dim=num_channels[-1],
			fpn_inplanes=num_channels[1:], 
			fpn_dim=256
		)
	else:
		seg_dec = None

	model = SceneNet(
		enc, 
		flow_dec=flow_dec, 
		disp_dec=disp_dec, 
		seg_dec=seg_dec,
		bn_type=args.bn_type,
		disp_with_ppm=not args.no_ppm,
		flow_with_ppm=not args.flow_no_ppm
	)

	if args.bn_type == 'plain':
		model = torch.nn.DataParallel(model).cuda()
	elif args.bn_type == 'syncbn':
		model = DataParallelWithCallback(model).cuda()
	elif args.bn_type == 'encoding':
		raise Exception('To be supported.')
		model = torch.nn.DataParallel(model).cuda()
		encoding.parallel.patch_replication_callback(model)
	else:
		raise Exception('Not supported bn type: {}'.format(args.bn_type))
	return model

def make_encoder(args):
	# encoder
	with_ppm = (not args.no_ppm) or (not args.flow_no_ppm)
	if args.enc_arch == 'pwc':
		enc = PWCEncoder(args.bn_type, with_ppm)
		num_channels = [16, 32, 64, 96, 128]
	elif args.enc_arch == 'pwc6l':
		# 6-layer PWC
		enc = PWC6LEncoder()
		num_channels = [16, 32, 64, 96, 128, 196]
	elif args.enc_arch == 'psm':
		enc = PSMEncoder(args.bn_type, with_ppm)
		num_channels = [32, 32, 64, 128, 128]
	elif args.enc_arch == 'psm2':
		enc = PSM2Encoder(args.bn_type, with_ppm)
		num_channels = [32, 32, 64, 128, 128]
	# elif args.enc_arch == 'resnet18' or args.enc_arch == 'resnet34':
	# 	enc = model_builder.build_encoder(arch=args.enc_arch)
	# 	num_channels = [32, 64, 128, 256, 512]
	# elif args.enc_arch == 'resnet50' or args.enc_arch == 'resnet101':
	# 	enc = model_builder.build_encoder(arch=args.enc_arch)
	# 	num_channels = [128, 256, 512, 1024, 2048]
	else:
		raise Exception('Unsupported encoder architecture: {}'.format(args.enc_arch))

	if args.bn_type == 'plain':
		enc = torch.nn.DataParallel(enc).cuda()
	elif args.bn_type == 'syncbn':
		enc = DataParallelWithCallback(enc).cuda()
	elif args.bn_type == 'encoding':
		raise Exception('To be supported.')
		enc = torch.nn.DataParallel(enc).cuda()
		encoding.parallel.patch_replication_callback(enc)
	else:
		raise Exception('Not supported bn type: {}'.format(args.bn_type))
	return enc

def make_seg_decoder(args):
	if args.enc_arch == 'pwc':
		num_channels = [16, 32, 64, 96, 128]
	elif args.enc_arch == 'pwc6l':
		# 6-layer PWC
		num_channels = [16, 32, 64, 96, 128, 196]
	elif args.enc_arch == 'psm':
		num_channels = [32, 32, 64, 128, 128]
	elif args.enc_arch == 'psm2':
		num_channels = [32, 32, 64, 128, 128]

	seg_dec = UPerNetLight(
			num_class=args.num_seg_class, 
			fc_dim=num_channels[-1],
			fpn_inplanes=num_channels[1:], 
			fpn_dim=256
			)
	if args.bn_type == 'plain':
		seg_dec = torch.nn.DataParallel(seg_dec).cuda()
	elif args.bn_type == 'syncbn':
		seg_dec = DataParallelWithCallback(seg_dec).cuda()
	elif args.bn_type == 'encoding':
		raise Exception('To be supported.')
		seg_dec = torch.nn.DataParallel(seg_dec).cuda()
		encoding.parallel.patch_replication_callback(seg_dec)
	else:
		raise Exception('Not supported bn type: {}'.format(args.bn_type))
	return seg_dec

def make_seg_teacher_model(args):
	seg_teacher_model = None
	if args.do_seg and args.seg_root_dir is None:
		# we need the segmentation model for distillation loss
		seg_teacher_model = make_resnet101_upernet(args)
		print('==> Made a segmentation teacher model.')

	if args.disp_semantic_consist_wt > 0 or args.flow_semantic_consist_wt > 0:
		if seg_teacher_model is None:
			seg_teacher_model = make_resnet101_upernet(args)
			print('==> Made a segmentation teacher model.')

	if seg_teacher_model is not None:
		if args.bn_type == 'plain':
			seg_teacher_model = torch.nn.DataParallel(seg_teacher_model).cuda()
		elif args.bn_type == 'syncbn':
			seg_teacher_model = DataParallelWithCallback(seg_teacher_model).cuda()
		else:
			raise Exception('Not supported bn type: {}'.format(args.bn_type))
	return seg_teacher_model

def make_disp_criteria(args):
	# # this weights works well only for 1/4 spatial resolution
	# # weights = np.array((0.0025, 0.005, 0.01, 0.02, 0.08))
	# # weights = np.array((0.005, 0.01, 0.01, 0.02, 0.08))
	# # downsample_factors = (1, 2, 2, 4, 8)
	# # change the weights according to spatial resolution
	# weights = np.array((0.005, 0.005, 0.01, 0.02, 0.08))
	# switch to use normalized per pixel loss, which is easier to interpret
	if args.dec_arch == 'pwcdc6l':
		downsample_factors = (1, 2, 4, 8, 16)
		if args.per_pix_loss:
			weights = np.array((4, 2, 1, 1, 1))
		else:
			# original weights were designed for 1/4 spatial resolution
			weights = np.array((0.005, 0.01, 0.02, 0.08, 0.32)) / 16
	elif args.dec_arch.find('pwcdc') >= 0 or args.dec_arch.find('pwcx') >= 0:
		if args.disp_refinement == 'none':
			downsample_factors = (1, 2, 4, 8)
			weights = np.array((4, 1, 1, 1), dtype=np.float32)
			if not args.per_pix_loss:
				orig_wts = np.array((0.00125, 0.005, 0.02, 0.08), dtype=np.float32) / 16
		else:
			downsample_factors = (1, 1, 2, 4, 8)
			weights = np.array((8, 4, 1, 1, 1), dtype=np.float32)
			if not args.per_pix_loss:
				orig_wts = np.array((0.00125, 0.00125, 0.005, 0.02, 0.08), dtype=np.float32) / 16
				weights *= orig_wts
	else:
		raise Exception('Not supported decoder architecture {} for disp.'.format(args.dec_arch))

	print('For disp: ', weights, downsample_factors)
	disp_crit = multiscaleloss(downsample_factors=downsample_factors,
							   weights=weights, 
							   loss='smooth_l1',
							   size_average=args.per_pix_loss
							   ).cuda()
	if args.no_occ:
		disp_occ_crit = None
	else:
		if args.soft_occ_gt:
			disp_occ_crit = multiscaleloss(
				downsample_factors=downsample_factors,
				weights=weights,
				loss='smooth_l1',
				size_average=args.per_pix_loss
				).cuda()
		else:
			disp_occ_crit = multiscaleloss(
				downsample_factors=downsample_factors,
				weights=weights,
				loss='xentropy_loss',
				size_average=args.per_pix_loss,
				class_weights=torch.Tensor(args.disp_occ_wts).float()
				).cuda()
	return (disp_crit, disp_occ_crit), downsample_factors, weights

def make_flow_criteria(args):
	if args.flow_dec_arch == 'pwcdc6l':
		downsample_factors = (4, 8, 16, 32, 64)
		if args.per_pix_loss:
			weights = (4, 2, 1, 1, 1)
		else:
			weights = (0.005, 0.01, 0.02, 0.08, 0.32)
	elif args.flow_dec_arch.find('pwcdc') >= 0 or args.flow_dec_arch.find('pwcx') >= 0:
		if args.flow_refinement == 'none':
			downsample_factors = (4, 8, 16, 32)
			spatial_weights = 1
			if args.upsample_flow_output:
				downsample_factors = (1, 4, 8, 16)
				spatial_weights = 16
			weights = np.array((4, 2, 1, 1), dtype=np.float32)
			if not args.per_pix_loss:
				orig_wts = np.array((0.00125, 0.005, 0.02, 0.08), dtype=np.float32) / spatial_weights
				weights *= orig_wts
				if args.upsample_flow_output:
					# orig_wts = np.array((0.00125, 0.00125, 0.00125, 0.00125), dtype=np.float32) / spatial_weights
					weights = np.array((0.001, 0.0025, 0.005, 0.02), dtype=np.float32)
		else:
			downsample_factors = (2, 4, 8, 16, 32)
			spatial_weights = 1
			if args.upsample_flow_output:
				downsample_factors = (1, 1, 4, 8, 16)
				spatial_weights = 16
			weights = np.array((8, 4, 2, 1, 1), dtype=np.float32)
			if not args.per_pix_loss:
				orig_wts = np.array((0.00125, 0.00125, 0.005, 0.02, 0.08), dtype=np.float32) / spatial_weights
				weights *= orig_wts
				if args.upsample_flow_output:
					# orig_wts = np.array((0.00125, 0.00125, 0.00125, 0.00125, 0.00125), dtype=np.float32) / spatial_weights
					weights = np.array((0.001, 0.001, 0.0025, 0.005, 0.02), dtype=np.float32)
	else:
		raise Exception('Not supported decoder architecture {} for flow.'.format(args.dec_arch))

	print('For flow: ', weights, downsample_factors)
	flow_crit = multiscaleloss(downsample_factors=downsample_factors,
							   weights=weights, 
							   loss=args.flow_loss_type,
							   p_robust=args.robust_loss_p,
							   size_average=args.per_pix_loss
							   ).cuda()

	if args.no_occ:
		flow_occ_crit = None
	else:
		if args.soft_occ_gt:
			flow_occ_crit = multiscaleloss(
				downsample_factors=downsample_factors,
				weights=weights,
				loss='smooth_l1',
				size_average=args.per_pix_loss
				).cuda()
		else:
			flow_occ_crit = multiscaleloss(
				downsample_factors=downsample_factors,
				weights=weights,
				loss='xentropy_loss',
				size_average=args.per_pix_loss,
				class_weights=torch.Tensor(args.flow_occ_wts).float()
				).cuda()
	return (flow_crit, flow_occ_crit), downsample_factors, weights

def make_seg_criterion(args, hard_lab=True):
	class DistillLoss(torch.nn.Module):
		def forward(self, pred, soft_gt, size_average=False):
			dist_loss = torch.sum(-soft_gt * pred, 1)
			if size_average:
				dist_loss = torch.mean(dist_loss)
			else:  
				dist_loss = torch.sum(dist_loss)
			return dist_loss

	if args.do_seg or args.do_seg_distill:
		if hard_lab:
			return torch.nn.NLLLoss(ignore_index=255, size_average=args.per_pix_loss).cuda()
		else:
			return DistillLoss().cuda()

	return None

	# loss = dist_loss * args.alpha * (args.T ** 2)

def make_disp_checkpoint_name(args, epoch):
	savefilename = args.dataset + '_disp'
	savefilename += '_' + args.enc_arch
	savefilename += '_' + args.dec_arch
	savefilename += '_' + ('woppm' if args.no_ppm else 'wppm')
	savefilename += '_' + args.disp_refinement
	savefilename += '_class' if args.do_class else '_reg' 
	savefilename += '_md_{}'.format(args.corr_radius)
	savefilename += '_' + args.bn_type
	savefilename += '_woocc' if args.no_occ else '_wocc'
	if not args.no_occ:
		savefilename += '_wcatocc' if args.cat_occ else '_wocatocc'
	savefilename += '_mean_loss' if args.per_pix_loss else '_sum_loss'
	savefilename += '_cropSize_' + str(args.disp_crop_imh) + 'x' + str(args.disp_crop_imw)
	savefilename += '_' + str(epoch)
	savefilename += '.tar'
	return savefilename

def make_flow_checkpoint_name(args, epoch):
	savefilename = args.dataset + '_flow'
	savefilename += '_' + args.enc_arch
	savefilename += '_' + args.flow_dec_arch
	savefilename += '_' + ('woppm' if args.flow_no_ppm else 'wppm')
	savefilename += '_' + args.flow_refinement
	savefilename += '_md_{}'.format(args.corr_radius)
	savefilename += '_' + args.bn_type
	savefilename += '_woocc' if args.no_occ else '_wocc'
	if not args.no_occ:
		savefilename += '_wcatocc' if args.cat_occ else '_wocatocc'
	savefilename += '_mean_loss' if args.per_pix_loss else '_sum_loss'
	savefilename += '_flowDimRatio_' + '{}'.format(args.flow_dim_ratio)
	savefilename += '_cropSize_' + str(args.flow_crop_imh) + 'x' + str(args.flow_crop_imw)
	savefilename += '_' + str(epoch)
	savefilename += '.tar'
	return savefilename

def make_joint_checkpoint_name(args, epoch):
	savefilename = args.dataset + '_joint'
	savefilename += '_' + args.enc_arch
	savefilename += '_' + args.dec_arch
	savefilename += '_' + ('woppm' if args.no_ppm else 'wppm')
	savefilename += '_' + args.disp_refinement
	savefilename += '_class' if args.do_class else '_reg' 
	savefilename += '_' + args.flow_dec_arch
	savefilename += '_' + ('woppm' if args.flow_no_ppm else 'wppm')
	savefilename += '_' + args.flow_refinement
	savefilename += '_md_{}'.format(args.corr_radius)
	savefilename += '_' + args.bn_type
	savefilename += '_woocc' if args.no_occ else '_wocc'
	if not args.no_occ:
		savefilename += '_wcatocc' if args.cat_occ else '_wocatocc'
	savefilename += '_mean_loss' if args.per_pix_loss else '_sum_loss'
	savefilename += '_dispCropSize_' + str(args.disp_crop_imh) + 'x' + str(args.disp_crop_imw)
	savefilename += '_flowDimRatio_' + '{}'.format(args.flow_dim_ratio)
	savefilename += '_flowCropSize_' + str(args.flow_crop_imh) + 'x' + str(args.flow_crop_imw)
	savefilename += '_wseg' if args.do_seg else '_woseg'
	if args.do_seg:
		if args.seg_root_dir is not None:
			savefilename += '_hardSegGt'
		else:
			# savefilename += '_softSegGt'
			raise NotImplementedError
		# savefilename += '_hardSegLossWt_{}'.format(args.hard_seg_loss_weight)

	if args.do_seg_distill:
		savefilename += '_segDistill'
		savefilename += '_distT_{}'.format(args.seg_distill_T)
		# savefilename += '_softSegLossWt_{}'.format(args.soft_seg_loss_weight)

	if args.pseudo_gt_dir is not None:
		savefilename += '_pseudoGt'
	if args.soft_occ_gt:
		savefilename += '_softOccGt'

	tmp_dir = savefilename

	# # self-supervised parameters
	# savefilename = 'dispPhotoConsistWt_{}'.format(args.disp_photo_consist_wt)
	# savefilename += '_dispSemanticConsistWt_{}'.format(args.disp_semantic_consist_wt) 
	# savefilename += '_flowPhotoConsistWt_{}'.format(args.flow_photo_consist_wt) 
	# savefilename += '_flowSemanticConsistWt_{}'.format(args.flow_semantic_consist_wt) 
	# savefilename += '_dispTemporalConsistWt_{}'.format(args.disp_temporal_consist_wt)
	# savefilename += '_flowDispConsistWt_{}'.format(args.flow_disp_consist_wt)
	# savefilename += '_flowSmoothnessWt_{}'.format(args.flow_smoothness_wt)
	# savefilename += '_dispSmoothnessWt_{}'.format(args.disp_smoothness_wt)
	# savefilename += '_flowSsimWt_{}'.format(args.flow_ssim_wt)
	# savefilename += '_dispSsimWt_{}'.format(args.disp_ssim_wt)
	# savefilename += '_flowOccPenalty_{}'.format(args.flow_occ_penalty)
	# savefilename += '_dispOccPenalty_{}'.format(args.disp_occ_penalty)

		# self-supervised parameters
	savefilename = 'dpcWt_{}'.format(args.disp_photo_consist_wt)
	savefilename += '_dscWt_{}'.format(args.disp_semantic_consist_wt) 
	savefilename += '_fpcWt_{}'.format(args.flow_photo_consist_wt) 
	savefilename += '_fscWt_{}'.format(args.flow_semantic_consist_wt) 
	savefilename += '_dtcWt_{}'.format(args.disp_temporal_consist_wt)
	savefilename += '_fdcWt_{}'.format(args.flow_disp_consist_wt)
	savefilename += '_fsWt_{}'.format(args.flow_smoothness_wt)
	savefilename += '_dsWt_{}'.format(args.disp_smoothness_wt)
	savefilename += '_flowSsimWt_{}'.format(args.flow_ssim_wt)
	savefilename += '_dispSsimWt_{}'.format(args.disp_ssim_wt)
	savefilename += '_fop_{}'.format(args.flow_occ_penalty)
	savefilename += '_dop_{}'.format(args.disp_occ_penalty)

	import os
	savefilename = os.path.join(tmp_dir, savefilename)



	# savefilename += '_' + str(epoch)
	# savefilename += '.tar'
	return savefilename

def make_pose_checkpoint_name(args, epoch):
	savefilename = args.dataset + '_pose'
	savefilename += '_' + args.enc_arch
	savefilename += '_' + args.dec_arch
	savefilename += '_' + ('woppm' if args.no_ppm else 'wppm')
	savefilename += '_' + args.disp_refinement
	savefilename += '_class' if args.do_class else '_reg' 
	savefilename += '_' + args.flow_dec_arch
	savefilename += '_' + ('woppm' if args.flow_no_ppm else 'wppm')
	savefilename += '_' + args.flow_refinement
	savefilename += '_md_{}'.format(args.corr_radius)
	savefilename += '_' + args.bn_type
	savefilename += '_woocc' if args.no_occ else '_wocc'
	if not args.no_occ:
		savefilename += '_wcatocc' if args.cat_occ else '_wocatocc'
	savefilename += '_mean_loss' if args.per_pix_loss else '_sum_loss'
	savefilename += '_dispCropSize_' + str(args.disp_crop_imh) + 'x' + str(args.disp_crop_imw)
	savefilename += '_flowDimRatio_' + '{}'.format(args.flow_dim_ratio)
	savefilename += '_flowCropSize_' + str(args.flow_crop_imh) + 'x' + str(args.flow_crop_imw)
	savefilename += '_' + str(epoch)
	savefilename += '.tar'
	return savefilename

def patch_model_state_dict(state_dict):
	# keys = list(state_dict.keys())
	# for k in keys:
	# 	if k.find('num_batches_tracked') >= 0:
	# 		state_dict.pop(k)
	return state_dict