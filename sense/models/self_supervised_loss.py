"""
Copyright (C) 2019 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from .common import flow_warp, disp_warp
from .mssim import MSSSIM

class OcclusionAwareLoss(nn.Module):
	def __init__(self, size_average, loss_type):
		super(OcclusionAwareLoss, self).__init__()
		# self.flow_occ_penalty = flow_occ_penalty
		self.size_average = size_average
		loss_reduce = False	#self.flow_occ_penalty <= 0
		if loss_type =='l1':
			self.lossFunc = torch.nn.L1Loss(size_average=size_average, reduce=loss_reduce)
		elif loss_type == 'smooth_l1':
			self.lossFunc = torch.nn.SmoothL1Loss(size_average=size_average, reduce=loss_reduce)
		elif loss_type == 'kl_div':
			self.lossFunc = torch.nn.KLDivLoss(size_average=size_average, reduce=loss_reduce)
		else:
			raise NotImplementedError("Loss not implemented")

	def forward(self, input, target, occ_mask, valid_mask):
		loss = self.lossFunc(input, target)
		if occ_mask is None:
			# return loss
			raise NotImplementedError

		# occ_mask: [Nx1xHxW]
		loss = loss * (1 - occ_mask)
		loss = loss * valid_mask
		if self.size_average:
			loss = torh.sum(loss) / (torch.sum(valid_mask) + 1e-30)
		else:
			loss = torch.sum(loss)

		# original version, with regularization term
		# if self.flow_occ_penalty > 0:
		# 	loss = loss * (1 - occ_mask) + self.flow_occ_penalty * occ_mask
		# 	if self.size_average:
		# 		loss = torh.mean(loss)
		# 	else:
		# 		loss = torch.sum(loss)
		# else:
		# 	loss = loss * (1 - occ_mask) + (1 - loss) * occ_mask
		# 	if self.size_average:
		# 		loss = torh.mean(loss)
		# 	else:
		# 		loss = torch.sum(loss)
		return loss

class OcclusionMaskRegularizer(nn.Module):
	def __init__(self, downsample_factors, weights, size_average):
		super(OcclusionMaskRegularizer, self).__init__()

		self.size_average = size_average
		self.downsample_factors = downsample_factors
		self.weights = weights
		self.weights = torch.Tensor(downsample_factors).fill_(1) if weights is None else torch.Tensor(weights)
		self.weights = self.weights.cuda()

	def forward(self, multi_scale_occ, valid_mask):
		loss = 0
		for i, occ_mask in enumerate(multi_scale_occ):
			downscale = int(self.downsample_factors[i])
			valid_mask_ = valid_mask[:, :, ::downscale, ::downscale]
			# occ_mask: [Nx1xHxW]
			if self.size_average:
				tmp = torch.sum(torch.abs(occ_mask) * valid_mask_) / (torch.sum(valid_mask_) + 1e-30)
			else:
				tmp = torch.sum(torch.abs(occ_mask) * valid_mask_)
			loss += tmp * self.weights[i]
		return loss

class FlowDispPhotoSemanticConsist(nn.Module):
	def __init__(self, downsample_factors, weights=None, 
				 loss= 'l1', size_average=False, is_disp=True):
		super(FlowDispPhotoSemanticConsist,self).__init__()
		self.lossFunc = OcclusionAwareLoss(size_average, loss)
		self.downsample_factors = downsample_factors
		self.weights = torch.Tensor(downsample_factors).fill_(1) if weights is None else torch.Tensor(weights)
		self.weights = self.weights.cuda()
		assert(len(weights) == len(downsample_factors))
		
		self.size_average = size_average
		self.warpFunc = disp_warp
		if not is_disp:
			self.warpFunc = flow_warp

	def forward(self, left_x, right_x, multi_scale_disp, 
		multi_scale_occ=None, valid_mask=None):
		assert type(multi_scale_disp) is tuple, ('multi-scale disp predictions are required.', type(multi_scale_disp))
		out = 0
		losses = []
		if type(multi_scale_disp) is tuple:
			if multi_scale_occ is not None:
				assert len(multi_scale_disp) == len(multi_scale_occ), \
					'Error in disparity and occlusions: {} vs {}'.format(len(multi_scale_disp), len(multi_scale_occ))
			for i, disp in enumerate(multi_scale_disp):
				downscale = int(self.downsample_factors[i])
				left_x_ = left_x[:, :, ::downscale, ::downscale]
				right_x_ = right_x[:, :, ::downscale, ::downscale]
				warp_right_x_ = self.warpFunc(right_x_, disp)
				if multi_scale_occ is not None:
					occ = multi_scale_occ[i]
				else:
					occ = None
				valid_mask_ = valid_mask[:, :, ::downscale, ::downscale]
				tmp = self.lossFunc(warp_right_x_, left_x_, occ, valid_mask_)
				out += self.weights[i] * tmp
				losses.append(self.weights[i].item() * tmp.item())
		else:
			raise NotImplementedError
			downscale = int(self.downsample_factors[0])
			left_x_ = left_x[:, :, ::downscale, ::downscale]
			right_x_ = right_x[:, :, ::downscale, ::downscale]
			warp_right_x_ = self.warpFunc(right_x_, disp)
			tmp = self.lossFunc(warp_right_x_, left_x_, occ)
			out += self.weights[0] * tmp
			losses.append(self.weights[0].item() * tmp.item())
		return out

class DispTemporalConsist(nn.Module):
	def __init__(self):
		pass

class FlowDispConsist(nn.Module):
	def __init__(self):
		pass

class SpatialSmoothness(nn.Module):
	def __init__(self, downsample_factors, weights=None, size_average=False):
		super(SpatialSmoothness, self).__init__()
		self.downsample_factors = downsample_factors
		self.weights = torch.Tensor(downsample_factors).fill_(1) if weights is None else torch.Tensor(weights)
		self.weights = self.weights.cuda()
		assert(len(weights) == len(downsample_factors))

		self.size_average = size_average

	def gradient_x(self, img):
		gx = img[:,:,:-1,:] - img[:,:,1:,:]
		return gx

	def gradient_y(self, img):
		gy = img[:,:,:,:-1] - img[:,:,:,:1]
		return gy

	def single_scale_loss(self, ref_im, pred):
		pred_grad_x = self.gradient_x(pred)
		pred_grad_y = self.gradient_y(pred)

		with torch.no_grad():
			im_grad_x = self.gradient_x(ref_im) * 255
			im_grad_y = self.gradient_y(ref_im) * 255

		weights_x = torch.exp(-torch.mean(torch.abs(im_grad_x), 1, keepdim=True))
		weights_y = torch.exp(-torch.mean(torch.abs(im_grad_y), 1, keepdim=True))

		# print('ref_im: ', ref_im.min().item(), ref_im.max().item())
		# print('im_grad_x: ', im_grad_x.min().item(), im_grad_x.max().item())
		# print('im_grad_y: ', im_grad_y.min().item(), im_grad_y.max().item())
		# print('weights_x: ', weights_x.min().item(), weights_x.max().item())
		# print('weights_y: ', weights_y.min().item(), weights_y.max().item())

		smoothness_x = torch.abs(pred_grad_x) * weights_x
		smoothness_y = torch.abs(pred_grad_y) * weights_y
		if self.size_average:
			loss = torch.mean(smoothness_x) + torch.mean(smoothness_y)
		else:
			loss = torch.sum(smoothness_x) + torch.sum(smoothness_y)
		return loss

	def forward(self, ref_im, multi_scale_pred):
		assert type(multi_scale_pred) is tuple, ('multi-scale predictions are required.', type(multi_scale_pred))
		out = 0
		losses = []
		if type(multi_scale_pred) is tuple:
			for i, pred in enumerate(multi_scale_pred):
				downscale = int(self.downsample_factors[i])
				ref_im_ = ref_im[:, :, ::downscale, ::downscale]
				tmp = self.single_scale_loss(ref_im_, pred)
				out += self.weights[i] * tmp
				losses.append(self.weights[i].item() * tmp.item())
		else:
			raise NotImplementedError
		return out, losses

class MultiScaleMSSSIM(nn.Module):
	def __init__(self, downsample_factors, weights=None, 
			size_average=False, is_disp=True):
		super(MultiScaleMSSSIM, self).__init__()
		self.downsample_factors = downsample_factors
		self.weights = torch.Tensor(downsample_factors).fill_(1) if weights is None else torch.Tensor(weights)
		self.weights = self.weights.cuda()
		assert(len(weights) == len(downsample_factors))

		self.size_average = size_average
		self.lossFunc = MSSSIM(size_average=True)
		self.warpFunc = disp_warp
		if not is_disp:
			self.warpFunc = flow_warp

	def forward(self, left_im, right_im, multi_scale_pred, multi_scale_occ=None):
		# assert type(multi_scale_pred) is tuple, ('multi-scale disp predictions are required.', type(multi_scale_disp))
		out = 0
		losses = []
		if type(multi_scale_pred) is tuple:
			raise NotImplementedError
			for i, disp in enumerate(multi_scale_pred):
				downscale = int(self.downsample_factors[i])
				left_im_ = left_im[:, :, ::downscale, ::downscale]
				right_im_ = right_im[:, :, ::downscale, ::downscale]
				warp_right_im_ = self.warpFunc(right_im_, disp)
				tmp = 1 - self.lossFunc(left_im_, warp_right_im_)
				if not self.size_average:
					tmp = tmp * float(disp.numel())
				out += self.weights[i] * tmp
				losses.append(self.weights[i].item() * tmp.item())
		else:
			disp = multi_scale_pred
			occ_mask = multi_scale_occ
			warp_right_im_ = self.warpFunc(right_im, disp)
			if occ_mask is not None:
				# warp_right_im_ = left_im * (1 - occ_mask) + warp_right_im_ * occ_mask
				warp_right_im_ = left_im * occ_mask + warp_right_im_ * (1 - occ_mask)
			out = self.lossFunc(left_im, warp_right_im_)
			if not self.size_average:
				out = (1 - out) * float(disp.numel()) * self.weights[0]
			losses.append(out.item())
		return out, losses

class SelfSupervisedLoss(nn.Module):
	def __init__(self, disp_downscales=None, disp_pyramid_weights=None,
				flow_downscales=None, flow_pyramid_weights=None,
				disp_photo_consist_wt=-1, disp_semantic_consist_wt=-1,
				flow_photo_consist_wt=-1, flow_semantic_consist_wt=-1,
				disp_temporal_consist_wt=-1, flow_disp_consist_wt=-1,
				flow_smoothness_wt=-1, disp_smoothness_wt=-1,
				flow_ssim_wt=-1, disp_ssim_wt=-1,
				size_average=False, 
				flow_occ_penalty=0.3, disp_occ_penalty=0.3):
		super(SelfSupervisedLoss, self).__init__()

		self.disp_photo_consist_wt = disp_photo_consist_wt
		self.disp_semantic_consist_wt = disp_semantic_consist_wt
		self.flow_photo_consist_wt = flow_photo_consist_wt
		self.flow_semantic_consist_wt = flow_semantic_consist_wt
		self.disp_temporal_consist_wt = disp_temporal_consist_wt
		self.flow_disp_consist_wt = flow_disp_consist_wt
		self.flow_smoothness_wt = flow_smoothness_wt
		self.disp_smoothness_wt = disp_smoothness_wt
		self.flow_ssim_wt = flow_ssim_wt
		self.disp_ssim_wt = disp_ssim_wt

		self.flow_occ_penalty = flow_occ_penalty
		self.disp_occ_penalty = disp_occ_penalty

		if disp_photo_consist_wt > 0:
			self.disp_photo_consist_crit = FlowDispPhotoSemanticConsist(
				disp_downscales, disp_pyramid_weights, 'l1', 
				size_average=size_average, is_disp=True)

		if disp_semantic_consist_wt > 0:
			self.disp_semantic_consist_crit = FlowDispPhotoSemanticConsist(
				disp_downscales, disp_pyramid_weights, 'l1', 
				size_average=size_average, is_disp=True)

		if flow_photo_consist_wt > 0:
			self.flow_photo_consist_crit = FlowDispPhotoSemanticConsist(
				flow_downscales, flow_pyramid_weights, 'l1', 
				size_average=size_average, is_disp=False)

		if flow_semantic_consist_wt > 0:
			self.flow_semantic_consist_crit = FlowDispPhotoSemanticConsist(
				flow_downscales, flow_pyramid_weights, 'l1', 
				size_average=size_average, is_disp=False)

		if flow_occ_penalty > 0:
			self.flow_occ_regularizer = OcclusionMaskRegularizer(
				flow_downscales, flow_pyramid_weights, size_average=size_average)
		
		if disp_occ_penalty > 0:
			self.disp_occ_regularizer = OcclusionMaskRegularizer(
				disp_downscales, disp_pyramid_weights, size_average=size_average)

		if disp_temporal_consist_wt > 0:
			self.disp_temporal_consist_crit = DispTemporalConsist()

		if flow_disp_consist_wt > 0:
			self.flow_disp_consist_crit = FlowDispConsist()

		if flow_smoothness_wt > 0:
			self.flow_smoothness_crit = SpatialSmoothness(
				flow_downscales, flow_pyramid_weights, 
				size_average=size_average)

		if disp_smoothness_wt > 0:
			self.disp_smoothness_crit = SpatialSmoothness(
				disp_downscales, disp_pyramid_weights, 
				size_average=size_average)

		if flow_ssim_wt > 0:
			self.flow_ssim_crit = MultiScaleMSSSIM(
				flow_downscales, flow_pyramid_weights, 
				is_disp=False, size_average=size_average)

		if disp_ssim_wt > 0:
			self.disp_ssim_crit = MultiScaleMSSSIM(
				disp_downscales, disp_pyramid_weights, 
				is_disp=True, size_average=size_average)

	def forward(self, cur_im=None, nxt_im=None, left_im=None, right_im=None,
				cur_seg=None, nxt_seg=None, left_seg=None, right_seg=None,
				multi_scale_disp=None, multi_scale_flow=None, camera_pose=None,
				multi_scale_disp_occ=None, multi_scale_flow_occ=None,
				disp_semi_loss_mask=None, flow_semi_loss_mask=None):
		def not_none(var):
			return var is not None

		loss = 0
		losses = {}

		if not_none(left_im) and not_none(right_im) and not_none(multi_scale_disp) and not_none(multi_scale_disp_occ):
			if disp_semi_loss_mask is None:
				B, _, H, W = left_im.size()
				disp_semi_loss_mask = left_im.new_ones((B, 1, H, W))

			if self.disp_photo_consist_wt > 0:
				tmp = self.disp_photo_consist_crit(
					left_im, right_im, multi_scale_disp, multi_scale_disp_occ, disp_semi_loss_mask)
				tmp *= self.disp_photo_consist_wt
				loss += tmp
				losses['disp_photo_consist'] = tmp.item()

			if self.disp_semantic_consist_wt > 0:
				tmp = self.disp_semantic_consist_crit(
					left_seg, right_seg, multi_scale_disp, multi_scale_disp_occ, disp_semi_loss_mask)
				tmp *= self.disp_semantic_consist_wt
				loss += tmp
				losses['disp_semantic_consist'] = tmp.item()

		if not_none(cur_im) and not_none(nxt_im) and not_none(multi_scale_flow) and not_none(multi_scale_flow_occ):
			if flow_semi_loss_mask is None:
				B, _, H, W = cur_im.size()
				flow_semi_loss_mask = cur_im.new_ones((B, 1, H, W))

			if self.flow_photo_consist_wt > 0:
				tmp = self.flow_photo_consist_crit(
					cur_im, nxt_im, multi_scale_flow, multi_scale_flow_occ, flow_semi_loss_mask)
				tmp *= self.flow_photo_consist_wt
				loss += tmp
				losses['flow_photo_consist'] = tmp.item()

			if self.flow_semantic_consist_wt > 0:
				tmp = self.flow_semantic_consist_crit(
					cur_seg, nxt_seg, multi_scale_flow, multi_scale_flow_occ, flow_semi_loss_mask)
				tmp *= self.flow_semantic_consist_wt
				loss += tmp
				losses['flow_semantic_consist'] = tmp.item()

		if self.flow_occ_penalty > 0 and not_none(multi_scale_flow_occ):
			if flow_semi_loss_mask is None:
				B, _, H, W = cur_im.size()
				flow_semi_loss_mask = cur_im.new_ones((B, 1, H, W))

			tmp = self.flow_occ_penalty * self.flow_occ_regularizer(multi_scale_flow_occ, flow_semi_loss_mask)
			loss += tmp
			losses['flow_occ_reg'] = tmp.item()

		if self.disp_occ_penalty > 0 and not_none(multi_scale_disp_occ):
			if disp_semi_loss_mask is None:
				B, _, H, W = left_im.size()
				disp_semi_loss_mask = left_im.new_ones((B, 1, H, W))

			tmp = self.disp_occ_penalty * self.disp_occ_regularizer(multi_scale_disp_occ, disp_semi_loss_mask)
			loss += tmp
			losses['disp_occ_reg'] = tmp.item()

		if self.disp_temporal_consist_wt > 0:
			raise NotImplementedError
			tmp = self.disp_temporal_consist_crit()
			tmp *= self.disp_temporal_consist_wt
			loss += tmp
			losses['disp_temporal_consist'] = tmp.item()

		if self.flow_disp_consist_wt > 0:
			raise NotImplementedError
			tmp = self.flow_disp_consist_crit()
			tmp *= self.flow_disp_consist_wt
			loss += tmp
			losses['flow_disp_consist'] = tmp.item()

		if not_none(cur_im) and not_none(multi_scale_flow) and self.flow_smoothness_wt > 0:
			tmp, _ = self.flow_smoothness_crit(
				cur_im, multi_scale_flow)
			tmp *= self.flow_smoothness_wt
			loss += tmp
			losses['flow_smoothness'] = tmp.item()

		if not_none(left_im) and not_none(multi_scale_disp) and self.disp_smoothness_wt > 0:
			tmp, _ = self.disp_smoothness_crit(
				left_im, multi_scale_disp)
			tmp *= self.disp_smoothness_wt
			loss += tmp
			losses['disp_smoothness'] = tmp.item()

		if not_none(cur_im) and not_none(nxt_im) and not_none(multi_scale_flow) and self.flow_ssim_wt > 0:
			tmp, _ = self.flow_ssim_crit(cur_im, nxt_im, multi_scale_flow[0], multi_scale_flow_occ[0])
			tmp = tmp * self.flow_ssim_wt
			loss += tmp
			losses['flow_ssim'] = tmp.item()

		if not_none(left_im) and not_none(right_im) and not_none(multi_scale_disp) and self.disp_ssim_wt > 0:
			tmp, _ = self.disp_ssim_crit(left_im, right_im, multi_scale_disp[0], multi_scale_disp_occ[0])
			tmp = tmp * self.disp_ssim_wt
			loss += tmp
			losses['disp_ssim'] = tmp.item()

		if len(losses.keys()) == 0:
			loss = torch.FloatTensor([0]).cuda()

		return loss, losses

def make_self_supervised_loss(
	args,
	disp_downscales=None, disp_pyramid_weights=None,
	flow_downscales=None, flow_pyramid_weights=None
	):
	criterion = SelfSupervisedLoss(
		disp_downscales, disp_pyramid_weights,
		flow_downscales, flow_pyramid_weights,
		disp_photo_consist_wt=args.disp_photo_consist_wt,
		disp_semantic_consist_wt=args.disp_semantic_consist_wt,
		flow_photo_consist_wt=args.flow_photo_consist_wt,
		flow_semantic_consist_wt=args.flow_semantic_consist_wt,
		disp_temporal_consist_wt=args.disp_temporal_consist_wt,
		flow_disp_consist_wt=args.flow_disp_consist_wt,
		flow_smoothness_wt=args.flow_smoothness_wt,
		disp_smoothness_wt=args.disp_smoothness_wt,
		flow_ssim_wt=args.flow_ssim_wt,
		disp_ssim_wt=args.disp_ssim_wt,
		size_average=args.per_pix_loss,
		flow_occ_penalty=args.flow_occ_penalty,
		disp_occ_penalty=args.disp_occ_penalty
		)
	return criterion

if __name__ == '__main__':
	from tools.arguments import parse_args

	parser = parse_args()
	args = parser.parse_args()
	print(args)

	crit = make_self_supervised_loss(args)
