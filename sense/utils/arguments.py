"""
Copyright (C) 2019 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""
import argparse

def parse_args():
	parser = argparse.ArgumentParser(description='Scene Flow PWC-Net')
	parser.add_argument('cmd', choices=['pre-train', 'finetune', 'eval', 'misc', 'demo'])
	parser.add_argument('--maxdisp', type=int ,default=192,
						help='maxium disparity')
	parser.add_argument('--dataset', default='sceneflow',
						help='datapath')
	parser.add_argument('--datapath', default='/media/jiaren/ImageNet/data_scene_flow_2015/training/',
						help='datapath')
	parser.add_argument('--batch-size', default=8, type=int)
	parser.add_argument('--epochs', type=int, default=300,
						help='number of epochs to train')
	parser.add_argument('--loadmodel', default=None,
						help='load model')
	parser.add_argument('--savemodel', default='./',
						help='save model')
	parser.add_argument('--seed', type=int, default=1, metavar='S',
						help='random seed (default: 1)')
	parser.add_argument('--workers', type=int, default=8,
						help='number of workers to load data')
	parser.add_argument('--enc-arch',
						help='architecture of the encoder')
	parser.add_argument('--dec-arch', 
						help='architecture of the decoder')
	parser.add_argument('--flow-dec-arch',
						help='architecture of the optical flow decoder')
	parser.add_argument('--do-class', action='store_true',
						help='wheter to do classification-based disparity estimation')
	parser.add_argument('--corr-radius', type=int, default=4,
						help='search radius of the correlation layer')
	parser.add_argument('--lr', default=0.0001, type=float,
						help='initial learning rate')
	parser.add_argument('--lr-steps', default=[100000000000], type=int, nargs='+',
						help='stepsize of changing the learning rate')
	parser.add_argument('--lr-gamma', default=0.1, type=float,
						help='learning rate will be multipled by this gamma')
	parser.add_argument('--print-freq', default=20, type=int,
						help='frequency to print information')
	parser.add_argument('--save-freq', default=1, type=int,
						help='frequency to save checkpoint')
	parser.add_argument('--per-pix-loss', action='store_true',
						help='whether to use per-pixel loss')
	parser.add_argument('--no-ppm', action='store_true',
						help='if to remove ppm (Pyramid Pooling Module) in the decoder')
	parser.add_argument('--flow-no-ppm', action='store_true',
						help='if to remove ppm in the flow decoder')
	parser.add_argument('--disp-refinement', choices=['none', 'lightweight', 'hourglass'],
						help='type of the disparity refiment module')
	parser.add_argument('--flow-refinement', choices=['none', 'lightweight', 'hourglass'],
						help='type of the optical flow refiment module')
	# parser.add_argument('--loss-weights', default=[4, 2, 1, 1, 1], type=float, nargs='+',
	# 					help='weights of multi-scale loss')
	parser.add_argument('--div-flow', default=20.0, type=float,
						help='value by which flow will be divided. Original value is 20 but 1 with batchNorm gives good results')
	parser.add_argument('--disp-loss-weight', type=float, default=0.75,
						help='weight to control the contribution of disparity loss (default: 2.0)'
						)
	parser.add_argument('--resume', default=None,
						help='saved checkpoint to resume training.')
	parser.add_argument('--eval-train', action='store_true',
						help='if to do evaluation on the training set')
	parser.add_argument('--save-dir', default=None, type=str,
						help='directory to save those visualization results')
	parser.add_argument('--bn-type', default='plain', choices=['plain', 'syncbn', 'encoding'],
						help='bantch normalization layer type.')
	parser.add_argument('--no-occ', action='store_true',
						help='no occlusion prediction.')
	parser.add_argument('--cat-occ', action='store_true',
						help='use occlusion predictions in the decoder hierarchy')
	parser.add_argument('--disp-crop-imh', type=int, default=256,
						help='height of cropped image')
	parser.add_argument('--disp-crop-imw', type=int, default=512,
						help='width of cropped image')
	parser.add_argument('--auto-resume', action='store_true',
						help='whether to resume training automatically from a latest checkpoint')
	parser.add_argument('--flow-dim-ratio', type=float, default=None,
						help='ratio to resize the optical flow data')
	parser.add_argument('--flow-crop-imh', type=int, default=256,
						help='height of cropped image')
	parser.add_argument('--flow-crop-imw', type=int, default=512,
						help='width of cropped image')
	parser.add_argument('--rot-loss-weight', type=float, default=1,
						help='weight of the rotation loss')
	parser.add_argument('--upsample-flow-output', action='store_true')
	parser.add_argument('--flow-loss-type', choices=['epe_loss', 'l1_robust'],
		default='l1_robust')

	# added by Huaizu Jiang, 06/22/2019, ICCV rebuttal
	parser.add_argument('--no-flow', action='store_true')
	parser.add_argument('--no-disp', action='store_true')

	parser.add_argument('--disp-occ-wts', type=float, nargs='+',
		default=[1.154, 7.481])
	parser.add_argument('--flow-occ-wts', type=float, nargs='+',
		default=[1.1726, 6.7944])
	parser.add_argument('--soft-occ-gt', action='store_true')
	parser.add_argument('--occ-loss-wt', type=float, default=1)

	parser.add_argument('--robust-loss-p', type=float, default=0.2)
	parser.add_argument('--pseudo-gt-dir', default=None)

	# semantic segmentation related
	parser.add_argument('--do-seg', action='store_true')
	parser.add_argument('--num-seg-class', default=19, type=int)
	parser.add_argument('--seg-root-dir', default=None,
		help='segmentation root data')
	parser.add_argument('--hard-seg-loss-weight', default=1, type=float,
		help='hard segmentation loss weight')
	parser.add_argument('--seg-teacher-encoder-weights', default=None)
	parser.add_argument('--seg-teacher-decoder-weights', default=None)
	parser.add_argument('--seg-distill-T', default=10.0, type=float,
		help='parameter for segmentation distillation loss')
	parser.add_argument('--saved-seg-res-dir', default=None,
		help='where the segmentation results (logits) are saved')
	parser.add_argument('--do-seg-distill', action='store_true')
	parser.add_argument('--soft-seg-loss-weight', default=1, type=float,
		help='soft segmentation loss weight')

	# added by Huaizu Jiang, 09/13/2018
	# self-supervised related
	parser.add_argument('--disp-photo-consist-wt', default=-1, type=float,
						help='weight of disparity photometric consistency (effective when >0)')
	parser.add_argument('--disp-semantic-consist-wt', default=-1, type=float,
						help='weight of disparity semantic consistency (effective when >0)')
	parser.add_argument('--flow-photo-consist-wt', default=-1, type=float,
						help='weight of optical flow photometric consistency (effective when >0)')
	parser.add_argument('--flow-semantic-consist-wt', default=-1, type=float,
						help='weight of optical flow semantic consistency (effective when >0)')
	parser.add_argument('--disp-temporal-consist-wt', default=-1, type=float,
						help='weight of disparity temporal consistency (effective when >0)')
	parser.add_argument('--flow-disp-consist-wt', default=-1, type=float,
						help='weight of optical flow and disparity consistency (effective when >0)')
	parser.add_argument('--flow-smoothness-wt', default=-1, type=float,
						help='flow smoothness term weight')
	parser.add_argument('--disp-smoothness-wt', default=-1, type=float,
						help='flow smoothness term weight')
	parser.add_argument('--flow-ssim-wt', default=-1, type=float,
						help='flow ssim weight')
	parser.add_argument('--disp-ssim-wt', default=-1, type=float,
						help='disp ssim weight')
	parser.add_argument('--flow-occ-penalty', default=-1, type=float,
						help='penalty for the loss value in the occlusion area.')
	parser.add_argument('--disp-occ-penalty', default=-1, type=float,
						help='penalty for the loss value in the occlusion area.')
	return parser