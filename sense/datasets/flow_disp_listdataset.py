"""
Copyright (C) 2019 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""
import torch
import torch.utils.data as data
import os
import os.path
import cv2
import numpy as np
import pdb
import pickle

def load_seg_logits(logits_paths):
	assert len(logits_paths) == 2, 'Unexpected logits path length: {}'.format(len(logits_paths))
	seg_logits = []
	for p in logits_paths:
		if p is not None:
			with open(p, 'rb') as f:
				seg_logits.append(pickle.load(f).transpose(1,2,0))
		else:
			seg_logits.append(np.array([-1]))
	return seg_logits

class SemiSupervisedFlowDispListDataset(data.Dataset):
	def __init__(self, root, path_list, flow_loader, disp_loader, transform, 
			flow_target_transform=None, disp_target_transform=None,
			flow_co_transform=None, disp_co_transform=None,
			flow_co_transform_test=None, disp_co_transform_test=None,
			transform_additional=None,
			ss_data_transform=None
		):
		super(SemiSupervisedFlowDispListDataset, self).__init__()

		self.root = root
		self.path_list = path_list
		self.flow_loader = flow_loader
		self.disp_loader = disp_loader
		self.transform = transform
		self.flow_target_transform = flow_target_transform
		self.disp_target_transform = disp_target_transform
		self.flow_co_transform = flow_co_transform
		self.disp_co_transform = disp_co_transform
		self.flow_co_transform_test = flow_co_transform_test
		self.disp_co_transform_test = disp_co_transform_test
		
		self.transform_additional = transform_additional

		# self-spervision data
		self.ss_data_transform = ss_data_transform

	def __getitem__(self, index):
		inputs, targets = self.path_list[index]
		# load image
		# need to fix this elegantly!
		assert len(inputs) == 8, 'Incorrect input path length: {}'.format(len(inputs))
		flow_im_paths = inputs[:2]
		flow_seg_paths = inputs[2:4]
		disp_im_paths = inputs[4:6]
		disp_seg_paths = inputs[6:8]
		flow_ims, flow_gt = self.flow_loader(self.root, flow_im_paths, targets[:2])
		disp_ims, disp_gt = self.disp_loader(self.root, disp_im_paths, targets[2:])

		# hard-coded here, fix them later
		flow_segs = load_seg_logits(flow_seg_paths)
		disp_segs = load_seg_logits(disp_seg_paths)

		flow_ims_orig = [im.copy() for im in flow_ims]
		disp_ims_orig = [im.copy() for im in disp_ims]

		if self.ss_data_transform is not None:
			flow_ims_orig = self.ss_data_transform(flow_ims_orig)
			disp_ims_orig = self.ss_data_transform(disp_ims_orig)

		flow_ims = self.transform(flow_ims)
		disp_ims = self.transform(disp_ims)

		# append the original unjittered images
		flow_ims.extend(flow_ims_orig)
		disp_ims.extend(disp_ims_orig)

		# append segmentation logits
		flow_ims.extend(flow_segs)
		disp_ims.extend(disp_segs)

		if self.flow_co_transform is not None:
			flow_ims, flow_gt = self.flow_co_transform(flow_ims, flow_gt)
		if self.disp_co_transform is not None:
			disp_ims, disp_gt = self.disp_co_transform(disp_ims, disp_gt)

		if self.flow_co_transform_test is not None:
			flow_ims, flow_gt = self.flow_co_transform_test(flow_ims, flow_gt)
		if self.disp_co_transform_test is not None:
			disp_ims, disp_gt = self.disp_co_transform_test(disp_ims, disp_gt)

		# deal with dilation of annotation masks (1 means annotated)
		disp_true = disp_gt[0]
		flow_true = flow_gt[0]
		disp_mask = disp_true > 0
		flow_mask = flow_true[:, :, 0] == flow_true[:, :, 0]

		# dilation
		element = cv2.getStructuringElement(
			cv2.MORPH_RECT, 
			(9, 9), 
			(-1, -1)
		)
		flow_mask = cv2.dilate(flow_mask.astype(np.float32), element)
		disp_mask = cv2.dilate(disp_mask.astype(np.float32), element)

		if self.flow_target_transform is not None :
			flow_gt = self.flow_target_transform(flow_gt)
		if self.disp_target_transform is not None :
			disp_gt = self.disp_target_transform(disp_gt)

		flow_ims_new = []
		disp_ims_new = []
		if self.transform_additional is not None:
			for i in range(len(flow_ims)):
				# flow_ims[i] = self.transform_additional(flow_ims[i])  
				flow_ims_new.append(self.transform_additional(flow_ims[i]))
			for i in range(len(disp_ims)):
				# disp_ims[i] = self.transform_additional(disp_ims[i])  
				disp_ims_new.append(self.transform_additional(disp_ims[i]))

		flow_mask = torch.from_numpy(flow_mask).unsqueeze(0)
		disp_mask = torch.from_numpy(disp_mask).unsqueeze(0)

		flow_gt.append(flow_mask)
		disp_gt.append(disp_mask)

		return flow_ims_new, flow_gt, disp_ims_new, disp_gt

	def __len__(self):
		return len(self.path_list)

class FlowDispListDataset(data.Dataset):
    def __init__(self, root, path_list, flow_loader, disp_loader, transform, 
			flow_target_transform=None, disp_target_transform=None,
			flow_co_transform=None, disp_co_transform=None,
			flow_co_transform_test=None, disp_co_transform_test=None,
			transform_additional=None
		):
        super(FlowDispListDataset, self).__init__()

        self.root = root
        self.path_list = path_list
        self.flow_loader = flow_loader
        self.disp_loader = disp_loader
        self.transform = transform
        self.flow_target_transform = flow_target_transform
        self.disp_target_transform = disp_target_transform
        self.flow_co_transform = flow_co_transform
        self.disp_co_transform = disp_co_transform
        self.flow_co_transform_test = flow_co_transform_test
        self.disp_co_transform_test = disp_co_transform_test
        
        self.transform_additional = transform_additional

    def __getitem__(self, index):
        inputs, targets = self.path_list[index]
        # load image
        # ims = [cv2.imread(n).astype(np.float32) / 255.0 for n in inputs]
        flow_ims, flow_gt = self.flow_loader(self.root, inputs[:2], targets[:2])
        disp_ims, disp_gt = self.disp_loader(self.root, inputs[2:], targets[2:])

        flow_ims = self.transform(flow_ims)
        disp_ims = self.transform(disp_ims)

        if self.flow_co_transform is not None:
            flow_ims, flow_gt = self.flow_co_transform(flow_ims, flow_gt)
        if self.disp_co_transform is not None:
            disp_ims, disp_gt = self.disp_co_transform(disp_ims, disp_gt)

        if self.flow_co_transform_test is not None:
            flow_ims, flow_gt = self.flow_co_transform_test(flow_ims, flow_gt)
        if self.disp_co_transform_test is not None:
            disp_ims, disp_gt = self.disp_co_transform_test(disp_ims, disp_gt)

        if self.flow_target_transform is not None :
            flow_gt = self.flow_target_transform(flow_gt)
        if self.disp_target_transform is not None :
            disp_gt = self.disp_target_transform(disp_gt)

        flow_ims_new = []
        disp_ims_new = []
        if self.transform_additional is not None:
            for i in range(len(flow_ims)):
                # flow_ims[i] = self.transform_additional(flow_ims[i])  
                flow_ims_new.append(self.transform_additional(flow_ims[i]))
            for i in range(len(disp_ims)):
                # disp_ims[i] = self.transform_additional(disp_ims[i])  
                disp_ims_new.append(self.transform_additional(disp_ims[i]))

        return flow_ims_new, flow_gt, disp_ims_new, disp_gt

    def __len__(self):
        return len(self.path_list)
