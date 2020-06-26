"""
Copyright (C) 2019 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""
import os
import glob
import pickle

from .dataset_utils import remove_flow_outliers, remove_disp_outliers, remove_flow_disp_outliers
from .flyingthings3d import make_train_flow_disp_data

def make_flow_data(base_dir, flow_thresh):
	in_paths = []
	target_paths = []
	img_dir = os.path.join(base_dir, 'frames_cleanpass')
	flow_dir = os.path.join(base_dir, 'optical_flow')
	for fcl in os.listdir(img_dir): 
		seqs_all = os.listdir(os.path.join(img_dir, fcl))
		for i, cur_seq in enumerate(seqs_all):
			for fs in os.listdir(os.path.join(img_dir, fcl, cur_seq)):
				for fb in ['into_future', 'into_past']:            
					for lr in ['left', 'right']:
						im_paths = sorted(glob.glob(os.path.join(img_dir, fcl, cur_seq, fs, lr, '*.png')))
						fl_paths = sorted(glob.glob(os.path.join(flow_dir, fcl, cur_seq, fs, fb, lr, '*.pfm')))
						if fb == 'into_past':
							im_paths = im_paths[::-1]
							fl_paths = fl_paths[::-1]
						for idx in range(len(im_paths) - 1):
							flow_fn = fl_paths[idx]
							if os.path.isfile(flow_fn): # file exists
								# statinfo = os.stat(flow_fn)
								# if  statinfo.st_size > 10000: # not corrupted
								#     print('too large')
								#     continue        
								in_paths.append([im_paths[idx], im_paths[idx+1]])
								target_paths.append([fl_paths[idx], None])
	
	raw_data = (in_paths, target_paths)
	data = remove_flow_outliers(raw_data, flow_thresh)
	print('{} left for Driving.'.format(len(data[0])/len(raw_data[0])))

	return data, ([], [])

def make_disp_data(base_dir, disp_thresh):
	in_paths = []
	disp_paths = []
	img_dir = os.path.join(base_dir, 'frames_cleanpass')
	disp_dir = os.path.join(base_dir, 'disparity')
	for fcl in os.listdir(img_dir): 
		seqs_all = os.listdir(os.path.join(img_dir, fcl))
		for i, cur_seq in enumerate(seqs_all):
			for fs in os.listdir(os.path.join(img_dir, fcl, cur_seq)):
				im_paths = sorted(glob.glob(os.path.join(img_dir, fcl, cur_seq, fs, 'left/*.png')))
				dp_paths = sorted(glob.glob(os.path.join(disp_dir, fcl, cur_seq, fs, 'left/*.pfm')))
				for idx in range(len(im_paths)):
					left_fn = im_paths[idx]
					right_fn = left_fn.replace('left', 'right')
					disp_fn = dp_paths[idx]
					disp_occ_fn = left_fn.replace('frames_cleanpass', 'disparity_occlusions')
					if not os.path.exists(disp_occ_fn):
						disp_occ_fn = None
					if os.path.isfile(disp_fn): # file exists
						# statinfo = os.stat(disp_fn)
						# if  statinfo.st_size > 10000: # not corrupted
						#     print('too large')
						#     continue        
						in_paths.append([left_fn, right_fn])
						disp_paths.append([disp_fn, disp_occ_fn])
	
	raw_data = (in_paths, disp_paths)
	data = remove_disp_outliers(raw_data, disp_thresh)
	print('{} left for Driving.'.format(len(data[0])/len(raw_data[0])))

	return data, ([], [])

def make_flow_disp_data_simple_merge(base_dir):
	input_paths = []        # (cur, nxt, left, right)
	target_paths = []       # (flow, disp)
	img_dir = os.path.join(base_dir, 'frames_cleanpass')
	flow_dir = os.path.join(base_dir, 'optical_flow')
	disp_dir = os.path.join(base_dir, 'disparity')

	for fcl in os.listdir(img_dir): 
		seqs_all = os.listdir(os.path.join(img_dir, fcl))
		for i, cur_seq in enumerate(seqs_all):
			for fs in os.listdir(os.path.join(img_dir, fcl, cur_seq)):
				for fb in ['into_future', 'into_past']:            
					for lr in ['left', 'right']:
						im_paths = sorted(glob.glob(os.path.join(img_dir, fcl, cur_seq, fs, lr, '*.png')))
						dp_paths = sorted(glob.glob(os.path.join(disp_dir, fcl, cur_seq, fs, 'left', '*.pfm')))
						fl_paths = sorted(glob.glob(os.path.join(flow_dir, fcl, cur_seq, fs, fb, lr, '*.pfm')))
						if fb == 'into_past':
							im_paths = im_paths[::-1]
							dp_paths = dp_paths[::-1]
							fl_paths = fl_paths[::-1]

						tmp_input, tmp_target = make_train_flow_disp_data(
							im_paths, fl_paths, None, dp_paths, None
						)
						# print('*** tmp input_paths')
						# for p in tmp_input:
						#     print(p)
						# print('*** tmp target_paths')
						# for p in tmp_target:
						#     print(p)
						input_paths.extend(tmp_input)
						target_paths.extend(tmp_target)
						# print('*** input_paths')
						# for p in input_paths:
						#     print(p)
						# print('*** target_paths')
						# for p in target_paths:
						#     print(p)
						# import numpy as np
						# if np.random.random() > 0.5:
						#     raise Exception

	# print('### input_paths')
	# for p in input_paths[:10]:
	#     print(p)
	# print('### target_paths')
	# for p in target_paths[:10]:
	#     print(p)
	return input_paths, target_paths

def make_flow_disp_data(base_dir, flow_thresh=500, disp_thresh=500, merge_crit='simple'):
	if merge_crit == 'simple':
		helper_func = make_flow_disp_data_simple_merge
	elif merge_crit == 'exhaust':
		raise NotImplementedError('Exhaustive merging for flow and disp data not supported yet.')
	else:
		raise NotImplementedError('Not supported merge criterion {}'.format(merge_crit))

	train_data = helper_func(base_dir)
	train_data = remove_flow_disp_outliers(train_data, flow_thresh)
	return train_data, ([], [])

if __name__ == '__main__':
	base_dir = '/scratch/Data/Driving'

	print('scene flow')
	# joint_train_data, joint_test_data = make_flow_disp_data(base_dir)
	# with open('Driving_joint_train.pkl', 'wb') as f:
	# 	pickle.dump({'joint_train_data': joint_train_data,
	# 				 'joint_test_data': joint_test_data},
	# 				 f, pickle.HIGHEST_PROTOCOL
	# 				 )
	with open('Driving_joint_train.pkl', 'rb') as f:
		joint_data = pickle.load(f)
		joint_train_data = joint_data['joint_train_data']
		joint_test_data = joint_data['joint_test_data']
	print(len(joint_train_data[0]), len(joint_test_data[0]))

	import numpy as np
	import skimage
	from skimage.io import imread, imsave
	from datasets.dataset_utils import load_pfm
	from misc.viz_flow import viz_flow
	dst_dir = 'sanity_vis'
	if not os.path.exists(dst_dir):
		os.makedirs(dst_dir)
	for i in range(10):
		idx = np.random.randint(len(joint_train_data[0]))
		input_paths = joint_train_data[0][idx]
		target_paths = joint_train_data[1][idx]
		print(input_paths)
		im = imread(input_paths[0])
		imsave(os.path.join(dst_dir, '{0:3d}_acurrent.png'.format(i)), im)
		im = imread(input_paths[1])
		imsave(os.path.join(dst_dir, '{0:3d}_bnext.png'.format(i)), im)
		im, _ = load_pfm(target_paths[0])
		im = viz_flow(im[:, :, 0], im[:, :, 1])
		imsave(os.path.join(dst_dir, '{0:3d}_cflow.png'.format(i)), im)
		im = imread(input_paths[2])
		imsave(os.path.join(dst_dir, '{0:3d}_dleft.png'.format(i)), im)
		im = imread(input_paths[3])
		imsave(os.path.join(dst_dir, '{0:3d}_eright.png'.format(i)), im)
		im, _ = load_pfm(target_paths[1])
		im = im.astype(np.float32)
		im = (im - np.min(im)) / (np.max(im) - np.min(im) + 1e-30)
		im = (im * 255).astype(np.uint8)
		imsave(os.path.join(dst_dir, '{0:3d}_fdisp.png'.format(i)), im)
		print(i+1)
		# print('+++++++++++++++++++++++++++++++++++++++++++')
		# print(joint_train_data[0][idx])
		# print(joint_train_data[1][idx])
		# print('+++++++++++++++++++++++++++++++++++++++++++\n')

	print('optical flow')
	# flow_train_data, flow_test_data = make_flow_data(base_dir, 500)
	# with open('Driving_flow_train.pkl', 'wb') as f:
	# 	pickle.dump({'flow_train_data': flow_train_data,
	# 				 'flow_test_data': flow_test_data},
	# 				 f, pickle.HIGHEST_PROTOCOL
	# 				 )
	with open('Driving_flow_train.pkl', 'rb') as f:
		flow_data = pickle.load(f)
		flow_train_data = flow_data['flow_train_data']
		flow_test_data = flow_data['flow_test_data']
	print(len(flow_train_data[0]), len(flow_test_data[0]))

	print('disparity')
	# disp_train_data, disp_test_data = make_disp_data(base_dir, -1)
	# with open('Driving_disp_train.pkl', 'wb') as f:
	# 	pickle.dump({'disp_train_data': disp_train_data,
	# 				 'disp_test_data': disp_test_data},
	# 				 f, pickle.HIGHEST_PROTOCOL
	# 				 )
	with open('Driving_disp_train.pkl', 'rb') as f:
		disp_data = pickle.load(f)
		disp_train_data = disp_data['disp_train_data']
		disp_test_data = disp_data['disp_test_data']
	print(len(disp_train_data[0]), len(disp_test_data[0]))

	def check_existence(data1, data2, idxes):
		for i, dt in enumerate(data1):
			tmp = dt[idxes]
			if len(tmp) == 1:
				tmp = tmp[0]
			if tmp not in data2:
				print(dt)
				print(tmp, idxes, i)
				if not isinstance(tmp, list):
					from datasets.dataset_utils import load_pfm
					import numpy as np
					flow, _ = load_pfm(tmp)
					flow = flow[:, :, :2]
					print(np.max(np.abs(flow)))
				raise Exception('An outlier found.')

	def check_exists_2(data1, data2, idxes):
		tmp_data = []
		for dt in data2:
			tmp = dt[idxes]
			if len(tmp) == 1:
				tmp_data.append(tmp[0])
			else:
				tmp_data.append(tmp)
		for i, dt in enumerate(data1):
			if dt not in tmp_data:
				print(dt)
				print(tmp_data[:3])
				raise Exception('An outlier found.')

	def sanity_check_flow_disp_data(joint_data, flow_data, disp_data):
		# check flow data first
		tmp_in_paths, tmp_tgt_paths = joint_data
		in_paths, tgt_paths = flow_data
		print('==> checking flow input')
		check_existence(tmp_in_paths, in_paths, slice(2))
		print('==> checking flow GT')
		for p in tgt_paths:
			if p[0].find('35mm_focallength/scene_backwards/fast/into_future') > 0:
				print(p[0])
		check_existence(tmp_tgt_paths, tgt_paths, slice(1))
		
		# in_paths, tgt_paths = disp_data
		# print('==> checking disp input')
		# check_existence(tmp_in_paths, in_paths, slice(2, 4))
		# print('==> checking disp GT')
		# check_existence(tmp_tgt_paths, tgt_paths, slice(1, 2))

		# print('checking in the opposite direction');
		# in_paths, tgt_paths = flow_data
		# print('==> checking flow input')
		# check_exists_2(in_paths, tmp_in_paths, slice(2))
		# print('==> checking flow GT')
		# check_exists_2(tgt_paths, tmp_tgt_paths, slice(1))
		
		# in_paths, tgt_paths = disp_data
		# print('==> checking disp input')
		# check_exists_2(in_paths, tmp_in_paths, slice(2, 4))
		# print('==> checking disp GT')
		# check_exists_2(tgt_paths, tmp_tgt_paths, slice(1, 2))

	# sanity_check_flow_disp_data(joint_train_data, flow_train_data, disp_train_data)

	# check
	def check_input_target_existence(joint_data, single_data, is_flow):
		input_paths, target_paths = joint_data
		joint_tmp_data = []
		for i in range(len(input_paths)):
			if is_flow:
				tmp = input_paths[i][:2]
				tmp.append(target_paths[i][0])
			else:
				tmp = input_paths[i][2:]
				tmp.append(target_paths[i][1])
			joint_tmp_data.append(tmp)

		input_paths, target_paths = single_data
		single_tmp_data = []
		for i in range(len(input_paths)):
			tmp = input_paths[i]
			tmp.append(target_paths[i])
			single_tmp_data.append(tmp)

		for td in joint_tmp_data:
			if td not in single_tmp_data:
				print(td)
				raise Exception('An outlier found.')

	print('==> checking flow')
	check_input_target_existence(joint_train_data, flow_train_data, True)
	print('==> checking disp')
	check_input_target_existence(joint_train_data, disp_train_data, False)

	# check number of unique disparity pairs
	input_dict = {}
	target_dict = {}
	input_paths, target_paths = joint_train_data
	for i in range(len(input_paths)):
		tmp_in = input_paths[i][2:4]
		tmp_in = tmp_in[0] + '@' + tmp_in[1]
		tmp_tgt = target_paths[i][1]
		if tmp_in not in input_dict:
			input_dict[tmp_in] = 1
		else:
			input_dict[tmp_in] += 1

		if tmp_tgt not in target_dict:
			target_dict[tmp_tgt] = 1
		else:
			target_dict[tmp_tgt] += 1

	print(len(input_dict.values()), len(target_dict.values()))