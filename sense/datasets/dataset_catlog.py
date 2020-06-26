"""
Copyright (C) 2019 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""
import os, sys
import pickle
import numpy as np

from joblib import Parallel, delayed

import sense.datasets.sceneflow as sf
import sense.datasets.flyingthings3d as fth3d
import sense.datasets.kitti2015 as kitti2015
import sense.datasets.kitti2012 as kitti2012
import sense.datasets.sintel as sintel

FLYINGTHINGS3D_DIR='/home/hzjiang/workspace/Data/SceneFlow/FlyingThings3D_subset'
MONKAA_DIR='/home/hzjiang/workspace/Data/SceneFlow/Monkaa'
DRIVING_DIR='/home/hzjiang/workspace/Data/SceneFlow/Driving'
SINTEL_DIR='/home/hzjiang/workspace/Data/MPI_Sintel/'
KITTI2012_DIR='/home/hzjiang/workspace/Data/KITTI_Stereo_2012'
KITTI2015_DIR='/home/hzjiang/workspace/Data/KITTI_scene_flow'

def make_warp_disp_refine_data(dataset_name):
	cache_file_path = 'cache/wdisp_refine_{}.pkl'.format(dataset_name)
	if dataset_name == 'flyingthings3d':
		base_dir = '/home/hzjiang/workspace/Data/SceneFlow/FlyingThings3D_subset'
		if os.path.exists(cache_file_path):
			with open(cache_file_path, 'rb') as f:
				cached_data = pickle.load(f)
				train_data = cached_data['train_data']
				test_data = cached_data['test_data']
		else:
			train_data, test_data = fth3d.make_warp_disp_refine_data(base_dir, filter_outlier=False)
			with open(cache_file_path, 'wb') as f:
				pickle.dump({'train_data': train_data,
							'test_data': test_data},
							f, pickle.HIGHEST_PROTOCOL
							)
		print('Warp disparity refine: there are {} and {} for the training and testing set of FlyingThings3D'\
			  .format(len(train_data), len(test_data))
			  )
		sys.stdout.flush()
	elif dataset_name == 'flyingthings3d_filtered':
		base_dir = '/home/hzjiang/workspace/Data/SceneFlow/FlyingThings3D_subset'
		if os.path.exists(cache_file_path):
			with open(cache_file_path, 'rb') as f:
				cached_data = pickle.load(f)
				train_data = cached_data['train_data']
				test_data = cached_data['test_data']
		else:
			train_data, test_data = fth3d.make_warp_disp_refine_data(base_dir, filter_outlier=True, flow_thresh=500)
			with open(cache_file_path, 'wb') as f:
				pickle.dump({'train_data': train_data,
							'test_data': test_data},
							f, pickle.HIGHEST_PROTOCOL
							)
		print('Warp disparity refine: there are {} and {} for the training and testing set of FlyingThings3D'\
			  .format(len(train_data), len(test_data))
			  )
		sys.stdout.flush()
	elif dataset_name == 'kitti2015':
		kitti_dir = '/home/hzjiang/workspace/Data/KITTI_scene_flow'
		raise NotImplementedError
	elif dataset_name == 'kitti2015_split1':
		kitti_dir = '/home/hzjiang/workspace/Data/KITTI_scene_flow'
		raise NotImplementedError
	else:
		raise Exception('Not supported dataset: {}'.format(dataset_name))
	return train_data, test_data

def make_flow_data(dataset_name, flow_thresh=500, pass_opt='clean'):
	cache_file_path = 'cache/optical_flow_{}.pkl'.format(dataset_name)
	if dataset_name == 'flyingthings3d':
		base_dir = FLYINGTHINGS3D_DIR
		if os.path.exists(cache_file_path):
			with open(cache_file_path, 'rb') as f:
				cached_data = pickle.load(f)
				train_data = cached_data['train_data']
				test_data = cached_data['test_data']
		else:
			train_data, test_data = fth3d.make_flow_data(base_dir, flow_thresh)
			tmp_train_data = []
			for i in range(len(train_data[0])):
				tmp_train_data.append([train_data[0][i], train_data[1][i]])
			train_data = tmp_train_data
			tmp_test_data = []
			for i in range(len(test_data[0])):
				tmp_test_data.append([test_data[0][i], test_data[1][i]])
			test_data = tmp_test_data
			with open(cache_file_path, 'wb') as f:
				pickle.dump(
					{
						'train_data': train_data,
						'test_data': test_data
					},
					f, pickle.HIGHEST_PROTOCOL
				)
		print('Flow: there are {} and {} for the training and testing set of FlyingThings3D'\
			  .format(len(train_data), len(test_data))
			  )
		sys.stdout.flush()
	elif dataset_name == 'sceneflow':
		flythings3d_base_dir = FLYINGTHINGS3D_DIR
		monkaa_base_dir = MONKAA_DIR
		driving_base_dir = DRIVING_DIR
		if os.path.exists(cache_file_path):
			with open(cache_file_path, 'rb') as f:
				cached_data = pickle.load(f)
				train_data = cached_data['train_data']
				test_data = cached_data['test_data']
		else:
			train_data, test_data = sf.make_flow_data(
				flythings3d_base_dir,
				monkaa_base_dir,
				driving_base_dir,
				flow_thresh
			)
			tmp_train_data = []
			for i in range(len(train_data[0])):
				tmp_train_data.append([train_data[0][i], train_data[1][i]])
			train_data = tmp_train_data
			tmp_test_data = []
			for i in range(len(test_data[0])):
				tmp_test_data.append([test_data[0][i], test_data[1][i]])
			test_data = tmp_test_data
			with open(cache_file_path, 'wb') as f:
				pickle.dump(
					{
						'train_data': train_data,
						'test_data': test_data
					},
					f, pickle.HIGHEST_PROTOCOL
				)
		print('Flow: there are {} and {} for the training and testing set of SceneFlow'\
			  .format(len(train_data), len(test_data))
			  )
		sys.stdout.flush()
	elif dataset_name == 'sintel':
		base_dir = SINTEL_DIR
		train_data, test_data = sintel.make_flow_data(base_dir, pass_opt=pass_opt)
		print('Flow: there are {} and {} for the training and testing set of Sintel'\
			.format(len(train_data), len(test_data))
			)
		# if os.path.exists(cache_file_path):
		# 	with open(cache_file_path, 'rb') as f:
		# 		cached_data = pickle.load(f)
		# 		train_data = cached_data['train_data']
		# 		test_data = cached_data['test_data']
		# else:
		# 	train_data, test_data = sintel.make_flow_data(base_dir, pass_opt=pass_opt)
		# 	with open(cache_file_path, 'wb') as f:
		# 		pickle.dump({'train_data': train_data,
		# 					'test_data': test_data},
		# 					f, pickle.HIGHEST_PROTOCOL
		# 					)
		# 	print('Flow: there are {} and {} for the training and testing set of Sintel'\
		# 	  .format(len(train_data), len(test_data))
		# 	  )
		# 	sys.stdout.flush()
	elif dataset_name == 'sintel_split1':
		base_dir = SINTEL_DIR
		train_data, test_data = sintel.make_custom_flow_data(base_dir, pass_opt=pass_opt)
		print('Flow: there are {} and {} for the training and testing set of Sintel split1'\
			.format(len(train_data), len(test_data))
			)
		sys.stdout.flush()
	elif dataset_name == 'kitti2015':
		kitti_dir = KITTI2015_DIR
		train_data, test_data = kitti2015.make_flow_dataset(kitti_dir)
	elif dataset_name == 'kitti2015_split1':
		kitti_dir = KITTI2015_DIR
		train_data, test_data = kitti2015.make_custom_flow_dataset(kitti_dir, split_id=1)
	elif dataset_name == 'kitti2015_split2':
		kitti_dir = KITTI2015_DIR
		train_data, test_data = kitti2015.make_custom_flow_dataset(kitti_dir, split_id=2)
	elif dataset_name == 'kitti2012':
		kitti2012_dir = KITTI2012_DIR
		train_data, test_data = kitti2012.make_flow_dataset(kitti2012_dir)
	elif dataset_name == 'vkitti':
		vkitti_dir = '/home/hzjiang/workspace/Data/VirtualKITTI'
		train_data, test_data = vkitti.make_flow_dataset(vkitti_dir)
	else:
		raise Exception('Not supported dataset: {}'.format(dataset_name))
	return train_data, test_data

def make_disp_data(dataset_name, disp_thresh=500, pass_opt='clean'):
	cache_file_path = 'cache/disparity_{}.pkl'.format(dataset_name)
	if dataset_name == 'flyingthings3d':
		base_dir = '/home/hzjiang/workspace/Data/SceneFlow/FlyingThings3D_subset'
		if os.path.exists(cache_file_path):
			with open(cache_file_path, 'rb') as f:
				cached_data = pickle.load(f)
				train_data = cached_data['train_data']
				test_data = cached_data['test_data']
		else:
			train_data, test_data = fth3d.make_disp_data(base_dir, disp_thresh)
			tmp_train_data = []
			for i in range(len(train_data[0])):
				tmp_train_data.append([train_data[0][i], train_data[1][i]])
			train_data = tmp_train_data
			tmp_test_data = []
			for i in range(len(test_data[0])):
				tmp_test_data.append([test_data[0][i], test_data[1][i]])
			test_data = tmp_test_data
			with open(cache_file_path, 'wb') as f:
				pickle.dump(
					{
						'train_data': train_data,
						'test_data': test_data
					},
					f, pickle.HIGHEST_PROTOCOL
				)
		print('Disp: there are {} and {} for the training and testing set of FlyingThings3D'\
			  .format(len(train_data), len(test_data))
			  )
		sys.stdout.flush()
	elif dataset_name == 'sceneflow':
		flythings3d_base_dir = '/home/hzjiang/workspace/Data/SceneFlow/FlyingThings3D_subset'
		monkaa_base_dir = '/home/hzjiang/workspace/Data/SceneFlow/Monkaa'
		driving_base_dir = '/home/hzjiang/workspace/Data/SceneFlow/Driving'
		if os.path.exists(cache_file_path):
			with open(cache_file_path, 'rb') as f:
				cached_data = pickle.load(f)
				train_data = cached_data['train_data']
				test_data = cached_data['test_data']
		else:
			train_data, test_data = sf.make_disp_data(
				flythings3d_base_dir,
				monkaa_base_dir,
				driving_base_dir,
				disp_thresh
			)
			tmp_train_data = []
			for i in range(len(train_data[0])):
				tmp_train_data.append([train_data[0][i], train_data[1][i]])
			train_data = tmp_train_data
			tmp_test_data = []
			for i in range(len(test_data[0])):
				tmp_test_data.append([test_data[0][i], test_data[1][i]])
			test_data = tmp_test_data
			with open(cache_file_path, 'wb') as f:
				pickle.dump(
					{
						'train_data': train_data,
						'test_data': test_data
					},
					f, pickle.HIGHEST_PROTOCOL
				)
		print('Disp: there are {} and {} for the training and testing set of SceneFlow'\
			.format(len(train_data), len(test_data))
		)
		sys.stdout.flush()
	elif dataset_name == 'kitti2015_split1':
		"""
		Randomly sample 160 images for training and 40 for testing (validation)
		"""
		kitti_dir = KITTI2015_DIR
		train_data, test_data = kitti2015.make_custom_disparity_dataset(kitti_dir, split_id=1)
	elif dataset_name == 'kitti2015_split2':
		"""
		Randomly sample 160 images for training and 40 for testing (validation)
		"""
		kitti_dir = KITTI2015_DIR
		train_data, test_data = kitti2015.make_custom_disparity_dataset(kitti_dir, split_id=2)
	elif dataset_name == 'kitti2015':
		kitti_dir = KITTI2015_DIR
		train_data, test_data = kitti2015.make_custom_disparity_dataset(kitti_dir, split_id=-1)
	elif dataset_name == 'kitti2012':
		kitti2012_dir = KITTI2012_DIR
		train_data, test_data = kitti2012.make_disparity_dataset(kitti2012_dir)
	elif dataset_name == 'sintel':
		sintel_dir = '/home/hzjiang/workspace/Data/MPI_Sintel'
		train_data, test_data = sintel.make_disp_data(sintel_dir, pass_opt)
	elif dataset_name == 'sintel_split1':
		sintel_dir = '/home/hzjiang/workspace/Data/MPI_Sintel'
		train_data, test_data = sintel.make_custom_disp_data(sintel_dir, pass_opt=pass_opt)
	else:
		raise Exception('Not supported dataset: {}'.format(dataset_name))
	return train_data, test_data

def make_flow_disp_data(dataset_name, flow_thresh=500, disp_thresh=500, 
	merge_crit='simple', pass_opt='clean+final', pseudo_gt_dir=None):
	os.makedirs('cache', exist_ok=True)
	cache_file_path = 'cache/flow_disp_{}.pkl'.format(dataset_name)
	if dataset_name == 'flyingthings3d':
		base_dir = FLYINGTHINGS3D_DIR
		if os.path.exists(cache_file_path):
			with open(cache_file_path, 'rb') as f:
				cached_data = pickle.load(f)
				train_data = cached_data['train_data']
				test_data = cached_data['test_data']
		else:
			train_data, test_data = fth3d.make_flow_disp_data(
				base_dir, 
				flow_thresh, 
				disp_thresh,
				merge_crit
			)
			tmp_train_data = []
			for i in range(len(train_data[0])):
				tmp_train_data.append([train_data[0][i], train_data[1][i]])
			train_data = tmp_train_data
			tmp_test_data = []
			for i in range(len(test_data[0])):
				tmp_test_data.append([test_data[0][i], test_data[1][i]])
			test_data = tmp_test_data
			with open(cache_file_path, 'wb') as f:
				pickle.dump(
					{
						'train_data': train_data,
						'test_data': test_data
					},
					f, pickle.HIGHEST_PROTOCOL
				)
		print('Flow+Disp: there are {} and {} for the training and testing set of FlyingThings3D'\
			.format(len(train_data), len(test_data))
		)
		sys.stdout.flush()
	elif dataset_name == 'sceneflow':
		flythings3d_base_dir = FLYINGTHINGS3D_DIR
		monkaa_base_dir = MONKAA_DIR
		driving_base_dir = DRIVING_DIR
		if os.path.exists(cache_file_path):
			with open(cache_file_path, 'rb') as f:
				cached_data = pickle.load(f)
				train_data = cached_data['train_data']
				test_data = cached_data['test_data']
		else:
			train_data, test_data = sf.make_flow_disp_data(
				flythings3d_base_dir,
				monkaa_base_dir,
				driving_base_dir,
				flow_thresh,
				disp_thresh,
				merge_crit
			)
			tmp_train_data = []
			for i in range(len(train_data[0])):
				tmp_train_data.append([train_data[0][i], train_data[1][i]])
			train_data = tmp_train_data
			tmp_test_data = []
			for i in range(len(test_data[0])):
				tmp_test_data.append([test_data[0][i], test_data[1][i]])
			test_data = tmp_test_data
			with open(cache_file_path, 'wb') as f:
				pickle.dump(
					{
						'train_data': train_data,
						'test_data': test_data
					},
					f, pickle.HIGHEST_PROTOCOL
				)
		print('Flow+Disp: there are {} and {} for the training and testing set of SceneFlow'\
			.format(len(train_data), len(test_data))
		)
		sys.stdout.flush()
	elif dataset_name == 'kitti2015':
		kitti_dir = KITTI2015_DIR
		train_data, test_data = kitti2015.make_custom_flow_disp_dataset(kitti_dir, split_id=-1, pseudo_gt_dir=pseudo_gt_dir)
	elif dataset_name == 'kitti2015_split1':
		kitti_dir = KITTI2015_DIR
		train_data, test_data = kitti2015.make_custom_flow_disp_dataset(kitti_dir, split_id=1, pseudo_gt_dir=pseudo_gt_dir)
	elif dataset_name == 'kitti2015_split2':
		kitti_dir = KITTI2015_DIR
		train_data, test_data = kitti2015.make_custom_flow_disp_dataset(kitti_dir, split_id=2, pseudo_gt_dir=pseudo_gt_dir)
	elif dataset_name == 'kitti2012':
		kitti2012_dir = KITTI2012_DIR
		train_data, test_data = kitti2012.make_flow_disp_dataset(kitti2012_dir, pseudo_gt_dir=pseudo_gt_dir)
	elif dataset_name == 'kitti2012+kitti2015_split1':
		kitti2012_dir = KITTI2012_DIR
		train_data, test_data = kitti2012.make_flow_disp_dataset(kitti2012_dir, pseudo_gt_dir=pseudo_gt_dir)
		kitti2015_dir = KITTI2015_DIR
		tmp_train_data, tmp_test_data = kitti2015.make_custom_flow_disp_dataset(kitti2015_dir, split_id=1, pseudo_gt_dir=pseudo_gt_dir)
		train_data.extend(tmp_train_data)
		test_data.extend(tmp_test_data)
	elif dataset_name == 'kitti2012+kitti2015':
		kitti2012_dir = KITTI2012_DIR
		train_data, test_data = kitti2012.make_flow_disp_dataset(kitti2012_dir, pseudo_gt_dir=pseudo_gt_dir)
		kitti2015_dir = KITTI2015_DIR
		tmp_train_data, tmp_test_data = kitti2015.make_custom_flow_disp_dataset(kitti2015_dir, pseudo_gt_dir=pseudo_gt_dir)
		train_data.extend(tmp_train_data)
		test_data.extend(tmp_test_data)
	elif dataset_name == 'sintel':
		sintel_dir = '/home/hzjiang/workspace/Data/MPI_Sintel'
		train_data, test_data = sintel.make_flow_disp_data(sintel_dir, pass_opt)
	elif dataset_name == 'sintel_split1':
		sintel_dir = '/home/hzjiang/workspace/Data/MPI_Sintel'
		train_data, test_data = sintel.make_custom_flow_disp_data(sintel_dir, pass_opt=pass_opt)
	elif dataset_name == 'cityscapes':
		if os.path.exists(cache_file_path):
			with open(cache_file_path, 'rb') as f:
				cached_data = pickle.load(f)
				train_data = cached_data['train_data']
				test_data = cached_data['test_data']
		else:
			cityscapes_dir = '/home/hzjiang/workspace/Data/CityScapes'
			# pseudo_gt_dir = '/home/hzjiang/Code/semi-scene-flow/pseudo-gt/CityScapes'
			train_data, test_data = cityscapes.make_flow_disp_data(cityscapes_dir, pseudo_gt_dir)
			with open(cache_file_path, 'wb') as f:
				pickle.dump(
					{
						'train_data': train_data,
						'test_data': test_data
					},
					f, pickle.HIGHEST_PROTOCOL
				)
	else:
		raise Exception('Not supported dataset: {}'.format(dataset_name))
	return train_data, test_data

def make_sceneflow_data():
	kitti_dir = KITTI2015_DIR
	train_data, test_data = kitti2015.make_custom_sceneflow_dataset(kitti_dir)
	return train_data, test_data

def read_seg_gt_data(seg_root_dir, dataset_name, split):
	seg_im_names = []
	with open(os.path.join(seg_root_dir, '{}_{}_images.txt'.format(dataset_name, split))) as f:
		for ln in f.readlines():
			seg_dir, im_name = os.path.split(ln.strip())
			_, seg_dir = os.path.split(seg_dir)
			seg_im_names.append(os.path.join(seg_dir, im_name))
	seg_gt_paths = []
	with open(os.path.join(seg_root_dir, '{}_{}_labels.txt'.format(dataset_name, split))) as f:
		for ln in f.readlines():
			seg_gt_paths.append(ln.strip())
	assert len(seg_im_names) == len(seg_gt_paths), [len(seg_im_names), len(seg_gt_paths)]

	seg_gt_data_dict = {}
	for i in range(len(seg_im_names)):
		im_name = seg_im_names[i]
		assert im_name not in seg_gt_data_dict, 'Something seriously wrong with segmentation data.'
		seg_gt_data_dict[im_name] = seg_gt_paths[i] 
	return seg_gt_data_dict

def patch_with_seg_gt(data, seg_root_dir, dataset_name, split):
	seg_gt_data_dict = read_seg_gt_data(seg_root_dir, dataset_name, split)

	for i in range(len(data)):
		im_path = data[i][0][2]
		im_dir, im_name = os.path.split(im_path)
		_, im_dir = os.path.split(im_dir)
		seg_path = seg_gt_data_dict[os.path.join(im_dir, im_name)]
		data[i][1].append(seg_path)
	return data

def get_saved_seg_logits_path(seg_res_dir, im_path):
	im_dir, fn = os.path.split(im_path)
	_, seq_name = os.path.split(im_dir)
	fn = os.path.join(seg_res_dir, '_logits', seq_name, fn[:-4] + '.pkl')
	assert os.path.exists(fn), fn
	return fn

def patched_with_saved_seg_logits(data, seg_res_dir, dataset_name, split):
	for i in range(len(data)):
		cur_im_path, nxt_im_path, left_im_path, right_im_path = data[i][0]
		if seg_res_dir is None:
			cur_seg_path = None
			nxt_seg_path = None
			left_seg_path = None
			right_seg_path = None
		else:
			cur_seg_path = get_saved_seg_logits_path(seg_res_dir, cur_im_path)
			nxt_seg_path = get_saved_seg_logits_path(seg_res_dir, nxt_im_path)
			left_seg_path = get_saved_seg_logits_path(seg_res_dir, left_im_path)
			right_seg_path = get_saved_seg_logits_path(seg_res_dir, right_im_path)
		data[i][0] = [cur_im_path, nxt_im_path, cur_seg_path, nxt_seg_path,
			left_im_path, right_im_path, left_seg_path, right_seg_path]
	return data

if __name__ == '__main__':
	# make_disp_data('sceneflow')
	make_flow_data('sceneflow')
	make_flow_disp_data('sceneflow')

	# train_data, _ = make_flow_data('sintel')
	# print(len(train_data))
