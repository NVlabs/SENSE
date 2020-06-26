"""
Copyright (C) 2019 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""
import os
import glob
import numpy as np

def make_flow_dataset_single_split(kitti_dir, split):
	im_dir = os.path.join(kitti_dir, split, 'image_2')
	flow_dir = os.path.join(kitti_dir, split, 'flow_occ')

	paths = []
	im_paths = sorted(glob.glob(os.path.join(im_dir, '*.png')))
	num_ims = len(im_paths) // 2
	assert num_ims == 200, 'Something wrong with number of images: {}'.format(num_ims)
	for i in range(num_ims):
		cur_im_name = '{:06d}_10.png'.format(i)
		nxt_im_name = '{:06d}_11.png'.format(i)
		cur_im_path = os.path.join(im_dir, cur_im_name)
		nxt_im_path = os.path.join(im_dir, nxt_im_name)
		assert os.path.exists(cur_im_path), cur_im_path
		assert os.path.exists(nxt_im_path), nxt_im_path
		flow_path = os.path.join(flow_dir, cur_im_name)
		if split.startswith('train'):
			assert os.path.exists(flow_path), flow_path
		paths.append([
			[cur_im_path, nxt_im_path], [flow_path, None]
		])
	return paths

def make_flow_dataset(kitti_dir):
	train_data = make_flow_dataset_single_split(kitti_dir, 'training')
	test_data = make_flow_dataset_single_split(kitti_dir, 'testing')
	return train_data, test_data

def make_custom_flow_dataset(kitti_dir, split_id=1):
	val_idxes_file = os.path.join(kitti_dir, 'val_idxes_split{}.txt'.format(split_id))
	assert os.path.exists(val_idxes_file), 'Val indexes file not found {}'.format(val_idxes_file)
	val_idxes = np.loadtxt(val_idxes_file, delimiter=',').astype(int).tolist()
	trn_idxes = [idx for idx in range(200) if idx not in val_idxes]

	im_dir = os.path.join(kitti_dir, 'training', 'image_2')
	flow_dir = os.path.join(kitti_dir, 'training', 'flow_occ')

	def make_file_list(idxes):
		paths = []
		for idx in idxes:
			cur_im_name = '{:06d}_10.png'.format(idx)
			nxt_im_name = '{:06d}_11.png'.format(idx)
			cur_im_path = os.path.join(im_dir, cur_im_name)
			nxt_im_path = os.path.join(im_dir, nxt_im_name)
			assert os.path.exists(cur_im_path), cur_im_path
			assert os.path.exists(nxt_im_path), nxt_im_path
			flow_path = os.path.join(flow_dir, cur_im_name)
			assert os.path.exists(flow_path), flow_path
			paths.append([
				[cur_im_path, nxt_im_path], [flow_path, None]
			])
		return paths

	train_data = make_file_list(trn_idxes)
	test_data = make_file_list(val_idxes)

	return train_data, test_data

def make_custom_disparity_dataset(kitti_dir, split_id=-1):
	left_dir  = os.path.join(kitti_dir, 'training/image_2')
	right_dir = os.path.join(kitti_dir, 'training/image_3')
	disp_dir = os.path.join(kitti_dir, 'training/disp_occ_0')

	if split_id > 0:
		val_idxes_file = os.path.join(kitti_dir, 'val_idxes_split{}.txt'.format(split_id))
		assert os.path.exists(val_idxes_file), 'Val indexes file not found {}'.format(val_idxes_file)
		val_idxes = np.loadtxt(val_idxes_file, delimiter=',').astype(int).tolist()
		val = ['%06d_10.png' % idx for idx in val_idxes]
		train = ['%06d_10.png' % idx for idx in range(200) if idx not in val_idxes]
	else:
		val = []
		train = ['%06d_10.png' % idx for idx in range(200)]

	def make_file_list(im_names, allow_no_disp_gt=False):
		left_im_paths = [os.path.join(left_dir, n) for n in im_names]
		right_im_paths = [os.path.join(right_dir, n) for n in im_names]
		disp_paths = [os.path.join(disp_dir, n) for n in im_names]
		paths = []
		for i in range(len(left_im_paths)):
			assert os.path.exists(left_im_paths[i]), left_im_paths[i]
			assert os.path.exists(right_im_paths[i]), right_im_paths[i]
			if not allow_no_disp_gt:
				assert os.path.exists(disp_paths[i]), disp_paths[i]
			paths.append([
				[left_im_paths[i], right_im_paths[i]], [disp_paths[i], None]
			])
		return paths

	if split_id > 0:
		train_data = make_file_list(train)
		test_data = make_file_list(val)
	else:
		train = ['%06d_10.png' % idx for idx in range(200)]
		# train_data = make_file_list(train)
		tmp_train = ['%06d_11.png' % idx for idx in range(200)]
		train.extend(tmp_train)
		train_data = make_file_list(train, allow_no_disp_gt=True)

		# tmp_train = ['%06d_11.png' % idx for idx in range(200)]
		# tmp_train_data = make_file_list(tmp_train, allow_no_disp_gt=True)
		# train_data.extend(tmp_train_data)

		left_dir  = os.path.join(kitti_dir, 'testing/image_2')
		right_dir = os.path.join(kitti_dir, 'testing/image_3')
		disp_dir = os.path.join(kitti_dir, 'testing/disp_occ_0')

		val = ['%06d_10.png' % idx for idx in range(200)]
		tmp_val = ['%06d_11.png' % idx for idx in range(200)]
		val.extend(tmp_val)
		
		test_data = make_file_list(val, allow_no_disp_gt=True)

	return train_data, test_data

def make_custom_flow_disp_dataset(kitti_dir, split_id=-1, pseudo_gt_dir=None):
	left_dir  = os.path.join(kitti_dir, 'training/image_2')
	right_dir = os.path.join(kitti_dir, 'training/image_3')
	disp_dir = os.path.join(kitti_dir, 'training/disp_occ_0')
	flow_dir = os.path.join(kitti_dir, 'training/flow_occ')

	if split_id > 0:
		val_idxes_file = os.path.join(kitti_dir, 'val_idxes_split{}.txt'.format(split_id))
		assert os.path.exists(val_idxes_file), 'Val indexes file not found {}'.format(val_idxes_file)
		val_idxes = np.loadtxt(val_idxes_file, delimiter=',').astype(int).tolist()
		trn_idxes = [idx for idx in range(200) if idx not in val_idxes]
	else:
		val_idxes = []
		trn_idxes = [idx for idx in range(200)]

	def make_file_list(idxes):
		paths = []
		for idx in idxes:
			cur_im_name = '{:06d}_10.png'.format(idx)
			nxt_im_name = '{:06d}_11.png'.format(idx)
			cur_im_path = os.path.join(left_dir, cur_im_name)
			nxt_im_path = os.path.join(left_dir, nxt_im_name)
			right_im_path = os.path.join(right_dir, cur_im_name)
			assert os.path.exists(cur_im_path), cur_im_path
			assert os.path.exists(nxt_im_path), nxt_im_path
			assert os.path.exists(right_im_path), right_im_path
			flow_path = os.path.join(flow_dir, cur_im_name)
			assert os.path.exists(flow_path), flow_path
			disp_path = os.path.join(disp_dir, cur_im_name)
			assert os.path.exists(disp_path), disp_path

			# occlusion mask path generated by a pre-trained model
			if pseudo_gt_dir is not None:
				flow_occ_path = os.path.join(pseudo_gt_dir, 'flow_occ', 'training/image_2', cur_im_name)
				assert os.path.exists(flow_occ_path), flow_occ_path
				disp_occ_path = os.path.join(pseudo_gt_dir, 'disp_occ', 'training/image_2', cur_im_name)
				assert os.path.exists(disp_occ_path), disp_occ_path
			else:
				flow_occ_path = None
				disp_occ_path = None
			paths.append([
				[cur_im_path, nxt_im_path, cur_im_path, right_im_path], 
				[flow_path, flow_occ_path, disp_path, disp_occ_path]
			])
		return paths

	train_data = make_file_list(trn_idxes)
	test_data = make_file_list(val_idxes)

	return train_data, test_data

def make_custom_sceneflow_dataset(kitti_dir, split_id=-1):
	left_dir  = os.path.join(kitti_dir, 'training/image_2')
	right_dir = os.path.join(kitti_dir, 'training/image_3')

	def make_file_list(idxes):
		paths = []
		for idx in idxes:
			cur_im_name = '{:06d}_10.png'.format(idx)
			nxt_im_name = '{:06d}_11.png'.format(idx)
			cur_left_im_path = os.path.join(left_dir, cur_im_name)
			nxt_left_im_path = os.path.join(left_dir, nxt_im_name)
			cur_right_im_path = os.path.join(right_dir, cur_im_name)
			nxt_right_im_path = os.path.join(right_dir, nxt_im_name)
			paths.append(
				[cur_left_im_path, cur_right_im_path, nxt_left_im_path, nxt_right_im_path]
			)
		return paths

	if split_id > 0:
		val_idxes_file = os.path.join(kitti_dir, 'val_idxes_split{}.txt'.format(split_id))
		assert os.path.exists(val_idxes_file), 'Val indexes file not found {}'.format(val_idxes_file)
		val_idxes = np.loadtxt(val_idxes_file, delimiter=',').astype(int).tolist()
		trn_idxes = [idx for idx in range(200) if idx not in val_idxes]

		train_data = make_file_list(trn_idxes)
		test_data = make_file_list(val_idxes)
	else:
		trn_idxes = [idx for idx in range(200)]
		train_data = make_file_list(trn_idxes)

		left_dir  = os.path.join(kitti_dir, 'testing/image_2')
		right_dir = os.path.join(kitti_dir, 'testing/image_3')

		val_idxes = [idx for idx in range(200)]
		test_data = make_file_list(val_idxes)	

	return train_data, test_data

if __name__ == '__main__':
	kitti_dir = '/home/hzjiang/workspace/Data/KITTI_scene_flow'
	train_data, test_data = make_custom_disparity_dataset(kitti_dir, -1)
	print(train_data[-1])