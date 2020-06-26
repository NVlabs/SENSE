"""
Copyright (C) 2019 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""
import os
import glob

def _make_optical_flow_helper(base_dir, pass_opt, split):
	seqs = glob.glob(os.path.join(base_dir, split, pass_opt, '*'))
	print('There are {} seqs in total.'.format(len(seqs)))

	paths = []

	for seq in seqs:
		im_paths = sorted(glob.glob(os.path.join(seq, '*.png')))
		for idx in range(len(im_paths) - 1):
			cur_im_path = im_paths[idx]
			nxt_im_path = im_paths[idx+1]
			flow_path = cur_im_path.replace(pass_opt, 'flow')
			flow_path = flow_path[:-4] + '.flo'
			assert os.path.exists(cur_im_path), cur_im_path
			assert os.path.exists(nxt_im_path), nxt_im_path
			flow_occ_path = cur_im_path.replace(pass_opt, 'occlusions')
			if split == 'training':
				assert os.path.exists(flow_path), flow_path
				assert os.path.exists(flow_occ_path), flow_occ_path
			if not os.path.exists(flow_occ_path):
				flow_occ_path = None
			paths.append([
				[cur_im_path, nxt_im_path], [flow_path, flow_occ_path]
			])
	return paths

def make_flow_data(base_dir, pass_opt='clean+final'):
	all_train_data = []
	all_test_data = []
	for p in pass_opt.split('+'):
		train_data = _make_optical_flow_helper(base_dir, p, 'training')
		all_train_data.extend(train_data)
		test_data = _make_optical_flow_helper(base_dir, pass_opt, 'test')
		all_test_data.extend(test_data)
	return all_train_data, all_test_data

def _make_custom_optical_flow_helper(base_dir, pass_opt, seq_names):
	print('There are {} seqs in total.'.format(len(seq_names)))

	paths = []

	for seq_name in seq_names:
		im_paths = sorted(glob.glob(os.path.join(base_dir, 'training', pass_opt, seq_name, '*.png')))
		for idx in range(len(im_paths) - 1):
			cur_im_path = im_paths[idx]
			nxt_im_path = im_paths[idx+1]
			flow_path = cur_im_path.replace(pass_opt, 'flow')
			flow_path = flow_path[:-4] + '.flo'
			assert os.path.exists(cur_im_path), cur_im_path
			assert os.path.exists(nxt_im_path), nxt_im_path
			flow_occ_path = cur_im_path.replace(pass_opt, 'occlusions')
			assert os.path.exists(flow_path), flow_path
			assert os.path.exists(flow_occ_path), flow_occ_path
			paths.append([
				[cur_im_path, nxt_im_path], [flow_path, flow_occ_path]
			])
	return paths

def make_custom_flow_data(base_dir, split_id=1, pass_opt='clean+final'):
	# load seq names
	with open(os.path.join(base_dir, 'split{}_train_seqs.txt'.format(split_id)), 'r') as f:
		train_seq_names = [ln.strip() for ln in f.readlines()]

	with open(os.path.join(base_dir, 'split{}_test_seqs.txt'.format(split_id)), 'r') as f:
		test_seq_names = [ln.strip() for ln in f.readlines()]

	all_train_data = []
	all_test_data = []
	for p in pass_opt.split('+'):
		train_data = _make_custom_optical_flow_helper(base_dir, p, train_seq_names)
		all_train_data.extend(train_data)
		test_data = _make_custom_optical_flow_helper(base_dir, p, test_seq_names)
		all_test_data.extend(test_data)
	return all_train_data, all_test_data

def _make_disp_helper(base_dir, pass_opt, split):
	seqs = glob.glob(os.path.join(base_dir, split, pass_opt, '*'))
	print('There are {} seqs in total.'.format(len(seqs)))

	stereo_base_dir = os.path.join(base_dir, 'stereo', split)

	paths = []

	for seq in seqs:
		im_paths = sorted(glob.glob(os.path.join(seq, '*.png')))
		for idx in range(len(im_paths) - 1):
			# disp data
			cur_im_path = im_paths[idx]
			_, im_name = os.path.split(cur_im_path)
			_, seq_name = os.path.split(seq)
			left_im_path = os.path.join(stereo_base_dir, pass_opt + '_left', seq_name, im_name)
			right_im_path = os.path.join(stereo_base_dir, pass_opt + '_right', seq_name, im_name)
			# assert left_im_path != cur_im_path, 'Something is seriously wrong.'
			assert os.path.exists(left_im_path), left_im_path
			assert os.path.exists(right_im_path), right_im_path
			disp_path = os.path.join(stereo_base_dir, 'disparities', seq_name, im_name)
			disp_occ_path = os.path.join(stereo_base_dir, 'occlusions', seq_name, im_name)
			if split == 'training':
				assert os.path.exists(disp_path), disp_path
				assert os.path.exists(disp_occ_path), disp_occ_path
			else:
				disp_occ_path = None
			paths.append([
				[left_im_path, right_im_path], 
				[disp_path, disp_occ_path]
			])
	return paths

def make_disp_data(base_dir, pass_opt='clean+pass'):
	all_train_data = []
	all_test_data = []
	for p in pass_opt.split('+'):
		train_data = _make_disp_helper(base_dir, p, 'training')
		all_train_data.extend(train_data)
		# test_data = _make_flow_disp_helper(base_dir, p, 'test')
		# all_test_data.append(test_data)
	return all_train_data, all_test_data

def _make_custom_disp_helper(base_dir, pass_opt, seq_names):
	print('There are {} seqs in total.'.format(len(seq_names)))

	stereo_base_dir = os.path.join(base_dir, 'stereo', 'training')

	paths = []

	for seq_name in seq_names:
		im_paths = sorted(glob.glob(os.path.join(base_dir, 'training', pass_opt, seq_name, '*.png')))
		for idx in range(len(im_paths) - 1):
			# disp data
			cur_im_path = im_paths[idx]
			_, im_name = os.path.split(cur_im_path)
			left_im_path = os.path.join(stereo_base_dir, pass_opt + '_left', seq_name, im_name)
			right_im_path = os.path.join(stereo_base_dir, pass_opt + '_right', seq_name, im_name)
			# assert left_im_path != cur_im_path, 'Something is seriously wrong.'
			assert os.path.exists(left_im_path), left_im_path
			assert os.path.exists(right_im_path), right_im_path
			disp_path = os.path.join(stereo_base_dir, 'disparities', seq_name, im_name)
			disp_occ_path = os.path.join(stereo_base_dir, 'occlusions', seq_name, im_name)
			assert os.path.exists(disp_path), disp_path
			assert os.path.exists(disp_occ_path), disp_occ_path
			paths.append([
				[left_im_path, right_im_path], 
				[disp_path, disp_occ_path]
			])
	return paths

def make_custom_disp_data(base_dir, split_id=1, pass_opt='clean+final'):
	# load seq names
	with open(os.path.join(base_dir, 'split{}_train_seqs.txt'.format(split_id)), 'r') as f:
		train_seq_names = [ln.strip() for ln in f.readlines()]

	with open(os.path.join(base_dir, 'split{}_test_seqs.txt'.format(split_id)), 'r') as f:
		test_seq_names = [ln.strip() for ln in f.readlines()]

	all_train_data = []
	all_test_data = []
	for p in pass_opt.split('+'):
		train_data = _make_custom_disp_helper(base_dir, p, train_seq_names)
		all_train_data.extend(train_data)
		test_data = _make_custom_disp_helper(base_dir, p, test_seq_names)
		all_test_data.extend(test_data)
	return all_train_data, all_test_data

def _make_flow_disp_helper(base_dir, pass_opt, split):
	seqs = glob.glob(os.path.join(base_dir, split, pass_opt, '*'))
	print('There are {} seqs in total.'.format(len(seqs)))

	stereo_base_dir = os.path.join(base_dir, 'stereo', split)
	print(stereo_base_dir)

	paths = []

	for seq in seqs:
		im_paths = sorted(glob.glob(os.path.join(seq, '*.png')))
		for idx in range(len(im_paths) - 1):
			# flow data
			cur_im_path = im_paths[idx]
			nxt_im_path = im_paths[idx+1]
			flow_path = cur_im_path.replace(pass_opt, 'flow')
			flow_path = flow_path[:-4] + '.flo'
			assert os.path.exists(cur_im_path), cur_im_path
			assert os.path.exists(nxt_im_path), nxt_im_path
			flow_occ_path = cur_im_path.replace(pass_opt, 'occlusions')
			if split == 'training':
				assert os.path.exists(flow_path), flow_path
				assert os.path.exists(flow_occ_path), flow_occ_path
			if not os.path.exists(flow_occ_path):
				flow_occ_path = None

			# disp data
			_, im_name = os.path.split(cur_im_path)
			_, seq_name = os.path.split(seq)
			left_im_path = os.path.join(stereo_base_dir, pass_opt + '_left', seq_name, im_name)
			right_im_path = os.path.join(stereo_base_dir, pass_opt + '_right', seq_name, im_name)
			# assert left_im_path != cur_im_path, 'Something is seriously wrong.'
			assert os.path.exists(left_im_path), left_im_path
			assert os.path.exists(right_im_path), right_im_path
			disp_path = os.path.join(stereo_base_dir, 'disparities', seq_name, im_name)
			disp_occ_path = os.path.join(stereo_base_dir, 'occlusions', seq_name, im_name)
			if split == 'training':
				assert os.path.exists(disp_path), disp_path
				assert os.path.exists(disp_occ_path), disp_occ_path
			else:
				disp_occ_path = None
			paths.append([
				[cur_im_path, nxt_im_path, left_im_path, right_im_path], 
				[flow_path, flow_occ_path, disp_path, disp_occ_path]
			])
	return paths

def make_flow_disp_data(base_dir, pass_opt='clean+pass'):
	all_train_data = []
	for p in pass_opt.split('+'):
		train_data = _make_flow_disp_helper(base_dir, p, 'training')
		all_train_data.extend(train_data)
		# test_data = _make_flow_disp_helper(base_dir, pass_opt, 'test')
	return all_train_data, []

def _make_custom_flow_disp_helper(base_dir, pass_opt, seq_names):
	print('There are {} seqs in total.'.format(len(seq_names)))

	stereo_base_dir = os.path.join(base_dir, 'stereo', 'training')

	paths = []

	for seq_name in seq_names:
		im_paths = sorted(glob.glob(os.path.join(base_dir, 'training', pass_opt, seq_name, '*.png')))
		for idx in range(len(im_paths) - 1):
			# flow data
			cur_im_path = im_paths[idx]
			nxt_im_path = im_paths[idx+1]
			flow_path = cur_im_path.replace(pass_opt, 'flow')
			flow_path = flow_path[:-4] + '.flo'
			assert os.path.exists(cur_im_path), cur_im_path
			assert os.path.exists(nxt_im_path), nxt_im_path
			flow_occ_path = cur_im_path.replace(pass_opt, 'occlusions')
			assert os.path.exists(flow_path), flow_path
			assert os.path.exists(flow_occ_path), flow_occ_path

			# disp data
			_, im_name = os.path.split(cur_im_path)
			left_im_path = os.path.join(stereo_base_dir, pass_opt + '_left', seq_name, im_name)
			right_im_path = os.path.join(stereo_base_dir, pass_opt + '_right', seq_name, im_name)
			# assert left_im_path != cur_im_path, 'Something is seriously wrong.'
			assert os.path.exists(left_im_path), left_im_path
			assert os.path.exists(right_im_path), right_im_path
			disp_path = os.path.join(stereo_base_dir, 'disparities', seq_name, im_name)
			disp_occ_path = os.path.join(stereo_base_dir, 'occlusions', seq_name, im_name)
			assert os.path.exists(disp_path), disp_path
			assert os.path.exists(disp_occ_path), disp_occ_path
			paths.append([
				[cur_im_path, nxt_im_path, left_im_path, right_im_path], 
				[flow_path, flow_occ_path, disp_path, disp_occ_path]
			])
	return paths

def make_custom_flow_disp_data(data_dir, split_id=1, pass_opt='clean+pass'):
	# load seq names
	with open(os.path.join(data_dir, 'split{}_train_seqs.txt'.format(split_id)), 'r') as f:
		train_seq_names = [ln.strip() for ln in f.readlines()]

	with open(os.path.join(data_dir, 'split{}_test_seqs.txt'.format(split_id)), 'r') as f:
		test_seq_names = [ln.strip() for ln in f.readlines()]

	all_train_data = []
	all_test_data = []
	for p in pass_opt.split('+'):
		train_data = _make_custom_flow_disp_helper(data_dir, p, train_seq_names)
		all_train_data.extend(train_data)
		test_data = _make_custom_flow_disp_helper(data_dir, p, test_seq_names)
		all_test_data.extend(test_data)
	return all_train_data, all_test_data

if __name__ == '__main__':
	train_list, test_list = make_custom_flow_data(
		'/home/hzjiang/Data/MPI_Sintel/',
		pass_opt='clean+final'
	)
	print(len(train_list))
	print(train_list[0])
	print(len(test_list))