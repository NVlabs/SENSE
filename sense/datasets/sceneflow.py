"""
Copyright (C) 2019 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""
import sense.datasets.flyingthings3d as fth3d
import sense.datasets.monkaa as mk
import sense.datasets.driving as drv

def make_flow_data(flyingthings3d_dir, monkaa_dir, driving_dir, flow_thresh):
	print('==> making flow data on FlyingThings3D.')
	train_data, test_data = fth3d.make_flow_data(flyingthings3d_dir, flow_thresh)

	print('==> making flow data on Monkaa.')
	tmp_train_data, tmp_test_data = mk.make_flow_data(monkaa_dir, flow_thresh)
	for i in range(2):
		train_data[i].extend(tmp_train_data[i])
		test_data[i].extend(tmp_test_data[i])

	print('==> making flow data on Driving.')
	tmp_train_data, tmp_test_data = drv.make_flow_data(driving_dir, flow_thresh)
	for i in range(2):
		train_data[i].extend(tmp_train_data[i])
		test_data[i].extend(tmp_test_data[i])
	return train_data, test_data

def make_disp_data(flyingthings3d_dir, monkaa_dir, driving_dir, disp_thresh):
	print('==> making disparity data on FlyingThings3D.')
	train_data, test_data = fth3d.make_disp_data(flyingthings3d_dir, disp_thresh)

	print('==> making disparity data on Monkaa.')
	tmp_train_data, tmp_test_data = mk.make_disp_data(monkaa_dir, disp_thresh)
	for i in range(2):
		train_data[i].extend(tmp_train_data[i])
		test_data[i].extend(tmp_test_data[i])

	print('==> making disparity data on Driving.')
	tmp_train_data, tmp_test_data = drv.make_disp_data(driving_dir, disp_thresh)
	for i in range(2):
		train_data[i].extend(tmp_train_data[i])
		test_data[i].extend(tmp_test_data[i])
	return train_data, test_data

def make_flow_disp_data(flyingthings3d_dir, monkaa_dir, driving_dir, 
						flow_thresh, disp_thresh, merge_crit='simple'):
	print('==> making flow and disparity data on FlyingThings3D. (This may take quite a while for the first time.)')
	train_data, test_data = fth3d.make_flow_disp_data(flyingthings3d_dir, flow_thresh)

	print('==> making flow and disparity data on Monkaa.')
	tmp_train_data, tmp_test_data = mk.make_flow_disp_data(monkaa_dir, flow_thresh)
	for i in range(2):
		train_data[i].extend(tmp_train_data[i])
		test_data[i].extend(tmp_test_data[i])

	print('==> making flow and disparity data on Driving.')
	tmp_train_data, tmp_test_data = drv.make_flow_disp_data(driving_dir, flow_thresh)
	for i in range(2):
		train_data[i].extend(tmp_train_data[i])
		test_data[i].extend(tmp_test_data[i])
	return train_data, test_data