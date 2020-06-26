"""
Copyright (C) 2019 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""
import os
import sys
import glob
import pickle
import numpy as np

from .dataset_utils import remove_flow_outliers, remove_disp_outliers, remove_flow_disp_outliers

"""
Optical flow only
"""
def make_flow_data_helper(base_dir, split):
    input_paths = []
    target_paths = []
    img_dir = os.path.join(base_dir, split, 'image_clean')
    flow_dir = os.path.join(base_dir, split, 'flow')
    flow_occ_dir = os.path.join(base_dir, split, 'flow_occlusions')
    # train/flow/left/into_future
    for lr in ['left', 'right']:
        for fb in ['into_future', 'into_past']:
            im_paths = sorted(glob.glob(os.path.join(img_dir, lr, '*.png')))
            # fl_paths = sorted(glob.glob(os.path.join(flow_dir, lr, fb, '*.flo')))
            # fl_occ_paths = sorted(glob.glob(os.path.join(flow_occ_dir, lr, fb, '*.png')))
            # print(len(im_paths), len(fl_paths), len(fl_occ_paths))
            if fb == 'into_past':
                im_paths = im_paths[::-1]
                # fl_paths = fl_paths[::-1]
                # fl_occ_paths = fl_occ_paths[::-1]
            for idx in range(len(im_paths) - 1):
                cur_fn = im_paths[idx]
                nxt_fn = im_paths[idx+1]
                _, base_name = os.path.split(cur_fn)
                flow_fn = os.path.join(flow_dir, lr, fb, base_name[:-4] + '.flo')
                flow_occ_fn = os.path.join(flow_occ_dir, lr, fb, base_name)
                # assert os.path.exists(cur_fn), cur_fn
                # assert os.path.exists(nxt_fn), nxt_fn
                # assert os.path.exists(flow_fn), flow_fn
                # assert os.path.exists(flow_occ_fn), flow_occ_fn
                if os.path.exists(flow_fn):
                    input_paths.append([cur_fn, nxt_fn])
                    target_paths.append([flow_fn, flow_occ_fn])
    return input_paths, target_paths

def make_flow_data(base_dir, flow_thresh):
    raw_train_data = make_flow_data_helper(base_dir, 'train')
    train_data = remove_flow_outliers(raw_train_data, flow_thresh)
    print('{} left for the training set of FlyingThings3D.'.format(len(train_data[0])/len(raw_train_data[0])))
    sys.stdout.flush()

    raw_test_data = make_flow_data_helper(base_dir, 'val')
    test_data = remove_flow_outliers(raw_test_data, flow_thresh)
    print('{} left for the testing set of FlyingThings3D.'.format(len(test_data[0])/len(raw_test_data[0])))
    sys.stdout.flush()

    return train_data, test_data

"""
Disparity only.
"""
def make_disp_data_helper(base_dir, split):
    input_paths = []
    target_paths = []
    img_dir = os.path.join(base_dir, split, 'image_clean')
    disp_dir = os.path.join(base_dir, split, 'disparity')
    disp_occ_dir = os.path.join(base_dir, split, 'disparity_occlusions')
    direct = 'left'
    im_paths = sorted(glob.glob(os.path.join(img_dir, direct, '*.png')))
    dp_paths = sorted(glob.glob(os.path.join(disp_dir, direct, '*.pfm')))
    dp_occ_paths = sorted(glob.glob(os.path.join(disp_occ_dir, direct, '*.png')))
    for idx in range(len(im_paths)):
        left_fn = im_paths[idx]
        right_fn = left_fn.replace('left', 'right')
        disp_fn = dp_paths[idx]
        disp_occ_fn = dp_occ_paths[idx]
        assert os.path.exists(left_fn), left_fn
        assert os.path.exists(right_fn), right_fn
        assert os.path.exists(disp_fn), disp_fn
        assert os.path.exists(disp_occ_fn), disp_occ_fn 
        input_paths.append([left_fn, right_fn])      
        target_paths.append([disp_fn, disp_occ_fn])
    return input_paths, target_paths

def make_disp_data(base_dir, disp_thresh):
    raw_train_data = make_disp_data_helper(base_dir, 'train')
    train_data = remove_disp_outliers(raw_train_data, disp_thresh)
    print('{} left for the training set of FlyingThings3D.'.format(len(train_data[0])/len(raw_train_data[0])))

    raw_test_data = make_disp_data_helper(base_dir, 'val')
    test_data = remove_disp_outliers(raw_test_data, disp_thresh)
    print('{} left for the testing set of FlyingThings3D.'.format(len(test_data[0])/len(raw_test_data[0])))

    return train_data, test_data

"""
Joint optical flow and disparity.
"""
def make_train_flow_disp_data(im_paths, fl_paths, fl_occ_paths, dp_paths, dp_occ_paths):
    input_paths = []        # (cur, nxt, left, right)
    target_paths = []       # (flow, disp)
    flow_dir, _ = os.path.split(fl_paths[0])
    if fl_occ_paths is not None:
        flow_occ_dir, _ = os.path.split(fl_occ_paths[0])
    else:
        flow_occ_dir = 'xxxxx'
    for idx in range(len(im_paths) - 1):
        cur_im_fn = im_paths[idx]
        nxt_im_fn = im_paths[idx+1]
        dp_fn = dp_paths[idx]
        dp_occ_fn = '' if dp_occ_paths is None else dp_occ_paths[idx]
        _, base_name = os.path.split(cur_im_fn)
        if cur_im_fn.find('image_clean') >= 0:
            flow_fn = os.path.join(flow_dir, base_name[:-4] + '.flo')
        else:
            # flow_fn = os.path.join(flow_dir, base_name[:-4] + '.pfm')
            flow_fn = fl_paths[idx]
        flow_occ_fn = os.path.join(flow_occ_dir, base_name)
        # if np.random.random() > 0.99:
        # print(flow_fn, flow_occ_fn, os.path.exists(flow_fn))
        if os.path.exists(flow_fn):
            assert os.path.exists(dp_fn), dp_fn
            assert os.path.exists(cur_im_fn), cur_im_fn
            assert os.path.exists(nxt_im_fn), nxt_im_fn
            if not os.path.exists(flow_occ_fn):
                flow_occ_fn = None
            if not os.path.exists(dp_occ_fn):
                dp_occ_fn = None
            if cur_im_fn.find('left/') > 0:
                right_im_fn = cur_im_fn.replace('left/', 'right/')
                input_paths.append(
                    [cur_im_fn, nxt_im_fn, cur_im_fn, right_im_fn]
                )
                target_paths.append([flow_fn, flow_occ_fn, dp_fn, dp_occ_fn])
                # right_im_fn = nxt_im_fn.replace('left/', 'right/')
                # input_paths.append(
                #     [cur_im_fn, nxt_im_fn, nxt_im_fn, right_im_fn]
                #     )
                # target_paths.append([flow_fn, dp_paths[idx+1]])
            else:
                left_im_fn = cur_im_fn.replace('right/', 'left/')
                input_paths.append(
                    [cur_im_fn, nxt_im_fn, left_im_fn, cur_im_fn]
                )
                target_paths.append([flow_fn, flow_occ_fn, dp_fn, dp_occ_fn])
                # left_im_fn = nxt_im_fn.replace('right/', 'left/')
                # input_paths.append(
                #     [cur_im_fn, nxt_im_fn, left_im_fn, nxt_im_fn]
                #     )
                # target_paths.append([flow_fn, dp_paths[idx+1]])
    return input_paths, target_paths

def _make_test_flow_disp_data(im_paths, fl_paths, fl_occ_paths, dp_paths, dp_occ_paths):
    input_paths = []        # (cur, nxt, left, right)
    target_paths = []       # (flow, disp)
    flow_dir, _ = os.path.split(fl_paths[0])
    flow_occ_dir, _ = os.path.split(fl_occ_paths[0])
    for idx in range(len(im_paths) - 1):
        cur_im_fn = im_paths[idx]
        nxt_im_fn = im_paths[idx+1]
        dp_fn = dp_paths[idx]
        dp_occ_fn = '' if dp_occ_paths is None else dp_occ_paths[idx]
        _, base_name = os.path.split(cur_im_fn)
        if cur_im_fn.find('image_clean') >= 0:
            flow_fn = os.path.join(flow_dir, base_name[:-4] + '.flo')
        else:
            flow_fn = os.path.join(flow_dir, base_name[:-4] + '.pfm')
        flow_occ_fn = os.path.join(flow_occ_dir, base_name)
        if os.path.exists(flow_fn):
            assert os.path.exists(dp_fn), dp_fn
            assert os.path.exists(cur_im_fn), cur_im_fn
            assert os.path.exists(nxt_im_fn), nxt_im_fn
            if not os.path.exists(flow_occ_fn):
                flow_occ_fn = None
            if not os.path.exists(dp_occ_fn):
                dp_occ_fn = None
            if cur_im_fn.find('left/') > 0:
                right_im_fn = cur_im_fn.replace('left/', 'right/')
                input_paths.append(
                    [cur_im_fn, nxt_im_fn, cur_im_fn, right_im_fn]
                )
                target_paths.append([flow_fn, flow_occ_fn, dp_fn, dp_occ_fn])
    return input_paths, target_paths
    
def make_flow_disp_data_simple_merge(base_dir, split):
    input_paths = []        # (cur, nxt, left, right)
    target_paths = []       # (flow, disp)
    img_dir = os.path.join(base_dir, split, 'image_clean')
    flow_dir = os.path.join(base_dir, split, 'flow')
    flow_occ_dir = os.path.join(base_dir, split, 'flow_occlusions')
    disp_dir = os.path.join(base_dir, split, 'disparity')
    disp_occ_dir = os.path.join(base_dir, split, 'disparity_occlusions')
    for lr in ['left', 'right']:
        for fb in ['into_future', 'into_past']:
            im_paths = sorted(glob.glob(os.path.join(img_dir, lr, '*.png')))
            fl_paths = sorted(glob.glob(os.path.join(flow_dir, lr, fb, '*.flo')))
            fl_occ_paths = sorted(glob.glob(os.path.join(flow_occ_dir, lr, fb, '*.png')))
            dp_paths = sorted(glob.glob(os.path.join(disp_dir, 'left', '*.pfm')))
            dp_occ_paths = sorted(glob.glob(os.path.join(disp_occ_dir, 'left', '*.png')))
            print(len(im_paths), len(fl_paths), len(fl_occ_paths))
            print(os.path.join(img_dir, lr))
            if fb == 'into_past':
                im_paths = im_paths[::-1]
                fl_paths = fl_paths[::-1]
                fl_occ_paths = fl_occ_paths[::-1]
                dp_paths = dp_paths[::-1]
                dp_occ_paths = dp_occ_paths[::-1]

            if split == 'train':
                tmp_input, tmp_target = make_train_flow_disp_data(
                    im_paths, fl_paths, fl_occ_paths, dp_paths, dp_occ_paths)
                input_paths.extend(tmp_input)
                target_paths.extend(tmp_target)
            else:
                if fb == 'into_future':
                    tmp_input, tmp_target = _make_test_flow_disp_data(
                        im_paths, fl_paths, fl_occ_paths, dp_paths, dp_occ_paths
                    )
                    input_paths.extend(tmp_input)
                    target_paths.extend(tmp_target)

    #         for idx in range(len(im_paths) - 1):
    #             cur_fn = im_paths[idx]
    #             nxt_fn = im_paths[idx+1]
    #             flow_fn = fl_paths[idx]
    #             flow_occ_fn = fl_occ_paths[idx]
    #             assert os.path.exists(cur_fn), cur_fn
    #             assert os.path.exists(nxt_fn), nxt_fn
    #             assert os.path.exists(flow_fn), flow_fn
    #             assert os.path.exists(flow_occ_fn), flow_occ_fn
    #             input_paths.append([cur_fn, nxt_fn])
    #             target_paths.append([flow_fn, flow_occ_fn])


    # for abc in ['A', 'B', 'C']: 
    #     seqs_all = os.listdir(os.path.join(img_dir, abc))
    #     for i, cur_seq in enumerate(seqs_all):
    #         for fb in ['into_future', 'into_past']:            
    #             for lr in ['left', 'right']:  
    #                 im_paths = sorted(glob.glob(os.path.join(img_dir, abc, cur_seq, lr, '*.png')))
    #                 dp_paths = sorted(glob.glob(os.path.join(disp_dir, abc, cur_seq, 'left', '*.pfm')))                  
    #                 fl_paths = sorted(glob.glob(os.path.join(flow_dir, abc, cur_seq, fb, lr, '*.pfm')))
    #                 if fb == 'into_past':
    #                     im_paths = im_paths[::-1]
    #                     fl_paths = fl_paths[::-1]
    #                     dp_paths = dp_paths[::-1]
    #                 if split == 'TRAIN':
    #                     tmp_input, tmp_target = make_train_flow_disp_data(
    #                                                 im_paths, fl_paths, dp_paths
    #                                                 )
    #                     # print('*** tmp input_paths')
    #                     # for p in tmp_input:
    #                     #     print(p)
    #                     # print('*** tmp target_paths')
    #                     # for p in tmp_target:
    #                     #     print(p)
    #                     input_paths.extend(tmp_input)
    #                     target_paths.extend(tmp_target)
    #                     # print('*** input_paths')
    #                     # for p in input_paths:
    #                     #     print(p)
    #                     # print('*** target_paths')
    #                     # for p in target_paths:
    #                     #     print(p)
    #                     # import numpy as np
    #                     # if np.random.random() > 0.5:
    #                     #     raise Exception
    #                 else:
    #                     if fb == 'into_future':
    #                         tmp_input, tmp_target = _make_test_flow_disp_data(
    #                                                     im_paths, fl_paths, dp_paths
    #                                                     )
    #                         input_paths.extend(tmp_input)
    #                         target_paths.extend(tmp_target)

    print('### input_paths')
    for p in input_paths[:10]:
        print(p)
    print('### target_paths')
    for p in target_paths[:10]:
        print(p)
    return input_paths, target_paths

def make_flow_disp_data_exhaust_merge(base_dir, split):
    cur_im_paths, nxt_im_paths, flow_paths = make_flow_data_helper(base_dir, split)
    left_im_paths, right_im_paths, disp_paths = make_disp_data_helper(base_dir, split)
    input_paths = []
    target_paths = []
    for i in range(len(cur_im_paths)):
        for j in range(len(left_im_paths)):
            input_paths.append([
                cur_im_paths[i], nxt_im_paths[i],
                left_im_paths[j], right_im_paths[j]
            ])
            target_paths.append([flow_paths[i], disp_paths[j]])
    return input_paths, target_paths

def make_flow_disp_data(base_dir, flow_thresh=500, disp_thresh=500, merge_crit='simple'):
    if merge_crit == 'simple':
        helper_func = make_flow_disp_data_simple_merge
    elif merge_crit == 'exhaust':
        helper_func = make_flow_disp_data_exhaust_merge
    else:
        raise NotImplementedError('Not supported merge criterion {}'.format(merge_crit))
    train_data = helper_func(base_dir, 'train')
    test_data = helper_func(base_dir, 'val')

    train_data = remove_flow_disp_outliers(train_data, flow_thresh)
    test_data = remove_flow_disp_outliers(test_data, flow_thresh)
    return train_data, test_data

if __name__ == '__main__':
    base_dir = '/home/hzjiang/Data/SceneFlow/FlyingThings3D_subset'
    
    print('scene flow')
    joint_train_data, joint_test_data = make_flow_disp_data(base_dir)
    # # with open('FlyingThings3D_joint_train.pkl', 'wb') as f:
    # #     pickle.dump({'joint_train_data': joint_train_data,
    # #                  'joint_test_data': joint_test_data},
    # #                  f, pickle.HIGHEST_PROTOCOL
    # #                  )
    # with open('FlyingThings3D_joint_train.pkl', 'rb') as f:
    #     joint_data = pickle.load(f)
    #     joint_train_data = joint_data['joint_train_data']
    #     joint_test_data = joint_data['joint_test_data']
    print(len(joint_train_data[0]), len(joint_test_data[0]))

    # import numpy as np
    # import skimage
    # from skimage.io import imread, imsave
    # from datasets.dataset_utils import load_pfm
    # from misc.viz_flow import viz_flow
    # dst_dir = 'sanity_vis'
    # if not os.path.exists(dst_dir):
    #     os.makedirs(dst_dir)
    # for i in range(10):
    #     idx = np.random.randint(len(joint_train_data[0]))
    #     input_paths = joint_train_data[0][idx]
    #     target_paths = joint_train_data[1][idx]
    #     print(input_paths)
    #     im = imread(input_paths[0])
    #     imsave(os.path.join(dst_dir, '{0:3d}_acurrent.png'.format(i)), im)
    #     im = imread(input_paths[1])
    #     imsave(os.path.join(dst_dir, '{0:3d}_bnext.png'.format(i)), im)
    #     im, _ = load_pfm(target_paths[0])
    #     im = viz_flow(im[:, :, 0], im[:, :, 1])
    #     imsave(os.path.join(dst_dir, '{0:3d}_cflow.png'.format(i)), im)
    #     im = imread(input_paths[2])
    #     imsave(os.path.join(dst_dir, '{0:3d}_dleft.png'.format(i)), im)
    #     im = imread(input_paths[3])
    #     imsave(os.path.join(dst_dir, '{0:3d}_eright.png'.format(i)), im)
    #     im, _ = load_pfm(target_paths[1])
    #     im = im.astype(np.float32)
    #     im = (im - np.min(im)) / (np.max(im) - np.min(im) + 1e-30)
    #     im = (im * 255).astype(np.uint8)
    #     imsave(os.path.join(dst_dir, '{0:3d}_fdisp.png'.format(i)), im)
    #     print(i+1)
    #     # print('+++++++++++++++++++++++++++++++++++++++++++')
    #     # print(joint_train_data[0][idx])
    #     # print(joint_train_data[1][idx])
    #     # print('+++++++++++++++++++++++++++++++++++++++++++\n')

    # print('optical flow')
    # flow_train_data, flow_test_data = make_flow_data(base_dir, 500)
    # # with open('FlyingThings3D_flow_train.pkl', 'wb') as f:
    # #     pickle.dump({'flow_train_data': flow_train_data,
    # #                  'flow_test_data': flow_test_data},
    # #                  f, pickle.HIGHEST_PROTOCOL
    # #                  )
    # with open('FlyingThings3D_flow_train.pkl', 'rb') as f:
    #     flow_data = pickle.load(f)
    #     flow_train_data = flow_data['flow_train_data']
    #     flow_test_data = flow_data['flow_test_data']
    # print(len(flow_train_data[0]), len(flow_test_data[0]))

    # print('disparity')
    # disp_train_data, disp_test_data = make_disp_data(base_dir, -1)
    # # with open('FlyingThings3D_disp_train.pkl', 'wb') as f:
    # #     pickle.dump({'disp_train_data': disp_train_data,
    # #                  'disp_test_data': disp_test_data},
    # #                  f, pickle.HIGHEST_PROTOCOL
    # #                  )
    # # with open('FlyingThings3D_disp_train.pkl', 'rb') as f:
    # #     disp_data = pickle.load(f)
    # #     disp_train_data = disp_data['disp_train_data']
    # #     disp_test_data = disp_data['disp_test_data']
    # print(len(disp_train_data[0]), len(disp_test_data[0]))

    # def check_existence(data1, data2, idxes):
    #     for i, dt in enumerate(data1):
    #         tmp = dt[idxes]
    #         if len(tmp) == 1:
    #             tmp = tmp[0]
    #         if tmp not in data2:
    #             print(dt)
    #             print(tmp, idxes, i)
    #             raise Exception('An outlier found.')

    # def check_exists_2(data1, data2, idxes):
    #     tmp_data = []
    #     for dt in data2:
    #         tmp = dt[idxes]
    #         if len(tmp) == 1:
    #             tmp_data.append(tmp[0])
    #         else:
    #             tmp_data.append(tmp)
    #     for i, dt in enumerate(data1):
    #         if dt not in tmp_data:
    #             print(dt)
    #             print(tmp_data[:3])
    #             raise Exception('An outlier found.')

    # def sanity_check_flow_disp_data(joint_data, flow_data, disp_data):
    #     # check flow data first
    #     tmp_in_paths, tmp_tgt_paths = joint_data
    #     in_paths, tgt_paths = flow_data
    #     print('==> checking flow input')
    #     check_existence(tmp_in_paths, in_paths, slice(2))
    #     print('==> checking flow GT')
    #     check_existence(tmp_tgt_paths, tgt_paths, slice(1))
        
    #     in_paths, tgt_paths = disp_data
    #     print('==> checking disp input')
    #     check_existence(tmp_in_paths, in_paths, slice(2, 4))
    #     print('==> checking disp GT')
    #     check_existence(tmp_tgt_paths, tgt_paths, slice(1, 2))

    #     # print('checking in the opposite direction');
    #     # in_paths, tgt_paths = flow_data
    #     # print('==> checking flow input')
    #     # check_exists_2(in_paths, tmp_in_paths, slice(2))
    #     # print('==> checking flow GT')
    #     # check_exists_2(tgt_paths, tmp_tgt_paths, slice(1))
        
    #     # in_paths, tgt_paths = disp_data
    #     # print('==> checking disp input')
    #     # check_exists_2(in_paths, tmp_in_paths, slice(2, 4))
    #     # print('==> checking disp GT')
    #     # check_exists_2(tgt_paths, tmp_tgt_paths, slice(1, 2))

    # # sanity_check_flow_disp_data(joint_train_data, flow_train_data, disp_train_data)

    # # check
    # def check_input_target_existence(joint_data, single_data, is_flow):
    #     input_paths, target_paths = joint_data
    #     joint_tmp_data = []
    #     for i in range(len(input_paths)):
    #         if is_flow:
    #             tmp = input_paths[i][:2]
    #             tmp.append(target_paths[i][0])
    #         else:
    #             tmp = input_paths[i][2:]
    #             tmp.append(target_paths[i][1])
    #         joint_tmp_data.append(tmp)

    #     input_paths, target_paths = single_data
    #     single_tmp_data = []
    #     for i in range(len(input_paths)):
    #         tmp = input_paths[i]
    #         tmp.append(target_paths[i])
    #         single_tmp_data.append(tmp)

    #     for td in joint_tmp_data:
    #         if td not in single_tmp_data:
    #             print(td)
    #             raise Exception('An outlier found.')

    # # print('==> checking flow')
    # # check_input_target_existence(joint_train_data, flow_train_data, True)
    # # print('==> checking disp')
    # # check_input_target_existence(joint_train_data, disp_train_data, False)

    # # check number of unique disparity pairs
    # input_dict = {}
    # target_dict = {}
    # input_paths, target_paths = joint_train_data
    # for i in range(len(input_paths)):
    #     tmp_in = input_paths[i][2:4]
    #     tmp_in = tmp_in[0] + '@' + tmp_in[1]
    #     tmp_tgt = target_paths[i][1]
    #     if tmp_in not in input_dict:
    #         input_dict[tmp_in] = 1
    #     else:
    #         input_dict[tmp_in] += 1

    #     if tmp_tgt not in target_dict:
    #         target_dict[tmp_tgt] = 1
    #     else:
    #         target_dict[tmp_tgt] += 1

    # print(len(input_dict.values()), len(target_dict.values()))

    # ------------------------ Deprecated -----------------------------
    # print('!!! input_paths')
    # for i in range(4):
    #     print(joint_train_data[0][i])
    # print('!!! target_paths')
    # for i in range(4):
    #     print(joint_train_data[1][i])

    # import pickle
    # with open('FlyingThings3D_flow_train.pkl', 'wb') as f:
    #     pickle.dump(train_data, f, pickle.HIGHEST_PROTOCOL)

    # compare with Deqing's list
    # import pickle
    # with open('FlyingThings3D_flow_train.pkl', 'rb') as f:
    #     train_data = pickle.load(f)

    # with open('/scratch/Code/semi-scene-flow/misc/flyingThings3D_frames_cleanpass_TRAIN.list', 'r') as f:
    #     all_paths = f.readlines()
    # with open('/scratch/Code/semi-scene-flow/misc/good_flyingThings3D_frames_cleanpass_TRAIN.list', 'r') as f:
    #     all_flags = f.readlines()
    # assert len(all_flags) == len(all_paths), [len(all_paths), len(all_flags)]

    # base_dir = '/scratch/Data/FlyingThings3D'
    # all_flow_paths = []
    # for data, flag in zip(all_paths, all_flags):
    #     if flag.strip() == '1':
    #         tokens = data.strip().split(' ')
    #         cur_im_path = os.path.join(base_dir, tokens[0])
    #         assert os.path.exists(cur_im_path), cur_im_path
    #         nxt_im_path = os.path.join(base_dir, tokens[1])
    #         assert os.path.exists(nxt_im_path), nxt_im_path
    #         flow_path = os.path.join(base_dir, tokens[2])
    #         assert os.path.exists(flow_path), flow_path
    #         flow_path = flow_path.replace('./', '')
    #         flow_path = flow_path.replace('optical_flow_flo_format', 'optical_flow')
    #         all_flow_paths.append(flow_path[:-4])

    # my_flow_paths = train_data[1]

    # print(len(all_flow_paths), len(my_flow_paths))
    # print(all_flow_paths[:5])
    # print(my_flow_paths[:5])

    # # all_flow_paths = all_flow_paths[:5]
    # # my_flow_paths = my_flow_paths[:500]

    # from datasets.dataset_utils import load_pfm, load_flo
    # from datasets.flow_io import flow_read_uv
    # import numpy as np
    # def check_flow_stats(flow_path):
    #     flow, _ = load_pfm(flow_path)
    #     flow = flow[:, :, :2]
    #     return np.max(np.abs(flow)), np.mean(np.abs(flow))

    # def check_flow_flo_stats(flow_path):
    #     flow = load_flo(flow_path)
    #     return np.max(np.abs(flow)), np.mean(np.abs(flow))

    # def check_flow_uv_stats(flow_path):
    #     uv = flow_read_uv(flow_path)
    #     return abs(uv).max()

    # cnt = 0
    # for fp in my_flow_paths:
    #     if fp[:-4] not in all_flow_paths:
    #         print(fp)
    #         print('pfm: ', check_flow_stats(fp))

    #         fp2 = fp.replace('optical_flow', 'optical_flow_flo_format')
    #         fp2 = fp2[:-4] + '.flo'
    #         print('flo: ', check_flow_flo_stats(fp2))
    #         print('uv: ', check_flow_uv_stats(fp2))
    #         cnt += 1

    # print('{} not found.'.format(cnt))

