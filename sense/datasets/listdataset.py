"""
Copyright (C) 2019 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""
import torch.utils.data as data
import os
import os.path
from scipy.ndimage import imread
import numpy as np
import pdb
import time
import sys

class ListDataset(data.Dataset):
    def __init__(self, root, path_list, loader, transform=None, target_transform=None,
            co_transform=None, co_transform_test=None, transform_additional=None
        ):
        super(ListDataset, self).__init__()

        self.root = root
        self.path_list = path_list
        self.transform = transform
        self.target_transform = target_transform
        self.co_transform = co_transform
        self.co_transform_test = co_transform_test
        self.loader = loader
        self.transform_additional = transform_additional

        # np.random.seed(0)
        # np.random.shuffle(self.path_list)

    def __getitem__(self, index):
        # start = time.time()
        inputs, targets = self.path_list[index]
        inputs, targets = self.loader(self.root, inputs, targets)
        # for _i, _inputs in enumerate(inputs):
        #     inputs[_i] = inputs[_i][:, :, ::-1]
        #     inputs[_i] = 1.0 * inputs[_i]/255.0
        # work = 'data loading'
        # print('{:<25s} {:>.6f}', .format(work, time.time() - start))
        # data_load_time = time.time() - start

        # start = time.time()
        if self.transform is not None:
            inputs = self.transform(inputs)    
        # work = 'transform'
        # print('{:<25s} {:>.6f}', .format(work, time.time() - start))
        # trans_time = time.time() - start

        # start = time.time()
        if self.co_transform is not None:
            inputs, targets = self.co_transform(inputs, targets)
        # work = 'co_transform'
        # print('{:<25s} {:>.6f}', .format(work, time.time() - start))
        # co_trans_time = time.time() - start

        # start = time.time()
        if self.co_transform_test is not None:
            inputs, targets = self.co_transform_test(inputs, targets)
        # work = 'co_transform_test'
        # print('{:<25s} {:>.6f}', .format(work, time.time() - start))
        # co_trans_test_time = time.time() - start

        # start = time.time()
        inputs_new = []
        if self.transform_additional is not None:
            for i in range(len(inputs)):
                # inputs[i] = self.transform_additional(inputs[i])  
                inputs_new.append(self.transform_additional(inputs[i]))
        # work = 'transform_additional'
        # print('{:<25s} {:>.6f}', .format(work, time.time() - start))
        # trans_add_time = time.time() - start
        
        # start = time.time()
        if self.target_transform is not None :
            targets = self.target_transform(targets)        
        # # work = 'target_transform'
        # # print('{:<25s} {:>.6f}', .format(work, time.time() - start))
        # tgt_trans_time = time.time() - start
        # print('{:>.15f}\t'
        #       '{:>.15f}\t'
        #       '{:>.15f}\t'
        #       '{:>.15f}\t'
        #       '{:>.15f}\t'
        #       '{:>.15f}\t'.format(
        #         data_load_time,
        #         trans_time,
        #         co_trans_time,
        #         co_trans_test_time,
        #         trans_add_time,
        #         tgt_trans_time))
        # sys.stdout.flush()
        return inputs_new, targets

    def __len__(self):
        return len(self.path_list)


