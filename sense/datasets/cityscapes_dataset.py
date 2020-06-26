"""
Copyright (C) 2019 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""
import os
from PIL import Image

import torch
import torch.utils.data as data
import numpy as np

from .dataset_utils import imread

CITYSCAPE_PALETTE = np.asarray([
    [128, 64, 128],
    [244, 35, 232],
    [70, 70, 70],
    [102, 102, 156],
    [190, 153, 153],
    [153, 153, 153],
    [250, 170, 30],
    [220, 220, 0],
    [107, 142, 35],
    [152, 251, 152],
    [70, 130, 180],
    [220, 20, 60],
    [255, 0, 0],
    [0, 0, 142],
    [0, 0, 70],
    [0, 60, 100],
    [0, 80, 100],
    [0, 0, 230],
    [119, 11, 32],
    [0, 0, 0]], dtype=np.uint8)


TRIPLET_PALETTE = np.asarray([
    [0, 0, 0, 255],
    [217, 83, 79, 255],
    [91, 192, 222, 255]], dtype=np.uint8)

class SegList(data.Dataset):
    def __init__(self, data_dir, phase, transforms, 
                 list_dir=None, out_name=False, im_format='pil'):
        self.list_dir = data_dir if list_dir is None else list_dir
        self.data_dir = data_dir
        self.out_name = out_name
        self.phase = phase
        self.transforms = transforms
        self.image_list = None
        self.label_list = None
        self.bbox_list = None
        self.im_format = im_format
        self.read_lists()

    def __getitem__(self, index):
        # data = []
        if self.im_format == 'cv2':
            im = imread(os.path.join(self.data_dir, self.image_list[index]))
        elif self.im_format == 'pil':
            im = Image.open(os.path.join(self.data_dir, self.image_list[index]))
            # RGB -> BGR
            try:
                r, g, b = im.split()
                im = Image.merge('RGB', (b, g, r))
            except:
                print('INCORRECT IMAGE ', self.image_list[index])
        else:
            raise Exception('Unsupported image format: {}'.format(self.im_format))
        data = [im]
        if self.label_list is not None:
            seg_im = Image.open(os.path.join(self.data_dir, self.label_list[index]))
            data.append(seg_im)
        data = list(self.transforms(*data))
        if self.out_name:
            if self.label_list is None:
                data.append(data[0][0, :, :])
            data.append(self.image_list[index])
        return tuple(data)

    def __len__(self):
        return len(self.image_list)

    def read_lists(self):
        image_path = os.path.join(self.list_dir, self.phase + '_images.txt')
        label_path = os.path.join(self.list_dir, self.phase + '_labels.txt')
        assert os.path.exists(image_path), image_path
        self.image_list = [line.strip() for line in open(image_path, 'r')]
        if os.path.exists(label_path):
            self.label_list = [line.strip() for line in open(label_path, 'r')]
            assert len(self.image_list) == len(self.label_list), [len(self.image_list), len(self.label_list)]

if __name__ == '__main__':
    import data_transforms as transforms

    t = []
    # if args.random_rotate > 0:
    #     t.append(transforms.RandomRotate(0))
    # if args.random_scale > 0:
    #     t.append(transforms.RandomScale(0))
    normalize = transforms.Normalize(mean=[102.9801, 115.9465, 122.7717], 
                                     std=[1., 1., 1.]
                                     )
    # t.extend([transforms.RandomCrop((768, 320), 4),
    #           transforms.RandomHorizontalFlip(),
    #           # transforms.ToNumpy(1/255.0),
    #           # transforms.RandomGammaImg((0.7,1.5)),
    #           # transforms.RandomBrightnessImg(0.2),
    #           # transforms.RandomContrastImg((0.8, 1.2)),
    #           # transforms.RandomGaussianNoiseImg(0.02),
    #           # transforms.ToNumpy(255.0),
    #           transforms.ToTensor(convert_pix_range=False),
    #           normalize])

    t.extend([transforms.RandomCrop((768, 320), 4),
              transforms.RandomHorizontalFlip(),
              # transforms.ToNumpy(1/255.0),
              # transforms.RandomGammaImg((0.7,1.5)),
              # transforms.RandomBrightnessImg(0.2),
              # transforms.RandomContrastImg((0.8, 1.2)),
              # transforms.RandomGaussianNoiseImg(0.02),
              # transforms.ToNumpy(255.0),
              transforms.ToTensor(convert_pix_range=False), normalize
              ])

    # data_dir = '/home/hzjiang/workspace/Data/CityScapes'
    data_dir = '/home/hzjiang/workspace/Data/KITTI_Semantics'
    train_data = SegList(data_dir, 'train', transforms.Compose(t),
                list_dir=data_dir)

    for i, (image, label) in enumerate(train_data):
        # for c in range(3):
        #     print('--- after norm: ', c, torch.max(image[:,:,c]), torch.min(image[:,:,c]))
        print(image.size(), label.size())
        image = image.numpy().transpose(1, 2, 0)
        image += [102.9801, 115.9465, 122.7717]
        print(np.max(image), np.min(image))
        import cv2
        cv2.imwrite('image_{}.png'.format(i), image.astype(np.uint8))
        if i > 4:
            break