"""
Copyright (C) 2019 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""

import numpy as np

class RandomHorizontalFlip(object):
    """Randomly horizontally flips the given PIL.Image with a probability of 0.5
    """
    def __call__(self, data, targets=None):
        if np.random.random() < 0.5:
            for i in range(len(data)):
                data[i] = np.fliplr(data[i]).copy()
        return data, targets

class RandomVerticalFlip(object):
    """Randomly horizontally flips the given PIL.Image with a probability of 0.5
    """
    def __call__(self, data, targets=None):
        prob = np.random.random()
        if prob < 0.5:
            for i, _ in enumerate(data):
                data[i] = np.flipud(data[i]).copy()
        return data, targets