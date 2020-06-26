# SENSE: a Shared Encoder for Scene Flow Estimation
PyTorch implementation of our ICCV 2019 Oral paper [SENSE: A Shared Encoder for Scene-flow Estimation](https://arxiv.org/pdf/1910.12361.pdf).

<p align="center">
  <img src="sense.png" width="500" />
</p>

## License

Copyright (C) 2019 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).

## Requirements
* Python (tested with Python3.6.10)
* PyTorch (tested with 1.3.0)
* SynchronizedBatchNorm (borrowed from https://github.com/CSAILVision/semantic-segmentation-pytorch)
* tensorboardX
* tqdm
* OpenCV
* scipy
* numpy

It is strongly recommended to use a conda environment to install all the dependencies. Simply run `sh scripts/install.sh` to install all dependencies and also compile the correlation package.

All experiments were conducted on 8 2080ti GPUs (each with 11GB memory) or 2 M40 GPUs (each with 24GB memory).

In our original implementation, we used a C++ implementation for the cost volume computation for both optical flow and stereo disparity estimations. But the C++ implementatyion strictly requires a PyTorch version of 0.4.0. In this relased version, we switched to use the implemtnation provided at <https://github.com/NVIDIA/flownet2-pytorch>. We use this implementation for stereo disparity estimation, although it only supports cost volume computation for optical flow (searching for correspondence in a local 2D range). Please consult our paper if you are interested in the running time and GPU memory usuage.

## Quick Start
First run `sh scripts/download_pretrained_models.sh` to download pre-trained models. Run `python tools/demo.py` then for a quick demo.

Run `sh scripits/make_kitti2015_submission.sh` to generate results that can be submitted to the online [KITTI scene flow estimation benchmark](http://www.cvlibs.net/datasets/kitti/eval_scene_flow.php). You should be able to get the following results.

| Error | D1-bg	| D1-fg	| D1-all	| D2-bg	| D2-fg	| D2-all	| Fl-bg	| Fl-fg	| Fl-all	| SF-bg	| SF-fg	| SF-all
| --- | :----: | :----: | :----: | :----: | :----: | :----: | :----: | :----: | :----: | :----: | :----: | :----: 
| All/Est | 2.07 | 3.01 | 2.22 | 4.90 | 10.83 | 5.89 | 7.30 | 9.33 | 7.64 | 8.36 | 15.49 | 9.55

## Training
See [TRAINING.md](TRAINING.md) for details.

## Citation
If you find SENSE useful for your research, please consider citing it.
```BibTeX
@InProceedings{jiang2019sense,
author = {Jiang, Huaizu and Sun, Deqing and Jampani, Varun and Lv, Zhaoyang and Learned-Miller, Erik and Kautz, Jan},
title = {SENSE: A Shared Encoder Network for Scene-Flow Estimation},
booktitle = {ICCV},
year = {2019}
}
```