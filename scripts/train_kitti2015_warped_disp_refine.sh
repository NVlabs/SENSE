#!/bin/bash

set -e

export PYTHONPATH=.

JOINT_MODEL_PATH=data/pretrained_models/kitti2012+kitti2015_new_lr_schedule_lr_disrupt+semi_loss_v3.pth
MODEL_ID=sense

ENC_ARCH=psm
FLOW_REFINEMENT=none

DATASET=kitti2015
SPLIT=training
VIS_DIR=output/${DATASET}_${MODEL_ID}/$SPLIT

KITTI_DIR=/home/hzjiang/workspace/Data/KITTI_scene_flow

# generate raw optical flow and stereo disparity
echo '### Running inference using the hoslitic scene model'
python tools/run_kitti2015_flow_disparity.py eval \
    --dataset $DATASET \
    --split $SPLIT \
    --kitti-dir ${KITTI_DIR} \
    --enc $ENC_ARCH \
    --dec pwcdc \
    --disp-refinement hourglass \
    --flow-dec pwcdc \
    --flow-refinement $FLOW_REFINEMENT \
    --flow-no-ppm \
    --corr-radius 4 \
    --bn-type syncbn \
    --joint-model \
    --do-seg \
    --upsample-flow-output \
    --loadmodel $JOINT_MODEL_PATH \
    --save-dir $VIS_DIR

# run rigidity refinement
echo '### Running rigidity-based warped disparity refinement'
python tools/run_kitti2015_warped_disparity_refinement.py \
    --res-dir ${VIS_DIR}

# train the warped disparity refinement model
echo '### Training network-based warped disparity refinement'
CUDA_VISIBLE_DEVICES=0 python tools/train_kitti2015_warp_disp_refine.py pre-train \
   --dataset kitti2015 \
   --split-id -1 \
   --savemodel ./output/kitti2015_warp_disp_refine_rigid_refine_unet_v3_scratch_full_training \
   --workers 8 \
   --lr 0.0005 \
   --lr-gamma 0.5 \
   --lr-steps 400 800 1000 1200 1400 1500 \
   --epochs 1500 \
   --batch-size 8 \
   --disp-crop-imh 320 \
   --disp-crop-imw 704 \
   --kitti-cache-dir ${VIS_DIR} \
   --use-rigid-refine \
   --print-freq 10 \
   --save-freq 100