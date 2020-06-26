#!/bin/bash

set -e

export PYTHONPATH=.

JOINT_MODEL_PATH=data/pretrained_models/kitti2012+kitti2015_new_lr_schedule_lr_disrupt+semi_loss_v3.pth
MODEL_ID=sense_release

DISP_REFINME_MODEL_PATH=data/pretrained_models/kitti2015_warp_disp_refine_1500.pth

ENC_ARCH=psm
FLOW_REFINEMENT=none

DATASET=kitti2015
SPLIT=testing
VIS_DIR=output/${DATASET}_${MODEL_ID}/$SPLIT

KITTI_DIR=/home/hzjiang/workspace/Data/KITTI_scene_flow

# generate raw optical flow and stereo disparity
echo '### Running hoslitic scene model'
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

# This step is not necessary. But due to the way of preparing the input images for semantic segmentation (padding vs resizing),
# it causes slightly different semantic segmentation results and thus scene flow results without doing this step.
# semantic segmentation
echo '### Running semantic segmentation'
python tools/run_kitti2015_segmentation.py eval \
    --dataset $DATASET \
    --datapath /home/hzjiang/workspace/Data/KITTI_Semantics \
    --split $SPLIT \
    --enc psm \
    --dec pwcdc \
    --disp-refinement hourglass \
    --flow-dec pwcdc \
    --flow-refinement none \
    --flow-no-ppm \
    --corr-radius 4 \
    --bn-type syncbn \
    --joint-model \
    --loadmodel $JOINT_MODEL_PATH \
    --save-dir $VIS_DIR

# rigidity and network-based refinement
echo '### Running warped disparity refinement for scene flow estimation'
python tools/run_kitti2015_warped_disparity_refinement.py \
    --loadmodel ${DISP_REFINME_MODEL_PATH} \
    --res-dir ${VIS_DIR}

CUR_DIR=$PWD
cd ${VIS_DIR}
cp -r flow_rigid flow
cp -r disp_1_nn disp_1
cd ${CUR_DIR}
