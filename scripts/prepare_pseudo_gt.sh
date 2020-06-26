#!/bin/bash

export PYTHONPATH=./

# ----------------------- semantic segmentation ---------------------
echo "### Preparing pseudo ground-truth for semantic segmentation."

UPER_NET_DIR=/home/hzjiang/Code/UPerNet-fork
export PYTHONPATH=${PYTHONPATH}:${UPER_NET_DIR}

CKPT_DIR=data/pretrained_models/UPerNet/more-data-aug-kitti-full-320x768-sgd-resnet101-upernet-ngpus1-batchSize8-cropSize840-randomScale2.0-randomRotate10-paddingConst32-segmDownsampleRate4-LR_encoder0.001-LR_decoder0.001-epoch1500-decay0.0001-fixBN0
KITTI2015_DIR=/home/hzjiang/workspace/Data/KITTI_scene_flow
KITTI2012_DIR=/home/hzjiang/workspace/Data/KITTI_Stereo_2012
RES_DIR=./output/KITTI2012+KITTI2015-UPerNet-R101

python tools/save_segmentation_results_cityscapes.py \
     --id '' \
     --data-dir ${KITTI2015_DIR} \
     --phase all \
     --ckpt $CKPT_DIR \
     --suffix _epoch_1500.pth \
     --arch_encoder resnet101    \
     --arch_decoder upernet \
     --batch-size 1 \
     --result ${RES_DIR}

python tools/save_segmentation_results_cityscapes.py \
     --id '' \
     --data-dir ${KITTI2012_DIR} \
     --phase all \
     --ckpt $CKPT_DIR \
     --suffix _epoch_1500.pth \
     --arch_encoder resnet101    \
     --arch_decoder upernet \
     --batch-size 1 \
     --result ${RES_DIR}

# ----------------------- occlusion for optical flow and stereo disparity ----------
echo "### Preparing pseudo ground-truth for occlusion detection."
SYNTHECI_PRETRAINED_MODEL=data/pretrained_models/sceneflow_joint_psm_pwcdc_wppm_hourglass_reg_pwcdc_woppm_none_md_4_upernet_wocc_wocatocc_sum_loss_dispCropSize_256x512_flowDimRatio_None_flowCropSize_384x640_96.pth

RES_DIR=output/KITTI2012+KITTI2015_occlusion_synthetic_sceneflow

python tools/save_occlusion_kitti.py eval \
     --dataset kitti2015 \
     --split training \
     --enc psm \
     --dec pwcdc \
     --disp-refinement hourglass \
     --flow-dec pwcdc \
     --flow-refinement none \
     --flow-no-ppm \
     --corr-radius 4 \
     --bn-type syncbn \
     --joint-model \
     --loadmodel $SYNTHECI_PRETRAINED_MODEL \
     --disp-occ-thresh 0.55 \
     --disp-occ-thresh 0.45 \
     --save-dir ${RES_DIR} \
     --save-occ-only \
     --soft-occ-gt

python tools/save_occlusion_kitti.py eval \
     --dataset kitti2012 \
     --split training \
     --enc psm \
     --dec pwcdc \
     --disp-refinement hourglass \
     --flow-dec pwcdc \
     --flow-refinement none \
     --flow-no-ppm \
     --corr-radius 4 \
     --bn-type syncbn \
     --joint-model \
     --loadmodel $SYNTHECI_PRETRAINED_MODEL \
     --disp-occ-thresh 0.55 \
     --disp-occ-thresh 0.45 \
     --save-dir ${RES_DIR} \
     --save-occ-only \
     --soft-occ-gt