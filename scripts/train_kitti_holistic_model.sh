#!/bin/bash

set -e

export PYTHONPATH=.

PRETRAINED_MODEL=data/pretrained_models/sceneflow_joint_psm_pwcdc_wppm_hourglass_reg_pwcdc_woppm_none_md_4_upernet_wocc_wocatocc_sum_loss_dispCropSize_256x512_flowDimRatio_None_flowCropSize_384x640_96.pth

PSEUDO_SEG_GT_DIR=./output/KITTI2012+KITTI2015-UPerNet-R101
PSEUDO_OCC_GT_DIR=./output/KITTI2012+KITTI2015_occlusion_synthetic_sceneflow

python tools/train_joint_kitti.py finetune \
   --dataset kitti2012+kitti2015 \
   --enc-arch psm \
   --dec-arch pwcdc \
   --disp-refinement hourglass \
   --flow-dec-arch pwcdc \
   --flow-no-ppm \
   --flow-refinement none \
   --maxdisp 192 \
   --savemodel ./output/kitti2012+kitti2015/stage1 \
   --workers 15 \
   --lr 0.001 \
   --lr-gamma 0.5 \
   --lr-steps 400 800 1000 1200 1400 1500 \
   --epochs 1500 \
   --bn-type syncbn \
   --batch-size 8 \
   --corr-radius 4  \
   --disp-crop-imh 320 \
   --disp-crop-imw 768 \
   --flow-crop-imh 320 \
   --flow-crop-imw 768 \
   --disp-loss-weight 1 \
   --flow-loss-type l1_robust \
   --do-seg-distill \
   --saved-seg-res-dir ${PSEUDO_SEG_GT_DIR} \
   --soft-seg-loss-weight 1 \
   --seg-distill-T 1 \
   --pseudo-gt-dir ${PSEUDO_OCC_GT_DIR} \
   --soft-occ-gt \
   --occ-loss-wt 0.05 \
   --disp-occ-penalty 0.5 \
   --flow-occ-penalty 0.5 \
   --disp-photo-consist-wt 0.5 \
   --flow-photo-consist-wt 0.5 \
   --disp-semantic-consist-wt 0.5 \
   --flow-semantic-consist-wt 0.5 \
   --mask-semi-loss \
   --flow-ssim-wt 0.01 \
   --disp-ssim-wt 0.005 \
   --print-freq 10 \
   --save-freq 100 \
   --loadmodel $PRETRAINED_MODEL

PRETRAINED_MODEL=output/kitti2012+kitti2015/stage1/kitti2012+kitti2015_joint_psm_pwcdc_wppm_hourglass_reg_pwcdc_woppm_none_md_4_syncbn_wocc_wocatocc_sum_loss_dispCropSize_320x768_flowDimRatio_None_flowCropSize_320x768_woseg_segDistill_distT_1.0_pseudoGt_softOccGt/dpcWt_0.5_dscWt_0.5_fpcWt_0.5_fscWt_0.5_dtcWt_-1_fdcWt_-1_fsWt_-1_dsWt_-1_flowSsimWt_0.01_dispSsimWt_0.005_fop_0.5_dop_0.5/model_1500.pth

python tools/train_joint_kitti.py finetune \
   --dataset kitti2012+kitti2015 \
   --enc-arch psm \
   --dec-arch pwcdc \
   --disp-refinement hourglass \
   --flow-dec-arch pwcdc \
   --flow-no-ppm \
   --flow-refinement none \
   --maxdisp 192 \
   --savemodel kitti2012+kitti2015/stage2 \
   --workers 15 \
   --lr 0.0002 \
   --lr-gamma 0.5 \
   --lr-steps 400 600 800 900 1000 \
   --epochs 1000 \
   --bn-type syncbn \
   --batch-size 8 \
   --corr-radius 4  \
   --disp-crop-imh 320 \
   --disp-crop-imw 768 \
   --flow-crop-imh 320 \
   --flow-crop-imw 768 \
   --disp-loss-weight 1 \
   --flow-loss-type l1_robust \
   --do-seg-distill \
   --saved-seg-res-dir ${PSEUDO_SEG_GT_DIR} \
   --soft-seg-loss-weight 1 \
   --seg-distill-T 1 \
   --pseudo-gt-dir ${PSEUDO_OCC_GT_DIR} \
   --soft-occ-gt \
   --occ-loss-wt 0.05 \
   --disp-occ-penalty 0.5 \
   --flow-occ-penalty 0.5 \
   --disp-photo-consist-wt 0.5 \
   --flow-photo-consist-wt 0.5 \
   --disp-semantic-consist-wt 0.5 \
   --flow-semantic-consist-wt 0.5 \
   --mask-semi-loss \
   --flow-ssim-wt 0.01 \
   --disp-ssim-wt 0.005 \
   --print-freq 10 \
   --save-freq 100 \
   --loadmodel $PRETRAINED_MODEL
