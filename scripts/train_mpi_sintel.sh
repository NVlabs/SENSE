#!/bin/bash

set -e

export PYTHONPATH=.

PRETRAINED_MODEL=data/pretrained_models/sceneflow_joint_psm_pwcdc_wppm_hourglass_reg_pwcdc_woppm_none_md_4_upernet_wocc_wocatocc_sum_loss_dispCropSize_256x512_flowDimRatio_None_flowCropSize_384x640_96.pth

# fist stage
python3 tools/train_joint_sintel.py pre-train \
   --dataset sintel \
   --enc-arch psm \
   --dec-arch pwcdc \
   --disp-refinement hourglass \
   --flow-dec-arch pwcdc \
   --flow-no-ppm \
   --flow-refinement none \
   --maxdisp 256 \
   --savemodel ./output/mpi_sintel_stage1 \
   --workers 16 \
   --lr 0.0005 \
   --epochs 500 \
   --bn-type syncbn \
   --batch-size 8 \
   --corr-radius 4  \
   --disp-crop-imh 384 \
   --disp-crop-imw 768 \
   --flow-crop-imh 384 \
   --flow-crop-imw 768 \
   --disp-loss-weight 0.25 \
   --flow-loss-type l1_robust \
   --disp-occ-wts 1.1176 9.5046 \
   --flow-occ-wts 1.0788 13.693 \
   --upsample-flow-output \
   --robust-loss-p 0.4 \
   --print-freq 10 \
   --save-freq 10 \
   --loadmodel $PRETRAINED_MODEL

# second stage
PRETRAINED_MODEL=output/sintel_joint_psm_pwcdc_wppm_hourglass_reg_pwcdc_woppm_none_md_4_upernet_wocc_wocatocc_sum_loss_dispCropSize_384x768_flowDimRatio_None_flowCropSize_384x768_woseg/dispPhotoConsistWt_-1_dispSemanticConsistWt_-1_flowPhotoConsistWt_-1_flowSemanticConsistWt_-1_dispTemporalConsistWt_-1_flowDispConsistWt_-1_flowSmoothnessWt_-1_dispSmoothnessWt_-1_flowSsimWt_-1_dispSsimWt_-1_occAreaPenalty_0.3/model_0500.pth

python tools/train_joint_sintel.py pre-train \
   --dataset sintel \
   --enc-arch psm \
   --dec-arch pwcdc \
   --disp-refinement hourglass \
   --flow-dec-arch pwcdc \
   --flow-no-ppm \
   --flow-refinement none \
   --maxdisp 256 \
   --savemodel ./output/lr_disruption \
   --workers 16 \
   --lr 0.0002 \
   --epochs 500 \
   --bn-type syncbn \
   --batch-size 8 \
   --corr-radius 4  \
   --disp-crop-imh 384 \
   --disp-crop-imw 768 \
   --flow-crop-imh 384 \
   --flow-crop-imw 768 \
   --disp-loss-weight 0.25 \
   --flow-loss-type l1_robust \
   --disp-occ-wts 1.1176 9.5046 \
   --flow-occ-wts 1.0788 13.693 \
   --upsample-flow-output \
   --robust-loss-p 0.4 \
   --print-freq 10 \
   --save-freq 10 \
   --no-gaussian-noise \
   --loadmodel $PRETRAINED_MODEL