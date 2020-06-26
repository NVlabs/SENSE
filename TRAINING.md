# Training of SENSE

### Before Getting Started
Since we use SynchronizedBatchNorm, use as few GPUs as possible depending on how much GPU memory you have. Otherwise, you will see significantly slow training.

### Pre-training on the synthetic SceneFlow dataset
Download FlyingThings3D_subset (**not the regular FlyingThings3D**), Driving, and Monkaa data from https://lmb.informatik.uni-freiburg.de/resources/datasets/SceneFlowDatasets.en.html. Suppose the downloaded data are put under the `SceneFlow` folder. We expect the following folder structure.
```
SceneFlow
|-- Driving
|   |-- disparity
|   |-- frames_cleanpass
|   |-- optical_flow
|-- FlyingThings3D_subset
|   |-- train
|   |   |-- disparity
|   |   |-- disparity_occlusions
|   |   |-- flow
|   |   |-- flow_occlusions
|   |   |-- image_clean
|   |-- val
|   |   |-- disparity
|   |   |-- disparity_occlusions
|   |   |-- flow
|   |   |-- flow_occlusions
|   |   |-- image_clean
|-- Monkaa
|   |-- disparity
|   |-- frames_cleanpass
|   |-- optical_flow
```
Configure the datasets' paths accordingly in `sense/datasets/dataset_catlog.py`.

Run `sh scripts/train_synthetic_sceneflow.sh` to start training the model from scratch. This model consists of estimations for optical flow, stereo disparity, and occlusions.

### Fine-tuning on MPI Sintel
Download both optical flow (http://sintel.is.tue.mpg.de/downloads) and stereo disparity (http://sintel.is.tue.mpg.de/stereo) data first. We expect the following folder structure.
```
|-- MPI_Sintel
|   |-- training
|   |-- test
|   |-- stereo
|   |   |-- training
```
After configuring the dataset's path in `sense/datasets/dataset_catlog.py`, we are ready to fine-tune the model that has been trained on the synthetic SceneFlow dataset introduced in the previous section. Simply run `sh scripts/train_mpi_sintel.sh`.

### Fine-tuning on KITTI (2012+2015)
Fine-tuning on KITTI2012 and KITTI2015 is not that straightforward. It consists of couple of steps.

#### Prepare Data
Again, download the data for KITTI2012 (http://www.cvlibs.net/datasets/kitti/eval_stereo_flow.php?benchmark=flow) and KITTI2015 (http://www.cvlibs.net/datasets/kitti/eval_scene_flow.php) first. Particularly, for KITTI2015, we need the camera calibration data. We expect the following folder structure.
```
KITTI2012
|-- training
|-- testing

KITTI2015
|-- training
|-- testing
```

#### Prepare Pseudo Ground-truth data
Semi-supervised training of SENSE requires pseudo ground-truth data for occlusion detection and semantic segmentation. For occlusion, we use the output of the model that is pre-trained on the synthetic SceneFlow dataset as pseudo ground-truth. For semantic segmentation, we use the model output from a ResNet101-UPerNet that is first trained on CityScapes and then fine-tuned on KITTI.  

First, clone the UPerNet code from https://github.com/playerkk/semantic-segmentation-pytorch. After configuring the variables in `scripts/prepare_pseudo_gt.sh`, simply run `sh scripts/prepare_pseudo_gt.sh`. **Note this step may take around 53GB space to store the results (mainly for semantic segmentation).**

Configure `PSEUDO_OCC_GT_DIR` and `PSEDU_SEG_GT_DIR` in the `scripts\train_kitti_holistic_model.sh` file then accordingly.

#### Training a Holistic Scene Model
Run `sh scripts\train_kitti_holistic_model.sh` to train a model for a set of holistic scene understanding tasks, including estimation of optical flow and stereo disparity, occlusion detection, and semantic segmentation. 

#### Training a Warped Disparity Refinement Model
We then need to train another neural network for wapred disparity refinement, which is essential for obtaining scene flow estimation results. Run `sh scripts/train_kitti2015_warped_disp_refine.sh`.