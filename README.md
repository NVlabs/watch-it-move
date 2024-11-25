# Watch It Move

Official implementation of the IEEE/CVF CVPR 2022 paper

**Watch It Move: Unsupervised Discovery of 3D Joints for Re-Posing of Articulated Objects**\
Atsuhiro Noguchi, Umar Iqbal, Jonathan Tremblay, Tatsuya Harada, Orazio Gallo\
[Project page](https://nvlabs.github.io/watch-it-move/) / [Paper](https://arxiv.org/abs/2112.11347)
/ [Video](https://www.youtube.com/watch?v=oRnnuCVV89o)

Abstract: Rendering articulated objects while controlling their poses is critical to applications such as virtual
reality or animation for movies. Manipulating the pose of an object, however, requires the understanding of its
underlying structure, that is, its joints and how they interact with each other. Unfortunately, assuming the structure
to be known, as existing methods do, precludes the ability to work on new object categories. We propose to learn both
the appearance and the structure of previously unseen articulated objects by observing them move from multiple views,
with no joints annotation supervision, or information about the structure. We observe that 3D points that are static
relative to one another should belong to the same part, and that adjacent parts that move relative to each other must be
connected by a joint. To leverage this insight, we model the object parts in 3D as ellipsoids, which allows us to
identify joints. We combine this explicit representation with an implicit one that compensates for the approximation
introduced. We show that our method works for different structures, from quadrupeds, to single-arm robots, to humans.

## Table of content
  * [Setup](#setup)
  * [Steps to replicate the teaser video for spot](#steps-to-replicate-the-teaser-video-for-spot)
  * [Steps to train for spot](#steps-to-train-for-spot)
  * [The WIM dataset](#the-wim-dataset)
  * [Dataset Preprocessing](#dataset-preprocessing)
  * [Training](#training)
  * [Pretrained Models](#pretrained-models)
  * [Demo](#demo)
  * [Evaluation (ZJU only)](#evaluation--zju-only-)
  * [Visualization](#visualization)
  * [Citation](#citation)

## Setup
Clone this repository and create the environment.
```angular2html
git clone --recursive git@github.com:NVlabs/watch-it-move.git
cd watch-it-move
bash install.sh

# To run the training and rendering examples below, download the data for Spot
mkdir -p data/robots/spot
gdown https://drive.google.com/u/1/uc\?id\=1HNzCa8olJgedpKe6jBCIi-_LffLX9f8R\&export\=download -O data/robots/spot/cache.pickle
```
### Disclaimer
We have only tested the following code on NVIDIA A100 GPUs.

## Steps to replicate the teaser video for spot

```angular2html
# download pretrained model for spot
mkdir -p data/output/result/spot_merge
gdown https://drive.google.com/u/1/uc\?id\=12_K-x-daAGqvIoDd3tRvRIe0LKLOALyC\&export\=download -O data/output/result/spot_merge/snapshot_latest.pth

cd <project_root>/src
# save the reconstruction video
python visualize/create_reconstruction_video.py --exp_name spot_merge
# save re-pose video
python visualize/create_repose_video.py --exp_name spot_merge --repose_config visualize/repose_configs/spot.yml --rotate
```
Videos in mp4 format and the png image for each frame will be saved to `<project_root>/data/output/result/spot_merge/`

<center>
<img src=figures/reconstruction_rotate_motion_0_0.gif width=50%><img src=figures/repose_rotate_0_0.gif width=50%>
</center>

## Steps to train for spot
```angular2html
cd <project_root>/src
CUDA_VISIBLE_DEVICES=[gpu_id] python train_single_video.py --config confs/spot.yml --default_config confs/default.yml
CUDA_VISIBLE_DEVICES=[gpu_id] python train_single_video.py --config confs/spot_merge.yml --default_config confs/default.yml
```
## The WIM dataset
<img src=figures/robot_example.jpg width=450px>

We provide multiview videos for seven different moving robots [here](https://drive.google.com/drive/folders/1i5rWanA8FgVLrWPO4bl0aaGKBYwhY6IQ) (see [LICENSE.md](LICENSE.md) for terms of use).
We provide both raw video data and preprocessed data. Please follow the instructions bellow to download and preprocess the data.
It includes: 1000 frame videos of moving robots from 20 different viewpoints and preprocessed data of 300 frames of 5 chosen viewpoints.


## Dataset Preprocessing
### WIM Dataset
- The WIM dataset is available [here](https://drive.google.com/drive/folders/1i5rWanA8FgVLrWPO4bl0aaGKBYwhY6IQ).
- We provide preprocessed data in the directory named [preprocessed](https://drive.google.com/drive/folders/1toiwb06VggqH1FOS9OnKYlqRFk3H6T9g). Download, uncompress, and place it in
  ```angular2html
  <project_root>data/robots/<name_of_robot>/cache.pickle
  ```
- If you run with pre-processing on your own, download tar.gz files from [here](https://drive.google.com/drive/folders/1i5rWanA8FgVLrWPO4bl0aaGKBYwhY6IQ), uncompress them, place them as
  ```
  <project_root>data/robots/<name_of_robot>/cam_<camera_id>.json
  <project_root>data/robots/<name_of_robot>/frame_<frame_id>_cam_<camera_id>.png
  ```
  and run
  ```angular2html
  cd <project_root>/data_preprocess/robot
  python preprocess.py --data_root ../../data/robots --robot_name atlas --robot_name baxter --robot_name spot --robot_name cassie --robot_name iiwa --robot_name nao --robot_name pandas
  ```

### ZJU MOCAP

- Requirements (installed by `install.sh`): [Adet](https://github.com/aim-uofa/AdelaiDet), [EasyMocap](https://github.com/zju3dv/EasyMocap)

- Download the COCO instance segmentation model
  named [R_101_dcni3_5x](https://github.com/aim-uofa/AdelaiDet#coco-instance-segmentation-baselines-with-blendmask) from
  Adet and copy it to `data_preprocess/zju/R_101_dcni3_5x.pth`.
- Download the [ZJU MOCAP LightStage dataset](https://github.com/zju3dv/EasyMocap#zju-mocap) and copy it in
  ```
  <project_root>/data/
                 └── zju_mocap
                     ├── 366
                     ├── 377
                     ├── 381
                     ├── 384
                     └── 387
  ```
- Download the SMPL models
  following [EasyMocap installation](https://github.com/zju3dv/EasyMocap/blob/master/doc/installation.md). You only need to download smplx models.
  ```
  <project_root>/data
                 └── smplx
                     ├── J_regressor_body25.npy
                     ├── J_regressor_body25_smplh.txt
                     ├── J_regressor_body25_smplx.txt
                     ├── J_regressor_mano_LEFT.txt
                     ├── J_regressor_mano_RIGHT.txt
                     └── smplx
                         ├── SMPLX_FEMALE.pkl
                         ├── SMPLX_MALE.pkl
                         └── SMPLX_NEUTRAL.pkl
  ```
- Run
  ```angular2html
  cd <project_root>/data_preprocess/zju
  python preprocess.py --smpl_model_path ../../data/smplx --zju_path ../../data/zju_mocap --person_id 366 --person_id 377 --person_id 381 --person_id 384 --person_id 387
  ```

### Dog dataset
- Requirement (installed by `install.sh`): [mask2former](https://github.com/facebookresearch/Mask2Former)
- Download `Mask2Former (200 queries)` model from https://github.com/facebookresearch/Mask2Former/blob/main/MODEL_ZOO.md#instance-segmentation and copy it to `<project_root>/data_preprocess/rgbd_dog/model_final_e5f453.pth`.
- Download [RGBD-Dog dataset](https://github.com/CAMERA-Bath/RGBD-Dog) as
  ```
  <project_root>/data/rgbd_dog
                 └── dog1
                     └── motion_testSeq
                         ├── kinect_depth
                         ├── kinect_rgb
                         ├── motion_capture
                         └── sony
  ```
  We used `motion_testSeq` for training.
- Run
  ```angular2html
  cd <project_root>/data_preprocess/rgbd_dog
  python preprocess.py --data_root ../../data/rgbd_dog/dog1/motion_testSeq
  ```

## Training
Run the following commands to train the model. Please specify the experiment name in `[exp_name]`
```
cd <project_root>/src
CUDA_VISIBLE_DEVICES=[gpu_id] python train_single_video.py --config confs/[exp_name].yml --default_config confs/default.yml
```

## Pretrained Models
Pretrained models for ZJU mocap dataset, robot dataset, and dog dataset are available [here](https://drive.google.com/drive/folders/1gmkkHXRr5-1w5W-kCSHcsInMY8ODEqyK).
The name of each directory corresponds to the name of a config fine under `src/confs`.
Please download and place these directories in `data/output/result`.
```angular2html
<project_root>/data/output/result
               ├── atlas
               │   └── snapshot_latest.pth
               ├── baxter
               ...
```

## Demo
Visualization code is available in `<project_root>/src/visualize/demo_notebook.ipynb`

## Evaluation (ZJU only)

### LPIPS and SSIM
Calculate lpips and ssim between generated and ground truth images.
```angular2html
cd <project_root>/src
python validation/reconsuruction.py --exp_name zju366 --exp_name zju377
python validation/lpips_ssim.py --exp_name zju366 --exp_name zju377
```
Results will be saved to `[output_dir]/result/[exp_name]/validation`

### Pose Regression
Calculate MPJPE (mm) between ground truth and regressed joint locations.
```angular2html
cd <project_root>/src
python validation/SMPL_regression.py --exp_name zju366 --exp_name zju377
```

## Visualization
### Reconstruction video
Results will be saved to `<project_root>/data/output/result/[exp_name]/reconstruction_...`.
```angular2html
cd <project_root>/src
python visualize/create_reconstruction_video.py --exp_name zju366 --exp_name zju377
```

### Manual re-posing
Results will be saved to `<project_root>/data/output/result/[exp_name]/repose_...`.
```angular2html
cd <project_root>/src
python visualize/create_repose_video.py --exp_name spot --repose_config visualize/repose_configs/spot.yml --rotate
```
repose_config (e.g., `<project_root>/src/visualize/repose_configs/spot.yml`) includes the following parameters:
```angular2html
camera_id: camera id of the reference frame
frame_id: frame id of the reference frame
root: part id of the root. 
first: part id and its rotation in rodrigues form for the first quarter of the video.
second: part id and its rotation in rodrigues form for the next quarter of the video.
```
`root`, `first`, and `second` vary depending on the training results, even when trained on the same data.
For your pretrained models, please follow `<project_root>/src/visualize/demo_notebook.ipynb` to adapt them.

### Merge Parts
Images of merged structure will be saved to `<project_root>/data/output/result/[exp_name]/merge`.
```angular2html
cd <project_root>/src
python visualize/part_merging.py --exp_name spot --camera_id 0
```

### Re-posing by driving frames (ZJU only)
Re-posing by test frames.
```angular2html
cd <project_root>/src
python visualize/repose_person_by_driving_pose.py --exp_name zju366 --camera_id 0 --num_video_frames 50
```
Results will be saved to `<project_root>/data/output/result/[exp_name]/drive_..`.

# Citation
```bibtex
@inproceedings{noguchi2022watch,
    title = {Watch It Move: {U}nsupervised Discovery of {3D} Joints for Re-Posing of Articulated Objects},
    author = {Atsuhiro Noguchi and Umar Iqbal and Jonathan Tremblay and Tatsuya Harada and Orazio Gallo},
    journal = {CVPR},
    year = {2022},
}
```
