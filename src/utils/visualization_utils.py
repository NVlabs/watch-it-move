"""
SPDX-FileCopyrightText: Copyright (c) 2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: LicenseRef-NvidiaProprietary
NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
property and proprietary rights in and to this material, related
documentation and any modifications thereto. Any use, reproduction,
disclosure or distribution of this material and related documentation
without an express license agreement from NVIDIA CORPORATION or
its affiliates is strictly prohibited.
"""

import os
import sys
from copy import deepcopy
from typing import List, Union, Dict, Optional, Tuple

import cv2
import numpy as np
import torch
from torch import nn
from tqdm import tqdm
from torch.utils.data import DataLoader

sys.path.append(".")
from models.model import SingleVideoPartDecomposition
from utils.get_args import get_args_jupyter
from utils.train_utils import create_dataloaders
from utils.train_utils import to_gpu


def save_video(frames: List[np.ndarray], file_name: str, fps: int = 10, n_repeat: int = 1) -> None:
    """

    Args:
        frames:
        file_name:
        fps:
        n_repeat:

    Returns:

    """
    size = (frames[0].shape[-2], frames[0].shape[-3])

    os.makedirs(os.path.dirname(file_name), exist_ok=True)
    fmt = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
    writer = cv2.VideoWriter(file_name, fmt, fps, size)

    for i in range(n_repeat):
        for frame in frames:
            writer.write(frame[:, :, ::-1])

    writer.release()


def save_gt_gen_seg(frames: List[np.ndarray], dir_name: str) -> None:
    """

    Args:
        frames:
        dir_name:

    Returns:

    """
    os.makedirs(dir_name, exist_ok=True)
    for i, frame in enumerate(frames):
        img_size = frame.shape[0]
        cv2.imwrite(f'{dir_name}/gt_{i:0>5}.png', frame[:, :img_size, ::-1])
        if frame.shape[1] > img_size:
            cv2.imwrite(f'{dir_name}/gen_{i:0>5}.png', frame[:, img_size:img_size * 2, ::-1])
            cv2.imwrite(f'{dir_name}/seg_{i:0>5}.png', frame[:, img_size * 2:, ::-1])


def generate_video(model: nn.Module, train_loader: DataLoader, frame_interval: int, num_video_frames: int,
                   frame_id: int, camera_id: int, increment_frame_id: bool = False, rotate: bool = False,
                   gradual_manipulation: bool = False,
                   manipulate_pose_config: Optional[Union[List[Dict[str, Union[str, list]]], List[None]]] = None,
                   is_dog: bool = False) -> List[np.ndarray]:
    """

    Args:
        model:
        train_loader:
        frame_interval:
        num_video_frames:
        frame_id:
        camera_id:
        increment_frame_id:
        rotate:
        gradual_manipulation:
        manipulate_pose_config:
        is_dog:

    Returns:

    """
    num_frames = train_loader.dataset.num_frames
    minibatch = train_loader.dataset[frame_id + camera_id * num_frames]
    minibatch = {k: torch.tensor(v) for k, v in minibatch.items()}
    minibatch = to_gpu(minibatch)
    camera_rotation = minibatch["camera_rotation"][None]
    camera_translation = minibatch["camera_translation"][None]
    inv_intrinsics = torch.inverse(minibatch["camera_intrinsic"])[None]
    frame_id = minibatch["frame_id"][None]
    camera_id = minibatch["camera_id"][None]

    video = []

    model.eval()
    rotate_angle = 0
    if not isinstance(manipulate_pose_config, list):
        manipulate_pose_config = [manipulate_pose_config]
    for m, mani_config in enumerate(manipulate_pose_config):
        with torch.no_grad():
            for i in tqdm(range(num_video_frames)):
                if increment_frame_id:
                    frame_id = frame_id + frame_interval

                minibatch = train_loader.dataset[frame_id[0].item() + camera_id[0].item() * num_frames]
                img = minibatch["img"][None]
                if is_dog:
                    camera_rotation = torch.tensor(minibatch["camera_rotation"][None], device="cuda")
                    camera_translation = torch.tensor(minibatch["camera_translation"][None], device="cuda")
                    inv_intrinsics = torch.inverse(torch.tensor(minibatch["camera_intrinsic"], device="cuda"))[None]

                if rotate:
                    rotate_angle = 2 * np.pi * (i + m * num_video_frames) / (
                            num_video_frames * len(manipulate_pose_config))
                if gradual_manipulation:
                    manipulate_pose_config_i = deepcopy(mani_config)
                    for conf in manipulate_pose_config_i["motion_config"]:
                        if "init_rodrigues" in conf:
                            conf["rodrigues"] = conf["rodrigues"] * i / (num_video_frames - 1) + conf[
                                "init_rodrigues"] * (1 - i / (num_video_frames - 1))
                        else:
                            conf["rodrigues"] = conf["rodrigues"] * i / (num_video_frames - 1)
                else:
                    manipulate_pose_config_i = mani_config
                out_dict = model.render_entire_img(frame_id, camera_rotation, camera_translation,
                                                   inv_intrinsics, segmentation_label=False,
                                                   rotate_angle=rotate_angle, ray_batchsize=10000,
                                                   manipulate_pose_config=manipulate_pose_config_i)
                color = out_dict["rendered_colors"]
                mask = out_dict["rendered_masks"]
                segmentation = out_dict["segmentation_colors"]

                color = (color + (1 - mask[None])).cpu().numpy().transpose(1, 2, 0)
                segmentation = (segmentation + (1 - mask[None])).cpu().numpy().transpose(1, 2, 0)
                color = np.concatenate([img[0].transpose(1, 2, 0), color, segmentation], axis=1)
                video.append(np.clip(color * 127.5 + 127.5, 0, 255).astype("uint8"))

    return video


def create_mani_config(root: int, first: List[List], second: List[List]
                       ) -> List[Dict[str, Union[str, list]]]:
    """

    Args:
        root:
        first:
        second:

    Returns:

    """
    motion_config_1 = []
    for move_id, angle in first:
        motion_config_1.append(
            {"move_id": move_id, "rodrigues": np.array(angle), "init_rodrigues": np.array([0, 0, 0])})

    motion_config_2 = []
    for move_id, angle in first:
        motion_config_2.append({"move_id": move_id, "rodrigues": np.array(angle), "init_rodrigues": np.array(angle)})
    for move_id, angle in second:
        motion_config_2.append(
            {"move_id": move_id, "rodrigues": np.array(angle), "init_rodrigues": np.array([0, 0, 0])})

    motion_config_3 = []
    for move_id, angle in first:
        motion_config_3.append(
            {"move_id": move_id, "rodrigues": np.array([0, 0, 0]), "init_rodrigues": np.array(angle)})
    for move_id, angle in second:
        motion_config_3.append({"move_id": move_id, "rodrigues": np.array(angle), "init_rodrigues": np.array(angle)})

    motion_config_4 = []
    for move_id, angle in first:
        motion_config_4.append(
            {"move_id": move_id, "rodrigues": np.array([0, 0, 0]), "init_rodrigues": np.array([0, 0, 0])})
    for move_id, angle in second:
        motion_config_4.append(
            {"move_id": move_id, "rodrigues": np.array([0, 0, 0]), "init_rodrigues": np.array(angle)})
    motion_configs = [motion_config_1, motion_config_2, motion_config_3, motion_config_4]
    mani_conf = [{"root_id": root, "motion_config": mc} for mc in motion_configs]

    return mani_conf


class GenerateVideoFromConfig:
    def __init__(self, config_path: str, default_config: str, iteration: int = -1) -> None:
        """

        Args:
            config_path:
            default_config:
        """
        args, config = get_args_jupyter(config_path, default_config)
        config.dataset.batchsize = 1

        train_loader = create_dataloaders(config.dataset, shuffle=True)

        train_loader.dataset.n_repetition_in_epoch = 1
        train_loader.dataset.color_augmentation = False
        train_loader.dataset.camera_dir_augmentation = False
        train_loader.dataset.background_color = 1

        out_dir = config.output_dir
        exp_name = config.exp_name

        # model
        model = SingleVideoPartDecomposition(config.network_params)
        model.cuda()

        save_dir = os.path.join(out_dir, "result", exp_name)
        if iteration == -1:
            model_path = os.path.join(save_dir, "snapshot_latest.pth")
        else:
            model_path = os.path.join(save_dir, f"snapshot_{iteration}.pth")
        if os.path.exists(model_path):
            snapshot = torch.load(model_path)
            state_dict = snapshot["model"]
            iteration = snapshot["iteration"]
            model.load_state_dict(state_dict, strict=False)
        else:
            assert False, "model is not loaded"

        train_loader.dataset.current_max_frame_id = train_loader.dataset.num_frames
        self.model = model
        self.train_loader = train_loader
        self.save_dir = save_dir
        self.exp_name = exp_name
        self.iteration = iteration
        self.is_dog = "dog" in config_path

    def __call__(self, frame_id: int = 0, camera_id: int = 4, rotate: bool = False, increment: bool = False,
                 num_video_frames: int = 50) -> None:
        """

        Args:
            frame_id:
            camera_id:
            rotate:
            increment:
            num_video_frames:

        Returns:

        """
        frame_interval = 5 if "zju" in self.exp_name or "dog" in self.exp_name else 5

        video = generate_video(self.model, self.train_loader, frame_interval, num_video_frames, frame_id, camera_id,
                               increment_frame_id=increment, rotate=rotate,
                               gradual_manipulation=False, manipulate_pose_config=[None], is_dog=self.is_dog)

        save_video(video, os.path.join(
            self.save_dir,
            'reconstruction_' + 'rotate_' * rotate + 'motion_' * increment + f'{frame_id}_{camera_id}.mp4'))
        save_gt_gen_seg(video, os.path.join(
            self.save_dir,
            'reconstruction_' + 'rotate_' * rotate + 'motion_' * increment + f'{frame_id}_{camera_id}'))

    def repose(self, root: int, first: List[List], second: List[List], frame_id: int = 393, camera_id: int = 4,
               rotate: bool = False, num_video_frames: int = 20) -> None:
        """

        Args:
            root:
            first:
            second:
            frame_id:
            camera_id:
            rotate:
            num_video_frames:

        Returns:

        """
        frame_interval = 8 if "zju" in self.exp_name else 2

        manipulate_pose_config = create_mani_config(root, first, second)

        video = generate_video(self.model, self.train_loader, frame_interval, num_video_frames, frame_id, camera_id,
                               increment_frame_id=False, rotate=rotate,
                               gradual_manipulation=True,
                               manipulate_pose_config=manipulate_pose_config, is_dog=self.is_dog)

        save_video(video, os.path.join(self.save_dir,
                                       'repose_' + 'rotate_' * rotate + f'{frame_id}_{camera_id}.mp4'))
        save_gt_gen_seg(video, os.path.join(self.save_dir,
                                            'repose_' + 'rotate_' * rotate + f'{frame_id}_{camera_id}'))


def render_and_get_joints(model: nn.Module, loader: DataLoader, frame_id: int, camera_id: int,
                          manipulate_pose_config: Optional[List] = None
                          ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor,
                                     torch.Tensor, np.ndarray, np.ndarray, np.ndarray, None]:
    """

    Args:
        model:
        loader:
        frame_id:
        camera_id:
        manipulate_pose_config:

    Returns:

    """
    num_frames = loader.dataset.num_frames
    minibatch = loader.dataset[frame_id + camera_id * num_frames]
    minibatch = {k: torch.tensor(v) for k, v in minibatch.items()}
    minibatch = to_gpu(minibatch)
    img = minibatch["img"][None]
    gt_mask = minibatch["mask"][None]
    camera_rotation = minibatch["camera_rotation"][None]
    camera_translation = minibatch["camera_translation"][None]
    inv_intrinsics = torch.inverse(minibatch["camera_intrinsic"])[None]
    frame_id = minibatch["frame_id"][None]

    model.eval()
    #     model.train()

    with torch.no_grad():
        rendered_dict = model.render_entire_img(frame_id, camera_rotation, camera_translation,
                                                inv_intrinsics, segmentation_label=False,
                                                ray_batchsize=16384,
                                                rotate_angle=0, manipulate_pose_config=manipulate_pose_config)
        color = rendered_dict["rendered_colors"]
        mask = rendered_dict["rendered_masks"]
        disparity = rendered_dict["rendered_disparities"]
        segmentation = rendered_dict["segmentation_colors"]

        joint_2d = model.joint_2d.cpu().numpy()
        child_root = model.child_root.cpu().numpy()
        self_root = model.self_root.cpu().numpy()
        bg = None

    return img, gt_mask, color, mask, disparity, segmentation, joint_2d, child_root, self_root, bg
