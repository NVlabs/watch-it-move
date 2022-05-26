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

import argparse
import os
import pickle
import sys
from typing import List

import cv2
import numpy as np
import torch
from easydict import EasyDict as edict
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.append(".")
from models.model import SingleVideoPartDecomposition
from utils.get_args import get_args_jupyter
from utils.train_utils import to_gpu, create_dataloaders


def generate_video_from_pose(model: nn.Module, train_loader: DataLoader, num_video_frames: int, camera_id: int,
                             joint_rotation: torch.Tensor, joint_translation: torch.Tensor,
                             rotate: bool = False, frame_id: int = 0):
    """

    Args:
        model:
        train_loader:
        num_video_frames:
        camera_id:
        joint_rotation:
        joint_translation:
        rotate:
        frame_id:

    Returns:

    """
    num_frames = train_loader.dataset.num_frames
    minibatch = train_loader.dataset[camera_id * num_frames]
    minibatch = {k: torch.tensor(v) for k, v in minibatch.items()}
    minibatch = to_gpu(minibatch)
    camera_rotation = minibatch["camera_rotation"][None]
    camera_translation = minibatch["camera_translation"][None]
    inv_intrinsics = torch.inverse(minibatch["camera_intrinsic"])[None]
    img = minibatch["img"].cpu().numpy()

    video = []

    model.eval()
    rotate_angle = 0
    frame_interval = 5
    with torch.no_grad():
        for i in tqdm(range(num_video_frames)):
            frame_id += frame_interval
            if rotate:
                rotate_angle = i / 20 * (2 * np.pi)
            if frame_id >= len(joint_translation):
                break
            out_dict = model.render_entire_img(None, camera_rotation, camera_translation,
                                               inv_intrinsics, segmentation_label=False,
                                               rotate_angle=rotate_angle, ray_batchsize=10000,
                                               part_pose=(joint_rotation[frame_id], joint_translation[frame_id]))
            color = out_dict["rendered_colors"]
            mask = out_dict["rendered_masks"]
            segmentation = out_dict["segmentation_colors"]

            color = (color + (1 - mask[None])).cpu().numpy().transpose(1, 2, 0)
            segmentation = (segmentation + (1 - mask[None])).cpu().numpy().transpose(1, 2, 0)
            color = np.concatenate([img.transpose(1, 2, 0), color, segmentation], axis=1)
            video.append(np.clip(color * 127.5 + 127.5, 0, 255).astype("uint8"))

    return video


def save_video(frames: List[np.ndarray], file_name: str, fps: int = 10, n_repeat: int = 10):
    """

    Args:
        frames:
        file_name:
        fps:
        n_repeat:

    Returns:

    """
    size = (frames[0].shape[-2], frames[0].shape[-3])

    fmt = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
    writer = cv2.VideoWriter(file_name, fmt, fps, size)

    for i in range(n_repeat):
        for frame in frames:
            writer.write(frame[:, :, ::-1])

    writer.release()


def save_png(frames: List[np.ndarray], dir_name: str):
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
        cv2.imwrite(f'{dir_name}/gen_{i:0>5}.png', frame[:, img_size:img_size * 2, ::-1])
        cv2.imwrite(f'{dir_name}/seg_{i:0>5}.png', frame[:, img_size * 2:, ::-1])


def get_mapping(model: nn.Module, config: edict, train_loader: DataLoader, use_smpl_verts: bool) -> torch.Tensor:
    """

    Args:
        model:
        config:
        train_loader:
        use_smpl_verts:

    Returns:

    """
    with torch.no_grad():
        num_frames = config.dataset.num_frames
        coordinate_scale = train_loader.dataset.coordinate_scale
        frame_id = torch.arange(config.dataset.num_frames, dtype=torch.float, device="cuda")
        trajectory = model.joint_trajectory(frame_id)

        rotation, translation = trajectory
        video_len = rotation.shape[0]

        child_root, self_root = model.decoder.joint_root_locations(rotation, translation)
        estimated_keypoints = torch.cat([child_root, self_root], dim=-1)

        estimated_keypoints = estimated_keypoints.permute(0, 1, 3, 2)

        if use_smpl_verts:
            smpl_verts_path = os.path.join(config.dataset.data_root, "smpl_verts.pickle")
            with open(smpl_verts_path, "rb") as f:
                smpl_keypoints = pickle.load(f)["smpl_verts"]
            smpl_keypoints = smpl_keypoints / coordinate_scale
            smpl_keypoints = torch.tensor(smpl_keypoints, device=rotation.device, dtype=torch.float)
            smpl_keypoints = smpl_keypoints[:num_frames]

        else:
            smpl_pose = torch.tensor(train_loader.dataset.video_cache["smpl_pose"], device=rotation.device,
                                     dtype=torch.float).clone()
            smpl_pose[:, :, :3, 3] /= coordinate_scale

            smpl_keypoints = smpl_pose[:num_frames, :, :3].clone()
            smpl_keypoints[:, :, :3, :3] *= 0.1
            smpl_keypoints[:, :, :3, :3] += smpl_keypoints[:, :, :3, 3:]
            smpl_keypoints = smpl_keypoints.permute(0, 1, 3, 2).reshape(video_len, -1, 3)

        _estimated_keypoints = estimated_keypoints.reshape(video_len, -1, 3).permute(0, 2, 1).reshape(video_len * 3,
                                                                                                      -1).cpu()
        _smpl_keypoints = smpl_keypoints.permute(0, 2, 1).reshape(video_len * 3, -1).cpu()

        lam = 5e-1 if use_smpl_verts else 1e-1
        s2j_mapping = torch.inverse(
            _smpl_keypoints.T.matmul(_smpl_keypoints) + torch.eye(_smpl_keypoints.shape[1]) * lam).matmul(
            _smpl_keypoints.T).matmul(_estimated_keypoints)

    return s2j_mapping


def repose(config_path: str, default_config: str, driving_person_id: int, num_video_frames: int, use_smpl_verts: bool
           ) -> None:
    """

    Args:
        config_path:
        default_config:
        driving_person_id:
        num_video_frames:
        use_smpl_verts: use smpl vertices for regression or not

    Returns:

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

    same_person = driving_person_id == -1

    # model
    model = SingleVideoPartDecomposition(config.network_params)
    model.cuda()

    save_dir = os.path.join(out_dir, "result", exp_name)
    model_path = os.path.join(save_dir, "snapshot_latest.pth")
    if os.path.exists(model_path):
        snapshot = torch.load(model_path)
        state_dict = snapshot["model"]
        model.load_state_dict(state_dict, strict=False)
    else:
        print("model is not loaded")

    train_loader.dataset.current_max_frame_id = train_loader.dataset.num_frames

    num_frames = train_loader.dataset.num_frames
    coordinate_scale = train_loader.dataset.coordinate_scale

    s2j_mapping = get_mapping(model, config, train_loader, use_smpl_verts)

    if use_smpl_verts:
        if same_person:
            data_root = config.dataset.data_root
        else:
            person_id = str(driving_person_id)
            data_root = f"../data/zju_mocap/cache512/{person_id}/"
        smpl_verts_path = os.path.join(data_root, "smpl_verts.pickle")
        with open(smpl_verts_path, "rb") as f:
            smpl_keypoints = pickle.load(f)["smpl_verts"]
        smpl_keypoints = smpl_keypoints / coordinate_scale
        smpl_translation = torch.tensor(smpl_keypoints, dtype=torch.float)
    else:
        if same_person:
            smpl_pose = torch.tensor(train_loader.dataset.video_cache["smpl_pose"], dtype=torch.float).clone()
        else:
            # read other smpl sequence
            person_id = str(driving_person_id)
            with open(f"../data/zju_mocap/cache512/{person_id}/cache_train.pickle", "rb") as f:
                smpl_pose = torch.tensor(pickle.load(f)["smpl_pose"], dtype=torch.float).clone()

        smpl_pose[:, :, :3, 3] /= coordinate_scale

        smpl_translation = smpl_pose[:, :, :3].clone()
        smpl_translation[:, :, :3, :3] *= 0.1
        smpl_translation[:, :, :3, :3] += smpl_translation[:, :, :3, 3:]
        smpl_translation = smpl_translation.permute(0, 1, 3, 2).reshape(-1, 23 * 4, 3)

    estimated_367_translation = s2j_mapping.T @ smpl_translation  # (L, 140, 3)

    # org pose
    with torch.no_grad():
        child_root_org, self_root_org = model.decoder.joint_root_locations(torch.eye(3, device="cuda",
                                                                                     dtype=torch.float),
                                                                           torch.zeros(model.num_parts, 3, 1,
                                                                                       device="cuda",
                                                                                       dtype=torch.float))
        estimated_keypoints_org = torch.cat([child_root_org, self_root_org], dim=-1)
        estimated_keypoints_org = estimated_keypoints_org.permute(0, 2, 1).cpu()

    estimated_367_translation = estimated_367_translation.reshape(estimated_367_translation.shape[0], model.num_parts,
                                                                  7, 3)
    estimated_367_translation_centered = estimated_367_translation - estimated_367_translation[:, :, -1:]
    U, S, Vh = torch.linalg.svd(estimated_367_translation_centered.permute(0, 1, 3, 2) @ estimated_keypoints_org)
    R = Vh.permute(0, 1, 3, 2) @ U.permute(0, 1, 3, 2)
    det = torch.linalg.det(R)
    Vh[:, :, 2] = Vh[:, :, 2] * det[:, :, None]
    R = Vh.permute(0, 1, 3, 2) @ U.permute(0, 1, 3, 2)

    frame_id = num_frames if same_person else 0
    camera_id = 0
    rotate = False

    video = generate_video_from_pose(model, train_loader, num_video_frames, camera_id,
                                     rotate=rotate,
                                     joint_rotation=R.permute(0, 1, 3, 2).cuda(),
                                     joint_translation=estimated_367_translation[:, :, -1, :, None].cuda(),
                                     frame_id=frame_id)
    save_video(video, os.path.join(
        save_dir,
        'drive_' + 'verts_' * use_smpl_verts + f'{driving_person_id}' * ~same_person + f'_{camera_id}.mp4'))

    save_png(video, os.path.join(
        save_dir,
        'drive_' + 'verts_' * use_smpl_verts + f'{driving_person_id}' * ~same_person + f'_{camera_id}'))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Create reposing videos with test poses')
    parser.add_argument('--exp_name', required=True, type=str)
    parser.add_argument('--camera_id', required=True, type=int)
    parser.add_argument('--num_video_frames', type=int, default=100)
    parser.add_argument('--driving_person_id', type=int, default=-1,
                        help="Driving person id is same as input person id if -1")
    args = parser.parse_args()

    exp_name = args.exp_name
    num_video_frames = args.num_video_frames
    driving_person_id = args.driving_person_id

    default_config = "confs/default.yml"

    config_path = f"confs/{exp_name}.yml"
    repose(config_path, default_config, driving_person_id, num_video_frames, use_smpl_verts=True)
