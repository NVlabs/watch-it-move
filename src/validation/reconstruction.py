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
from typing import Tuple, Any, Optional

import numpy as np
import torch
from easydict import EasyDict as edict
from torch import nn
from torch.utils.data import Dataset
from tqdm import tqdm

sys.path.append(".")
from models.model import SingleVideoPartDecomposition
from utils.get_args import get_args_jupyter
from utils.train_utils import to_gpu, create_dataloaders


def render(model: nn.Module, test_dataset: Dataset, data_idx: int, bg_color: np.ndarray,
           part_pose: Optional[Tuple[Any, Any]] = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """

    Args:
        model:
        test_dataset:
        data_idx:
        bg_color:
        part_pose:

    Returns:

    """
    minibatch = test_dataset[data_idx]
    minibatch = {k: torch.tensor(v) for k, v in minibatch.items()}
    minibatch = to_gpu(minibatch)
    img = minibatch["img"]
    gt_mask = minibatch["mask"]
    camera_rotation = minibatch["camera_rotation"][None]
    camera_translation = minibatch["camera_translation"][None]
    inv_intrinsics = torch.inverse(minibatch["camera_intrinsic"])[None]
    frame_id = minibatch["frame_id"][None]

    model.eval()

    with torch.no_grad():
        if part_pose is not None:
            _part_pose = (part_pose[0][frame_id], part_pose[1][frame_id])
        else:
            _part_pose = None

        rendered_dict = model.render_entire_img(frame_id, camera_rotation, camera_translation,
                                                inv_intrinsics, segmentation_label=False,
                                                ray_batchsize=16384,
                                                rotate_angle=0, manipulate_pose_config=None,
                                                part_pose=_part_pose)
        color = rendered_dict["rendered_colors"]
        mask = rendered_dict["rendered_masks"]

        color = color + (1 - mask) * bg_color

    img = img.cpu().numpy() * 127.5 + 127.5
    gt_mask = gt_mask.cpu().numpy()
    color = color.cpu().numpy() * 127.5 + 127.5
    mask = mask.cpu().numpy()

    return img, gt_mask, color, mask


def test(config_path: str, default_config: str, mode: str = "test") -> None:
    """

    Args:
        config_path:
        default_config:
        mode:

    Returns:

    """
    assert mode in ["test", "novel_pose"]

    args, config = get_args_jupyter(config_path, default_config)
    config.dataset.batchsize = 1

    config.dataset.set_name = mode  # test and novel_pose

    test_dataset = create_dataloaders(config.dataset, shuffle=True).dataset
    num_train_data = test_dataset.num_frames

    test_dataset.n_repetition_in_epoch = 1
    test_dataset.color_augmentation = False
    test_dataset.camera_dir_augmentation = False
    test_dataset.thin_out_interval = 1

    num_test_data = len(test_dataset.video_cache["img"])
    test_dataset.current_max_frame_id = 100000000
    test_dataset.num_frames = 20
    num_test_view = num_test_data // test_dataset.num_frames
    num_camera_to_use = 5
    camera_for_test = np.linspace(0, num_test_view, num_camera_to_use, endpoint=False, dtype="int")

    out_dir = config.output_dir
    exp_name = config.exp_name
    save_dir = os.path.join(out_dir, "result", exp_name)

    # model
    model = SingleVideoPartDecomposition(config.network_params)
    model.cuda()

    model_path = os.path.join(save_dir, "snapshot_latest.pth")
    if os.path.exists(model_path):
        snapshot = torch.load(model_path)
        state_dict = snapshot["model"]
        model.load_state_dict(state_dict, strict=False)
    else:
        raise FileNotFoundError()

    if mode == "novel_pose":
        part_pose = regress_learned_from_smpl(num_train_data, model, test_dataset, config, use_smpl_verts=True)
    else:
        part_pose = None

    background_color = config.dataset.background_color

    gt_imgs, gt_masks, gen_imgs, gen_masks = [], [], [], []
    for fra_idx in tqdm(range(test_dataset.num_frames)):
        for cam_idx in camera_for_test:
            data_idx = fra_idx + cam_idx * test_dataset.num_frames
            gt_img, gt_mask, gen_img, gen_mask = render(model, test_dataset, data_idx, background_color, part_pose)
            gt_imgs.append(gt_img)
            gt_masks.append(gt_mask)
            gen_imgs.append(gen_img)
            gen_masks.append(gen_mask)

    gt_imgs = np.array(gt_imgs)
    gt_masks = np.array(gt_masks)
    gen_imgs = np.array(gen_imgs)
    gen_masks = np.array(gen_masks)

    save_dict = {"gt_img": gt_imgs, "gt_mask": gt_masks, "gen_img": gen_imgs, "gen_mask": gen_masks}

    os.makedirs(f"{save_dir}/validation", exist_ok=True)
    with open(f"{save_dir}/validation/reconstruction_{mode}.pkl", "wb") as f:
        pickle.dump(save_dict, f)


def regress_learned_from_smpl(num_train_data: int, model: nn.Module, test_dataset: Dataset, config: edict,
                              use_smpl_verts: bool = True) -> Tuple[torch.Tensor, torch.Tensor]:
    """

    Args:
        num_train_data:
        model:
        test_dataset:
        config:
        use_smpl_verts:

    Returns:

    """
    with torch.no_grad():
        frame_id = torch.arange(num_train_data, dtype=torch.float, device="cuda")
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
            smpl_keypoints = smpl_keypoints / 1.5
            smpl_keypoints = torch.tensor(smpl_keypoints, dtype=torch.float)
            smpl_translation = smpl_keypoints  # (L, n_verts, 3)
            smpl_keypoints = smpl_keypoints[:num_train_data].cpu()
        else:
            smpl_pose = torch.tensor(test_dataset.video_cache["smpl_pose"], device="cpu", dtype=torch.float).clone()
            smpl_pose[:, :, :3, 3] /= 1.5
            smpl_keypoints = smpl_pose[:num_train_data, :, :3, 3]

            smpl_translation = smpl_pose[:, :, :3, 3]  # (L, 22, 3)

        _estimated_keypoints = estimated_keypoints.reshape(video_len, -1, 3).permute(0, 2, 1).reshape(video_len * 3,
                                                                                                      -1).cpu()
        _smpl_keypoints = smpl_keypoints.permute(0, 2, 1).reshape(video_len * 3, -1)

        lam = 5e-1 if use_smpl_verts else 1e-1
        s2j_mapping = torch.inverse(
            _smpl_keypoints.T.matmul(_smpl_keypoints) + torch.eye(_smpl_keypoints.shape[1]) * lam).matmul(
            _smpl_keypoints.T).matmul(_estimated_keypoints)

    regressed = s2j_mapping.T @ smpl_translation  # (L, 3, 140)

    # canonical pose
    with torch.no_grad():
        child_root_can, self_root_can = model.decoder.joint_root_locations(torch.eye(3, device="cuda",
                                                                                     dtype=torch.float),
                                                                           torch.zeros(model.num_parts, 3, 1,
                                                                                       device="cuda",
                                                                                       dtype=torch.float))
        estimated_keypoints_can = torch.cat([child_root_can, self_root_can], dim=-1)
        estimated_keypoints_can = estimated_keypoints_can.permute(0, 2, 1).cpu()

    regressed_translation = regressed.reshape(regressed.shape[0], model.num_parts, 7, 3)
    U, S, Vh = torch.linalg.svd(regressed_translation.permute(0, 1, 3, 2) @ estimated_keypoints_can)
    R = Vh.permute(0, 1, 3, 2) @ U.permute(0, 1, 3, 2)
    det = torch.linalg.det(R)
    Vh[:, :, 2] = Vh[:, :, 2] * det[:, :, None]
    R = Vh.permute(0, 1, 3, 2) @ U.permute(0, 1, 3, 2)
    joint_rotation = R.permute(0, 1, 3, 2).cuda()
    joint_translation = regressed_translation[:, :, -1, :, None].cuda()

    return joint_rotation, joint_translation  # (L, num_parts, 3, 3)


if __name__ == "__main__":
    # evaluate novel view and novel pose reconstruction
    # novel view -> learned pose, new camera
    # novel pose -> novel pose, all camera. Requires smpl regression
    parser = argparse.ArgumentParser(description='Save reconstructed images')
    parser.add_argument('--exp_name', action='append', required=True)
    args = parser.parse_args()
    exp_names = args.exp_name

    default_config = "confs/default.yml"
    for exp_name in exp_names:
        config_path = f"confs/{exp_name}.yml"
        test(config_path, default_config, mode="test")  # novel view
        test(config_path, default_config, mode="novel_pose")
