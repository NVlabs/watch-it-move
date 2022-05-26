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
import sys
from typing import Dict, Tuple

import torch

sys.path.append(".")

from models.model import SingleVideoPartDecomposition
from datasets.dataset import SingleVideoDataset as HumanVideoDataset
from utils.get_args import get_args_jupyter
from utils.train_utils import create_dataloaders


def regress(model: SingleVideoPartDecomposition, config: Dict, dataset: HumanVideoDataset
            ) -> Tuple[torch.Tensor, torch.Tensor, float]:
    """

    Args:
        model:
        config:
        dataset:

    Returns:

    """
    with torch.no_grad():
        frame_id = torch.arange(config.dataset.num_frames, dtype=torch.float, device="cuda")
        trajectory = model.joint_trajectory(frame_id)

        rotation, translation = trajectory
        video_len = rotation.shape[0]

        smpl_pose = torch.tensor(dataset.video_cache["smpl_pose"], device=rotation.device,
                                 dtype=torch.float) / dataset.coordinate_scale

        child_root, self_root = model.decoder.joint_root_locations(rotation, translation)
        child_root = child_root.permute(0, 1, 3, 2).reshape(video_len, -1, 3)
        estimated_keypoints = torch.cat([child_root, self_root.squeeze(-1)], dim=1)
        smpl_keypoints = smpl_pose[:, :, :3, 3]

        # train test split
        train_idx = range(0, video_len, 10)
        test_idx = [i for i in range(video_len) if i % 10 != 0]
        estimated_keypoints_train = estimated_keypoints[train_idx]
        estimated_keypoints_test = estimated_keypoints[test_idx]
        smpl_keypoints_train = smpl_keypoints[train_idx]
        smpl_keypoints_test = smpl_keypoints[test_idx]

        estimated_keypoints_train = estimated_keypoints_train.permute(0, 2, 1).reshape(len(train_idx) * 3, -1).cpu()
        estimated_keypoints_test = estimated_keypoints_test.permute(0, 2, 1).reshape(len(test_idx) * 3, -1).cpu()
        smpl_keypoints_train = smpl_keypoints_train.permute(0, 2, 1).reshape(len(train_idx) * 3, -1).cpu()
        smpl_keypoints_test = smpl_keypoints_test.permute(0, 2, 1).reshape(len(test_idx) * 3, -1).cpu()

    lstsq_result_train = torch.linalg.lstsq(estimated_keypoints_train, smpl_keypoints_train, driver="gelsd")
    j2s_mapping_train = lstsq_result_train.solution
    test_error = estimated_keypoints_test @ j2s_mapping_train - smpl_keypoints_test
    test_error = test_error.reshape(len(test_idx), 3, smpl_keypoints_test.shape[-1])
    test_error = test_error.norm(dim=1).mean()

    return j2s_mapping_train, estimated_keypoints, test_error * dataset.coordinate_scale * 1000  # millimeter


def smpl_regression(config_path: str, default_config: str) -> None:
    """

    Args:
        config_path:
        default_config:

    Returns:

    """
    args, config = get_args_jupyter(config_path, default_config)
    config.dataset.batchsize = 1

    train_dataset: HumanVideoDataset = create_dataloaders(config.dataset, shuffle=True).dataset

    out_dir = config.output_dir
    exp_name = config.exp_name

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
        assert False, "model is not loaded"

    _, _, mpjpe = regress(model, config, train_dataset)

    print(f"{exp_name}: MPJPE={mpjpe:.4f}mm")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='SMPL regression evaluation')
    parser.add_argument('--exp_name', action='append', required=True)
    args = parser.parse_args()
    default_config = "confs/default.yml"

    exp_names = args.exp_name
    for exp_name in exp_names:
        config_path = f"confs/{exp_name}.yml"
        smpl_regression(config_path, default_config)
