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
import sys

import yaml
from easydict import EasyDict as edict

sys.path.append(".")
from utils.visualization_utils import GenerateVideoFromConfig

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Manual re-posing')
    parser.add_argument('--exp_name', required=True, type=str)
    parser.add_argument('--repose_config', required=True, type=str)
    parser.add_argument('--rotate', action="store_true")
    parser.add_argument('--num_video_frames', type=int, default=20)
    parser.add_argument('--iteration', type=int, default=-1)

    args = parser.parse_args()

    config_path = f"confs/{args.exp_name}.yml"

    default_config = "confs/default.yml"

    repose_config = edict(yaml.load(open(args.repose_config), Loader=yaml.SafeLoader))
    frame_id = repose_config.frame_id
    camera_id = repose_config.camera_id
    root = repose_config.root
    first = repose_config.first
    second = repose_config.second
    rotate = args.rotate
    num_video_frames = args.num_video_frames
    iteration = args.iteration

    generate_vide_from_conf = GenerateVideoFromConfig(config_path, default_config, iteration=iteration)
    generate_vide_from_conf.repose(root, first, second, frame_id, camera_id,
                                   rotate=rotate, num_video_frames=num_video_frames)
