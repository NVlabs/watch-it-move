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

sys.path.append(".")
from utils.visualization_utils import GenerateVideoFromConfig

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Save reconstruction video')
    parser.add_argument('--exp_name', action='append', required=True)
    parser.add_argument('--camera_id', type=int, default=0)
    parser.add_argument('--num_video_frames', type=int, default=50)
    args = parser.parse_args()

    exp_names = args.exp_name
    default_config = "confs/default.yml"
    camera_id = args.camera_id
    num_video_frames = args.num_video_frames

    for exp_name in exp_names:
        config_path = f"confs/{exp_name}.yml"
        generate_vide_from_conf = GenerateVideoFromConfig(config_path, default_config)
        generate_vide_from_conf(rotate=True, increment=True, camera_id=camera_id, num_video_frames=num_video_frames)
