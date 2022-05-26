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

import yaml
from easydict import EasyDict as edict


def check_config(config: edict):
    """

    Args:
        config:

    Returns:

    """
    if "temporal_consistency_loss_coef" in config.loss_params:
        assert (config.loss_params.temporal_consistency_loss_coef > 0) == config.network_params.temporal_consistency
        assert not config.network_params.temporal_consistency or (config.dataset.video_len > 1)

    if "surface_loss" in config.network_params:
        assert (config.loss_params.surface_loss_coef > 0) == config.network_params.surface_loss
    if "structure_loss" in config.network_params:
        assert (config.loss_params.structure_loss_coef > 0) == config.network_params.structure_loss

    if "transformation_equivariance" in config.network_params:
        assert ((config.loss_params.heatmap_2d_equivariance_loss_coef >
                 0) == config.network_params.transformation_equivariance) or \
               ((config.loss_params.depth_map_equivariance_loss_coef >
                 0) == config.network_params.transformation_equivariance) or \
               ((config.loss_params.pose_equivariance_loss_coef >
                 0) == config.network_params.transformation_equivariance)


def yaml_config(config_path: str, default_config_path: str) -> edict:
    """

    Args:
        config_path:
        default_config_path:

    Returns:

    """
    default_config = edict(yaml.load(open(default_config_path), Loader=yaml.SafeLoader))
    current_config = edict(yaml.load(open(config_path), Loader=yaml.SafeLoader))

    def _copy(conf: dict, default_conf: dict):
        for key in conf:
            if isinstance(default_conf[key], edict):
                _copy(conf[key], default_conf[key])
            else:
                default_conf[key] = conf[key]

    _copy(current_config, default_config)

    # copy params
    default_config.network_params.size = default_config.dataset.size
    default_config.network_params.num_parts = default_config.dataset.num_parts

    if "video_len" in default_config.dataset:
        default_config.network_params.video_len = default_config.dataset.video_len

    if "transformation_equivariance" in default_config.network_params:
        default_config.dataset.transformation_equivariance = default_config.network_params.transformation_equivariance
        default_config.test_dataset.transformation_equivariance = False

    if "decoder_params" in default_config.network_params:
        default_config.network_params.decoder_params.num_parts = default_config.dataset.num_parts
        default_config.network_params.decoder_params.num_camera = default_config.dataset.num_view

    if "multiview" in default_config.dataset:
        default_config.network_params.multiview = default_config.dataset.multiview

    if "num_frames" in default_config.dataset:
        default_config.network_params.video_length = default_config.dataset.num_frames
        default_config.network_params.num_view = default_config.dataset.num_view

    return_neighboring_frames = False

    default_config.dataset.return_neighboring_frames = return_neighboring_frames
    default_config.test_dataset.return_neighboring_frames = return_neighboring_frames

    return_random_frames = False

    default_config.dataset.return_random_frames = return_random_frames
    default_config.test_dataset.return_random_frames = return_random_frames

    default_config.network_params.background_color = default_config.dataset.background_color

    check_config(default_config)

    return default_config
