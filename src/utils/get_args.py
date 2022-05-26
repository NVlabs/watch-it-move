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
from typing import Optional, Any, Tuple

from easydict import EasyDict as edict

from .config import yaml_config


def get_config(args: Any) -> edict:
    """

    Args:
        args:

    Returns:

    """
    config = yaml_config(args.config, args.default_config)
    config.resume_latest = args.resume_latest
    if config.resume_model_path is None:
        config.resume_model_path = args.resume_model_path

    return config


def get_args(config_path: Optional[str] = None) -> Tuple[Any, edict]:
    """

    Args:
        config_path:

    Returns:

    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default="confs/default.yml")
    parser.add_argument('--default_config', type=str, default="confs/default.yml")
    parser.add_argument('--resume_latest', action='store_true')
    parser.add_argument('--resume_model_path', type=str, default=None)

    args = parser.parse_args()
    if config_path is not None:
        args.config = config_path

    config = get_config(args)

    return args, config


def get_ddp_args(config_path: Optional[str] = None) -> Tuple[Any, edict]:
    """

    Args:
        config_path:

    Returns:

    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default="confs/default.yml")
    parser.add_argument('--default_config', type=str, default="confs/default.yml")
    parser.add_argument('--resume_latest', action='store_true')
    parser.add_argument('--resume_model_path', type=str, default=None)
    parser.add_argument('--gpus', type=int, default=1)
    parser.add_argument('--nodes', type=int, default=1)

    args = parser.parse_args()
    if config_path is not None:
        args.config = config_path

    config = get_config(args)

    return args, config


def get_args_jupyter(config_path: str = "cons/default.yml", default_config: str = "confs/default.yml"
                     ) -> Tuple[None, edict]:
    """

    Args:
        config_path:
        default_config:

    Returns:

    """
    config = yaml_config(config_path, default_config)

    return None, config
