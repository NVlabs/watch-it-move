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

from typing import Tuple, Dict, Any

import torch
from torch import nn

from easydict import EasyDict as edict
from models.loss import SupervisedLoss
from models.model import SingleVideoPartDecomposition
from utils.get_args import get_args
from utils.train_utils import (create_optimizer,
                               send_model_to_gpu)
from utils.trainer import TrainerBase

mse = nn.MSELoss()


def loss_reconstruction_based(minibatch: dict, model: nn.Module, loss_func: SupervisedLoss, config: edict,
                              pull_rigid_parts: bool = False) -> Tuple[torch.Tensor, Dict[str, float]]:
    """

    Args:
        minibatch:
        model:
        loss_func:
        config:
        pull_rigid_parts:

    Returns:

    """
    img = minibatch["img"]
    mask = minibatch["mask"]
    camera_rotation = minibatch["camera_rotation"]
    camera_translation = minibatch["camera_translation"]
    inv_intrinsics = torch.inverse(minibatch["camera_intrinsic"])
    frame_id = minibatch["frame_id"]

    model_output_dict = model(frame_id, camera_rotation, camera_translation,
                              inv_intrinsics, num_ray=config.render_setting.num_ray, mask=mask)

    loss, loss_dict = loss_func(img, mask, model_output_dict, pull_rigid_parts=pull_rigid_parts)

    if "sdf_grad" in model_output_dict and config.loss_params.sdf_loss_coef > 0:
        sdf_grad = model_output_dict["sdf_grad"]
        sdf_loss = mse(torch.norm(sdf_grad, dim=1), torch.ones_like(sdf_grad[:, 0]))
        loss += sdf_loss * config.loss_params.sdf_loss_coef
        loss_dict["sdf_loss"] = sdf_loss.item()

    return loss, loss_dict


class Trainer(TrainerBase):
    def __init__(self):
        self.snapshot_prefix = "snapshot"
        self.only_update_joint = False
        self.pull_rigid_parts = False

    def prepare_model_and_optimizer(self, config: edict, rank: int, ddp: int) -> Tuple[nn.Module, nn.Module, Any]:
        """

        Args:
            config:
            rank:
            ddp:

        Returns:

        """
        self.config = config
        model = SingleVideoPartDecomposition(config.network_params)
        optimizer = create_optimizer(config.train_setting, model)  # optimizer works locally, define before DDP_model

        model, model_module = send_model_to_gpu(rank, model, ddp)
        return model, model_module, optimizer

    def define_loss_func(self, config: edict, model_module: nn.Module, ddp: bool) -> None:
        """

        Args:
            config:
            model_module:
            ddp:

        Returns:

        """
        self.reconstruction_loss_func = SupervisedLoss(config.loss_params, model_module, ddp, coarse_rate=64)

    def process_incremental(self, schedule_config: edict, iteration: int) -> None:
        """

        Args:
            schedule_config:
            iteration:

        Returns:

        """
        initial_frame = schedule_config.initial_frame
        start = schedule_config.start
        incremental_period = schedule_config.incremental_period
        num_frames = self.config.dataset.num_frames // self.config.dataset.thin_out_interval

        if start is None:
            start = 1e10
        if incremental_period is None:
            incremental_period = 1e10
        self.train_loader.dataset.current_max_frame_id = \
            int(initial_frame + min(max(0, iteration - start), incremental_period) *
                (num_frames - initial_frame) / incremental_period)
        self.model.current_max_frame_id = self.train_loader.dataset.current_max_frame_id

        loss_config = self.reconstruction_loss_func.config
        if loss_config.initial_structure_loss_coef > 0:
            if iteration > start:
                loss_config.structure_loss_coef = loss_config.max_structure_loss_coef
            else:
                loss_config.structure_loss_coef = loss_config.initial_structure_loss_coef * (1 - iteration / start) + \
                                                  loss_config.max_structure_loss_coef * (iteration / start)

        if iteration > start + incremental_period:
            self.pull_rigid_parts = True
        assert self.train_loader.num_workers == 0

    def process_before_train_step(self, iteration: int) -> None:
        """

        Args:
            iteration:

        Returns:

        """
        dataset_schedule_type = self.config.train_setting.dataset_schedule_type
        schedule_config = self.config.train_setting.frame_schedule[dataset_schedule_type]
        if dataset_schedule_type == "incremental":
            self.process_incremental(schedule_config, iteration)
        else:
            raise ValueError("Invalid dataset schedule type")

    def lossfunc(self, config: edict, minibatch: dict, model: nn.Module, model_module: nn.Module,
                 pull_rigid_parts: bool = False) -> Tuple[torch.Tensor, dict]:
        """

        Args:
            config:
            minibatch:
            model:
            model_module:
            pull_rigid_parts:

        Returns:

        """
        loss_dict = {}

        # reconstruction branch
        recon_loss_func = self.reconstruction_loss_func
        loss, _loss_dict = loss_reconstruction_based(minibatch, model, recon_loss_func, config,
                                                     pull_rigid_parts=self.pull_rigid_parts)
        loss_dict.update(_loss_dict)

        return loss, loss_dict


if __name__ == "__main__":
    args, config = get_args()

    trainer = Trainer()
    trainer.run(config)
