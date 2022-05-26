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

import tensorboardX as tbx
import torch
import torch.distributed as dist
from torch.cuda.amp import GradScaler

from utils.train_utils import (all_reduce_dict, cat_dim0_dict, create_dataloaders, create_ddp_dataloaders,
                               load_snapshot, to_gpu, save_model)
from utils.train_utils import set_port


class TrainerBase:
    def run(self, config: dict, rank: int = 0, world_size: int = 1) -> None:
        """

        Args:
            config:
            rank:
            world_size:

        Returns:

        """
        torch.backends.cudnn.benchmark = True
        ddp = False
        assert world_size == 1
        train_loader = create_dataloaders(config.dataset)

        self.train_loader = train_loader

        self.train_func(config, train_loader, None, rank=rank, ddp=ddp, world_size=world_size)

    def ddp_run(self, rank: int, config: dict, world_size: int = 1) -> None:
        """

        Args:
            rank:
            config:
            world_size:

        Returns:

        """
        assert world_size > 1
        torch.backends.cudnn.benchmark = True
        ddp = True

        set_port(config.train_setting)
        backend = config.train_setting.backend
        dist.init_process_group(backend=backend, init_method='env://', rank=rank,
                                world_size=world_size)
        torch.manual_seed(0)

        train_loader = create_ddp_dataloaders(config.dataset, rank, world_size)

        self.train_loader = train_loader

        try:
            self.train_func(config, train_loader, None, rank=rank, ddp=ddp, world_size=world_size)
        except KeyboardInterrupt:
            print('interrupted')

        dist.destroy_process_group()

    def prepare_model_and_optimizer(self, *args, **kwargs):
        raise NotImplementedError("Please implement prepare_model_and_optimizer")

    def define_loss_func(self, *args, **kwargs):
        raise NotImplementedError("Please implement define_loss_func")

    def lossfunc(self, *args, **kwargs):
        raise NotImplementedError("Please implement lossfunc")

    def process_before_train_step(self, iteration: int):
        pass

    def train_func(self, config: dict, train_loader, val_loader=None, rank: int = 0,
                   ddp: bool = False, world_size: int = 1) -> None:
        """

        Args:
            config:
            train_loader:
            val_loader:
            rank:
            ddp:
            world_size:

        Returns:

        """
        num_iter = config.train_setting.num_iter
        log_interval = config.train_setting.log_interval
        save_interval = config.train_setting.save_interval
        out_dir = config.output_dir
        exp_name = config.exp_name

        model, model_module, optimizer = self.prepare_model_and_optimizer(config, rank, ddp)
        self.model = model

        save_dir = os.path.join(out_dir, "result", exp_name)
        if rank == 0:
            writer = tbx.SummaryWriter(os.path.join(out_dir, "tensorboard", exp_name))
            os.makedirs(save_dir, exist_ok=True)
            os.chmod(save_dir, 0o755)

        iteration = 0

        if config.resume_model_path or config.resume_latest:
            if config.resume_model_path is not None:
                model_path = config.resume_model_path
            else:
                model_path = os.path.join(save_dir, f"{self.snapshot_prefix}_latest.pth")

            iteration = load_snapshot(model_module, optimizer, model_path, load_optimizer=config.load_optimizer)
            if config.iteration is not None:
                iteration = config.iteration

        # define loss
        self.define_loss_func(config, model_module, ddp)

        self.process_before_train_step(iteration)
        while iteration < num_iter:
            for i, minibatch in enumerate(train_loader):
                self.process_before_train_step(iteration)

                iteration += 1
                model.train()
                minibatch = to_gpu(minibatch)

                if minibatch["img"].ndim == 5:
                    # reshape (B, video_len, *) -> (B * video_len, *)
                    minibatch = cat_dim0_dict(minibatch)

                optimizer.zero_grad(set_to_none=True)

                # loss calculation
                loss, loss_dict = self.lossfunc(config, minibatch, model, model_module)

                if config.fp16:
                    scaler = GradScaler()
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:

                    # with torch.autograd.detect_anomaly():
                    loss.backward()

                    # detect nan
                    nan = any([p.grad.isnan().any() for p in model_module.parameters() if p.grad is not None])
                    if nan:
                        print("NaN is detected!!!!")
                        del loss
                        torch.cuda.empty_cache()
                    else:
                        if config.train_setting.clip_grad:
                            torch.nn.utils.clip_grad_norm_(model.parameters(),
                                                           max_norm=2.0, norm_type=2)

                        optimizer.step()

                if ddp:
                    loss_dict = all_reduce_dict(loss_dict, world_size)

                if iteration % 10 == 0 and rank == 0:
                    print(iteration, loss_dict)
                # tensorboard
                if iteration % log_interval == 0 and rank == 0:
                    print("log")
                    for key, val in loss_dict.items():
                        writer.add_scalar("metrics/" + key, val, iteration)

                if iteration % save_interval == 0:
                    iteration = save_model(model_module, optimizer, save_dir, iteration, rank, self.snapshot_prefix)
