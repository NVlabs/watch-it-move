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
from typing import Any, Tuple

import torch
import torch.optim as optim
from easydict import EasyDict as edict
from torch import nn
from torch.utils.data import DataLoader, Dataset

from datasets.dataset import SingleVideoDataset as HumanVideoDataset


def create_dataloaders(config: dict, shuffle: bool = True) -> DataLoader:
    """create train and val dataloaders

    Args:
        config (dict): config.dataset
        shuffle

    Returns:
        data_loader (DataLoader): dataloader
    """
    dataset = HumanVideoDataset(config)

    batchsize = config.batchsize
    num_workers = config.num_workers
    data_loader = DataLoader(dataset, batch_size=batchsize, num_workers=num_workers,
                             shuffle=shuffle, drop_last=True, pin_memory=True)

    return data_loader


def ddp_data_sampler(dataset: Dataset, rank: int, world_size: int, shuffle: bool, drop_last: bool
                     ) -> torch.utils.data.distributed.DistributedSampler:
    """

    Args:
        dataset:
        rank:
        world_size:
        shuffle:
        drop_last:

    Returns:

    """
    dist_sampler = torch.utils.data.distributed.DistributedSampler(
        dataset, rank=rank, num_replicas=world_size, shuffle=shuffle,
        drop_last=drop_last)

    return dist_sampler


def create_ddp_dataloaders(config: dict, rank: int, world_size: int) -> DataLoader:
    """create train and val dataloaders for ddp

    Args:
        config (dict): config.dataset
        rank
        world_size

    Returns:
        train_loader (DataLoader): dataloader for train
        val_loader (DataLoader): dataloader for val
    """

    dataset = HumanVideoDataset(config)

    batchsize = config.batchsize
    num_workers = config.num_workers
    ddp_sampler = ddp_data_sampler(dataset, rank, world_size, shuffle=True, drop_last=True)
    data_loader = DataLoader(dataset, batch_size=batchsize, num_workers=num_workers,
                             sampler=ddp_sampler, pin_memory=True)

    return data_loader


def to_gpu(minibatch: dict) -> dict:
    """send minibatch dict to gpu

    Args:
        minibatch (dict): [description]

    Returns:
        dict: [description]
    """

    return {key: val.cuda(non_blocking=True) for key, val in minibatch.items()}


def to_tensor(minibatch: dict) -> dict:
    """numpy to torch.tensor
    Args:
        minibatch (dict): [description]

    Returns:
        dict: [description]
    """

    return {key: torch.tensor(val).cuda(non_blocking=True).float() for key, val in minibatch.items()}


def cat_dim0(tensor: torch.Tensor) -> torch.Tensor:
    """

    Args:
        tensor:

    Returns:

    """
    shape = tensor.shape

    return tensor.reshape((shape[0] * shape[1],) + shape[2:])


def cat_dim0_dict(minibatch: dict) -> dict:
    """

    Args:
        minibatch:

    Returns:

    """
    out_dict = {}
    for key, val in minibatch.items():
        shape = val.shape
        if len(shape) <= 2:
            reshaped = val.reshape(-1)
        else:
            reshaped = val.reshape((shape[0] * shape[1],) + shape[2:])
        out_dict[key] = reshaped

    return out_dict


def set_port(config: edict) -> None:
    """

    Args:
        config:

    Returns:

    """
    master_addr = config.master_addr
    master_port = config.master_port
    os.environ['MASTER_ADDR'] = master_addr
    os.environ['MASTER_PORT'] = master_port


def all_reduce_scalar(scalar: float) -> float:
    """

    Args:
        scalar:

    Returns:

    """
    scalar = torch.tensor(scalar).cuda(non_blocking=True)
    torch.distributed.all_reduce(scalar)

    return scalar.item()


def all_reduce_dict(dictionary: dict, world_size: int) -> dict:
    """

    Args:
        dictionary:
        world_size:

    Returns:

    """
    reduced_dict = {}
    for key, val in dictionary.items():
        reduced_dict[key] = all_reduce_scalar(val) / world_size

    return reduced_dict


def grid_coordinates(size: int, device: str, scale: int = 2) -> torch.Tensor:
    """

    Args:
        size:
        device:
        scale:

    Returns:

    """
    grid = torch.meshgrid(torch.arange(size, device=device),
                          torch.arange(size, device=device), indexing='ij')[::-1]
    grid = torch.stack(grid, dim=-1) * scale + 0.5
    grid = grid.reshape(1, size ** 2, 2)

    return grid


def check_nan(model: nn.Module) -> bool:
    """

    Args:
        model:

    Returns:

    """
    state_dict = model.state_dict()

    isnan = False
    for val in state_dict.values():
        if val.isnan().any():
            isnan = True
            break

    return isnan


def load_snapshot(model: nn.Module, optimizer, path: str, load_optimizer: bool = True) -> int:
    """

    Args:
        model:
        optimizer:
        path:
        load_optimizer:

    Returns:

    """
    snapshot = torch.load(path, map_location=lambda storage, loc: storage)  # avoid OOM

    name_in_model = [n for n, _ in model.named_parameters()]
    for name in list(snapshot["model"].keys()):
        if name not in name_in_model:
            snapshot["model"].pop(name)

    model.load_state_dict(snapshot["model"], strict=False)

    if load_optimizer:
        optimizer.load_state_dict(snapshot["optimizer"])
    iter = snapshot["iteration"]
    del snapshot
    torch.cuda.empty_cache()  # remove cache for resuming

    return iter


def create_optimizer(config: dict, model: nn.Module) -> Any:
    """create optimizer

    Args:
        config (dict): config.tran_setting
        model (nn.Module): target model

    Returns:
        [type]: optimizer
    """
    if config.optimizer == "Adam":
        lr = config.lr
        decay = config.decay
        optimizer = optim.Adam(model.parameters(), lr=lr, betas=(0.9, 0.999), eps=1e-20, weight_decay=decay)
    elif config.optimizer == "AdamW":
        lr = config.lr
        decay = config.decay
        print("adamw", decay)
        optimizer = optim.AdamW(model.parameters(), lr=lr, betas=(0.9, 0.999), eps=1e-20, weight_decay=decay)
    else:
        raise ValueError()

    return optimizer


def send_model_to_gpu(rank: int, model: nn.Module, ddp: bool) -> Tuple[nn.Module, nn.Module]:
    """

    Args:
        rank:
        model:
        ddp:

    Returns:

    """
    num_gpus = torch.cuda.device_count()
    n_gpu = rank % num_gpus

    torch.cuda.set_device(n_gpu)
    model.cuda(n_gpu)

    if ddp:
        print(n_gpu)
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
        model = nn.parallel.DistributedDataParallel(model, device_ids=[n_gpu], find_unused_parameters=True)
        model_module = model.module
    else:
        model_module = model

    return model, model_module,


def save_model(model_module: nn.Module, optimizer, save_dir: str, iteration: int, rank: int,
               snapshot_prefix: str = "snapshot") -> int:
    """
    Save model. If nan is detected, load the latest snapshot
    Args:
        model_module:
        optimizer:
        save_dir:
        iteration:
        rank:
        snapshot_prefix:

    Returns:

    """
    isnan = check_nan(model_module)

    if isnan:
        print("nan detected")
        model_path = os.path.join(save_dir, f"{snapshot_prefix}_latest.pth")
        assert os.path.exists(model_path), "model snapshot is not saved"

        iteration = load_snapshot(model_module, optimizer, model_path)
    else:
        if rank == 0:
            params_to_save = {"iteration": iteration,
                              "model": model_module.state_dict(),
                              "optimizer": optimizer.state_dict()}
            torch.save(params_to_save, os.path.join(save_dir, f"{snapshot_prefix}_latest.pth"))
            torch.save(
                params_to_save, os.path.join(
                    save_dir, f"{snapshot_prefix}_{(iteration // 10000 + 1) * 10000}.pth"))

    return iteration
