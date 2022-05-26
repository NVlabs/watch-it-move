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

from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from easydict import EasyDict as edict
from pytorch3d.transforms.rotation_conversions import rotation_6d_to_matrix
from torch import nn


def expand_mask(mask: torch.Tensor, coarse_rate: int = 32, stride: int = 32):
    """
    Expand mask by max pooling
    Args:
        mask:
        coarse_rate:
        stride:

    Returns:

    """
    pad = 0 if coarse_rate == stride else (coarse_rate - 1) // 2
    dilate_mask = F.max_pool2d(mask[:, None], coarse_rate, stride, pad)
    if stride > 1:
        dilate_mask = F.interpolate(dilate_mask, scale_factor=stride, mode="nearest")

    return dilate_mask


def foreground_sampler(img_size: int, num_ray: int, mask: torch.Tensor, coarse_rate: int = 32,
                       stride: int = 32, dim: int = 1, expand=True) -> torch.Tensor:
    """uniformly sample around foreground mask

    Args:
        img_size (int): image size
        num_ray (int): number of points to sample
        mask (int): shape: (B, img_size, img_size)
        coarse_rate
        stride
        dim
        expand

    Returns:
        torch.Tensor: sampled coordinates, shape: (B, 2, num_ray) if dim==1
    """
    if expand:
        dilate_mask = expand_mask(mask, coarse_rate, stride).squeeze(1) > 0.5
    else:
        dilate_mask = mask > 0.5
    unreliable_mask = mask > 1
    dilate_mask = dilate_mask.float() - unreliable_mask * 2
    noised_dilate_mask = dilate_mask + torch.empty_like(dilate_mask, dtype=torch.float).uniform_()
    noised_dilate_mask = noised_dilate_mask.reshape(-1, img_size ** 2)
    _, coordinates = torch.topk(noised_dilate_mask, num_ray, dim=1, sorted=False)
    coordinates = torch.stack([coordinates % img_size,
                               torch.div(coordinates, img_size, rounding_mode='trunc')], dim=dim)

    return coordinates


def patch_sampler(img_size: int, num_ray: int, mask: torch.Tensor, coarse_rate: int = 32,
                  dim: int = 1, expand=True) -> torch.Tensor:
    """sample patch

    Args:
        img_size (int): image size
        num_ray (int): number of points to sample
        mask (torch.Tensor): shape: (B, img_size, img_size)
        coarse_rate
        dim
        expand

    Returns:
        torch.Tensor: sampled coordinates, shape: (B, 2, num_ray) if dim==1
    """
    assert (num_ray ** 0.5).is_integer()
    assert expand

    patch_size = int(num_ray ** 0.5)
    expansion_size = max(0, coarse_rate - patch_size // 2)

    dilate_mask = expand_mask(mask, expansion_size + 1, 1)

    noised_dilate_mask = (dilate_mask > 0.5) + torch.empty_like(dilate_mask, dtype=torch.float).uniform_()
    noised_dilate_mask = noised_dilate_mask.reshape(-1, img_size ** 2)
    patch_center = torch.argmax(noised_dilate_mask, dim=1, keepdim=True)

    device = mask.device
    coordinates = torch.stack([patch_center % img_size,
                               torch.div(patch_center, img_size, rounding_mode='trunc')], dim=dim)
    coordinates = coordinates.clamp(patch_size // 2, img_size - patch_size // 2 - 1)

    grid = torch.meshgrid(torch.arange(-patch_size // 2, patch_size // 2, device=device),
                          torch.arange(-patch_size // 2, patch_size // 2, device=device), indexing='ij')
    grid = torch.stack([grid[1].reshape(1, -1), grid[0].reshape(1, -1)], dim=dim)

    coordinates = coordinates + grid

    return coordinates


class PixelSampler:
    def __init__(self, sample_strategy: str = "uniform"):
        """

        Args:
            sample_strategy:
        """
        self.sample_strategy = sample_strategy

    @staticmethod
    def unifrom_sampler(img_size: int, num_ray: int, batchsize: int) -> torch.Tensor:
        """uniformly sample pixel coordinates

        Args:
            img_size (int): image size
            num_ray (int): number of points to sample
            batchsize:

        Returns:
            torch.Tensor: sampled coordinates, shape: (B, 2, num_ray)
        """
        coordinates = torch.randint(high=img_size, size=(batchsize, 2, num_ray), device="cuda")

        return coordinates

    def __call__(self, img_size: int, num_ray: int, batchsize: int,
                 mask: Optional[torch.Tensor] = None, expand=True,
                 coarse_rate: int = 32, stride: int = 32) -> torch.Tensor:
        """

        Args:
            img_size:
            num_ray:
            batchsize:
            mask:
            expand:
            coarse_rate:
            stride:

        Returns:

        """
        if self.sample_strategy == "uniform":
            return self.unifrom_sampler(img_size, num_ray, batchsize)
        elif self.sample_strategy == "foreground":
            return foreground_sampler(img_size, num_ray, mask, expand=expand,
                                      coarse_rate=coarse_rate, stride=stride)
        elif self.sample_strategy == "patch":
            return patch_sampler(img_size, num_ray, mask, expand=expand,
                                 coarse_rate=coarse_rate)
        else:
            raise ValueError()


class PoseTrajectoryMLP(nn.Module):
    def __init__(self, video_len: int, n_keypoints: int, hidden_dim: int = 128, n_mlp: int = 4, k: int = 100,
                 n_split: int = 1, **kwargs):
        """

        Args:
            video_len:
            n_keypoints:
            hidden_dim:
            n_mlp:
            k:
            n_split:
            **kwargs:
        """
        super(PoseTrajectoryMLP, self).__init__()
        self.video_len = video_len
        self.n_keypoints = n_keypoints
        self.k = k
        self.n_split = n_split
        if n_split > 1:
            split_loc = [-1 / n_split] + [(i + 1) / n_split for i in range(n_split - 1)] + [1 + 1 / n_split]
            self.split_loc = np.array(split_loc)

            layers = [nn.Conv1d(self.k, hidden_dim * n_split, 1), nn.ELU(inplace=True)]
            for i in range(n_mlp - 1):
                layers.append(nn.Conv1d(hidden_dim * n_split, hidden_dim * n_split, 1, groups=n_split))
                layers.append(nn.ELU(inplace=True))

            layers.append(nn.Conv1d(hidden_dim * n_split, n_keypoints * 9 * n_split, 1, groups=n_split))
        else:
            layers = [nn.Linear(self.k, hidden_dim), nn.ELU(inplace=True)]
            for i in range(n_mlp - 1):
                layers.append(nn.Linear(hidden_dim, hidden_dim))
                layers.append(nn.ELU(inplace=True))

            layers.append(nn.Linear(hidden_dim, n_keypoints * 9))

        self.model = nn.Sequential(*layers)

    def backbone(self, t: torch.Tensor) -> torch.Tensor:
        """

        Args:
            t:

        Returns:

        """
        batchsize = t.shape[0]
        device = t.device
        freq = (t[:, None] + 0.5 / self.video_len) * np.pi * torch.arange(0, self.k, device=device)  # (B, k)
        if self.n_split > 1:
            freq = freq[:, :, None]  # (B, k)

        trajectory = self.model(torch.cos(freq))  # (B, n_kpts * 9)

        if self.n_split > 1:
            trajectory = trajectory.reshape(batchsize, self.n_split, self.n_keypoints * 9)
            split_loc = torch.tensor(self.split_loc, device=device, dtype=torch.float)
            sigmoid_scale = 12 * self.n_split
            weight = torch.sigmoid((t[:, None] - split_loc[None, :-1]) * sigmoid_scale) * \
                     torch.sigmoid(-(t[:, None] - split_loc[None, 1:]) * sigmoid_scale)
            trajectory = torch.sum(trajectory * weight[:, :, None], dim=1)

        return trajectory

    def forward(self, idx: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute rotation and translation matrices from positionally encoded time
        Args:
            idx: frame index: (B, )

        Returns:
            joint rotation: (B, n_kpts, 3, 3)
            joint_translation: (B, n_kpts, 3, 1)

        """
        batchsize = idx.shape[0]

        t = idx / self.video_len

        trajectory = self.backbone(t)

        rot, trans = torch.split(trajectory, [6 * self.n_keypoints,
                                              3 * self.n_keypoints], dim=1)
        rot = rotation_6d_to_matrix(rot.reshape(batchsize * self.n_keypoints, 6))
        rot = rot.reshape(batchsize, self.n_keypoints, 3, 3)
        trans = trans.reshape(batchsize, self.n_keypoints, 3, 1)

        return rot, trans


def get_pose_trajectory(config: edict) -> PoseTrajectoryMLP:
    """

    Args:
        config:

    Returns:

    """
    video_len = config.video_length
    num_parts = config.num_parts
    params = config.trajectory_params.dct

    return PoseTrajectoryMLP(video_len, num_parts, **params)
