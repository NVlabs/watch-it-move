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

from typing import Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn


def _get_ray_direction(pixel_location: torch.Tensor, inv_intrinsics: torch.Tensor) -> torch.Tensor:
    """

    Args:
        pixel_location:
        inv_intrinsics:

    Returns:

    """
    batchsize, _, num_ray = pixel_location.shape
    # + 0.5 is required
    homogeneous = torch.cat(
        [pixel_location + 0.5, torch.ones(batchsize, 1, num_ray, device="cuda")], dim=1)

    ray_direction = torch.matmul(inv_intrinsics,
                                 homogeneous)  # shape: (B, 3, num_ray), not unit vector

    return ray_direction


def _coarse_sample(pixel_location: torch.Tensor, inv_intrinsics: torch.Tensor,
                   joint_translation: torch.Tensor = None, near_plane: float = 1,
                   far_plane: float = 5, num_coarse: int = 64
                   ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """coarse sampling. uniform sampling in camera frustrum

    Args:
        pixel_location (torch.Tensor): 2D location on image,
                                       shape: (B, 2, num_ray)
        inv_intrinsics (torch.Tensor): inverse of camera intrinsics
                                       shape: (B, 3, 3)
        joint_translation (torch.Tensor, optional): (B, num_parts, 3, 1)
        near_plane (float, optional): [description]. Defaults to 1.
        far_plane (float, optional): [description]. Defaults to 4.
        num_coarse (int, optional): number of sample in each ray

    Returns:
        coarse_location (torch.Tensor): shape: (B, 3, num_ray, num_coarse)
        coarse_depth (torch.Tensor): shape: (B, 1, num_ray, num_coarse)
        ray_direction (torch.Tensor): shape: (B, 3, num_ray)
    """
    batchsize, _, num_ray = pixel_location.shape

    ray_direction = _get_ray_direction(pixel_location, inv_intrinsics)

    if joint_translation is None:
        uniform_depth = torch.linspace(near_plane, far_plane, num_coarse, device="cuda")
        coarse_depth = uniform_depth[None, None, None].repeat(batchsize, 1, num_ray, 1)
    else:
        max_depth = joint_translation[:, :, 2, 0].max(dim=1)[0]
        min_depth = joint_translation[:, :, 2, 0].min(dim=1)[0]

        far = max_depth + 0.5
        near = torch.clamp_min(min_depth - 0.5, near_plane)
        eps = torch.linspace(0, 1, num_coarse, device="cuda")
        uniform_depth = near[:, None] * (1 - eps) + far[:, None] * eps
        uniform_depth = uniform_depth[:, None, None]
        coarse_depth = uniform_depth.repeat(1, 1, num_ray, 1)

    coarse_location = ray_direction[:, :, :, None] * uniform_depth
    ray_direction = F.normalize(ray_direction, dim=1)

    return coarse_location, coarse_depth, ray_direction,


def _weight_for_volume_rendering(density: torch.Tensor, depth: torch.Tensor) -> torch.Tensor:
    """weight for volume rendering

    Args:
        density (torch.Tensor): [description]
        depth (torch.Tensor): [description]

    Returns:
        torch.Tensor: weight for each coarse bin, shape: (B, 1, 1, num_ray, num - 1)
    """
    assert density.ndim == 4
    assert depth.ndim == 4
    sigmoid = torch.sigmoid(density)  # shape: (B, 1, num_ray, num_on_ray)
    alpha = torch.clamp_min((sigmoid[..., :-1] - sigmoid[..., 1:]) / (sigmoid[..., :-1] + 1e-10), 0)
    _alpha = torch.cat([torch.zeros_like(alpha[..., :1]), alpha], dim=-1)
    alpha_ = torch.cat([alpha, torch.zeros_like(alpha[..., :1])], dim=-1)
    T_i = torch.cumprod(1 - _alpha, dim=-1)
    weights = T_i * alpha_

    return weights


def _multinomial_sample(weights: torch.Tensor, num_fine: int) -> torch.Tensor:
    """multinomial sample for fine sampling

    Args:
        weights (torch.Tensor): [description]
        num_fine (int): [description]

    Returns:
        torch.Tensor: normalized sampled position
    """
    batchsize, _, num_ray, num_coarse = weights.shape
    weights = weights.reshape(batchsize * num_ray, num_coarse)
    sampled_bins = torch.multinomial(torch.clamp_min(weights, 1e-8), num_fine,
                                     replacement=True).reshape(batchsize, 1, 1, num_ray, num_fine) / (num_coarse - 1)
    offset_in_bins = torch.cuda.FloatTensor(
        batchsize, 1, 1, num_ray, num_fine).uniform_() / (num_coarse - 1)
    sampled_normalized_depth = sampled_bins + offset_in_bins

    return sampled_normalized_depth


def _get_fine_location(coarse_location: torch.Tensor, coarse_depth: torch.Tensor,
                       sampled_normalized_depth: torch.Tensor
                       ) -> Tuple[torch.Tensor, torch.Tensor]:
    """

    Args:
        coarse_location:
        coarse_depth:
        sampled_normalized_depth:

    Returns:

    """
    near_location = coarse_location[:, :, :, :1]
    far_location = coarse_location[:, :, :, -1:]
    fine_location = (near_location * (1 - sampled_normalized_depth) +
                     far_location * sampled_normalized_depth)

    near_depth = coarse_depth[:, :, :, :1]
    far_depth = coarse_depth[:, :, :, -1:]
    fine_depth = (near_depth * (1 - sampled_normalized_depth) +
                  far_depth * sampled_normalized_depth)

    fine_location = torch.cat([coarse_location, fine_location], dim=3)
    fine_depth = torch.cat([coarse_depth, fine_depth], dim=3)

    return fine_location, fine_depth


def fine_sample(implicit_model: nn.Module, joint_rotation,
                joint_translation: torch.Tensor, pixel_location: torch.Tensor,
                inv_intrinsics: torch.Tensor, near_plane: float = 1, far_plane: float = 4,
                num_coarse: int = 64, num_fine: int = 64
                ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """fine sampling for nerf

    Args:
        implicit_model (nn.Module): implicit decoder
        joint_rotation (nn.Module): rotation matrix for each part
        joint_translation (nn.Module): translation for each part
        pixel_location (torch.Tensor): location of pixels
        inv_intrinsics (torch.Tensor): inverse of intrinsic matrix
        near_plane (float, optional): Defaults to 1.
        far_plane (float, optional): Defaults to 4.
        num_coarse (int, optional): number of sampling points for coarse sampling. Defaults to 64.
        num_fine (int, optional): number of sampling points for fine sampling. Defaults to 32.

    Returns:
        torch.Tensor: fine points
    """
    (coarse_location, coarse_depth, ray_direction) = _coarse_sample(pixel_location, inv_intrinsics, joint_translation,
                                                                    near_plane, far_plane, num_coarse)

    _, _, _, num_coarse = coarse_location.shape
    with torch.no_grad():
        decoder_output = implicit_model(coarse_location, joint_rotation, joint_translation,
                                        coarse_sample=True)

    coarse_density = decoder_output["density"]
    sdf_scale = decoder_output.get("sdf_scale")

    if sdf_scale is not None:
        coarse_density = coarse_density * sdf_scale

    weights = _weight_for_volume_rendering(coarse_density, coarse_depth)

    # normalised fine points, shape: (B, 1, num_ray, num_fine)
    sampled_normalized_depth = _multinomial_sample(weights, num_fine)[:, 0]

    # fine points, shape: (B, 3, num_ray, num_fine)
    fine_location, fine_depth = _get_fine_location(coarse_location, coarse_depth, sampled_normalized_depth)

    sort_idx = torch.argsort(fine_depth, dim=3)
    fine_location = torch.gather(fine_location, dim=3, index=sort_idx.repeat(1, 3, 1, 1))
    fine_depth = torch.gather(fine_depth, dim=3, index=sort_idx)

    return (fine_location,  # (batchsize, 3, num_ray, (num_coarse + num_fine))
            fine_depth,  # (batchsize, 1, num_ray, (num_coarse + num_fine))
            ray_direction)  # (batchsize, 3, num_ray)


def volume_rendering(density: torch.Tensor, color: torch.Tensor, depth: torch.tensor
                     ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """volume rendering

    Args:
        density (torch.Tensor): shape: (B, 1, num_ray, num_coarse + num_fine)
        color (torch.Tensor): shape: (B, 3, num_ray, num_coarse + num_fine)
        depth (torch.Tensor): shape: (B, 1, num_ray, num_coarse + num_fine)

    Returns:
        rendered_color (torch.Tensor): shape: (B, 3, num_ray)
        rendered_mask (torch.Tensor): shape: (B, 1, num_ray)
        rendered_disparity (torch.Tensor): shape: (B, 1, num_ray)
        weights
    """
    weights = _weight_for_volume_rendering(density, depth)

    rendered_color = torch.sum(weights * color, dim=3)
    rendered_mask = torch.sum(weights, dim=3).squeeze(1)
    rendered_disparity = torch.sum(weights / depth, dim=3).squeeze(1)

    return (rendered_color,  # (B, 3, num_ray)
            rendered_mask,  # (B, num_ray)
            rendered_disparity,  # (B, num_ray)
            weights)  # (B, 1, 1, num_ray, num_points)


def gather_pixel(img: torch.Tensor, pixel_location: torch.Tensor) -> torch.Tensor:
    """

    Args:
        img:
        pixel_location:

    Returns:

    """
    single_channel = (img.ndim == 3)
    if single_channel:
        img = img[:, None]

    batchsize, ch, height, width = img.shape

    if pixel_location.dtype == torch.int64:  # pixel index
        img = img.reshape(batchsize, ch, height * width)
        # gather pixel values from pixel_location
        x_coord = pixel_location[:, 0]
        y_coord = pixel_location[:, 1]
        flattened_location = y_coord * width + x_coord  # (B, num_ray)
        gathered_img = torch.gather(img, dim=2, index=flattened_location[:, None].repeat(1, ch, 1))
    elif pixel_location.dtype == torch.float32:  # in pixel index space (top-left = (0, 0))
        _pixel_location = pixel_location.permute(0, 2, 1)[:, :, None] + 0.5  # (B, n_rays, 1, 2)
        _pixel_location = _pixel_location / (height / 2) - 1
        gathered_img = F.grid_sample(img, _pixel_location, mode='bicubic')  # (B, ch, n_rays, 1)
        gathered_img = gathered_img.squeeze(3)
    else:
        raise TypeError("Invalid type for pixel_location")
    if single_channel:
        gathered_img = gathered_img[:, 0]

    return gathered_img


def rotation_matrix(theta: float, axis: str = "y") -> torch.Tensor:
    """

    Args:
        theta:
        axis:

    Returns:
        R: rotation matrix
    """
    c = np.cos(theta)
    s = np.sin(theta)
    if axis == "y":
        R = torch.tensor(np.array([[c, 0, -s, 0],
                                   [0, 1, 0, 0],
                                   [s, 0, c, 0],
                                   [0, 0, 0, 1]])).float().cuda()
    elif axis == "z":
        R = torch.tensor(np.array([[c, -s, 0, 0],
                                   [s, c, 0, 0],
                                   [0, 0, 1, 0],
                                   [0, 0, 0, 1]])).float().cuda()
    else:
        raise ValueError("invalid axis")

    return R


def rotate_pose(pose_camera: torch.Tensor, R: torch.Tensor) -> torch.Tensor:
    """

    Args:
        pose_camera:
        R:

    Returns:

    """
    center = torch.zeros(4, 4, device=R.device, dtype=torch.float)
    center[:3, 3] = pose_camera[0, :, :3, 3].mean(dim=0)
    center[3, 3] = 1
    rotated_pose = torch.matmul(R, (pose_camera - center)) + center

    return rotated_pose
