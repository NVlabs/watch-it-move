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

from typing import Optional, Tuple, Dict

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from utils.sdf_utils import ellipsoid_sdf


class MLP(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int,
                 groups: int, num_layers: int = 4):
        """

        Args:
            in_dim:
            hidden_dim:
            groups:
            num_layers:
        """
        super().__init__()
        layers = [nn.Conv1d(in_dim, hidden_dim, 1, groups=groups),
                  nn.LeakyReLU(negative_slope=0.2)]
        for _ in range(num_layers):
            layers += [nn.Conv1d(hidden_dim, hidden_dim, 1, groups=groups),
                       nn.LeakyReLU(negative_slope=0.2)]
        self.layers = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """MLP used for implicit function

        Args:
            x (torch.Tensor): position, shape: (B, num_parts*?, num_points)

        Returns:
            torch.Tensor:
        """
        return self.layers(x)


class EllipsoidDecoder(nn.Module):
    def __init__(self, num_parts: int, hidden_dim: int, n_power: int, num_layers: int,
                 sdf_residual_range: float, child_root: list, sdf_scale: float,
                 initial_sdf_weight: int, residual_sdf: bool = True, *args, **kwargs):
        """

        Args:
            num_parts:
            hidden_dim:
            n_power:
            num_layers:
            sdf_residual_range:
            child_root:
            sdf_scale:
            initial_sdf_weight:
            residual_sdf:
            *args:
            **kwargs:
        """
        super().__init__()
        self.num_parts = num_parts
        self.n_power = n_power
        self.sdf_residual_range = sdf_residual_range
        self.sdf_scale = sdf_scale
        self.residual_sdf = residual_sdf
        self.render_sphere = False  # used for visualization

        self.radius = nn.Parameter(torch.ones(num_parts, 3, dtype=torch.float) * 0.1)

        self.conv_xyz = nn.Conv1d(3 * num_parts * n_power * 2, hidden_dim, 1)
        self.mlp = MLP(in_dim=hidden_dim, hidden_dim=hidden_dim,
                       groups=1, num_layers=num_layers - 2)

        self.conv_sdf = nn.Conv1d(hidden_dim, 1, 1)

        self.conv_color = nn.Conv1d(hidden_dim, 3, 1)

        self.sdf_weight = nn.Parameter(torch.tensor(initial_sdf_weight))
        self.logsumexp_scale = nn.Parameter(torch.tensor(5.))

        self.register_buffer("child_root", torch.tensor(child_root).float())  # (-1, 3)

        self.sdf_residual_activation = torch.tanh

    def positional_encoding(self, x: torch.Tensor, n_power: int, num_parts: int) -> torch.Tensor:
        """positional encoding for multi-part

        Args:
            x (torch.Tensor): position, shape: (B, num_parts * ?, num_points)
            n_power (int): L in NeRF paper
            num_parts (int): number of parts

        Returns:
            torch.Tensor: encoded x, shape: (B,  num_parts * dim_per_part * n_power * 2, num_points)
        """
        batchsize, n_part, dim, n = x.shape
        h = x.reshape(batchsize, num_parts, dim, n)
        h = [torch.cos(h * (2 ** i)) for i in range(n_power)] + \
            [torch.sin(h * (2 ** i)) for i in range(n_power)]
        h = torch.cat(h, dim=2)
        h = h.reshape(batchsize, num_parts, dim * n_power * 2, n)

        return h

    def positional_encoding_1d(self, x: torch.Tensor, n_power: int) -> torch.Tensor:
        """Positional encoding for scalar

        Args:
            x: (B, )
            n_power: L in NeRF paper

        Returns:
            torch.Tensor: encoded x, (B, n_power * 2)

        """
        h = [torch.cos(x * (2 ** i)) for i in range(n_power)] + \
            [torch.sin(x * (2 ** i)) for i in range(n_power)]
        h = torch.stack(h, dim=-1)

        return h  # (B, n_poset * 2)

    def forward(self, xyz: torch.Tensor, joint_rotation: torch.Tensor,
                joint_translation: torch.Tensor, coarse_sample: bool = False,
                not_return_color: bool = False
                ) -> Dict[str, torch.Tensor]:
        """implicit function for each part. implemented with grouped convolution

        Args:
            xyz (torch.Tensor): position in camera coordinate,
                                shape: (B, 3, num_ray, num_on_ray)
            joint_rotation (torch.Tensor): shape: (B, num_parts, 3, 3)
            joint_translation (torch.Tensor): shape: (B, num_parts, 3, 1)
            coarse_sample (bool): if true, used for coarse sampling
            not_return_color: whether to return color

        Returns:
            density (torch.Tensor): density, shape: (B, num_parts, num_points)
            color (torch.Tensor): color, shape: (B, 3 * num_parts, num_points)
        """
        assert xyz.ndim == 4, f"xyz.ndim={xyz.ndim} should be 4"
        assert xyz.shape[1] == 3
        batchsize, _, num_ray, num_on_ray = xyz.shape
        num_points = num_ray * num_on_ray  # number of sampled 3d points
        device = xyz.device
        xyz = xyz.reshape(batchsize, 1, 3, num_points)
        # camera coord -> part coord
        inv_rotation = joint_rotation.permute(0, 1, 3, 2)
        transformed_xyz = torch.matmul(inv_rotation, xyz - joint_translation)  # (B, n_part, 3, n_pts)
        radius = torch.clamp_min(self.radius, 0.01)
        sdf = ellipsoid_sdf(radius, transformed_xyz)  # (B, n_part, n_pts)

        ignore_background = False
        _transformed_xyz = transformed_xyz
        if not self.training:
            if batchsize == 1:
                ignore_background = True
                sdf = sdf.reshape(batchsize, self.num_parts, num_ray, num_on_ray)
                foreground = (sdf < self.sdf_residual_range + 0.05).any(dim=1).any(dim=-1)[0]
                sdf = sdf[:, :, foreground]
                sdf = sdf.reshape(batchsize, self.num_parts, -1)
                _transformed_xyz = _transformed_xyz.clone().reshape(batchsize, self.num_parts, 3, num_ray, num_on_ray)
                _transformed_xyz = _transformed_xyz[:, :, :, foreground]
                _transformed_xyz = _transformed_xyz.reshape(batchsize, self.num_parts, 3, -1)

        if sdf.shape[-1] > 0:
            part_prob = torch.softmax(-sdf * F.softplus(self.sdf_weight), dim=1)  # (B, n_part, n_pts)

            # positional encoding
            encoded_xyz = self.positional_encoding(_transformed_xyz, n_power=self.n_power, num_parts=self.num_parts)

            h = encoded_xyz * part_prob[:, :, None]
            h = h.reshape(batchsize, -1, h.shape[-1])

            h = self.conv_xyz(h)  # (B, h_dim, n_pts)
            h = self.mlp(h)

            sdf_residual = self.conv_sdf(h)  # (B, 1, n_pts)

            if self.render_sphere:
                final_sdf = sdf.min(dim=1, keepdim=True)[0]
            else:
                if self.residual_sdf:
                    sdf_residual = self.sdf_residual_activation(sdf_residual) * self.sdf_residual_range
                    scale = F.softplus(self.logsumexp_scale * 10)
                    final_sdf = -torch.logsumexp(-sdf * scale, dim=1, keepdim=True) / scale
                    final_sdf = final_sdf + sdf_residual
                else:
                    final_sdf = sdf_residual
            final_sdf = final_sdf.reshape(batchsize, 1, -1, num_on_ray)

            if coarse_sample or not_return_color:
                color = None
            else:
                color = torch.tanh(self.conv_color(h))  # (B, 3, n_pts)
                color = color.reshape(batchsize, 3, -1, num_on_ray)

            part_prob = -sdf * 1000  # sharpen part probabilities
            part_prob = torch.softmax(part_prob, dim=1)  # (B, n_part, n_pts)

            part_prob = part_prob.reshape(batchsize, self.num_parts, 1, -1, num_on_ray)

            sdf = sdf.reshape(batchsize, self.num_parts, -1, num_on_ray)

            if ignore_background:
                _final_sdf = torch.full(size=(batchsize, 1, num_ray, num_on_ray), fill_value=0.05, device=device)
                _final_sdf[:, :, foreground] = final_sdf
                final_sdf = _final_sdf

                _sdf = torch.full(size=(batchsize, self.num_parts, num_ray, num_on_ray), fill_value=0.05, device=device)
                _sdf[:, :, foreground] = sdf
                sdf = _sdf

                if color is not None:
                    _color = torch.full(size=(batchsize, 3, num_ray, num_on_ray), fill_value=0., device=device)
                    _color[:, :, foreground] = color
                    color = _color

                _part_prob = torch.full(size=(batchsize, self.num_parts, 1, num_ray, num_on_ray),
                                        fill_value=1 / self.num_parts, device=device)
                _part_prob[:, :, :, foreground] = part_prob
                part_prob = _part_prob

        else:
            final_sdf = torch.full(size=(batchsize, 1, num_ray, num_on_ray), fill_value=0.05, device=device)
            sdf = torch.full(size=(batchsize, 1, num_ray, num_on_ray), fill_value=0.05, device=device)
            if coarse_sample or not_return_color:
                color = None
            else:
                color = torch.full(size=(batchsize, 3, num_ray, num_on_ray), fill_value=0., device=device)
            part_prob = torch.full(size=(batchsize, self.num_parts, 1, num_ray, num_on_ray),
                                   fill_value=1 / self.num_parts, device=device)

        return_data = {"density": final_sdf,  # shape: (B, 1, num_ray, num_on_ray)
                       "color": color,  # shape: (B, 3, num_ray, num_on_ray)
                       "part_weight": part_prob,  # shape: (B, num_parts, 1, num_ray, num_on_ray)
                       "sdf_scale": self.sdf_scale,  # scalar
                       "ellipsoid_sdf": sdf,  # shape: (B, num_parts, num_ray, num_on_ray)
                       "radius": radius,
                       "xyz": transformed_xyz,
                       }

        return return_data

    def joint_root_locations(self, joint_rotation: torch.Tensor, joint_translation: torch.Tensor
                             ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns part center and joint candidates
        Args:
            joint_rotation: (B, num_parts, 3, 3)
            joint_translation:(B, num_parts, 3, 1)

        Returns:
            joint_candidates: (B, num_parts, 3, n_candidates)
            part_center: (B, num_parts, 3, 1)

        """
        radius = torch.clamp_min(self.radius, 0.01)
        joint_candidates = self.child_root.permute(1, 0) * radius[:, :, None]  # (num_parts, 3, n_cand)
        joint_candidates = torch.matmul(joint_rotation, joint_candidates) + joint_translation
        part_center = joint_translation

        return joint_candidates, part_center

    def points_on_primitives(self, joint_rotation: torch.Tensor, joint_translation: torch.Tensor,
                             num_points: int = 200
                             ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Sample points from ellipsoid surface
        Args:
            joint_rotation: (B, num_parts, 3, 3)
            joint_translation: (B, num_parts, 3, 1)
            num_points: number of points to sample

        Returns:
            points: sampled points, (B, num_points, num_parts, 3)
        """
        # sample points from sphere
        radius = torch.clamp_min(self.radius, 0.01)
        batchsize, num_parts, _, _ = joint_rotation.shape
        _z = joint_rotation.new_empty(batchsize, num_points, num_parts, 1).uniform_(-1, 1)
        _theta = joint_rotation.new_empty(batchsize, num_points, num_parts, 1).uniform_(-np.pi, np.pi)
        _x = torch.sqrt(1 - _z ** 2) * torch.cos(_theta)
        _y = torch.sqrt(1 - _z ** 2) * torch.sin(_theta)
        points = torch.cat([_x, _y, _z], dim=3) * radius  # (B, n_pts, n_part, 3)

        points = points.permute(0, 2, 3, 1)
        points = torch.matmul(joint_rotation, points) + joint_translation

        return points, None


def ImplicitDecoder(kwargs: dict):
    """

    Args:
        kwargs:

    Returns:

    """
    return EllipsoidDecoder(**kwargs)
