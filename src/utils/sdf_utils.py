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

import torch


def f(radius: torch.Tensor, x: torch.Tensor, lam: torch.Tensor) -> torch.Tensor:
    """func to optimize

    Args:
        radius: radius of ellipsoids, (..., n_part, 3, 1)
        x: position, (..., n_part, 3, n_pts)
        lam: Lagrange multiplier, (..., n_part, n_pts)

    Returns:

    """
    lam = lam.unsqueeze(-2)
    h = radius.square() * x.square() / torch.clamp_min((radius.square() + lam).square(), 1e-15)
    h = torch.sum(h, dim=-2)

    return h  # (..., n_part, n_pts)


def d_f(radius: torch.Tensor, x: torch.Tensor, lam: torch.Tensor) -> torch.Tensor:
    """derivative of f

    Args:
        radius: radius of ellipsoids, (..., n_part, 3, 1)
        x: position, (..., n_part, 3, n_pts)
        lam: Lagrange multiplier, (..., n_part, n_pts)

    Returns:

    """
    lam = lam.unsqueeze(-2)
    eps = (((radius.square() + lam) > 0) * 2 - 1) * 1e-20
    h = radius.square() * x.square() / ((radius.square() + lam) ** 3 + eps)
    h = -2 * torch.sum(h, dim=-2)

    return h  # (..., n_part, n_pts)


def newton_step(radius: torch.Tensor, x: torch.Tensor, lam: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """

    Args:
        radius: radius of ellipsoids, (..., n_part, 3, 1)
        x: position, (..., n_part, 3, n_pts)
        lam: Lagrange multiplier, (..., n_part, n_pts)

    Returns:

    """
    with torch.no_grad():
        diff = 1 - f(radius, x, lam)
        df = d_f(radius, x, lam)
        eps = ((df > 0) * 2 - 1) * 1e-15
        update = diff / (df + eps)

    return lam + update, diff


def search_lam(radius: torch.Tensor, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """

    Args:
        radius: radius of ellipsoids, (..., n_part, 3)
        x: position, (..., n_part, 3, n_pts)

    Returns:

    """
    radius = radius.unsqueeze(-1)
    lam = torch.max(radius * x.abs() - radius.square(), dim=-2)[0]  # (..., n_part, n_pts)
    diff = torch.tensor(0)
    for _ in range(10):
        lam, diff = newton_step(radius, x, lam)

    valid = torch.lt(diff, 1e-5)

    valid_lam = torch.lt(-torch.square(radius.min(dim=-2)[0]), lam)
    valid = torch.logical_and(valid, valid_lam)

    return lam, valid


def lam_to_sdf(radius: torch.Tensor, x: torch.Tensor, lam: torch.Tensor) -> torch.Tensor:
    """

    Args:
        radius: radius of ellipsoids, (..., n_part, 3)
        x: position, (..., n_part, 3, n_pts)
        lam: Lagrange multiplier, (..., n_part, n_pts)

    Returns:

    """
    with torch.no_grad():
        radius = radius.unsqueeze(-1)
        lam = lam.unsqueeze(-2)
        foot_on_sphere = radius / (radius.square() + lam) * x

    # differentiable from here!
    foot_on_ellipsoid = foot_on_sphere * radius  # (..., n_part, 3, n_pts)
    with torch.no_grad():
        sign = torch.sign(torch.sum(x.square() / radius.square(), dim=-2) - 1)
    sdf = torch.norm(foot_on_ellipsoid - x, dim=-2) * sign  # (..., n_part, n_pts)

    return sdf


@torch.jit.script
def ellipsoid_sdf(radius: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """

    Args:
        radius: radius of ellipsoids, (..., n_part, 3)
        x: position, (..., n_part, 3, n_pts)

    Returns:

    """
    lam, valid = search_lam(radius, x)
    sdf = lam_to_sdf(radius, x, lam)
    min_sdf = -radius.min(dim=-1)[0].unsqueeze(-1)
    sdf = torch.where(valid, sdf, min_sdf)
    sdf = torch.where(sdf < min_sdf, min_sdf, sdf)

    return sdf
