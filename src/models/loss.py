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

from typing import Tuple, Dict, Any, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.render_utils import gather_pixel
from utils.train_utils import grid_coordinates


def joint_separation_loss(joint_location: torch.Tensor, sigma: float = 6) -> torch.Tensor:
    """maximize distance between parts

    Args:
        joint_location (torch.Tensor): 2D/3D joint location on image, (B, num_parts, 2or3)
        sigma: Variance of normal distribution
    Returns:
        torch.Tensor: loss
    """
    batchsize, num_parts, _ = joint_location.shape
    distance = torch.sum((joint_location[:, :, None] - joint_location[:, None]) ** 2, dim=-1)
    exp_dist = torch.exp(-distance / (2 * sigma ** 2))  # (B, num_parts, num_parts)
    exp_dist = exp_dist * (1 - torch.eye(num_parts, device=joint_location.device))
    loss = exp_dist.sum() / (batchsize * (num_parts - 1))

    return loss


class SupervisedLoss:
    def __init__(self, config: dict, model: nn.Module, ddp: bool, coarse_rate: int = 64):
        """

        Args:
            config:
            model:
            ddp:
            coarse_rate:
        """
        super().__init__()
        self.config = config
        self.mse = nn.MSELoss()
        self.model = model
        self.ddp = ddp
        self.coarse_rate = coarse_rate

    def __call__(self, gt_img: torch.Tensor, gt_mask: torch.Tensor, model_output_dict: dict,
                 pull_rigid_parts: bool = False) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """

        Args:
            gt_img:
            gt_mask:
            model_output_dict:
            pull_rigid_parts:

        Returns:

        """
        gen_img = model_output_dict["rendered_color"]
        gen_mask = model_output_dict["rendered_mask"]
        pixel_location = model_output_dict["pixel_location"]
        device = gen_img.device
        loss_dict = {}
        l2_loss = self.l2_loss(gt_img, gt_mask, gen_img, gen_mask, pixel_location)
        loss = l2_loss
        loss_dict["l2_loss"] = l2_loss.item()

        if "surface_points_2d" in model_output_dict:
            surface_points_2d = model_output_dict["surface_points_2d"]
            surface_loss_coef = self.config.surface_loss_coef
            if surface_loss_coef > 0:
                surface_2d_loss = self.surface_2d_loss(surface_points_2d, gt_mask)
                loss += surface_2d_loss * surface_loss_coef
                loss_dict["surface_loss"] = surface_2d_loss.item()

        if "child_root_locations" in model_output_dict:
            child_root_locations = model_output_dict["child_root_locations"]
            self_root_locations = model_output_dict["self_root_locations"]

            structure_loss_coef = self.config.structure_loss_coef
            if structure_loss_coef > 0:
                structure_loss = self.structure_loss_v3(self_root_locations, child_root_locations)
                loss += structure_loss * structure_loss_coef
                loss_dict["structure_loss"] = structure_loss.item()

        if "joint_2d" in model_output_dict:
            joint_2d_loss_coef = self.config.joint_2d_loss_coef
            if joint_2d_loss_coef > 0:
                joint_2d = model_output_dict["joint_2d"]  # (B, num_parts, 2)
                joint_2d_loss = self.joint_2d_loss(joint_2d, gt_mask)

                loss += joint_2d_loss * joint_2d_loss_coef
                loss_dict["joint_2d_loss"] = joint_2d_loss.item()

        if "joint_3d_separation_loss_coef" in self.config:
            joint_3d_separation_loss_coef = self.config.joint_3d_separation_loss_coef
            if joint_3d_separation_loss_coef > 0:
                joint_translation = model_output_dict["joint_translation"]
                joint_3d_separation_loss = joint_separation_loss(joint_translation[:, :, :, 0], sigma=0.05)
                loss += joint_3d_separation_loss * joint_3d_separation_loss_coef
                loss_dict["joint_3d_separation_loss"] = joint_3d_separation_loss.item()

        if pull_rigid_parts:
            pull_rigid_parts_loss_coef = self.config.pull_rigid_parts_loss_coef
            if pull_rigid_parts_loss_coef is not None and pull_rigid_parts_loss_coef > 0:
                pull_rigid_parts_loss = self.pull_rigid_parts(device)
                loss += pull_rigid_parts_loss * pull_rigid_parts_loss_coef
                loss_dict["pull_rigid_parts_loss"] = pull_rigid_parts_loss.item()

        return loss, loss_dict

    def l2_loss(self, gt_img: torch.Tensor, gt_mask: torch.Tensor, gen_img: torch.Tensor,
                gen_mask: torch.Tensor, pixel_location: torch.Tensor) -> Union[float, torch.Tensor]:
        """
        L2 loss for RGB and mask
        Args:
            gt_img: (B, 3, img_size, img_size)
            gt_mask: (B, img_size, img_size)
            gen_img: (B, 3, img_size, img_size)
            gen_mask: (B, img_size, img_size)
            pixel_location: sampled pixels

        Returns:

        """

        reliable_gt_mask = gt_mask <= 1
        valid_mask = (gt_mask.sum(dim=(1, 2)) > 1).float()  # (B, )
        gt_img = gather_pixel(gt_img, pixel_location)
        if gt_mask is not None:
            # multiply weight
            dilate_mask = F.max_pool2d(gt_mask[:, None], self.coarse_rate, self.coarse_rate, 0)
            dilate_mask = F.interpolate(dilate_mask, scale_factor=self.coarse_rate, mode="nearest")
            dilate_mask = gather_pixel(dilate_mask[:, 0], pixel_location)
            weight = torch.where(dilate_mask > 0.5, 2, 1)

            if pixel_location.isnan().any():
                return 0

            gt_mask = gather_pixel(gt_mask, pixel_location)
            reliable_gt_mask = gather_pixel(reliable_gt_mask, pixel_location)
            gt_img = gt_img * weight[:, None] * reliable_gt_mask[:, None]
            gt_mask = gt_mask * weight * valid_mask[:, None, None] * reliable_gt_mask
            gen_img = gen_img * weight[:, None] * reliable_gt_mask[:, None]
            gen_mask = gen_mask * weight * valid_mask[:, None, None] * reliable_gt_mask

            return self.mse(gt_img, gen_img) + self.mse(gt_mask, gen_mask) * self.config.mask_loss_multiplier
        else:
            return self.mse(gt_img, gen_img)

    def surface_2d_loss(self, surface_points_2d: torch.Tensor, gt_mask: torch.Tensor,
                        pixel_radius_threshold: float = 3) -> torch.Tensor:
        """surface distance in image

        Args:
            surface_points_2d (torch.Tensor):   # (B, num_parts, 2, num_points)
            gt_mask (torch.Tensor): # (B, size, size)
            pixel_radius_threshold (float): Ignore distances smaller than this.

        Returns:
            torch.Tensor: [description]
        """
        batchsize, img_size, _ = gt_mask.shape
        valid_mask = (gt_mask.sum(dim=(1, 2)) > 1).float()

        def _foreground_sampler(img_size: int, num_ray: int, mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
            """uniformly sample from foreground mask

            Args:
                img_size (int): image size
                num_ray (int): number of points to sample
                mask (int): shape: (B, img_size, img_size)

            Returns:
                coordinates: sampled coordinates, shape: (B, num_ray, 2)
                mask_value: mask value at sampled location
            """
            mask = mask.reshape(-1, img_size ** 2)
            noised_mask = (mask > 0.5) + torch.empty_like(mask, dtype=torch.float).uniform_(0, 0.1)
            _, coordinates = torch.topk(noised_mask, num_ray, dim=1, sorted=False)
            mask_value = torch.gather(mask, dim=1, index=coordinates)
            coordinates = torch.stack([coordinates % img_size,
                                       torch.div(coordinates, img_size, rounding_mode='trunc')], dim=-1) + 0.5

            return coordinates, mask_value

        surface_points_2d = surface_points_2d.permute(0, 1, 3, 2).reshape(batchsize, -1, 2)
        # chamfer
        with torch.no_grad():
            # randomly sample foreground pixels
            grid, gt_mask = _foreground_sampler(img_size, img_size ** 2 // 16, gt_mask)
            reliable_gt_mask = gt_mask == 1
            reliable_gt_background = gt_mask == 0

            # distance
            torch.cuda.empty_cache()
            distance = torch.sum((grid[:, :, None].half() - surface_points_2d[:, None].half()) ** 2, dim=-1)
            distance = distance + reliable_gt_background[:, :, None] * 1e10  # mask outside

            # nearest gt mask
            nearest_gt_mask_idx = torch.argmin(distance, dim=1)  # (B, num_parts * num_points)

            # nearest surface points
            nearest_surface_idx = torch.argmin(distance, dim=2)  # (B, num_grid_points)

        # surface -> mask
        nearest_gt_mask = torch.gather(grid.expand(batchsize, -1, 2), dim=1,
                                       index=nearest_gt_mask_idx[:, :, None].expand(-1, -1, 2))
        nearest_gt_mask_reliability = torch.gather(gt_mask, dim=1, index=nearest_gt_mask_idx) == 1
        surface_to_mask = torch.sum((surface_points_2d - nearest_gt_mask) ** 2, dim=-1) * valid_mask[:, None]
        surface_to_mask = surface_to_mask * nearest_gt_mask_reliability
        surface_to_mask = surface_to_mask.masked_fill(surface_to_mask < pixel_radius_threshold ** 2, 0)
        loss = surface_to_mask.mean()

        # mask -> surface
        nearest_surface = torch.gather(surface_points_2d, dim=1,
                                       index=nearest_surface_idx[:, :, None].expand(-1, -1, 2))
        mask_to_surface = torch.sum((grid - nearest_surface) ** 2, dim=-1)
        mask_to_surface = mask_to_surface.masked_fill(mask_to_surface < pixel_radius_threshold ** 2, 0)
        loss += torch.sum(mask_to_surface * reliable_gt_mask) / (reliable_gt_mask.sum() + 1e-8)

        return loss / img_size ** 2

    def structure_loss_v3(self, part_center: torch.Tensor, candidates: torch.Tensor) -> torch.Tensor:
        """iteratively optimize structure and joint location
        no distinction between self joint and child joint

        Args:
            part_center (torch.Tensor):   # (B, num_parts, 3, 1)
            candidates (torch.Tensor):   # (B, num_parts, 3, num_child)

        Returns:
            structure loss
        """
        batchsize = part_center.shape[0]
        num_child = candidates.shape[3]
        assert num_child > 1
        num_parts = part_center.shape[1]
        part_center = part_center.squeeze(dim=3)
        candidates = candidates.permute(0, 1, 3, 2).reshape(batchsize, -1, 3)
        center_coef = self.model.center_coef
        device = part_center.device

        with torch.no_grad():
            # l2 distance between self and self's parent's children
            relative_cand_position = candidates[:, :, None] - candidates[:, None]  # (B, n_all_child, n_all_child, 3)
            cand_distance = torch.sum(relative_cand_position ** 2, dim=-1).mean(dim=0)  # (n_all_child, n_all_child)

            if center_coef > 0:
                relative_center_position = part_center[:, :, None] - part_center[:, None]  # (B, n_parts, n_parts, 3)
                center_distance = torch.sum(relative_center_position ** 2, dim=-1).mean(dim=0)  # (n_parts, n_parts)

            if self.ddp:
                # sum across gpu
                torch.distributed.all_reduce(cand_distance)
                torch.distributed.all_reduce(center_distance)

            if not hasattr(self, "child_distance_ema"):
                self.child_distance_ema = cand_distance
                self.center_distance_ema = center_distance
            else:
                eps = 0.05
                self.child_distance_ema = cand_distance * eps + self.child_distance_ema * (1 - eps)
                self.center_distance_ema = center_distance * eps + self.center_distance_ema * (1 - eps)

            # structure step
            # choose best child pairs
            best_distance, best_idx = F.max_pool2d(-self.child_distance_ema[None, None], num_child,
                                                   return_indices=True)
            best_distance = -best_distance[0, 0]  # (n_parts, n_parts)
            best_idx_0 = (torch.div(best_idx[0, 0], (num_parts * num_child),
                                    rounding_mode='trunc')) % num_child  # (n_parts, n_parts)
            best_idx_1 = best_idx[0, 0] % num_child  # (n_parts, n_parts)

            # greedy estimation
            connectivity = torch.eye(num_parts, device=device, dtype=torch.long)  # (n_parts, n_parts)

            joint_connection = torch.zeros(num_parts - 1, 2, device=device, dtype=torch.long)
            child_ids = torch.zeros(num_parts - 1, 2, device=device, dtype=torch.long)

            if center_coef > 0:
                best_distance = best_distance + self.center_distance_ema * center_coef  # consider center position

            for j in range(num_parts - 1):  # there are n_parts-1 connection
                # find minimum distance
                invalid_connection_bias = connectivity * 1e10
                connected = torch.argmin(best_distance + invalid_connection_bias)
                connected_idx_0 = torch.div(connected, num_parts, rounding_mode='trunc')
                connected_idx_1 = connected % num_parts

                # update connectivity
                connectivity[connected_idx_0] = torch.maximum(connectivity[connected_idx_0].clone(),
                                                              connectivity[connected_idx_1].clone())
                connectivity[torch.where(connectivity[connected_idx_0] == 1)] = connectivity[connected_idx_0].clone()

                joint_connection[j, 0] = connected_idx_0
                joint_connection[j, 1] = connected_idx_1
                child_ids[j, 0] = best_idx_0[connected_idx_0, connected_idx_1]
                child_ids[j, 1] = best_idx_1[connected_idx_0, connected_idx_1]
            self.model.joint_connection.data = joint_connection
            self.model.child_ids.data = child_ids

        # parent-child distance of current minibatch
        # (B, num_parts - 1, 3)
        joint_index = joint_connection * num_child + child_ids
        joint_0 = torch.gather(candidates, dim=1,
                               index=(joint_index[:, 0])[None, :, None].expand(batchsize, -1, 3))
        joint_1 = torch.gather(candidates, dim=1,
                               index=(joint_index[:, 1])[None, :, None].expand(batchsize, -1, 3))
        parent_child_distance = torch.sum((joint_0 - joint_1) ** 2, dim=2)

        return parent_child_distance.mean()

    def joint_2d_loss(self, joint_2d: torch.Tensor, gt_mask: torch.Tensor) -> torch.Tensor:
        """estimated joints should be inside the foreground mask

        Args:
            joint_2d (torch.Tensor): 2D joint location on image, (B, num_parts, 2)
            gt_mask (torch.Tensor): (B, size, size)

        Returns:
            loss
        """
        batchsize, img_size, _ = gt_mask.shape
        reliable_fg_mask = gt_mask == 1
        valid_mask = (reliable_fg_mask.sum(dim=(1, 2)) > 1).float()

        with torch.no_grad():
            thin_ratio = 2
            grid = grid_coordinates(img_size // thin_ratio, joint_2d.device, scale=thin_ratio)
            gt_mask = gt_mask[:, ::thin_ratio, ::thin_ratio].reshape(batchsize, -1)
            reliable_fg_mask = gt_mask == 1
            # distance
            distance = torch.sum((grid[:, :, None] - joint_2d[:, None]) ** 2, dim=-1)  # (B, size**2, num_parts)
            distance = distance + (gt_mask[:, :, None] < 0.5) * 1e10  # mask outside

            # nearest gt mask
            nearest_gt_mask_idx = torch.argmin(distance, dim=1)  # (B, num_parts)
            # nearest joint
            nearest_joint_idx = torch.argmin(distance, dim=2)  # (B, num_grid_points)

        # joint -> mask
        nearest_gt_mask = torch.gather(grid.expand(batchsize, -1, 2), dim=1,
                                       index=nearest_gt_mask_idx[:, :, None].expand(-1, -1, 2))
        nearest_gt_mask_reliability = torch.gather(gt_mask, dim=1, index=nearest_gt_mask_idx) == 1
        joint_to_mask = torch.sum((joint_2d - nearest_gt_mask) ** 2, dim=-1) * valid_mask[:, None]
        joint_to_mask = joint_to_mask * nearest_gt_mask_reliability
        loss = joint_to_mask.mean()

        # mask -> joint
        nearest_joint = torch.gather(joint_2d, dim=1,
                                     index=nearest_joint_idx[:, :, None].expand(-1, -1, 2))
        mask_to_joint = torch.sum((grid - nearest_joint) ** 2, dim=-1)
        loss += torch.sum(mask_to_joint * reliable_fg_mask) / reliable_fg_mask.sum()

        return loss / img_size ** 2

    def pull_rigid_parts(self, device, thres=0.1) -> torch.Tensor:
        """
        Merge loss
        Args:
            device:
            thres: threshold for loss calculation

        Returns:
            loss

        """
        rotation, translation = self.model.joint_trajectory(
            torch.arange(self.model.video_length, device=device).float())
        relative_rotation = torch.matmul(rotation[:, :, None].transpose(-1, -2), rotation[:, None])
        relative_translation = torch.matmul(rotation[:, :, None].transpose(-1, -2),
                                            translation[:, None] - translation[:, :, None])
        mat = (relative_rotation.std(dim=0).mean(dim=(2, 3)) +  # std of rotation matrix
               relative_translation.std(dim=0).mean(dim=(2, 3)) * 3)  # std of translation matrix
        eye = torch.eye(mat.shape[0], device=device, dtype=torch.float)
        mat = mat * (1 - eye)  # mask diag elements
        thres = max(thres, (mat + eye * 1e4).min().item())
        loss = torch.mean(mat * torch.sigmoid((thres - mat) / thres).detach())

        return loss
