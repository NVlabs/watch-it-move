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

from typing import Optional, Tuple, Dict, Any

import cv2
import torch
from torch import nn

from utils.graph_utils import get_parent_and_children_id
from utils.model_utils import PixelSampler, get_pose_trajectory
from utils.render_utils import (fine_sample, rotate_pose, rotation_matrix, volume_rendering)
from .decoder import ImplicitDecoder


class SingleVideoPartDecomposition(nn.Module):
    def __init__(self, config: Dict):
        """

        Args:
            config:
        """
        super(SingleVideoPartDecomposition, self).__init__()
        self.video_length = config.video_length
        self.config = config
        self.size = config.size
        self.num_parts = config.num_parts
        self.joint_trajectory = get_pose_trajectory(config)

        self.background_color = config.background_color
        self.decoder = ImplicitDecoder(config.decoder_params)
        self.surface_loss = config.surface_loss
        self.structure_loss = config.structure_loss

        self.pixel_sampler = PixelSampler(config.pixel_sampler)

        initial_connection = torch.stack([torch.arange(self.num_parts - 1),
                                          torch.arange(self.num_parts - 1) + 1], dim=1)  # (n_parts - 1, 2)
        self.register_buffer("joint_connection", initial_connection.long())
        self.register_buffer("child_ids", torch.zeros(self.num_parts - 1, 2).long())

        self.center_coef = config.center_coef_for_structure_loss

    def segmentation_rendering(self, density: torch.Tensor, part_weight: torch.Tensor,
                               fine_depth: torch.Tensor, one_hot=False) -> torch.Tensor:
        """render segmentation image

        Args:
            density (torch.Tensor): shape: (B, 1, num_ray, num_on_ray)
            part_weight (torch.Tensor): weight per part, shape: (B, num_parts, 1, num_ray, num_on_ray)
            fine_depth (torch.Tensor): depth for fine sampling
            one_hot: return part label or not

        Returns:
            torch.Tensor: segmentation image
        """
        batchsize, num_parts, _, _, _ = part_weight.shape
        if one_hot:
            part_color = torch.eye(num_parts,
                                   device=density.device)[None, :, :, None, None]  # (1, num_parts, num_parts, 1, 1)
            color = torch.softmax(part_weight * 60, dim=1) * part_color
            color = color.sum(dim=1)  # sum across parts
        else:
            part_idx = torch.arange(num_parts)
            part_color = torch.stack([torch.div(part_idx, 9, rounding_mode='trunc'),
                                      torch.div(part_idx, 3, rounding_mode='trunc') % 3, part_idx % 3], dim=1) - 1
            part_color[::2] = part_color.flip(dims=(0,))[1 - num_parts % 2::2]  # (num_parts, 3)
            part_color = part_color[None, :, :, None, None]  # (1, num_parts, 3, 1, 1)

            color = part_weight * part_color.cuda(non_blocking=True)
            color = color.sum(dim=1)  # sum across parts

        # neural rendering
        (segmentation_color, _, _, _) = volume_rendering(density, color, fine_depth)

        return segmentation_color

    def render_from_feature(self, decoder: nn.Module, joint_rotation: torch.Tensor,
                            joint_translation: torch.Tensor, pixel_location: torch.Tensor,
                            inv_intrinsics: torch.Tensor, segmentation: bool = False,
                            segmentation_label: bool = False,
                            ) -> dict:
        """render pixel color/mask from joint posture

        Args:
            decoder
            joint_rotation (torch.Tensor): shape: (B, num_parts, 3, 3)
            joint_translation (torch.Tensor): shape: (B, num_parts, 3, 1)
            pixel_location (torch.Tensor): [description]
            inv_intrinsics (torch.Tensor): [description]
            segmentation: if true, return segmentation value
            segmentation_label:

        Returns:
            dict: [description]
        """
        # 3d point sampling (fine sampling)
        fine_location, fine_depth, ray_direction = fine_sample(decoder, joint_rotation,
                                                               joint_translation, pixel_location,
                                                               inv_intrinsics)
        # decode density and color for each location
        decoder_output = decoder(fine_location, joint_rotation, joint_translation)
        density = decoder_output["density"]
        color = decoder_output["color"]
        part_weight = decoder_output["part_weight"]
        sdf_scale = decoder_output.get("sdf_scale")

        # volume rendering
        density_for_rendering = density if sdf_scale is None else density * sdf_scale
        (rendered_color, rendered_mask,
         rendered_disparity, vr_weight) = volume_rendering(density_for_rendering, color, fine_depth)
        return_dict = {"rendered_color": rendered_color,  # (B, 3, num_ray)
                       "rendered_mask": rendered_mask,  # (B, num_ray)
                       "rendered_disparity": rendered_disparity,  # (B, num_ray)
                       "density": density,  # (B, num_parts, num_ray, num_on_ray)
                       "fine_location": fine_location,  # (B, 3, num_ray, num_on_ray)
                       }

        # segmentation rendering
        if segmentation:
            segmentation_color = self.segmentation_rendering(density_for_rendering, part_weight, fine_depth,
                                                             one_hot=segmentation_label)
            return_dict["segmentation_color"] = segmentation_color

        if self.training:
            batchsize, _, num_ray, num_points = fine_location.shape
            sample_index = torch.randint(num_points, size=(batchsize, 1, num_ray, 4), device=fine_location.device)
            sample_index = sample_index.expand(-1, 3, -1, -1)
            random_location = torch.gather(fine_location, dim=3, index=sample_index).data  # (B, 3, num_ray, 4)
            random_location.requires_grad = True
            sdf = decoder(random_location, joint_rotation, joint_translation,
                          not_return_color=True)["density"]  # shape: (B, 1, num_ray, 4)
            sdf_grad = torch.autograd.grad([sdf.sum()], [random_location], create_graph=True)[0]  # (B, 3, num_ray, 4)
            return_dict["sdf_grad"] = sdf_grad

        return return_dict

    def render_entire_img_from_feature(self, joint_rotation: torch.Tensor,
                                       joint_translation: torch.Tensor, inv_intrinsics: torch.Tensor,
                                       ray_batchsize: int = 1000,
                                       segmentation_label: bool = False, rotate_angle: float = 0,
                                       ) -> Dict[str, Any]:
        """

        Args:
            joint_rotation:
            joint_translation:
            inv_intrinsics:
            ray_batchsize:
            segmentation_label:
            rotate_angle:

        Returns:

        """
        device = joint_rotation.device
        if rotate_angle != 0:
            R = rotation_matrix(rotate_angle)
            _zeros = torch.tensor([0, 0, 0, 1], dtype=torch.float,
                                  device=device)[None, None, None].expand(1, self.num_parts, 1, 4)
            rotated_pose = torch.cat([torch.cat([joint_rotation, joint_translation], dim=-1),
                                      _zeros], dim=-2)
            rotated_pose = rotate_pose(rotated_pose, R)

            joint_rotation = rotated_pose[:, :, :3, :3]
            joint_translation = rotated_pose[:, :, :3, 3:]

        rendered_colors = []
        rendered_masks = []
        rendered_disparities = []
        segmentation_colors = []

        # sample ray
        y_loc, x_loc = torch.meshgrid(torch.arange(self.size, device="cuda"),
                                      torch.arange(self.size, device="cuda"), indexing='ij')
        pixel_location = torch.stack([x_loc.reshape(-1), y_loc.reshape(-1)])  # (2, size**2)
        pixel_location = pixel_location[None]

        for i in range(0, self.size ** 2, ray_batchsize):
            pixel_batch = pixel_location[:, :, i:i + ray_batchsize]
            result_dict = self.render_from_feature(self.decoder, joint_rotation,
                                                   joint_translation,
                                                   pixel_batch, inv_intrinsics,
                                                   segmentation=True,
                                                   segmentation_label=segmentation_label)
            rendered_color = result_dict["rendered_color"]
            rendered_mask = result_dict["rendered_mask"]
            rendered_disparity = result_dict["rendered_disparity"]
            segmentation_color = result_dict["segmentation_color"]

            rendered_colors.append(rendered_color)
            rendered_masks.append(rendered_mask)
            rendered_disparities.append(rendered_disparity)
            segmentation_colors.append(segmentation_color)

        rendered_colors = torch.cat(rendered_colors, dim=2)
        segmentation_colors = torch.cat(segmentation_colors, dim=2)
        rendered_masks = torch.cat(rendered_masks, dim=1)
        rendered_disparities = torch.cat(rendered_disparities, dim=1)

        rendered_colors = rendered_colors.reshape(3, self.size, self.size)
        segmentation_colors = segmentation_colors.reshape(-1, self.size, self.size)
        rendered_masks = rendered_masks.reshape(self.size, self.size)
        rendered_disparities = rendered_disparities.reshape(self.size, self.size)

        return_dict = {"rendered_colors": rendered_colors,  # (3, size, size)
                       "rendered_masks": rendered_masks,  # (size, size)
                       "rendered_disparities": rendered_disparities,  # (size, size)
                       "segmentation_colors": segmentation_colors}  # (3, size, size)

        return return_dict

    def points_on_primitives(self, joint_rotation: torch.Tensor, joint_translation: torch.Tensor,
                             num_points: int = 200,
                             ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """

        Args:
            joint_rotation:
            joint_translation:
            num_points:

        Returns:

        """
        return self.decoder.points_on_primitives(joint_rotation, joint_translation, num_points)

    @staticmethod
    def _to_image_coord(x: torch.Tensor, intrinsics: torch.Tensor):
        """

        Args:
            x:
            intrinsics:

        Returns:

        """
        point_on_img = torch.matmul(intrinsics[:, None], x)
        point_on_img = point_on_img[:, :, :2, :] / point_on_img[:, :, 2:, :]

        return point_on_img  # (B, num_parts, 2, num_points)

    def sample_surface_points(self, joint_rotation: torch.Tensor, joint_translation: torch.Tensor,
                              num_points: int = 200,
                              intrinsics: Optional[torch.Tensor] = None
                              ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """

        Args:
            joint_rotation:
            joint_translation:
            num_points:
            intrinsics:

        Returns:

        """
        surface_points, _ = self.points_on_primitives(joint_rotation, joint_translation,
                                                      num_points=num_points)  # (B, num_parts, 3, num_points)

        if intrinsics is not None:
            surface_points_2d = self._to_image_coord(surface_points, intrinsics)  # (B, num_parts, 2, num_points)
        else:
            surface_points_2d = None

        return surface_points, surface_points_2d

    def joint_root_locations(self, joint_rotation: torch.Tensor, joint_translation: torch.Tensor,
                             intrinsics: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """

        Args:
            joint_rotation:
            joint_translation:
            intrinsics:

        Returns:

        """
        child_root, self_root = self.decoder.joint_root_locations(
            joint_rotation, joint_translation)  # (B, num_parts, 3, num_child), (B, num_parts, 3, 1)

        if intrinsics is not None:  # for visualization
            child_root = self._to_image_coord(child_root, intrinsics)  # (B, num_parts, 2, num_child)
            self_root = self._to_image_coord(self_root, intrinsics)  # (B, num_parts, 2, 1)

        return child_root, self_root

    def forward(self, frame_id: torch.Tensor, camera_rot: torch.Tensor,
                camera_trans: torch.Tensor,
                inv_intrinsics: torch.Tensor, num_ray: int, mask: torch.Tensor,
                ) -> Dict:
        """

        Args:
            frame_id: frame id of image to generate
            camera_rot: rotation matrix of camera
            camera_trans: translation of camera
            inv_intrinsics (torch.Tensor): [description]
            num_ray (int): [description]
            mask: bbox mask of object

        Returns:
            rendered_color (torch.Tensor): shape: (B, 3, num_ray)
            rendered_mask (torch.Tensor): shape: (B, num_ray)
            rendered_disparity (torch.Tensor): shape: (B, num_ray)
            pixel_location (torch.Tensor): shape: (B, 2, num_ray)
        """
        batchsize = frame_id.shape[0]

        # sample ray based on org view
        rate = 64
        pixel_location = self.pixel_sampler(self.size, num_ray, batchsize, mask, coarse_rate=rate, stride=rate)

        # joint pose
        # # in world
        joint_rotation, joint_translation = self.joint_trajectory(frame_id.float())

        # # in camera coordinate
        joint_rotation = torch.matmul(camera_rot[:, None], joint_rotation)
        joint_translation = torch.matmul(camera_rot[:, None], joint_translation) + camera_trans[:, None]

        # render
        result_dict = self.render_from_feature(self.decoder, joint_rotation, joint_translation,
                                               pixel_location, inv_intrinsics)

        # add background
        rendered_color = result_dict["rendered_color"]
        rendered_mask = result_dict["rendered_mask"]
        background_color = self.background_color
        result_dict["rendered_color"] = rendered_color + (1 - rendered_mask[:, None]) * background_color

        # others
        result_dict["pixel_location"] = pixel_location
        result_dict["joint_rotation"] = joint_rotation
        result_dict["joint_translation"] = joint_translation

        # # joint 2d
        intrinsics = torch.inverse(inv_intrinsics)
        joint_2d = torch.matmul(intrinsics[:, None], joint_translation)
        joint_2d = joint_2d[:, :, :2, 0] / joint_2d[:, :, 2:, 0]  # (B, n_kpts, 2)
        result_dict["joint_2d"] = joint_2d

        if self.surface_loss:
            surface_points, surface_points_2d = self.sample_surface_points(joint_rotation, joint_translation,
                                                                           intrinsics=intrinsics)
            result_dict["surface_points"] = surface_points
            result_dict["surface_points_2d"] = surface_points_2d
        if self.structure_loss:
            child_root, self_root = self.joint_root_locations(joint_rotation, joint_translation,
                                                              intrinsics=None)
            result_dict["child_root_locations"] = child_root
            result_dict["self_root_locations"] = self_root

        return result_dict

    def render_entire_img(self, frame_id: torch.Tensor, camera_rot: torch.Tensor,
                          camera_trans: torch.Tensor,
                          inv_intrinsics: torch.Tensor, ray_batchsize: int = 1000,
                          segmentation_label: bool = False,
                          rotate_angle: float = 0,
                          relative_rotation: Optional[torch.Tensor] = None,
                          inv_intrinsics_decoder: Optional[torch.Tensor] = None,
                          manipulate_pose_config: Optional[Dict[str, Any]] = None,
                          part_pose: Optional[Tuple] = None
                          ) -> Dict[str, Any]:
        """render an entire img with the same setting as forward()

        Args:
            inv_intrinsics (torch.Tensor): [description]
            ray_batchsize
            segmentation_label
            rotate_angle
            relative_rotation
            inv_intrinsics_decoder : intrinsic for decoder
            manipulate_pose_config (Optional[Dict[str, Any]]): pose manipulation configuration
            part_pose: to input pose directly (rotation, translation) of each part
        Returns:
            rendered_colors (torch.Tensor): shape: (3, size, size)
            rendered_masks (torch.Tensor): shape: (size, size)
            rendered_disparities (torch.Tensor): shape: (size, size)
        """
        assert part_pose is not None or frame_id.shape[0] == 1, "batchsize should be 1"

        if inv_intrinsics_decoder is None:
            inv_intrinsics_decoder = inv_intrinsics

        intrinsics = torch.inverse(inv_intrinsics_decoder)

        # joint pose
        # # in world
        if part_pose is not None:
            joint_rotation, joint_translation = part_pose
        else:
            joint_rotation, joint_translation = self.joint_trajectory(frame_id.float())

        # manipulate pose if necessary
        if manipulate_pose_config is not None:
            candidates, _ = self.joint_root_locations(joint_rotation, joint_translation)
            joint_rotation, joint_translation = self.manipulate_pose(joint_rotation, joint_translation,
                                                                     candidates, manipulate_pose_config)
        # rotate
        if rotate_angle != 0:
            R = rotation_matrix(rotate_angle, axis="z")
            _zeros = torch.tensor([0, 0, 0, 1], dtype=torch.float,
                                  device=R.device)[None, None, None].expand(1, self.num_parts, 1, 4)
            rotated_pose = torch.cat([torch.cat([joint_rotation, joint_translation], dim=-1),
                                      _zeros], dim=-2)
            rotated_pose = rotate_pose(rotated_pose, R)

            joint_rotation = rotated_pose[:, :, :3, :3]
            joint_translation = rotated_pose[:, :, :3, 3:]

        # # in camera coordinate
        joint_rotation = torch.matmul(camera_rot[:, None], joint_rotation)
        joint_translation = torch.matmul(camera_rot[:, None], joint_translation) + camera_trans[:, None]

        if relative_rotation is not None:  # multiview
            relative_rot = relative_rotation[:, None, :3, :3]
            relative_trans = relative_rotation[:, None, :3, 3:]

            joint_rotation = torch.matmul(relative_rot, joint_rotation)
            joint_translation = torch.matmul(relative_rot, joint_translation) + relative_trans

        rendered_dict = self.render_entire_img_from_feature(joint_rotation,
                                                            joint_translation, inv_intrinsics_decoder,
                                                            ray_batchsize,
                                                            segmentation_label, 0)

        # # joint 2d
        joint_2d = torch.matmul(intrinsics[:, None], joint_translation)
        joint_2d = joint_2d[:, :, :2, 0] / joint_2d[:, :, 2:, 0]  # (B, n_kpts, 2)
        self.joint_2d = joint_2d.detach()

        child_root, self_root = self.joint_root_locations(joint_rotation, joint_translation,
                                                          intrinsics=intrinsics)
        self.child_root = child_root
        self.self_root = self_root

        return rendered_dict

    def manipulate_pose(self, joint_rotation: torch.Tensor, joint_translation: torch.Tensor,
                        candidates: torch.Tensor,
                        manipulate_pose_config: Dict) -> Tuple[torch.Tensor, torch.tensor]:
        """manipulate pose

        Args:
            joint_rotation: (B, n_parts, 3, 3)
            joint_translation: (B, n_parts, 3, 1)
            candidates (torch.Tensor):  (B, n_parts, 3, n_child)
            manipulate_pose_config:

        Returns:
            joint_rotation
            joint_translation

        """
        assert joint_rotation.shape[0] == 1  # batchsize == 1

        # sort direct graph
        joint_connection = self.joint_connection.detach().cpu().numpy()
        child_id = self.child_ids.cpu().numpy()
        root_id = manipulate_pose_config["root_id"]
        device = joint_rotation.device

        # store parent ids
        parent_id, children_ids, selected_candidates = get_parent_and_children_id(self.num_parts, joint_connection,
                                                                                  child_id, root_id)

        for conf in manipulate_pose_config["motion_config"]:
            move_id = conf["move_id"]
            rodrigues = conf["rodrigues"]  # (3, )

            parent = parent_id[move_id]
            if parent >= 0:
                cand_move = selected_candidates[move_id][0]
                cand_parent = selected_candidates[move_id][1]
                rot_center = (candidates[0, move_id, :, cand_move, None] +
                              candidates[0, parent, :, cand_parent, None]) / 2  # (3, 1)

                rot_matrix = cv2.Rodrigues(rodrigues)[0]  # (3, 3)

                rot_matix = torch.tensor(rot_matrix, device=device).float()
                joint_rotation_to_move = joint_rotation[:, children_ids[move_id]].clone()
                joint_translation_to_move = joint_translation[:, children_ids[move_id]].clone()

                joint_translation[:, children_ids[move_id]] = torch.matmul(rot_matix,
                                                                           joint_translation_to_move - rot_center) + rot_center
                joint_rotation[:, children_ids[move_id]] = torch.matmul(rot_matix, joint_rotation_to_move)

        return joint_rotation, joint_translation
