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
from typing import Any

import cv2
import numpy as np
import torch
from easymocap.smplmodel import load_model

from EasyMocap.easymocap.smplmodel.lbs import lbs as extract_bone


class PoseLoader:
    def __init__(self, smpl_model_path: str):
        """

        Args:
            smpl_model_path:
        """
        self.body_model = load_model(
            gender="neutral",
            model_type="smplx",
            model_path=smpl_model_path,
            device="cpu")

    def __call__(self, smpl_param: Any) -> np.ndarray:
        """

        Args:
            smpl_param:

        Returns:

        """
        Rh = np.array(smpl_param["Rh"])  # 1 x 3
        Th = np.array(smpl_param["Th"])  # 1 x 3
        poses = np.array(smpl_param["poses"])[:, :72]  # 1 x 72
        shapes = smpl_param["shapes"]  # 1 x 10
        expression = smpl_param["expression"]  # 1 x 10

        shapes = torch.tensor(shapes).float()
        expression = torch.tensor(expression).float()
        shapes = torch.cat([shapes, expression], dim=1)
        poses = torch.tensor(poses).float()
        v_template = self.body_model.j_v_template
        joints, transformation = extract_bone(shapes, poses, v_template,
                                              self.body_model.j_shapedirs, None,
                                              self.body_model.j_J_regressor, self.body_model.parents,
                                              None, dtype=self.body_model.dtype,
                                              use_pose_blending=False)
        bone_pose = transformation.clone()
        bone_pose[:, :, :3, 3] = joints

        trans = np.eye(4)
        trans[:3, :3] = cv2.Rodrigues(Rh[0])[0]
        trans[:3, 3] = Th

        bone_pose_world = np.matmul(trans, bone_pose.numpy()[0])

        return bone_pose_world
