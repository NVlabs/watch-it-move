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

import argparse
import io
import os
import sys
from copy import deepcopy
from typing import List, Tuple

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch

sys.path.append(".")
from utils.visualization_utils import GenerateVideoFromConfig, render_and_get_joints


def draw_polygon(joints: np.ndarray, color: str) -> None:
    """

    Args:
        joints: (n, 2)
        color:

    Returns:

    """
    indices = []

    visited = np.zeros(len(joints))
    current_idx = joints[:, 0].argmin()
    current_angle = np.pi / 2
    indices.append(current_idx)

    while visited[current_idx] == 0:
        visited[current_idx] = 1
        vec = joints - joints[current_idx]
        angle = np.arctan2(vec[:, 1], vec[:, 0])
        rel_angle = (current_angle - angle) % (2 * np.pi)
        rel_angle[current_idx] = 1e4  # ignore self
        current_idx = rel_angle.argmin()
        current_angle = angle[current_idx]
        indices.append(current_idx)
    indices.append(indices[0])

    plt.plot(joints[indices, 0],
             joints[indices, 1], c=f"#{color}")


def main(config_path: str, default_config: str, cam_id: int) -> None:
    """

    Args:
        config_path:
        default_config:
        cam_id:

    Returns:

    """
    self = GenerateVideoFromConfig(config_path, default_config, args.iteration)
    model = self.model
    train_loader = self.train_loader

    with torch.no_grad():
        frame_id = torch.arange(train_loader.dataset.num_frames, dtype=torch.float, device="cuda")
        trajectory = model.joint_trajectory(frame_id)

    rotation, translation = trajectory
    relative_rotation = torch.matmul(rotation[:, :, None].transpose(-1, -2), rotation[:, None])
    relative_translation = torch.matmul(rotation[:, :, None].transpose(-1, -2),
                                        translation[:, None] - translation[:, :, None])
    mat = relative_rotation.std(dim=0).mean(dim=(2, 3)) + relative_translation.std(dim=0).mean(dim=(2, 3)) * 3
    mat = mat + mat.transpose(0, 1)
    mat = mat

    asort = (mat + torch.eye(model.num_parts, device="cuda") * 1e10).reshape(-1).argsort()
    merged = torch.stack([torch.div(asort, model.num_parts, rounding_mode='trunc'),
                          asort % model.num_parts], dim=1)[::2]
    val = mat[merged[:, 0], merged[:, 1]]
    for i in range(len(merged[:10])):
        print(merged[i].cpu().numpy(), val[i].item())

    frame_id = 0
    color_each_view = []
    mask_each_view = []
    segmentation_each_view = []
    gt_img_each_view = []
    gt_mask_each_view = []
    joint_2d_each_view = []
    background_each_view = []
    disparity_each_view = []
    child_root_each_view = []
    self_root_each_view = []

    (img, gt_mask, color, mask, disparity, segmentation, joint_2d, child_root,
     self_root, background) = render_and_get_joints(model, train_loader, frame_id, cam_id)
    color_each_view.append(color)
    mask_each_view.append(mask)
    segmentation_each_view.append(segmentation)
    gt_img_each_view.append(img)
    gt_mask_each_view.append(gt_mask)
    joint_2d_each_view.append(joint_2d)
    background_each_view.append(background)
    disparity_each_view.append(disparity)
    child_root_each_view.append(child_root)
    self_root_each_view.append(self_root)

    num_prune = 40

    i = 0

    joint_2d = joint_2d_each_view[i][0, :, :]
    child_root = child_root_each_view[i]

    joint_connection = model.joint_connection.cpu().numpy()
    child_ids = model.child_ids.cpu().numpy()

    new_to_old_ = {_: [_] for _ in range(model.num_parts)}
    old_to_new_ = {_: _ for _ in range(model.num_parts)}

    def create_figures(text: bool) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        """

        Args:
            text:

        Returns:

        """
        old_to_new = deepcopy(old_to_new_)
        new_to_old = deepcopy(new_to_old_)
        joint_figure = []
        center_figure = []
        fig_id = 0
        for _ in range(num_prune):
            should_merge = True
            if _ > 0:
                merged_idx = merged[_ - 1].cpu().numpy()
                From = np.max(merged_idx)
                To = np.min(merged_idx)
                if old_to_new[To] != old_to_new[From]:
                    new_to_old[old_to_new[To]] += new_to_old[old_to_new[From]].copy()
                    new_to_old[old_to_new[From]] = []
                    old_to_new = {}
                    for ii in range(model.num_parts):
                        connected_to_ii = new_to_old[ii]
                        for jj in connected_to_ii:
                            old_to_new[jj] = ii
                else:
                    should_merge = False

            if should_merge:
                fig_id += 1
                out = 1 - mask_each_view[i].cpu().numpy()[:, :, None][:, :, [0, 0, 0]] / 2
                plt.imshow(out, vmin=0, vmax=1, alpha=0.2)
                joint_location = (child_root[0, joint_connection[:, 0], :, child_ids[:, 0]] +
                                  child_root[0, joint_connection[:, 1], :, child_ids[:, 1]]) / 2
                new_joint_connection = np.array([[old_to_new[jc[0]], old_to_new[jc[1]]] for jc in joint_connection])
                for j in range(model.num_parts):
                    if len(new_to_old[j]) > 0:
                        joints = joint_location[np.where((new_joint_connection == old_to_new[j]) & (
                                new_joint_connection[:, :1] != new_joint_connection[:, 1:]))[0]]
                        if len(joints) > 1:
                            color = format(j * 600000, '06x')
                            draw_polygon(joints, color)
                        elif len(joints) == 1:
                            plt.plot([joints[0, 0], joint_2d[j, 0]],
                                     [joints[0, 1], joint_2d[j, 1]], c="b")
                        else:
                            break
                        if text:
                            plt.text(joint_2d[j, 0], joint_2d[j, 1], j, fontsize="small")

                plt.axis("off")
                plt.subplots_adjust(left=0, right=1, bottom=0, top=1)

                buf = io.BytesIO()
                plt.savefig(buf, format='png', dpi=150)
                enc = np.frombuffer(buf.getvalue(), dtype=np.uint8)
                dst = cv2.imdecode(enc, 1)[:, :, ::-1]
                joint_figure.append(dst)
                plt.clf()

                out = 1 - mask_each_view[i].cpu().numpy()[:, :, None][:, :, [0, 0, 0]] / 2
                plt.imshow(out, vmin=0, vmax=1, alpha=0.2)

                new_joint_2d = [np.mean(joint_2d[new_to_old[_]], axis=0) if len(new_to_old[_]) > 0 else None
                                for _ in range(model.num_parts)]
                for j in range(model.num_parts):
                    if len(new_to_old[j]) > 0:
                        if text:
                            plt.text(new_joint_2d[old_to_new[j]][0], new_joint_2d[old_to_new[j]][1], j,
                                     fontsize="small")
                        if j == model.num_parts - 1:
                            break
                for njc in new_joint_connection:
                    if njc[0] != njc[1]:
                        plt.plot([new_joint_2d[njc[0]][0], new_joint_2d[njc[1]][0]],
                                 [new_joint_2d[njc[0]][1], new_joint_2d[njc[1]][1]])

                plt.axis("off")
                plt.subplots_adjust(left=0, right=1, bottom=0, top=1)

                buf = io.BytesIO()
                plt.savefig(buf, format='png', dpi=150)
                enc = np.frombuffer(buf.getvalue(), dtype=np.uint8)
                dst = cv2.imdecode(enc, 1)[:, :, ::-1]
                center_figure.append(dst)
                plt.clf()

        return joint_figure, center_figure

    joint_figure, center_figure = create_figures(text=True)
    os.makedirs(f"{self.save_dir}/merge", exist_ok=True)
    for idx, jf in enumerate(joint_figure):
        cv2.imwrite(f"{self.save_dir}/merge/joints_{idx:0>4}.png", jf)
    for idx, cf in enumerate(center_figure):
        cv2.imwrite(f"{self.save_dir}/merge/centers_{idx:0>4}.png", cf)

    joint_figure, center_figure = create_figures(text=False)
    os.makedirs(f"{self.save_dir}/merge", exist_ok=True)
    for idx, jf in enumerate(joint_figure):
        cv2.imwrite(f"{self.save_dir}/merge/joints_notext_{idx:0>4}.png", jf)
    for idx, cf in enumerate(center_figure):
        cv2.imwrite(f"{self.save_dir}/merge/centers_notext_{idx:0>4}.png", cf)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Part merging')
    parser.add_argument('--exp_name', required=True, type=str)
    parser.add_argument('--camera_id', required=True, type=int)
    parser.add_argument('--iteration', default=-1, type=int)
    args = parser.parse_args()

    exp_name = args.exp_name
    camera_id = args.camera_id
    config_path = f"confs/{exp_name}.yml"

    default_config = "confs/default.yml"

    main(config_path, default_config, camera_id)
