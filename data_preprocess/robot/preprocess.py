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
import json
import pickle
from typing import List

import blosc
import cv2
import numpy as np
from tqdm import tqdm


def read_frames(chosen_camera_id: List[int], video_len: int, data_dir: str):
    """

    Args:
        chosen_camera_id:
        video_len:
        data_dir:

    Returns:

    """
    all_video = []
    all_mask = []
    all_camera_intrinsic = []
    all_camera_rotation = []
    all_camera_translation = []
    for c_id in tqdm(chosen_camera_id):
        for f_id in range(video_len):
            img_path = f"{data_dir}/frame_{f_id:0>5}_cam_{c_id:0>3}.png"
            config_path = f"{data_dir}/cam_{c_id:0>3}.json"
            img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
            img_scale = 1
            if img.shape[0] != 512:
                img_scale = 512 / img.shape[0]
                img = cv2.resize(img, (512, 512), interpolation=cv2.INTER_AREA)
            with open(config_path, "r") as f:
                config = json.load(f)
            frame = img[:, :, [2, 1, 0]]
            mask = img[:, :, 3] > 127.5
            all_video.append(frame)
            all_mask.append(mask)

            intrinsic = config["camera_data"]["intrinsics"]
            camera_intrinsic = np.zeros((3, 3), dtype="float32")
            camera_intrinsic[0, 0] = intrinsic['fx'] * img_scale
            camera_intrinsic[1, 1] = intrinsic['fy'] * img_scale
            camera_intrinsic[0, 2] = intrinsic['cx'] * img_scale
            camera_intrinsic[1, 2] = intrinsic['cy'] * img_scale
            camera_intrinsic[2, 2] = 1
            all_camera_intrinsic.append(camera_intrinsic)

            extrinsic = np.array(config["camera_data"]["camera_view_matrix"])
            extrinsic[:, 1] = -extrinsic[:, 1]
            extrinsic[:, 2] = -extrinsic[:, 2]
            all_camera_rotation.append(extrinsic[:3, :3].transpose())  # (3, 3)
            all_camera_translation.append(extrinsic[3, :3, None])  # (3, 1)

    all_video = np.array(all_video)
    all_mask = np.array(all_mask)
    all_camera_intrinsic = np.array(all_camera_intrinsic)
    all_camera_rotation = np.array(all_camera_rotation)
    all_camera_translation = np.array(all_camera_translation)

    frame_id = [np.arange(video_len) for cam in chosen_camera_id]
    frame_id = np.concatenate(frame_id, axis=0)

    camera_id = [np.ones(video_len, dtype="int") * (cam - 1) for cam in chosen_camera_id]
    camera_id = np.concatenate(camera_id, axis=0)

    return all_video, all_mask, frame_id, camera_id, all_camera_intrinsic, all_camera_rotation, all_camera_translation


def preprocess_robot(robot_name: str):
    if robot_name == "atlas":
        data_dir = f"{DATA_ROOT}/atlas"
        chosen_camera_id = [0, 2, 5, 13, 18]
        video_len = 300
    elif robot_name == "baxter":
        data_dir = f"{DATA_ROOT}/baxter"
        chosen_camera_id = [0, 2, 5, 13, 18]
        video_len = 300
    elif robot_name == "spot":
        data_dir = f"{DATA_ROOT}/spot"
        chosen_camera_id = [0, 2, 5, 10, 13]
        video_len = 300
    elif robot_name == "cassie":
        data_dir = f"{DATA_ROOT}/cassie"
        chosen_camera_id = [0, 3, 5, 10, 13]
        video_len = 300
    elif robot_name == "iiwa":
        data_dir = f"{DATA_ROOT}/iiwa"
        chosen_camera_id = [0, 2, 4, 5, 13]
        video_len = 300
    elif robot_name == "nao":
        data_dir = f"{DATA_ROOT}/nao"
        chosen_camera_id = [7, 10, 11, 14, 15]
        video_len = 300
    elif robot_name == "pandas":
        data_dir = f"{DATA_ROOT}/pandas"
        chosen_camera_id = [0, 2, 5, 10, 13]
        video_len = 300
    else:
        raise ValueError("invalid robot name")

    data_dict = {}

    # read frame
    all_video, all_mask, frame_id, camera_id, all_intrinsic, all_rot, all_trans = read_frames(chosen_camera_id,
                                                                                              video_len, data_dir)

    data_dict["frame_id"] = frame_id
    data_dict["img"] = np.array([blosc.pack_array(frame.transpose(2, 0, 1)) for frame in tqdm(all_video)],
                                dtype="object")
    data_dict["mask"] = np.array([blosc.pack_array(mask) for mask in tqdm(all_mask)], dtype="object")
    data_dict["camera_intrinsic"] = all_intrinsic
    data_dict["camera_rotation"] = all_rot
    data_dict["camera_translation"] = all_trans

    data_dict["camera_id"] = np.arange(len(all_video)) // (len(all_video) // len(chosen_camera_id))

    with open(data_dir + '/cache.pickle', 'wb') as f:
        pickle.dump(data_dict, f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Robot data preprocessing')
    parser.add_argument('--data_root', type=str, required=True)
    parser.add_argument('--robot_name', action='append',
                        required=True)
    args = parser.parse_args()

    DATA_ROOT = args.data_root
    robot_names = args.robot_name

    for robot_name in robot_names:
        preprocess_robot(robot_name)
