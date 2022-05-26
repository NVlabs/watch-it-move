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
import glob
import json
import os
import pickle
from typing import Tuple, List, Dict

import blosc
import cv2
import numpy as np
import torch
from easymocap.smplmodel import SMPLlayer
from tqdm import tqdm

from detect_person import PersonDetector
from read_smpl import PoseLoader


def read_frames(person_id: int, save_size: int, crop_size: int, chosen_camera_id: np.ndarray
                ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    """

    Args:
        person_id:
        save_size:
        crop_size:
        chosen_camera_id:

    Returns:

    """
    all_video = []
    for cam in tqdm(chosen_camera_id):
        video_path = f"{ZJU_PATH}/{person_id}/videos/{cam:0>2}.mp4"
        video = cv2.VideoCapture(video_path)
        frames = []
        while True:
            ret, frame = video.read()
            if not ret:
                break
            frame = frame[:crop_size, :crop_size]
            frame = cv2.resize(frame, (save_size, save_size), interpolation=cv2.INTER_CUBIC)
            frames.append(frame[:, :, ::-1])
        frames = np.array(frames)
        all_video.append(frames)
    video_len = np.array([video.shape[0] for video in all_video])
    assert (video_len == video_len[0]).all()
    frame_id = [np.arange(video_len[0]) for _ in range(NUM_CAMERA)]
    frame_id = np.stack(frame_id, axis=0)

    all_video = np.stack(all_video, axis=0)

    camera_id = [np.ones(video_len[0], dtype="int") * (cam - 1) for cam in chosen_camera_id]
    camera_id = np.stack(camera_id, axis=0)

    return all_video, frame_id, camera_id, video_len[0]


class DetectPerson:
    def __init__(self):
        self.detector = PersonDetector()

    def __call__(self, all_video: np.ndarray):
        """

        Args:
            all_video:

        Returns:

        """
        detected = self.detector.run_on_video(all_video)

        return detected


def read_intrinsic(person_id: int, save_scale: float) -> np.ndarray:
    """

    Args:
        person_id:
        save_scale:

    Returns:

    """
    fs = cv2.FileStorage(f"{ZJU_PATH}/{person_id}/intri.yml", cv2.FILE_STORAGE_READ)
    all_intrinsic = []
    for cam in range(1, NUM_CAMERA + 1):
        matrix = fs.getNode(f"K_{cam:0>2}").mat()
        matrix = np.array(matrix).reshape(3, 3)
        all_intrinsic.append(matrix)
    all_intrinsic = np.array(all_intrinsic)
    all_intrinsic[:, :2] /= save_scale

    return all_intrinsic


def read_extrinsic(person_id: int) -> Tuple[np.ndarray, np.ndarray]:
    """

    Args:
        person_id:

    Returns:

    """
    fs = cv2.FileStorage(f"{ZJU_PATH}/{person_id}/extri.yml", cv2.FILE_STORAGE_READ)
    all_rot = []
    all_trans = []
    for cam in range(1, NUM_CAMERA + 1):
        rot = fs.getNode(f"Rot_{cam:0>2}").mat()
        rot = np.array(rot).reshape(3, 3)
        trans = fs.getNode(f"T_{cam:0>2}").mat()
        trans = np.array(trans).reshape(3, 1)
        all_rot.append(rot)
        all_trans.append(trans)
    all_rot = np.array(all_rot)
    all_trans = np.array(all_trans)

    return all_rot, all_trans


def read_smpl_parameters(person_id: int, video_len: int) -> np.ndarray:
    """

    Args:
        person_id:
        video_len:

    Returns:

    """
    all_smpl_param = []
    for frame_id in tqdm(range(video_len)):
        smpl_path = f"{ZJU_PATH}/{person_id}/smplx/{frame_id:0>6}.json"

        with open(smpl_path, "r") as f:
            smpl_param = json.load(f)[0]

        smpl_param = pose_loader(smpl_param)
        all_smpl_param.append(smpl_param)
    all_smpl_param = np.array(all_smpl_param)

    return all_smpl_param


def read_smpl_verts(person_id: int, smpllayer: SMPLlayer) -> np.ndarray:
    """

    Args:
        person_id:
        smpllayer:

    Returns:

    """
    all_smpl_verts = []
    smpl_paths = sorted(glob.glob(f"{ZJU_PATH}/{person_id}/smplx/*.json"))
    for frame_id in tqdm(range(len(smpl_paths))):
        smpl_path = smpl_paths[frame_id]

        with open(smpl_path, "r") as f:
            smpl_param = json.load(f)[0]
        with torch.no_grad():
            Rh = torch.tensor(np.array(smpl_param["Rh"])).float()  # 1 x 3
            Th = torch.tensor(np.array(smpl_param["Th"])).float()  # 1 x 3
            poses = torch.tensor(np.array(smpl_param["poses"])).float()  # 1 x 72
            shapes = torch.tensor(smpl_param["shapes"]).float()  # 1 x 10
            expression = torch.tensor(smpl_param["expression"]).float()  # 1 x 10
            verts = smpllayer(poses, shapes, Rh, Th, expression)

        all_smpl_verts.append(verts[0].cpu().numpy())
    all_smpl_verts = np.array(all_smpl_verts)

    return all_smpl_verts  # (video_len, n_verts, 3)


def create_dict(video: np.ndarray, mask: List[np.ndarray], frame: np.ndarray, camera: np.ndarray,
                all_intrinsic: np.ndarray, all_rot: np.ndarray, all_trans: np.ndarray, smpl: np.ndarray, set_size: int
                ) -> Dict[str, np.ndarray]:
    """

    Args:
        video:
        mask:
        frame:
        camera:
        all_intrinsic:
        all_rot:
        all_trans:
        smpl:
        set_size:

    Returns:

    """
    data_dict = {}
    data_dict["frame_id"] = frame.reshape(-1)
    data_dict["img"] = np.array([blosc.pack_array(frame.transpose(2, 0, 1)) for frame in tqdm(video)],
                                dtype="object")

    data_dict["mask"] = np.array([blosc.pack_array(det[:, :, 0]) for det in tqdm(mask)],
                                 dtype="object")
    data_dict["camera_intrinsic"] = all_intrinsic[camera]
    data_dict["camera_rotation"] = all_rot[camera]
    data_dict["camera_translation"] = all_trans[camera]

    data_dict["camera_id"] = np.arange(len(frame)) // set_size
    data_dict["smpl_pose"] = smpl

    return data_dict


def process_train_set(person_id: int, all_video, all_intrinsic: np.ndarray, all_rot: np.ndarray, all_trans: np.ndarray,
                      all_smpl_param: np.ndarray, frame_id, camera_id, video_len: int, train_set_rate: float) -> int:
    """

    Args:
        person_id:
        all_video:
        all_intrinsic:
        all_rot:
        all_trans:
        all_smpl_param:
        frame_id:
        camera_id:
        video_len:
        train_set_rate:

    Returns:

    """
    train_set_size = int(video_len * train_set_rate)
    train_video = all_video[TRAIN_CAMERA_ID - 1, :train_set_size].reshape(-1, *all_video.shape[2:])
    train_frame = frame_id[TRAIN_CAMERA_ID - 1, :train_set_size].reshape(-1, *frame_id.shape[2:])
    train_camera = camera_id[TRAIN_CAMERA_ID - 1, :train_set_size].reshape(-1, *camera_id.shape[2:])
    train_mask = person_detector(train_video)

    train_dict = create_dict(train_video, train_mask, train_frame, train_camera, all_intrinsic,
                             all_rot, all_trans, all_smpl_param, train_set_size)

    with open(f'{ZJU_PATH}/cache{SAVE_SIZE}/{person_id}/cache_train.pickle', 'wb') as f:
        pickle.dump(train_dict, f)

    print("person id:", person_id, "train set size", train_set_size)

    return train_set_size


def process_test_set(person_id: int, train_set_size: int, test_set_size: int, video_len: int, all_video: np.ndarray,
                     all_intrinsic: np.ndarray, all_rot: np.ndarray, all_trans: np.ndarray,
                     all_smpl_param: np.ndarray, frame_id: np.ndarray, camera_id: np.ndarray, mode: str) -> None:
    """
    
    Args:
        person_id:
        train_set_size:
        test_set_size:
        video_len:
        all_video:
        all_intrinsic:
        all_rot:
        all_trans:
        all_smpl_param:
        frame_id:
        camera_id:
        mode:

    Returns:

    """
    if mode == "novel_view":
        test_frame_id = np.linspace(0, train_set_size - 1, test_set_size).astype("int")
        camera_idx = TEST_CAMERA_ID
        cache_name = "cache_test"
    elif mode == "novel_pose":
        test_frame_id = np.linspace(train_set_size, video_len - 1, test_set_size).astype("int")
        camera_idx = ALL_CAMERA_ID
        cache_name = "cache_novel_pose"
    else:
        raise ValueError()

    test_video = all_video[camera_idx - 1][:, test_frame_id].reshape(-1, *all_video.shape[2:])
    test_frame = frame_id[camera_idx - 1][:, test_frame_id].reshape(-1, *frame_id.shape[2:])
    test_camera = camera_id[camera_idx - 1][:, test_frame_id].reshape(-1, *camera_id.shape[2:])
    test_mask = person_detector(test_video)

    test_dict = create_dict(test_video, test_mask, test_frame, test_camera, all_intrinsic,
                            all_rot, all_trans, all_smpl_param, test_set_size)

    with open(f'{ZJU_PATH}/cache{SAVE_SIZE}/{person_id}/{cache_name}.pickle', 'wb') as f:
        pickle.dump(test_dict, f)


def main():
    person_ids = args.person_id
    train_set_rate = 0.8
    test_set_size = 20

    for person_id in person_ids:
        # smpl verts
        all_verts = read_smpl_verts(person_id, smpllayer)
        data_dict = {"smpl_verts": all_verts}
        os.makedirs(f'{ZJU_PATH}/cache{SAVE_SIZE}/{person_id}', exist_ok=True)
        with open(f'{ZJU_PATH}/cache{SAVE_SIZE}/{person_id}/smpl_verts.pickle', 'wb') as f:
            pickle.dump(data_dict, f)

        # read frame
        all_video, frame_id, camera_id, video_len = read_frames(person_id, SAVE_SIZE, CROP_SIZE, ALL_CAMERA_ID)
        all_smpl_param = read_smpl_parameters(person_id, video_len)
        all_smpl_param = all_smpl_param[:, :23]
        all_intrinsic = read_intrinsic(person_id, SAVE_SCALE)
        all_rot, all_trans = read_extrinsic(person_id)

        # train set
        train_set_size = process_train_set(person_id, all_video, all_intrinsic, all_rot, all_trans, all_smpl_param,
                                           frame_id, camera_id, video_len, train_set_rate)

        # novel view
        process_test_set(person_id, train_set_size, test_set_size, video_len, all_video, all_intrinsic, all_rot,
                         all_trans, all_smpl_param, frame_id, camera_id, "novel_view")

        # novel pose
        process_test_set(person_id, train_set_size, test_set_size, video_len, all_video, all_intrinsic, all_rot,
                         all_trans, all_smpl_param, frame_id, camera_id, "novel_pose")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='ZJU data preprocessing')
    parser.add_argument('--smpl_model_path', type=str, required=True)
    parser.add_argument('--zju_path', type=str, required=True)
    parser.add_argument('--person_id', action='append',
                        required=True)
    args = parser.parse_args()

    SMPL_MODEL_PATH = args.smpl_model_path

    ZJU_PATH = args.zju_path
    SAVE_SCALE = 2
    CROP_SIZE = 1024
    NUM_CAMERA = 23
    SAVE_SIZE = CROP_SIZE // SAVE_SCALE

    TRAIN_CAMERA_ID = np.array([1, 5, 9, 13, 17, 21])
    TEST_CAMERA_ID = np.array([2, 3, 4, 6, 7, 8, 10, 11, 12, 14, 15, 16, 18, 19, 20, 22, 23])
    ALL_CAMERA_ID = np.arange(1, NUM_CAMERA + 1)

    pose_loader = PoseLoader(SMPL_MODEL_PATH)
    smpllayer = SMPLlayer(SMPL_MODEL_PATH + "/smplx", model_type='smplx')
    person_detector = DetectPerson()

    main()
