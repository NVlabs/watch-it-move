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
import os
import pickle
from typing import Optional, List, Tuple, Any

import blosc
import cv2
import numpy as np
import torch
from tqdm import tqdm

from detect_dog_mask2former import DogDetector


def load_video(video_name: str, mask_path: str, K: Optional[np.ndarray] = None, D: Optional[np.ndarray] = None
               ) -> Tuple[np.ndarray, np.ndarray, List[np.ndarray]]:
    """

    Args:
        video_name:
        mask_path:
        K:
        D:

    Returns:

    """
    cap = cv2.VideoCapture(video_name)

    frames = []

    while cap.isOpened():
        ret, frame = cap.read()
        if ret:
            frames.append(frame)
        else:
            break

    cap.release()

    org_masks = []
    for frame, mask_p in tqdm(zip(frames, mask_path)):
        bgr = frame
        h, w, _ = bgr.shape
        mask = cv2.resize(cv2.imread(mask_p), (w, h), interpolation=cv2.INTER_NEAREST)
        org_masks.append(mask)

    # undistort
    if K is not None:
        frames = [cv2.undistort(frame, K, D) for frame in frames]
        org_masks = [(cv2.undistort(mask, K, D) == 255).astype("uint8") * 255 for mask in org_masks]

    frames = np.array(frames)
    background = torch.tensor(frames, device="cuda").median(dim=0)[0].cpu().numpy()

    return frames, background, org_masks


def crop_frames(frames: np.ndarray, org_masks: List[np.ndarray]
                ) -> Tuple[List[np.ndarray], List[np.ndarray], List[Tuple]]:
    """

    Args:
        frames:
        org_masks:

    Returns:

    """
    cropped_frames = []
    cropped_masks = []
    crop_box = []
    for frame, mask in zip(frames, org_masks):
        bgr = frame
        y, x, _ = np.where(mask > 0)
        y_min, y_max = y.min(), y.max()
        x_min, x_max = x.min(), x.max()
        crop_box.append((x_min, x_max, y_min, y_max))
        cropped_frames.append(bgr[y_min - DETECTION_MARGIN:y_max + DETECTION_MARGIN,
                              x_min - DETECTION_MARGIN:x_max + DETECTION_MARGIN])
        cropped_masks.append(mask[y_min - DETECTION_MARGIN:y_max + DETECTION_MARGIN,
                             x_min - DETECTION_MARGIN:x_max + DETECTION_MARGIN])

    return cropped_frames, cropped_masks, crop_box


def detect_objects(cropped_frames: List[np.ndarray]) -> List[Any]:
    """

    Args:
        cropped_frames:

    Returns:

    """
    object_masks = detector.run_on_video(cropped_frames)

    return object_masks


def moving_region(frame: np.ndarray, org_mask: np.ndarray, background: np.ndarray, reliable_person_mask: np.ndarray,
                  crop_box: Tuple[float, float, float, float]) -> np.ndarray:
    """

    Args:
        frame:
        org_mask:
        background:
        reliable_person_mask:
        crop_box:

    Returns:

    """
    (x_min, x_max, y_min, y_max) = crop_box
    foreground = frame.astype("float") - background.astype("float")
    foreground = np.linalg.norm(foreground, axis=2) > 18
    foreground = foreground.astype("uint8") * 255

    kernel = np.ones((9, 9), np.uint8)
    for i in range(1):
        foreground = cv2.dilate(foreground, kernel, iterations=1)
        foreground = cv2.erode(foreground, kernel, iterations=1)

    # delete dots
    kernel = np.ones((3, 3), np.uint8)
    foreground = cv2.erode(foreground, kernel, iterations=1)
    foreground = cv2.dilate(foreground, kernel, iterations=1)

    # delete holes
    kernel = np.zeros((15, 15), np.uint8)
    cv2.circle(kernel, (7, 7), 8, 1, thickness=-1)
    for i in range(1):
        foreground = cv2.dilate(foreground, kernel, iterations=1)
        foreground = cv2.erode(foreground, kernel, iterations=1)

    foreground = foreground[y_min:y_max, x_min:x_max]
    org_mask = org_mask[y_min:y_max, x_min:x_max]

    foreground = foreground | reliable_person_mask

    # apply org mask
    foreground = foreground | (org_mask[:, :, 0] > 0)
    # dilated org mask
    kernel = np.ones((25, 25), np.uint8)
    org_mask = (org_mask[:, :, 0] > 0).astype("uint8")
    org_mask = cv2.dilate(org_mask, kernel, iterations=1)
    foreground = foreground * org_mask

    return foreground > 0


def _preprocess_mask(pred_mask: np.ndarray, gt_mask: np.ndarray, scores: np.ndarray, pred_classes: np.ndarray,
                     thres: float = 0.8, occlusion_thres: float = 0.007,
                     person_thres=0.03) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], np.ndarray]:
    """

    Args:
        pred_mask:
        gt_mask:
        scores:
        pred_classes:
        thres:
        occlusion_thres:
        person_thres:

    Returns:

    """
    pred_dog_mask = pred_mask[(pred_classes == 16) & (scores > 0.005)]  # | (pred_classes == 0)]
    pred_person_mask = pred_mask[(pred_classes == 0) & (scores > person_thres)]  # | (pred_classes == 0)]
    reliable_person_mask = pred_mask[(pred_classes == 0) & (scores > 0.3)]

    reliable_person_mask = reliable_person_mask > 0
    reliable_person_mask = False if len(reliable_person_mask) == 0 else reliable_person_mask.max(axis=0)

    # no detected
    if len(pred_dog_mask) == 0:
        return None, None, reliable_person_mask
    pred_dog_mask = pred_dog_mask > 0
    pred_person_mask = pred_person_mask > 0
    gt_mask = gt_mask > 0
    dog_precision = np.sum(pred_dog_mask * gt_mask, axis=(1, 2)) / (np.sum(pred_dog_mask, axis=(1, 2)) + 1e-3)

    if max(dog_precision) >= thres:
        object_mask = pred_dog_mask[dog_precision >= thres].max(axis=0)
    else:
        object_mask = pred_dog_mask[0] * False

    object_mask = object_mask & ~reliable_person_mask

    pred_person_mask = pred_person_mask & ~object_mask
    person_recall = np.sum(pred_person_mask * gt_mask, axis=(1, 2)) / (np.sum(gt_mask) + 1e-3)
    if (person_recall > occlusion_thres).any() == True:
        occlusion = pred_person_mask[person_recall > occlusion_thres].max(axis=0)
    else:
        occlusion = pred_dog_mask[0] * False

    # gt_maskと94% overlap
    recall = np.sum((object_mask | occlusion) * gt_mask) / (np.sum(gt_mask) + 1e-3)
    if recall < 0.94:
        occlusion = occlusion | gt_mask

    occlusion = occlusion & ~object_mask

    return object_mask, occlusion, reliable_person_mask


def postprocess_detected(object_mask: List[Any], cropped_masks: List[np.ndarray], crop_box: List[tuple],
                         frames: np.ndarray, org_masks: List[np.ndarray], background: np.ndarray
                         ) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    """

    Args:
        object_mask:
        cropped_masks:
        crop_box:
        frames:
        org_masks:
        background:

    Returns:

    """
    foreground_masks = []
    occlusion_masks = []
    for i in tqdm(range(len(object_mask))):
        instances = object_mask[i].pred_masks.float().cpu().numpy()
        scores = object_mask[i].scores.cpu().numpy()
        pred_classes = object_mask[i].pred_classes.cpu().numpy()
        foreground_mask, occlusion_mask, reliable_person_mask = _preprocess_mask(instances, cropped_masks[i][:, :, 0],
                                                                                 scores, pred_classes)
        x_min, x_max, y_min, y_max = crop_box[i]
        foreground = moving_region(frames[i], org_masks[i], background, reliable_person_mask,
                                   (x_min - DETECTION_MARGIN, x_max + DETECTION_MARGIN,
                                    y_min - DETECTION_MARGIN, y_max + DETECTION_MARGIN))
        if foreground_mask is None:
            occlusion_mask = foreground
            foreground_mask = foreground * False
        else:
            foreground_mask = foreground_mask & foreground
            occlusion_mask = occlusion_mask & foreground
        foreground_masks.append(foreground_mask)
        occlusion_masks.append(occlusion_mask)

    return foreground_masks, occlusion_masks


def undo_crop(frames: np.ndarray, foreground_masks: List[np.ndarray], occlusion_masks: List[np.ndarray],
              crop_box: List[tuple]) -> Tuple[np.ndarray, np.ndarray]:
    """

    Args:
        frames:
        foreground_masks:
        occlusion_masks:
        crop_box:

    Returns:

    """
    org_foreground_masks = np.zeros((len(frames), *frames[0].shape[:-1]), dtype="bool")
    org_occlusion_masks = np.zeros((len(frames), *frames[0].shape[:-1]), dtype="bool")
    for i, (fm, om, (x_min, x_max, y_min, y_max)) in tqdm(enumerate(zip(foreground_masks, occlusion_masks, crop_box))):
        org_foreground_masks[i, y_min - DETECTION_MARGIN:y_max + DETECTION_MARGIN,
        x_min - DETECTION_MARGIN:x_max + DETECTION_MARGIN] = fm
        org_occlusion_masks[i, y_min - DETECTION_MARGIN:y_max + DETECTION_MARGIN,
        x_min - DETECTION_MARGIN:x_max + DETECTION_MARGIN] = om

    return org_foreground_masks, org_occlusion_masks


def square_crop(frames: np.ndarray, foreground_masks: np.ndarray, occlusion_masks: np.ndarray, crop_box: List[tuple],
                K: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """

    Args:
        frames:
        foreground_masks:
        occlusion_masks:
        crop_box:
        K:

    Returns:

    """
    cropped_frames = []
    cropped_fg_masks = []
    cropped_oc_masks = []
    all_intrinsics = []
    for i in range(len(frames)):
        frame = frames[i]
        foreground_mask = foreground_masks[i]
        occlusion_mask = occlusion_masks[i]
        H, W, _ = frame.shape
        x_min, x_max, y_min, y_max = crop_box[i]
        w = x_max - x_min
        h = y_max - y_min
        _crop_size = max(CROP_SIZE, int(h * 1.1), int(w * 1.1)) // 2 * 2  # even number
        x_center = int((x_min + x_max) / 2)
        y_center = int((y_min + y_max) / 2)

        left = x_center - _crop_size // 2
        right = x_center + _crop_size // 2
        top = y_center - _crop_size // 2
        bottom = y_center + _crop_size // 2

        padded_frame = np.pad(frame, ((max(0, -top), max(0, bottom - H)),
                                      (max(0, -left), max(0, right - H)),
                                      (0, 0)))
        padded_fg_mask = np.pad(foreground_mask, ((max(0, -top), max(0, bottom - H)),
                                                  (max(0, -left), max(0, right - H))))
        padded_oc_mask = np.pad(occlusion_mask, ((max(0, -top), max(0, bottom - H)),
                                                 (max(0, -left), max(0, right - H))), constant_values=True)
        cropped_frame = padded_frame[
                        max(0, top): max(0, top) + _crop_size,
                        max(0, left): max(0, left) + _crop_size]
        cropped_fg_mask = padded_fg_mask[
                          max(0, top): max(0, top) + _crop_size,
                          max(0, left): max(0, left) + _crop_size]
        cropped_oc_mask = padded_oc_mask[
                          max(0, top): max(0, top) + _crop_size,
                          max(0, left): max(0, left) + _crop_size]

        intrinsics = K.copy()
        intrinsics[0, 2] -= left
        intrinsics[1, 2] -= top

        if x_min == 0 or x_max == W - 1 or y_min == 0 or y_max == H - 1:
            cropped_oc_mask = cropped_oc_mask | True
            cropped_fg_mask = cropped_fg_mask & False

        # resize
        if _crop_size >= CROP_SIZE:
            cropped_frame = cv2.resize(cropped_frame, (CROP_SIZE, CROP_SIZE),
                                       interpolation=cv2.INTER_AREA)
            cropped_fg_mask = cv2.resize(cropped_fg_mask.astype("uint8"), (CROP_SIZE, CROP_SIZE),
                                         interpolation=cv2.INTER_NEAREST).astype("bool")
            cropped_oc_mask = cv2.resize(cropped_oc_mask.astype("uint8"), (CROP_SIZE, CROP_SIZE),
                                         interpolation=cv2.INTER_NEAREST).astype("bool")
            intrinsics = intrinsics * np.array([CROP_SIZE / _crop_size, CROP_SIZE / _crop_size, 1.])[:, None]

        cropped_frames.append(cropped_frame)
        cropped_fg_masks.append(cropped_fg_mask)
        cropped_oc_masks.append(cropped_oc_mask)
        all_intrinsics.append(intrinsics)

    return np.array(cropped_frames), np.array(cropped_fg_masks), np.array(cropped_oc_masks), np.array(all_intrinsics)


def load_camera_info(camera_id: str):
    """

    Args:
        camera_id:

    Returns:

    """
    camera_path = os.path.join(os.path.dirname(DATA_ROOT), "calibration/sony")

    with open(f"{camera_path}/calibFile{camera_id}", "r") as f:
        camera_params = f.read()

    camera_params = camera_params.split("\n")

    W, H = int(camera_params[0]), int(camera_params[1])
    K = np.array([cp.split(" ") for cp in camera_params[2:5]], dtype="float")
    T = np.array([cp.split(" ") for cp in camera_params[6:10]], dtype="float")
    D = np.array(camera_params[11].split(" "), dtype="float")

    # intrinsics is 4K but video is 2K
    K = K * np.array([0.5, 0.5, 1])[:, None]

    return W, H, K, T, D


def read_frames(chosen_camera_id: List[int]):
    """

    Args:
        chosen_camera_id:

    Returns:

    """
    all_video = []
    all_mask = []
    all_camera_intrinsic = []
    all_camera_rotation = []
    all_camera_translation = []
    n_frames = None
    for c_id in tqdm(chosen_camera_id):
        W, H, K, T, D = load_camera_info(f"{c_id:0>2d}")
        video_name = f"{DATA_ROOT}/sony/camera{c_id:0>2d}/camera{c_id:0>2d}_2K.mp4"
        mask_path = f"{DATA_ROOT}/sony/camera{c_id:0>2d}/masks/*.png"
        mask_path = sorted(glob.glob(mask_path))
        frames, background, org_masks = load_video(video_name, mask_path, K, D)
        cropped_frames, cropped_masks, crop_box = crop_frames(frames, org_masks)
        object_masks = detect_objects(cropped_frames)
        foreground_masks, occlusion_masks = postprocess_detected(object_masks, cropped_masks, crop_box,
                                                                 frames, org_masks, background)
        org_foreground_masks, org_occlusion_masks = undo_crop(frames, foreground_masks, occlusion_masks, crop_box)
        frames, org_foreground_masks, org_occlusion_masks, intrinsics = square_crop(frames, org_foreground_masks,
                                                                                    org_occlusion_masks, crop_box, K)
        frames = frames * org_foreground_masks[:, :, :, None]

        org_foreground_masks = org_foreground_masks * 1 + org_occlusion_masks * 2
        assert n_frames is None or n_frames == len(frames)
        n_frames = len(frames)

        all_video.append(frames)
        all_mask.append(org_foreground_masks)
        all_camera_intrinsic.append(intrinsics)
        all_camera_rotation.append(np.broadcast_to(T[None, :3, :3], (n_frames, 3, 3)))
        all_camera_translation.append(np.broadcast_to(T[None, :3, 3:], (n_frames, 3, 1)))

    all_video = np.concatenate(all_video)
    all_mask = np.concatenate(all_mask)
    all_camera_intrinsic = np.concatenate(all_camera_intrinsic)
    all_camera_rotation = np.concatenate(all_camera_rotation)
    all_camera_translation = np.concatenate(all_camera_translation)

    frame_id = [np.arange(n_frames) for cam in chosen_camera_id]
    frame_id = np.concatenate(frame_id, axis=0)

    return all_video, all_mask, frame_id, all_camera_intrinsic, all_camera_rotation, all_camera_translation


def main():
    data_dict = {}

    # read frame
    all_video, all_mask, frame_id, all_intrinsic, all_rot, all_trans = read_frames(CHOSEN_CAMERA_ID)

    data_dict["frame_id"] = frame_id
    data_dict["img"] = np.array([blosc.pack_array(frame.transpose(2, 0, 1)[[2, 1, 0]]) for frame in tqdm(all_video)],
                                dtype="object")
    data_dict["mask"] = np.array([blosc.pack_array(mask) for mask in tqdm(all_mask)], dtype="object")
    data_dict["camera_intrinsic"] = all_intrinsic
    data_dict["camera_rotation"] = all_rot
    data_dict["camera_translation"] = all_trans

    data_dict["camera_id"] = np.arange(len(all_video)) // (len(all_video) // len(CHOSEN_CAMERA_ID))

    with open(DATA_ROOT + '/cache.pickle', 'wb') as f:
        pickle.dump(data_dict, f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Robot data preprocessing')
    parser.add_argument('--data_root', type=str, required=True)
    args = parser.parse_args()

    DATA_ROOT = args.data_root

    DETECTION_MARGIN = 50
    CROP_SIZE = 512
    CHOSEN_CAMERA_ID = [0, 1, 4, 5, 6, 7, 8, 9]

    detector = DogDetector()
    main()
