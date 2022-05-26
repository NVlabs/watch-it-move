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

import pickle
import random
from typing import Dict, Any, Optional

import blosc
import numpy as np


class SingleVideoDataset:
    def __init__(self, config: Dict[str, Any]) -> None:
        """

        Args:
            config:
        """
        self.config = config
        self.num_parts = config.num_parts
        self.num_frames = config.num_frames
        self.num_view = config.num_view
        self.img_size = config.size
        self.thin_out_interval = config.thin_out_interval
        self.return_neighboring_frames = config.return_neighboring_frames
        self.return_random_frames = config.return_random_frames
        self.compression = config.compression
        self.video_cache = self.cache_data(config.set_name)
        self.n_repetition_in_epoch = config.n_repetition_in_epoch
        self.coordinate_scale = config.coordinate_scale
        self.current_max_frame_id = self.num_frames // self.thin_out_interval
        self.current_min_frame_id = 0
        self.prob_sample_latest = config.prob_sample_latest
        self.background_color = config.background_color

    @staticmethod
    def seed():
        np.random.seed()
        random.seed()

    def cache_data(self, set_name: Optional[str] = None) -> Dict:
        """
        cache data into a dictionary of numpy array
        Args:
            set_name:

        Returns:
            video_cache (dict): cached data
        """
        file_name = "cache.pickle" if set_name is None else f"cache_{set_name}.pickle"
        cache_path = f"{self.config.data_root}/{file_name}"
        with open(cache_path, "rb") as f:
            video_cache = pickle.load(f)

        return video_cache

    def __len__(self) -> int:
        return self.num_frames * self.num_view // self.thin_out_interval * \
               self.n_repetition_in_epoch  # number of frames

    def get_index(self, index: int) -> np.ndarray:
        """

        Args:
            index:

        Returns:

        """
        num_frames = self.num_frames // self.thin_out_interval
        if self.current_max_frame_id >= num_frames:
            index = index // self.n_repetition_in_epoch
        else:
            current_max_frame_id = min(num_frames, self.current_max_frame_id)
            current_min_frame_id = self.current_min_frame_id
            if random.random() < self.prob_sample_latest:
                min_frame_id = max(0, current_max_frame_id - 6)
                frame_id = random.randint(min_frame_id, current_max_frame_id - 1)
            else:
                frame_id = random.randint(current_min_frame_id, current_max_frame_id - 1)
            camera_id = random.randint(0, self.num_view - 1)
            index = self.num_frames * camera_id + frame_id * self.thin_out_interval

        return index

    def __getitem__(self, index: int) -> dict:
        """

        Args:
            index:

        Returns:

        """
        self.seed()
        index = self.get_index(index)

        frame_id = self.video_cache["frame_id"][index]

        img = self.video_cache["img"][index]
        mask = self.video_cache["mask"][index]
        if self.compression:
            img = blosc.unpack_array(img)
            mask = blosc.unpack_array(mask)
        img = img / 127.5 - 1

        # remove background
        fg_mask = (mask == 1)  # ignore unreliable pixels
        img = img * fg_mask + (1 - fg_mask) * self.background_color

        camera_rotation = self.video_cache["camera_rotation"][index]
        camera_translation = self.video_cache["camera_translation"][index] / self.coordinate_scale
        camera_id = self.video_cache["camera_id"][index]
        camera_intrinsic = self.video_cache["camera_intrinsic"][index]
        minibatch = {"frame_id": frame_id,
                     "img": img.astype("float32"),
                     "mask": mask.astype("float32"),
                     "camera_rotation": camera_rotation.astype("float32"),
                     "camera_translation": camera_translation.astype("float32"),
                     "camera_id": camera_id,
                     "camera_intrinsic": camera_intrinsic.astype("float32")}

        return minibatch
