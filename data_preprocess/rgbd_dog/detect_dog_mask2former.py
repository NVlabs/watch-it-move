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

import sys
from typing import Any, List

import numpy as np
import torch
from detectron2.config import get_cfg
from detectron2.engine.defaults import DefaultPredictor
from detectron2.projects.deeplab import add_deeplab_config
from tqdm import tqdm

sys.path.append("Mask2Former")
from mask2former import add_maskformer2_config


def setup_cfg():
    # load config from file and command-line arguments
    cfg = get_cfg()
    add_deeplab_config(cfg)
    add_maskformer2_config(cfg)
    cfg.merge_from_file(
        "Mask2Former/configs/coco/instance-segmentation/swin/maskformer2_swin_large_IN21k_384_bs16_100ep.yaml")
    cfg.merge_from_list(["MODEL.WEIGHTS", "model_final_e5f453.pkl"])
    cfg.freeze()

    return cfg


class DogDetector(object):
    def __init__(self):
        cfg = setup_cfg()
        self.cpu_device = torch.device("cpu")
        self.vis_text = cfg.MODEL.ROI_HEADS.NAME == "TextHead"

        self.predictor = DefaultPredictor(cfg)

    def run_on_video(self, video: List[np.ndarray]) -> List[Any]:
        """
        Visualizes predictions on frames of the input video.
        Args:
            video (np.array):
        Returns:
            ndarray: RGB
        """

        def process_predictions(predictions: Any):
            predictions = predictions["instances"].to(self.cpu_device)
            return predictions

        detected_video = []
        for frame in tqdm(video):
            detected_video.append(process_predictions(self.predictor(frame)))

        return detected_video
