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
from typing import Any, List

import numpy as np
import torch
from adet.config import get_cfg
from detectron2.data import MetadataCatalog
from detectron2.engine.defaults import DefaultPredictor
from tqdm import tqdm


def setup_cfg():
    # load config from file and command-line arguments
    confidence_threshold = 0.3
    cfg = get_cfg()
    cfg.merge_from_file("AdeliDet/configs/BlendMask/R_101_dcni3_5x.yaml")
    cfg.merge_from_list(["MODEL.WEIGHTS", "R_101_dcni3_5x.pth"])
    # Set score_threshold for builtin models
    cfg.MODEL.RETINANET.SCORE_THRESH_TEST = confidence_threshold
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = confidence_threshold
    cfg.MODEL.FCOS.INFERENCE_TH_TEST = confidence_threshold
    cfg.MODEL.MEInst.INFERENCE_TH_TEST = confidence_threshold
    cfg.MODEL.PANOPTIC_FPN.COMBINE.INSTANCES_CONFIDENCE_THRESH = confidence_threshold
    cfg.freeze()

    return cfg


class PersonDetector(object):
    def __init__(self):
        cfg = setup_cfg()
        self.metadata = MetadataCatalog.get(
            cfg.DATASETS.TEST[0] if len(cfg.DATASETS.TEST) else "__unused"
        )
        self.cpu_device = torch.device("cpu")
        self.vis_text = cfg.MODEL.ROI_HEADS.NAME == "TextHead"

        self.predictor = DefaultPredictor(cfg)

    def process_predictions(self, frame: np.ndarray, predictions: Any) -> np.ndarray:
        """

        Args:
            frame:
            predictions:

        Returns:

        """
        predictions = predictions["instances"].to(self.cpu_device)
        if predictions.pred_masks.shape[0] == 0:
            print("No mask detected")
            return np.zeros((frame.shape[0], frame.shape[1], 1))

        mask = predictions.pred_masks[0, :, :, None].cpu().numpy()

        return mask

    def run_on_video(self, video: np.ndarray) -> List[np.ndarray]:
        """
        Detect person from video
        Args:
            video:

        Returns:

        """
        detected_video = []
        for frame in tqdm(video):
            detected_video.append(self.process_predictions(frame, self.predictor(frame)))

        return detected_video
