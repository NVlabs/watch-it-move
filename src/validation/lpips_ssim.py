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
import argparse
import os
import pickle
from typing import Tuple

import lpips
import numpy as np
import torch
from skimage.metrics import structural_similarity as ssim

sys.path.append(".")
from utils.get_args import get_args_jupyter

loss_fn_vgg = lpips.LPIPS(net='vgg').cuda()


def evaluate(path: str) -> Tuple[np.float, np.float]:
    """

    Args:
        path:

    Returns:

    """
    with open(path, "rb") as f:
        data = pickle.load(f)

    gt_img = np.clip(data["gt_img"], 0, 255) / 127.5 - 1
    gen_img = np.clip(data["gen_img"], 0, 255) / 127.5 - 1

    batchsize = 16
    lpips_vals = []
    with torch.no_grad():
        for i in range(0, gt_img.shape[0], batchsize):
            lpips_val = loss_fn_vgg(torch.tensor(gt_img[i:i + batchsize]).cuda(),
                                    torch.tensor(gen_img[i:i + batchsize]).cuda()).squeeze().cpu().numpy()
            lpips_vals.append(lpips_val)

    mean_lpips = np.concatenate(lpips_vals).mean()

    ssim_vals = []
    for i in range(gt_img.shape[0]):
        gt = gt_img[i].transpose(1, 2, 0)
        gen = gen_img[i].transpose(1, 2, 0)
        ssim_vals.append(ssim(gt, gen, data_range=gt.max() - gt.min(), multichannel=True))

    mean_ssim = np.array(ssim_vals).mean()

    return mean_lpips, mean_ssim


def eval_all(exp_name: str) -> None:
    """

    Args:
        exp_name:

    Returns:

    """
    default_config = "confs/default.yml"
    config_path = f"confs/{exp_name}.yml"
    args, config = get_args_jupyter(config_path, default_config)
    out_dir = config.output_dir
    exp_name = config.exp_name
    root = os.path.join(out_dir, "result")
    result = {}
    validation_dir_name = f"{root}/{exp_name}/validation"
    mean_lpips, mean_ssim = evaluate(f"{validation_dir_name}/reconstruction_test.pkl")
    print(exp_name)
    print("NV", mean_lpips, mean_ssim)
    result["novel_view"] = {"lpips": mean_lpips, "ssim": mean_ssim}
    mean_lpips, mean_ssim = evaluate(f"{validation_dir_name}/reconstruction_novel_pose.pkl")
    print("NP", mean_lpips, mean_ssim)
    result["novel_pose"] = {"lpips": mean_lpips, "ssim": mean_ssim}

    with open(f"{validation_dir_name}/lpips_ssim.pkl", "wb") as f:
        pickle.dump(result, f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Compute lpips and ssim')
    parser.add_argument('--exp_name', action='append', required=True)
    args = parser.parse_args()

    exp_names = args.exp_name

    for exp_name in exp_names:
        eval_all(exp_name)
