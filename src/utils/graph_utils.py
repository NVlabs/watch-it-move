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


from typing import Tuple, List
from copy import deepcopy
import numpy as np
import queue


def get_parent_and_children_id(num_parts: int, joint_connection: np.ndarray, selected_candidate_id: np.ndarray,
                               root_id: int) -> Tuple[np.ndarray, List[List], List[List]]:
    """
    Get parent and children id of each part
    Args:
        num_parts:
        joint_connection:
        selected_candidate_id:
        root_id:

    Returns:

    """
    parent_id = -np.ones(num_parts, dtype="int64")  # initialize parent id with -1
    children_id = [[_] for _ in range(num_parts)]
    connected_to = [[] for _ in range(num_parts)]
    for j1, j2 in joint_connection:
        connected_to[j1].append(j2)
        connected_to[j2].append(j1)

    que = queue.Queue()
    visited = [root_id]
    [que.put(ct) for ct in connected_to[root_id]]
    while True:
        current = que.get()
        visited.append(current)
        parent_id[current] = list(set(visited) & set(connected_to[current]))[0]
        if len(visited) == num_parts:
            break
        not_visited = list(set(connected_to[current]) - set(visited))
        [que.put(ct) for ct in not_visited]

    for idx in reversed(visited):
        if parent_id[idx] >= 0:
            children_id[parent_id[idx]] += deepcopy(children_id[idx])

    selected_candidate = []
    for i in range(num_parts):
        parent = parent_id[i]
        if parent < 0:
            cand_i, cand_parent = -1, -1
        else:
            matched = (joint_connection == np.array([i, parent])).all(axis=1)

            if matched.any():
                assert matched.sum() == 1
                cand_i, cand_parent = selected_candidate_id[matched][0]
            else:
                matched = (joint_connection == np.array([parent, i])).all(axis=1)
                assert matched.sum() == 1
                cand_parent, cand_i = selected_candidate_id[matched][0]

        selected_candidate.append([cand_i, cand_parent])

    return parent_id, children_id, selected_candidate
