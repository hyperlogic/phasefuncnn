from typing import Tuple, TypedDict

from tqdm import std

import torch
from dataclasses import dataclass, field

from skeleton import Skeleton

TRAJ_WINDOW_SIZE = 12

class ColumnLens:
    size: int
    indices: list[int]
    def __init__(self, size: int, indices: list[int]):
        self.size = size
        self.indices = indices

    def get(self, data: torch.Tensor, index: int):
        start = self.indices[index]
        return data[start : start + self.size]

    def set(self, data: torch.Tensor, index: int, value):
        start = self.indices[index]
        data[start : start + self.size] = value

def build_column_indices(start: int, stride: int, repeat: int = 1) -> Tuple[int, list[int]]:
    indices = [i * stride + start for i in range(repeat)]
    offset = repeat * stride + start
    return offset, indices


class InputLens:
    traj_pos_i: ColumnLens
    traj_vel_i: ColumnLens
    joint_pos_im1: ColumnLens
    joint_vel_im1: ColumnLens
    num_cols: int
    def __init__(self, num_joints: int, mean: torch.Tensor, std: torch.Tensor, w: torch.Tensor):

        offset = 0
        next_offset, indices = build_column_indices(offset, 4, TRAJ_WINDOW_SIZE)
        self.traj_pos_i = ColumnLens(2, indices)
        _, indices = build_column_indices(offset + 2, 4, TRAJ_WINDOW_SIZE)
        self.traj_vel_i = ColumnLens(2, indices)

        offset = next_offset
        next_offset, indices = build_column_indices(offset, 6, num_joints)
        self.joint_pos_im1 = ColumnLens(3, indices)
        _, indices = build_column_indices(offset + 3, 6, num_joints)
        self.joint_vel_im1 = ColumnLens(3,indices)

        self.num_cols = next_offset

        self.mean = mean
        self.std = std
        self.w = w

    def unnormalize(self, data: torch.Tensor) -> torch.Tensor:
        return data * (self.std / self.w) + self.mean

    def normalize(self, data: torch.Tensor) -> torch.Tensor:
        return (data - self.mean) * (self.w / self.std)

