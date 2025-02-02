from typing import Tuple, TypedDict

from tqdm import std

import torch
from dataclasses import dataclass, field

from skeleton import Skeleton

MAX_COLUMNS_PER_LINE = 8

def tensor_fmt(tensor: torch.tensor):
    assert tensor.dim() == 1
    return "[" + (", ".join(map(lambda x: f"{x:10.3f}", tensor.tolist()))) + " ]"

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
    traj_dir_i: ColumnLens
    joint_pos_im1: ColumnLens
    joint_vel_im1: ColumnLens
    gait_i: ColumnLens
    num_cols: int

    def __init__(self, traj_count: int, joint_count: int):

        offset = 0
        next_offset, indices = build_column_indices(offset, 4, traj_count)
        self.traj_pos_i = ColumnLens(2, indices)
        _, indices = build_column_indices(offset + 2, 4, traj_count)
        self.traj_dir_i = ColumnLens(2, indices)

        offset = next_offset
        next_offset, indices = build_column_indices(offset, 6, joint_count)
        self.joint_pos_im1 = ColumnLens(3, indices)
        _, indices = build_column_indices(offset + 3, 6, joint_count)
        self.joint_vel_im1 = ColumnLens(3, indices)

        offset = next_offset
        next_offset, indices = build_column_indices(offset, 8, 1)
        self.gait_i = ColumnLens(8, indices)

        self.num_cols = next_offset

    def unnormalize(self, data: torch.Tensor, mean: torch.Tensor, std: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
        return data * (std / w) + mean

    def normalize(self, data: torch.Tensor, mean: torch.Tensor, std: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
        return (data - mean) * (w / std)

    def print(self, data: torch.Tensor):
        for attr_name, attr_type in InputLens.__annotations__.items():
            if attr_type == ColumnLens:
                print(f"    {attr_name} =")
                tensor_strings = []
                tensors_per_line = MAX_COLUMNS_PER_LINE // self.__dict__[attr_name].size
                for i, index in enumerate(self.__dict__[attr_name].indices):
                    tensor = data[index : index + self.__dict__[attr_name].size]
                    tensor_strings.append(tensor_fmt(tensor))
                for i in range(0, len(tensor_strings), tensors_per_line):
                    print("        " + (", ".join(tensor_strings[i:i+tensors_per_line])))

class OutputLens:
    traj_pos_ip1: ColumnLens
    traj_dir_ip1: ColumnLens
    joint_pos_i: ColumnLens
    joint_vel_i: ColumnLens
    joint_rot_i: ColumnLens
    root_vel_i: ColumnLens
    root_angvel_i: ColumnLens
    phase_vel_i: ColumnLens
    contacts_i: ColumnLens
    num_cols: int

    def __init__(self, traj_count: int, joint_count: int):
        offset = 0
        next_offset, indices = build_column_indices(offset, 4, traj_count)
        self.traj_pos_ip1 = ColumnLens(2, indices)
        _, indices = build_column_indices(offset + 2, 4, traj_count)
        self.traj_dir_ip1 = ColumnLens(2, indices)

        offset = next_offset
        next_offset, indices = build_column_indices(offset, 9, joint_count)
        self.joint_pos_i = ColumnLens(3, indices)
        _, indices = build_column_indices(offset + 3, 9, joint_count)
        self.joint_vel_i = ColumnLens(3, indices)
        _, indices = build_column_indices(offset + 6, 9, joint_count)
        self.joint_rot_i = ColumnLens(3, indices)

        offset = next_offset
        next_offset, indices = build_column_indices(offset, 3, 1)
        self.root_vel_i = ColumnLens(2, indices)
        _, indices = build_column_indices(offset + 2, 3, 1)
        self.root_angvel_i = ColumnLens(1, indices)

        offset = next_offset
        next_offset, indices = build_column_indices(offset, 1, 1)
        self.phase_vel_i = ColumnLens(1, indices)

        offset = next_offset
        next_offset, indices = build_column_indices(offset, 4, 1)
        self.contacts_i = ColumnLens(4, indices)

        self.num_cols = next_offset

    def unnormalize(self, data: torch.Tensor, mean: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
        return data * std + mean

    def normalize(self, data: torch.Tensor, mean: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
        return (data - mean) / std

    def print(self, data: torch.Tensor):
        for attr_name, attr_type in OutputLens.__annotations__.items():
            if attr_type == ColumnLens:
                print(f"    {attr_name} =")
                tensor_strings = []
                tensors_per_line = MAX_COLUMNS_PER_LINE // self.__dict__[attr_name].size
                for i, index in enumerate(self.__dict__[attr_name].indices):
                    tensor = data[index : index + self.__dict__[attr_name].size]
                    tensor_strings.append(tensor_fmt(tensor))
                for i in range(0, len(tensor_strings), tensors_per_line):
                    print("        " + (", ".join(tensor_strings[i:i+tensors_per_line])))
