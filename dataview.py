from typing import Tuple, TypedDict

import torch

from skeleton import Skeleton

TRAJ_WINDOW_SIZE = 12


class ColumnView(TypedDict):
    size: int
    indices: list[int]


class InputView(TypedDict):
    traj_pos_i: ColumnView
    traj_vel_i: ColumnView
    joint_pos_im1: ColumnView
    joint_vel_im1: ColumnView


class OutputView(TypedDict):
    traj_pos_ip1: ColumnView
    traj_dir_ip1: ColumnView
    joint_pos_i: ColumnView
    joint_vel_i: ColumnView
    root_vel_i: ColumnView
    root_angvel_i: ColumnView
    joint_angvel_i: ColumnView
    phase_vel_i: ColumnView
    contacts_i: ColumnView


def ref(row: torch.Tensor, output_view: OutputView, key: str, index: str) -> torch.Tensor:
    index = output_view[key]["indices"][index]
    size = output_view[key]["size"]
    return row[index : index + size]


def build_column_indices(start: int, stride: int, repeat: int = 1) -> Tuple[int, list[int]]:
    indices = [i * stride + start for i in range(repeat)]
    offset = repeat * stride + start
    return offset, indices


def build_input_view(skeleton: Skeleton) -> InputView:
    num_joints = skeleton.num_joints

    input_view = {}
    offset = 0
    next_offset, indices = build_column_indices(offset, 4, TRAJ_WINDOW_SIZE)
    input_view["traj_pos_i"] = {"size": 2, "indices": indices}
    _, indices = build_column_indices(offset + 2, 4, TRAJ_WINDOW_SIZE)
    input_view["traj_vel_i"] = {"size": 2, "indices": indices}

    offset = next_offset
    next_offset, indices = build_column_indices(offset, 6, num_joints)
    input_view["joint_pos_im1"] = {"size": 3, "indices": indices}
    _, indices = build_column_indices(offset + 3, 6, num_joints)
    input_view["joint_vel_im1"] = {"size": 3, "indices": indices}

    for k, v in input_view.items():
        print(f"{k}: {v}")

    return InputView(**input_view)


def build_output_view(skeleton: Skeleton) -> OutputView:
    num_joints = skeleton.num_joints

    output_view = {}
    offset = 0
    next_offset, indices = build_column_indices(offset, 4, TRAJ_WINDOW_SIZE)
    output_view["traj_pos_ip1"] = {"size": 2, "indices": indices}
    _, indices = build_column_indices(offset + 2, 4, TRAJ_WINDOW_SIZE)
    output_view["traj_dir_ip1"] = {"size": 2, "indices": indices}

    offset = next_offset
    next_offset, indices = build_column_indices(offset, 9, num_joints)
    output_view["joint_pos_i"] = {"size": 3, "indices": indices}
    _, indices = build_column_indices(offset + 3, 9, num_joints)
    output_view["joint_vel_i"] = {"size": 3, "indices": indices}
    _, indices = build_column_indices(offset + 6, 9, num_joints)
    output_view["joint_angvel_i"] = {"size": 3, "indices": indices}

    offset = next_offset
    next_offset, indices = build_column_indices(offset, 3, 1)
    output_view["root_vel_i"] = {"size": 2, "indices": indices}
    _, indices = build_column_indices(offset + 2, 3, 1)
    output_view["root_angvel_i"] = {"size": 1, "indices": indices}

    offset = next_offset
    next_offset, indices = build_column_indices(offset, 1, 1)
    output_view["phase_vel_i"] = {"size": 1, "indices": indices}

    offset = next_offset
    next_offset, indices = build_column_indices(offset, 4, 1)
    output_view["contacts_i"] = {"size": 4, "indices": indices}

    for k, v in output_view.items():
        print(f"{k}: {v}")

    return OutputView(**output_view)
