import os
import pickle
import sys
from typing import Tuple, TypedDict

import numpy as np
import pygfx as gfx
import torch
from wgpu.gui.auto import WgpuCanvas, run

import mocap

OUTPUT_DIR = "output"
TRAJ_WINDOW_SIZE = 12
TRAJ_ELEMENT_SIZE = 4

class ColumnView(TypedDict):
    size: int
    indices: list[int]

class InputView(TypedDict):
    traj_pos_i: ColumnView
    traj_vel_i: ColumnView
    joint_pos_im1: ColumnView
    joint_vel_im1: ColumnView


class RenderBuddy:
    def __init__(self, skeleton: mocap.Skeleton, input_view: InputView):
        self.skeleton = skeleton
        self.input_view = input_view


def build_column_indices(start: int, stride: int, repeat: int = 1) -> Tuple[int, list[int]]:
    indices = [i * stride + start for i in range(repeat)]
    offset = repeat * stride + start
    return offset, indices


def build_input_view(skeleton: mocap.Skeleton) -> InputView:
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


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Error: expected mocap filename (without .bvh extension)")
        exit(1)

    mocap_basename = sys.argv[1]
    outbasepath = os.path.join(OUTPUT_DIR, mocap_basename)

    # unpickle/load data
    skeleton = mocap.unpickle_obj(outbasepath + "_skeleton.pkl")
    input_view = build_input_view(skeleton)

    print(f"skeleton.num_joints = {skeleton.num_joints}")
    X = torch.load(os.path.join(OUTPUT_DIR, "X.pth"), weights_only=True)
    print(f"X.shape = {X.shape}")

    num_cols = input_view["joint_vel_im1"]["indices"][-1] + input_view["joint_vel_im1"]["size"]
    assert num_cols == X.shape[1]
