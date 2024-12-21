import os
import pickle
import sys
from typing import Tuple, TypedDict
from time import perf_counter

import numpy as np
import pygfx as gfx
import flycam
import torch
from wgpu.gui.auto import WgpuCanvas, run

import math_util as mu
from skeleton import Skeleton
from renderbuddy import RenderBuddy

OUTPUT_DIR = "output"
TRAJ_WINDOW_SIZE = 12
TRAJ_ELEMENT_SIZE = 4


def unpickle_obj(filename):
    with open(filename, "rb") as f:
        return pickle.load(f)


class ColumnView(TypedDict):
    size: int
    indices: list[int]


class InputView(TypedDict):
    traj_pos_i: ColumnView
    traj_vel_i: ColumnView
    joint_pos_im1: ColumnView
    joint_vel_im1: ColumnView


def ref(row: torch.Tensor, input_view: InputView, key: str, index: str) -> torch.Tensor:
    index = input_view[key]["indices"][index]
    size = input_view[key]["size"]
    return row[index : index + size]


class VisInputRenderBuddy(RenderBuddy):
    skeleton: Skeleton
    input_view: InputView
    X: torch.Tensor
    row_group: gfx.Group
    row: int
    camera: gfx.PerspectiveCamera
    canvas: WgpuCanvas
    playing: bool

    def __init__(self, skeleton: Skeleton, input_view: InputView, X: torch.Tensor):
        super().__init__()

        self.skeleton = skeleton
        self.input_view = input_view
        self.X = X

        self.row_group = gfx.Group()
        self.scene.add(self.row_group)

        self.camera.show_object(self.scene, up=(0, 1, 0), scale=1.4)

        self.playing = False
        self.retain_row(0)

    def retain_row(self, row: int):
        self.row = row
        self.scene.remove(self.row_group)
        self.row_group = gfx.Group()

        positions = []
        colors = []
        X_row = self.X[row]
        for child in skeleton.joint_names:
            child_index = skeleton.get_joint_index(child)
            parent_index = skeleton.get_parent_index(child)
            if parent_index >= 0:
                # line from p.offset
                p0 = ref(X_row, self.input_view, "joint_pos_im1", parent_index).tolist()
                p1 = ref(X_row, self.input_view, "joint_pos_im1", child_index).tolist()
                positions += [p0, p1]
                colors += [[1, 1, 1, 1], [0.5, 0.5, 1, 1]]
        joint_line = gfx.Line(
            gfx.Geometry(positions=positions, colors=colors), gfx.LineSegmentMaterial(thickness=2, color_mode="vertex")
        )

        positions = []
        colors = []
        for i in range(TRAJ_WINDOW_SIZE - 1):
            p0 = ref(X_row, self.input_view, "traj_pos_i", i)
            p1 = ref(X_row, self.input_view, "traj_pos_i", i + 1)
            positions += [[p0[0], 0.0, p0[1]], [p1[0], 0.0, p1[1]]]
            if i % 2 == 0:
                colors += [[1, 1, 1, 1], [1, 1, 1, 1]]
            else:
                colors += [[1, 0, 0, 1], [1, 0, 0, 1]]

        traj_line = gfx.Line(
            gfx.Geometry(positions=positions, colors=colors), gfx.LineSegmentMaterial(thickness=2, color_mode="vertex")
        )

        self.row_group.add(joint_line)
        self.row_group.add(traj_line)
        self.scene.add(self.row_group)

    def on_animate(self, dt: float):
        super().on_animate(dt)
        if self.playing:
            row = self.row + 1
            if row >= self.X.shape[0]:
                row = 0
            self.retain_row(row)

    def on_key_down(self, event):
        super().on_key_down(event)
        if event.key == " ":
            self.playing = not self.playing

    def on_dpad_left(self):
        super().on_dpad_left()
        print("DPAD LEFT!")

    def on_dpad_right(self):
        super().on_dpad_right()
        print("DPAD RIGHT!")

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


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Error: expected mocap filename (without .bvh extension)")
        exit(1)

    mocap_basename = sys.argv[1]
    outbasepath = os.path.join(OUTPUT_DIR, mocap_basename)

    # unpickle/load data
    skeleton = unpickle_obj(outbasepath + "_skeleton.pkl")
    input_view = build_input_view(skeleton)

    print(f"skeleton.num_joints = {skeleton.num_joints}")
    X = torch.load(os.path.join(OUTPUT_DIR, "X.pth"), weights_only=True)
    X_mean = torch.load(os.path.join(OUTPUT_DIR, "X_mean.pth"), weights_only=True)
    X_std = torch.load(os.path.join(OUTPUT_DIR, "X_std.pth"), weights_only=True)
    X_w = torch.load(os.path.join(OUTPUT_DIR, "X_w.pth"), weights_only=True)
    print(f"X.shape = {X.shape}")

    num_cols = input_view["joint_vel_im1"]["indices"][-1] + input_view["joint_vel_im1"]["size"]
    assert num_cols == X.shape[1]

    # un-normalize input for visualiztion
    X = X * (X_std / X_w) + X_mean

    renderBuddy = VisInputRenderBuddy(skeleton, input_view, X)
    run()
