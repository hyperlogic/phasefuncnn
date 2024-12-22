import os
import pickle
import sys
from typing import Tuple, TypedDict

import numpy as np
import pygfx as gfx
import flycam
import torch
import torch.nn as nn
import torch.nn.functional as F
from wgpu.gui.auto import WgpuCanvas, run

import dataview
import math_util as mu
from pfnn import PFNN
from skeleton import Skeleton
from renderbuddy import RenderBuddy

OUTPUT_DIR = "output"
TRAJ_WINDOW_SIZE = 12
TRAJ_ELEMENT_SIZE = 4


def unpickle_obj(filename):
    with open(filename, "rb") as f:
        return pickle.load(f)


class VisOutputRenderBuddy(RenderBuddy):
    skeleton: Skeleton
    output_view: dataview.OutputView
    Y: torch.Tensor
    row_group: gfx.Group
    row: int
    camera: gfx.PerspectiveCamera
    canvas: WgpuCanvas
    playing: bool

    def __init__(self, skeleton: Skeleton, output_view: dataview.OutputView, Y: torch.Tensor):
        super().__init__()

        self.skeleton = skeleton
        self.output_view = output_view
        self.Y = Y

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

        Y_row = self.Y[row]
        for child in skeleton.joint_names:
            child_index = skeleton.get_joint_index(child)
            parent_index = skeleton.get_parent_index(child)
            if parent_index >= 0:
                # line from p.offset
                p0 = dataview.ref(Y_row, self.output_view, "joint_pos_i", parent_index).tolist()
                p1 = dataview.ref(Y_row, self.output_view, "joint_pos_i", child_index).tolist()
                positions += [p0, p1]
                colors += [[1, 1, 1, 1], [0.5, 0.5, 1, 1]]
        joint_line = gfx.Line(
            gfx.Geometry(positions=positions, colors=colors), gfx.LineSegmentMaterial(thickness=2, color_mode="vertex")
        )

        positions = []
        colors = []
        for i in range(TRAJ_WINDOW_SIZE - 1):
            p0 = dataview.ref(Y_row, self.output_view, "traj_pos_ip1", i)
            p1 = dataview.ref(Y_row, self.output_view, "traj_pos_ip1", i + 1)
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
            if row >= self.Y.shape[0]:
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


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Error: expected mocap filename (without .bvh extension)")
        exit(1)

    mocap_basename = sys.argv[1]
    outbasepath = os.path.join(OUTPUT_DIR, mocap_basename)

    # unpickle/load data
    skeleton = unpickle_obj(outbasepath + "_skeleton.pkl")

    input_view = dataview.build_input_view(skeleton)
    output_view = dataview.build_output_view(skeleton)

    print(f"skeleton.num_joints = {skeleton.num_joints}")

    device = torch.device("cpu")
    print(f"cuda.is_available() = {torch.cuda.is_available()}")
    print(f"device = {device}")

    # load model
    in_features = input_view["num_cols"]
    out_features = output_view["num_cols"]
    print(f"PFNN(in_features = {in_features}, out_features = {out_features}, device = {device}")
    model = PFNN(in_features, out_features, device=device)
    state_dict = torch.load(os.path.join(OUTPUT_DIR, "final_checkpoint.pth"), weights_only=False)
    model.load_state_dict(state_dict)

    # load input, and phase
    X = torch.load(os.path.join(OUTPUT_DIR, "X.pth"), weights_only=True)
    X_mean = torch.load(os.path.join(OUTPUT_DIR, "X_mean.pth"), weights_only=True)
    X_std = torch.load(os.path.join(OUTPUT_DIR, "X_std.pth"), weights_only=True)
    X_w = torch.load(os.path.join(OUTPUT_DIR, "X_w.pth"), weights_only=True)
    # un-normalize input for visualization
    X = X * (X_std / X_w) + X_mean
    print(f"X.shape = {X.shape}")

    # load output
    Y = torch.load(os.path.join(OUTPUT_DIR, "Y.pth"), weights_only=True)
    Y_mean = torch.load(os.path.join(OUTPUT_DIR, "Y_mean.pth"), weights_only=True)
    Y_std = torch.load(os.path.join(OUTPUT_DIR, "Y_std.pth"), weights_only=True)
    # un-normalize input for visualiztion
    Y = Y * Y_std + Y_mean

    # load phase
    P = torch.load(os.path.join(OUTPUT_DIR, "P.pth"), weights_only=True)
    print(f"P.shape = {P.shape}")

    def make_batch(t: torch.Tensor, start: int, batch_size: int) -> torch.Tensor:
        return t[start:start+batch_size]

    ii = torch.randint(0, X.shape[0] - 10, (1,)).item()
    print(f"ii = {ii}")
    xx = make_batch(X, ii, 2)
    pp = make_batch(P, ii, 2)
    yy = make_batch(Y, ii, 2)

    print(f"xx = {xx.shape}, pp.shape = {pp.shape}")
    # x = torch.Size([NN, 234]), p.shape = torch.Size([NN])
    output = model(xx, pp)

    print(f"xx[0] = {xx[0]}")
    print(f"pp[0] = {pp[0]}")
    print(f"output[0] = {output[0]}")

    criterion = nn.L1Loss()
    loss = criterion(output, yy)

    print(f"loss = {loss}")

    """
    render_buddy = VisOutputRenderBuddy(skeleton, output_view, Y)
    run()
    """
