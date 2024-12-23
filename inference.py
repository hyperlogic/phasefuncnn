import math
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
SAMPLE_RATE = 60


def unpickle_obj(filename):
    with open(filename, "rb") as f:
        return pickle.load(f)


class VisOutputRenderBuddy(RenderBuddy):
    skeleton: Skeleton
    x_view: dataview.InputView
    y_view: dataview.OutputView
    model: nn.Module
    x: torch.Tensor
    p: float
    y_mean: torch.Tensor
    y_std: torch.Tensor
    x_mean: torch.Tensor
    x_std: torch.Tensor
    x_w: torch.Tensor
    y: torch.Tensor
    group: gfx.Group
    camera: gfx.PerspectiveCamera
    canvas: WgpuCanvas
    playing: bool

    def __init__(
        self,
        skeleton: Skeleton,
        x_view: dataview.InputView,
        y_view: dataview.OutputView,
        model: nn.Module,
        x: torch.Tensor,
        p: float,
        y_mean: torch.Tensor,
        y_std: torch.Tensor,
        x_mean: torch.Tensor,
        x_std: torch.Tensor,
        x_w: torch.Tensor,
    ):
        super().__init__()

        self.skeleton = skeleton
        self.x_view = x_view
        self.y_view = y_view
        self.model = model
        self.x = x.clone()
        self.p = p
        self.y_mean = y_mean
        self.y_std = y_std
        self.x_mean = x_mean
        self.x_std = x_std
        self.x_w = x_w

        # run inference
        self.y = self.model(self.x, self.p).detach()

        self.group = gfx.Group()
        self.scene.add(self.group)

        self.camera.show_object(self.scene, up=(0, 1, 0), scale=1.4)

        self.retain_output()

        self.playing = False
        self.t = 0.0

    def retain_output(self):
        self.scene.remove(self.group)
        self.group = gfx.Group()

        positions = []
        colors = []

        # un-normalize output
        y = self.y * self.y_std + self.y_mean

        for child in skeleton.joint_names:
            child_index = skeleton.get_joint_index(child)
            parent_index = skeleton.get_parent_index(child)
            if parent_index >= 0:
                # line from p.offset
                p0 = dataview.get(y, self.y_view, "joint_pos_i", parent_index).tolist()
                p1 = dataview.get(y, self.y_view, "joint_pos_i", child_index).tolist()
                positions += [p0, p1]
                colors += [[1, 1, 1, 1], [0.5, 0.5, 1, 1]]
        joint_line = gfx.Line(
            gfx.Geometry(positions=positions, colors=colors), gfx.LineSegmentMaterial(thickness=2, color_mode="vertex")
        )

        positions = []
        colors = []
        for i in range(TRAJ_WINDOW_SIZE - 1):
            p0 = dataview.get(y, self.y_view, "traj_pos_ip1", i)
            p1 = dataview.get(y, self.y_view, "traj_pos_ip1", i + 1)
            positions += [[p0[0], 0.0, p0[1]], [p1[0], 0.0, p1[1]]]
            if i % 2 == 0:
                colors += [[1, 1, 1, 1], [1, 1, 1, 1]]
            else:
                colors += [[1, 0, 0, 1], [1, 0, 0, 1]]

        traj_line = gfx.Line(
            gfx.Geometry(positions=positions, colors=colors), gfx.LineSegmentMaterial(thickness=2, color_mode="vertex")
        )

        self.group.add(joint_line)
        self.group.add(traj_line)
        self.scene.add(self.group)

    def on_animate(self, dt: float):
        super().on_animate(dt)
        """
        if self.playing:
            self.t += dt
            if self.t > (1 / SAMPLE_RATE):
                self.tick_model()
                self.retain_output()
        """

    def tick_model(self):
        self.x = torch.zeros(self.x.shape)

        """
        print("x = ")
        for i in range(TRAJ_WINDOW_SIZE):
            # traj_pos = dataview.get(self.y, self.y_view, "traj_pos_ip1", i)
            # traj_dir = dataview.get(self.y, self.y_view, "traj_dir_ip1", i)

            traj_pos = torch.tensor([0.0, 0.0], dtype=torch.float32, requires_grad=False)
            traj_dir = torch.tensor([1.0, 0.0], dtype=torch.float32, requires_grad=False)
            dataview.set(self.x, self.x_view, "traj_pos_i", i, traj_pos)
            dataview.set(self.x, self.x_view, "traj_vel_i", i, traj_dir * (1 / SAMPLE_RATE))

            print(f"    traj_pos_i[{i}] = {dataview.get(self.x, self.x_view, 'traj_pos_i', i)}")
            print(f"    traj_vel_i[{i}] = {dataview.get(self.x, self.x_view, 'traj_vel_i', i)}")

        # global joint_pos
        g_joint_positions = torch.zeros((self.skeleton.num_joints, 3))
        for i in range(self.skeleton.num_joints):
            joint_name = self.skeleton.get_joint_name(i)
            parent_i = self.skeleton.get_parent_index(joint_name)
            joint_pos = torch.tensor(self.skeleton.get_joint_offset(joint_name), dtype=torch.float32, requires_grad=False)
            if parent_i >= 0:
                g_joint_positions[i] = g_joint_positions[parent_i] + joint_pos
            else:
                g_joint_positions[i] = joint_pos

        for i in range(self.skeleton.num_joints):
            #joint_pos = dataview.get(self.y, self.y_view, "joint_pos_i", i)
            #joint_vel = dataview.get(self.y, self.y_view, "joint_vel_i", i)

            joint_name = self.skeleton.get_joint_name(i)
            joint_pos = g_joint_positions[i]
            joint_vel = torch.tensor([0.0, 0.0, 0.0], dtype=torch.float32, requires_grad=False)

            dataview.set(self.x, self.x_view, "joint_pos_im1", i, joint_pos)
            dataview.set(self.x, self.x_view, "joint_vel_im1", i, joint_vel)

            print(f"    joint_name[{i}] = {joint_name}")
            print(f"    joint_pos_im1[{i}] = {dataview.get(self.x, self.x_view, 'joint_pos_im1', i)}")
            print(f"    joint_vel_im1[{i}] = {dataview.get(self.x, self.x_view, 'joint_vel_im1', i)}")

        phase_vel = dataview.get(self.y, self.y_view, "phase_vel_i", 0).item()
        self.p += phase_vel * (1 / SAMPLE_RATE)
        self.p = self.p % (2 * math.pi)

        print(f"phase = {self.p}")

        # re-normalize input
        self.x = (self.x - self.x_mean) * (self.x_w / self.x_std)
        """

        # model inference
        self.y = self.model(self.x, self.p).detach()

    def on_key_down(self, event):
        super().on_key_down(event)
        if event.key == " ":
            # self.playing = not self.playing
            self.tick_model()
            self.retain_output()

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

    torch.no_grad()

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
    # X = X * (X_std / X_w) + X_mean
    print(f"X.shape = {X.shape}")

    # load output
    Y = torch.load(os.path.join(OUTPUT_DIR, "Y.pth"), weights_only=True)
    Y_mean = torch.load(os.path.join(OUTPUT_DIR, "Y_mean.pth"), weights_only=True)
    Y_std = torch.load(os.path.join(OUTPUT_DIR, "Y_std.pth"), weights_only=True)
    # un-normalize input for visualiztion
    # Y = Y * Y_std + Y_mean

    # load phase
    P = torch.load(os.path.join(OUTPUT_DIR, "P.pth"), weights_only=True)
    print(f"P.shape = {P.shape}")

    def make_batch(t: torch.Tensor, start: int, batch_size: int) -> torch.Tensor:
        return t[start : start + batch_size]

    ii = torch.randint(0, X.shape[0], (1,)).item()
    output = model(X[ii], P[ii])

    criterion = nn.L1Loss()
    loss = criterion(output, Y[ii])

    print(f"loss = {loss}")

    render_buddy = VisOutputRenderBuddy(
        skeleton, input_view, output_view, model, X[1000], P[1000], Y_mean, Y_std, X_mean, X_std, X_w
    )
    run()
