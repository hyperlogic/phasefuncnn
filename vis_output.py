#
# Copyright (c) 2025 Anthony J. Thibault
# This software is licensed under the MIT License. See LICENSE for more details.
#

import glob
import os
import pickle
from typing import Tuple

import numpy as np
import pygfx as gfx
import torch
from wgpu.gui.auto import WgpuCanvas, run

import datalens
import math_util as mu
import skeleton
import skeleton_mesh
from renderbuddy import RenderBuddy
from skeleton import Skeleton

OUTPUT_DIR = "output"
TRAJ_WINDOW_SIZE = 12
TRAJ_ELEMENT_SIZE = 4
SAMPLE_RATE = 60


def unpickle_obj(filename):
    with open(filename, "rb") as f:
        return pickle.load(f)


class VisOutputRenderBuddy(RenderBuddy):
    skeleton: Skeleton
    y_lens: datalens.OutputLens
    Y: torch.Tensor
    P: torch.Tensor
    row_group: gfx.Group
    row: int
    camera: gfx.PerspectiveCamera
    canvas: WgpuCanvas
    playing: bool

    def __init__(self, skeleton: Skeleton, y_lens: datalens.OutputLens, Y: torch.Tensor, P: torch.Tensor):
        super().__init__()

        self.skeleton = skeleton
        self.y_lens = y_lens
        self.Y = Y
        self.P = P

        self.skeleton_group = gfx.Group()
        self.traj_line = gfx.Group()
        self.skeleton_group.add(self.traj_line)
        self.bones = skeleton_mesh.add_skeleton_mesh(self.skeleton, self.skeleton_group)
        self.scene.add(self.skeleton_group)

        self.camera.show_object(self.scene, up=(0, 1, 0), scale=1.4)

        self.playing = False
        self.animate_skeleton(0)

    def animate_skeleton(self, row: int):
        self.row = row

        Y_row = self.Y[row]

        # rotate the bones!
        global_rots = np.array([0, 0, 0, 1]) * np.ones((self.skeleton.num_joints, 4))
        for i in range(self.skeleton.num_joints):
            # extract rotation exponent from output
            rot6d = self.y_lens.joint_rot_i.get(Y_row, i)

            # convert into a quaternion
            mat = np.eye(3)
            x_axis = rot6d[0:3].numpy()
            y_axis = rot6d[3:6].numpy()
            z_axis = np.linalg.cross(x_axis, y_axis)
            y_axis = np.linalg.cross(z_axis, x_axis)
            mat[0:3, 0] = x_axis
            mat[0:3, 1] = y_axis
            mat[0:3, 2] = z_axis
            global_rots[i] = mu.quat_from_mat(mat)

            # transform rotations from root_relative to local to parent joint
            joint_name = self.skeleton.get_joint_name(i)
            parent_index = self.skeleton.get_parent_index(joint_name)
            if parent_index >= 0:
                local_rot = mu.quat_mul(mu.quat_inverse(global_rots[parent_index]), global_rots[i])
            else:
                local_rot = global_rots[i]

            self.bones[i].local.rotation = local_rot

        # update pelvis pos
        pelvis_pos = self.y_lens.joint_pos_i.get(Y_row, 0).tolist()
        self.bones[0].local.position = pelvis_pos

        # apply root motion!
        root_vel = np.array([y_lens.root_vel_i.get(Y_row, 0)[0], 0, y_lens.root_vel_i.get(Y_row, 0)[1]])
        root_angvel = y_lens.root_angvel_i.get(Y_row, 0).item()
        dt = 1 / SAMPLE_RATE
        delta_xform = np.eye(4)
        mu.build_mat_from_quat_pos(
            delta_xform, mu.quat_from_angle_axis(root_angvel * dt, np.array([0, 1, 0])), root_vel * dt
        )
        root_xform = np.eye(4)
        mu.build_mat_from_quat_pos(root_xform, self.skeleton_group.local.rotation, self.skeleton_group.local.position)
        final_xform = root_xform @ delta_xform
        self.skeleton_group.local.position = final_xform[0:3, 3]
        self.skeleton_group.local.rotation = mu.quat_from_mat(final_xform)

        # create a new line mesh for the trajectory
        positions = []
        colors = []
        for i in range(TRAJ_WINDOW_SIZE - 1):
            p0 = self.y_lens.traj_pos_ip1.get(Y_row, i)
            p1 = self.y_lens.traj_pos_ip1.get(Y_row, i + 1)
            positions += [[p0[0], 0.0, p0[1]], [p1[0], 0.0, p1[1]]]
            if i % 2 == 0:
                colors += [[1, 1, 1, 1], [1, 1, 1, 1]]
            else:
                colors += [[1, 0, 0, 1], [1, 0, 0, 1]]

        traj_line = gfx.Line(
            gfx.Geometry(positions=positions, colors=colors), gfx.LineSegmentMaterial(thickness=2, color_mode="vertex")
        )

        phase = (P[row] / (2.0 * torch.pi)).float()
        text_node = gfx.Text(
            text=f"frame={row},phase={phase:.2}",
            font_size=20,
            screen_space=True,
            text_align="left",
            anchor="top-left",
            material=gfx.TextMaterial(color="#ffffff", outline_color="#000", outline_thickness=1),
        )

        self.skeleton_group.add(text_node)
        self.skeleton_group.remove(self.traj_line)
        self.skeleton_group.add(traj_line)
        self.traj_line = traj_line

    def on_animate(self, dt: float):
        super().on_animate(dt)
        if self.playing:
            row = self.row + 1
            if row >= self.Y.shape[0]:
                row = 0
            self.animate_skeleton(row)

    def on_key_down(self, event):
        super().on_key_down(event)
        if event.key == " ":
            self.playing = not self.playing

    def on_dpad_left(self):
        super().on_dpad_left()
        row = self.row - 1
        if row < 0:
            row = self.Y.shape[0] - 1
        self.animate_skeleton(row)

    def on_dpad_right(self):
        super().on_dpad_right()
        row = self.row + 1
        if row >= self.Y.shape[0]:
            row = 0
        self.animate_skeleton(row)


if __name__ == "__main__":
    # unpickle skeleton
    # pick ANY skeleton in the output dir, they should all be the same.
    skeleton_files = glob.glob(os.path.join(OUTPUT_DIR, "skeleton/*.pkl"))
    assert len(skeleton_files) > 0, "could not find any pickled skeletons in output folder"
    skeleton = unpickle_obj(skeleton_files[0])

    # load output
    y_lens = datalens.OutputLens(TRAJ_WINDOW_SIZE, skeleton.num_joints)
    Y = torch.load(os.path.join(OUTPUT_DIR, "Y.pth"), weights_only=True)
    Y_mean = torch.load(os.path.join(OUTPUT_DIR, "Y_mean.pth"), weights_only=True)
    Y_std = torch.load(os.path.join(OUTPUT_DIR, "Y_std.pth"), weights_only=True)

    # load phase
    P = torch.load(os.path.join(OUTPUT_DIR, "P.pth"), weights_only=True)

    assert y_lens.num_cols == Y.shape[1]

    # un-normalize input for visualiztion
    Y = y_lens.unnormalize(Y, Y_mean, Y_std)

    render_buddy = VisOutputRenderBuddy(skeleton, y_lens, Y, P)
    run()
