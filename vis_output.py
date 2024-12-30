import glob
import os
import pickle
import sys
from typing import Tuple, TypedDict

import numpy as np
import pygfx as gfx
import flycam
import torch
from wgpu.gui.auto import WgpuCanvas, run

import datalens
import math_util as mu
from skeleton import Skeleton
from renderbuddy import RenderBuddy

OUTPUT_DIR = "output"
TRAJ_WINDOW_SIZE = 12
TRAJ_ELEMENT_SIZE = 4


def unpickle_obj(filename):
    with open(filename, "rb") as f:
        return pickle.load(f)


def bone_geometry(base: np.ndarray) -> gfx.Geometry:
    zero = np.array([0, 0, 0])
    up = np.array([0, 1, 0])
    mat = mu.build_look_at_mat(base, zero, up)
    l = np.linalg.norm(base)
    width = l * 0.1
    z_offset = l * 0.2
    local_positions = np.array([[0, 0, 0, 1],
                                [ width, width, -z_offset, 1],
                                [-width, width, -z_offset, 1],
                                [-width, -width, -z_offset, 1],
                                [ width, -width, -z_offset, 1],
                                [0, 0, -l, 1]])
    positions = (mat @ np.expand_dims(local_positions, axis=-1))[:, 0:3].squeeze().astype(np.float32)
    indices = np.array([[0, 1, 2], [0, 2, 3], [0, 3, 4], [0, 4, 1],
                        [5, 2, 1], [5, 3, 2], [5, 4, 3], [5, 1, 4]], dtype=np.int32)
    geom = gfx.Geometry(
        indices=indices,
        positions=positions,
    )
    return geom


def add_skeleton_mesh(skeleton: Skeleton, node: gfx.WorldObject):
    world_offsets = np.zeros((skeleton.num_joints, 3))
    for child in skeleton.joint_names:
        child_index = skeleton.get_joint_index(child)
        parent_index = skeleton.get_parent_index(child)
        if parent_index >= 0:
            offset = world_offsets[parent_index] + skeleton.get_joint_offset(child)
        else:
            offset = skeleton.get_joint_offset(child)
        world_offsets[child_index] = offset

    for child in skeleton.joint_names:
        child_index = skeleton.get_joint_index(child)
        parent_index = skeleton.get_parent_index(child)
        if parent_index >= 0:
            bone = gfx.Mesh(
                bone_geometry(-np.array(skeleton.get_joint_offset(child))),
                gfx.MeshPhongMaterial(color=(0.5, 0.5, 1.0, 1.0), flat_shading=True),
            )
            bone.local.position = world_offsets[child_index]
            node.add(bone)
        else:
            bone = gfx.Mesh(
                bone_geometry(np.array([0, 0, 0])),
                gfx.MeshPhongMaterial(color=(0.5, 0.5, 1.0, 1.0), flat_shading=True),
            )
            bone.local.position = world_offsets[child_index]
            node.add(bone)


class VisOutputRenderBuddy(RenderBuddy):
    skeleton: Skeleton
    y_lens: datalens.OutputLens
    Y: torch.Tensor
    row_group: gfx.Group
    row: int
    camera: gfx.PerspectiveCamera
    canvas: WgpuCanvas
    playing: bool

    def __init__(self, skeleton: Skeleton, y_lens: datalens.OutputLens, Y: torch.Tensor):
        super().__init__()

        self.skeleton = skeleton
        self.y_lens = y_lens
        self.Y = Y

        self.row_group = gfx.Group()
        self.scene.add(self.row_group)

        self.skeleton_group = gfx.Group()
        add_skeleton_mesh(self.skeleton, self.skeleton_group)
        self.scene.add(self.skeleton_group)

        self.camera.show_object(self.scene, up=(0, 1, 0), scale=1.4)

        self.playing = False
        self.retain_row(0)

    def retain_row(self, row: int):
        self.row = row
        self.scene.remove(self.row_group)
        self.row_group = gfx.Group()

        Y_row = self.Y[row]

        print(f"num_joints {self.skeleton.num_joints}, num_children = {len(self.skeleton_group.children)}")
        for i in range(self.skeleton.num_joints):
            exp = self.y_lens.joint_rot_i.get(Y_row, i)
            rot = mu.expmap(exp)
            self.skeleton_group.children[i].local.rotation = rot

        """
        positions = []
        colors = []

        for child in skeleton.joint_names:
            child_index = skeleton.get_joint_index(child)
            parent_index = skeleton.get_parent_index(child)
            if parent_index >= 0:
                # line from p.offset
                p0 = self.y_lens.joint_pos_i.get(Y_row, parent_index).tolist()
                p1 = self.y_lens.joint_pos_i.get(Y_row, child_index).tolist()
                positions += [p0, p1]
                colors += [[1, 1, 1, 1], [0.5, 0.5, 1, 1]]
        joint_line = gfx.Line(
            gfx.Geometry(positions=positions, colors=colors), gfx.LineSegmentMaterial(thickness=2, color_mode="vertex")
        )
        """

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

        #self.row_group.add(joint_line)
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
    # unpickle skeleton
    # pick ANY skeleton in the output dir, they should all be the same.
    skeleton_files = glob.glob(os.path.join(OUTPUT_DIR, "*_skeleton.pkl"))
    assert len(skeleton_files) > 0, "could not find any pickled skeletons in output folder"
    skeleton = unpickle_obj(skeleton_files[0])

    # load output
    y_lens = datalens.OutputLens(TRAJ_WINDOW_SIZE, skeleton.num_joints)
    Y = torch.load(os.path.join(OUTPUT_DIR, "Y.pth"), weights_only=True)
    Y_mean = torch.load(os.path.join(OUTPUT_DIR, "Y_mean.pth"), weights_only=True)
    Y_std = torch.load(os.path.join(OUTPUT_DIR, "Y_std.pth"), weights_only=True)

    assert y_lens.num_cols == Y.shape[1]

    # un-normalize input for visualiztion
    Y = y_lens.unnormalize(Y, Y_mean, Y_std)

    render_buddy = VisOutputRenderBuddy(skeleton, y_lens, Y)
    run()
