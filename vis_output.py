import glob
import os
import pickle
from typing import Tuple

import numpy as np
import pygfx as gfx
import skeleton
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


def build_bone_mesh(base: np.ndarray, tip: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    up = np.array([0, 1, 0])
    mat = mu.build_look_at_mat(base, tip, up)
    l = np.linalg.norm(base - tip)
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
    return positions, indices


def add_skeleton_mesh(skeleton: Skeleton, root_node: gfx.WorldObject) -> list[gfx.WorldObject]:
    zero = np.array([0, 0, 0], dtype=np.float32)
    BONE_VERTEX_COUNT = 6
    BONE_TRIANGLE_COUNT = 8
    bones = []
    for joint_name in skeleton.joint_names:
        children_indices = skeleton.get_children_indices(joint_name)
        num_children = len(children_indices)
        if num_children > 0:
            # pre allocate verts and tris with correct size.
            verts = np.zeros((len(children_indices) * BONE_VERTEX_COUNT, 3), dtype=np.float32)
            tris = np.zeros((len(children_indices) * BONE_TRIANGLE_COUNT, 3), dtype=np.int32)

            # for each bone, insert attributes into verts and tris
            for i, child_index in enumerate(children_indices):
                child_name = skeleton.get_joint_name(child_index)
                tip = np.array(skeleton.get_joint_offset(child_name))
                pp, ii = build_bone_mesh(zero, tip)
                vert_start_index = i * BONE_VERTEX_COUNT
                tri_start_index = i * BONE_TRIANGLE_COUNT
                verts[vert_start_index:vert_start_index + BONE_VERTEX_COUNT] = pp
                tris[tri_start_index:tri_start_index + BONE_TRIANGLE_COUNT] = ii + vert_start_index

            # build the geometry
            geom = gfx.Geometry(positions=verts, indices=tris)
            bone = gfx.Mesh(geom, gfx.MeshPhongMaterial(color=(0.5, 0.5, 1.0, 1.0), flat_shading=True))

            # set local transform, zero rot
            bone.local.position = np.array(skeleton.get_joint_offset(joint_name))

            bones.append(bone)
        else:
            # this joint has no children so just create a group node. (TODO: maybe add a sphere? or a small joint?)
            bones.append(gfx.Group())

    # link nodes up to their parents
    for joint_name in skeleton.joint_names:
        joint_index = skeleton.get_joint_index(joint_name)
        parent_index = skeleton.get_parent_index(joint_name)
        if parent_index >= 0:
            bones[parent_index].add(bones[joint_index])

    # add hips of skeleton to root_node
    root_node.add(bones[0])

    return bones


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
        self.bones = add_skeleton_mesh(self.skeleton, self.skeleton_group)

        self.scene.add(self.skeleton_group)

        self.camera.show_object(self.scene, up=(0, 1, 0), scale=1.4)

        self.playing = False
        self.retain_row(0)

    def retain_row(self, row: int):
        self.row = row
        self.scene.remove(self.row_group)
        self.row_group = gfx.Group()

        Y_row = self.Y[row]

        # rotate the bones!
        global_rots = np.array([0, 0, 0, 1]) * np.ones((self.skeleton.num_joints, 4))
        for i in range(self.skeleton.num_joints):
            # extract rotation exponent from output
            exp = self.y_lens.joint_rot_i.get(Y_row, i)

            # convert into a quaternion
            global_rots[i] = mu.expmap(exp)

            # transform rotations from global_rot to local_rot
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

        # create a line mesh for the trajectory
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
