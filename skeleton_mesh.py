from typing import Tuple

import numpy as np
import pygfx as gfx

import math_util as mu
from skeleton import Skeleton


def build_bone_mesh(base: np.ndarray, tip: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    up = np.array([0, 1, 0])
    mat = mu.build_look_at_mat(base, tip, up)
    l = np.linalg.norm(base - tip)
    width = l * 0.1
    z_offset = l * 0.2
    local_positions = np.array(
        [
            [0, 0, 0, 1],
            [width, width, -z_offset, 1],
            [-width, width, -z_offset, 1],
            [-width, -width, -z_offset, 1],
            [width, -width, -z_offset, 1],
            [0, 0, -l, 1],
        ]
    )
    positions = (mat @ np.expand_dims(local_positions, axis=-1))[:, 0:3].squeeze().astype(np.float32)
    indices = np.array(
        [[0, 1, 2], [0, 2, 3], [0, 3, 4], [0, 4, 1], [5, 2, 1], [5, 3, 2], [5, 4, 3], [5, 1, 4]], dtype=np.int32
    )
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
                verts[vert_start_index : vert_start_index + BONE_VERTEX_COUNT] = pp
                tris[tri_start_index : tri_start_index + BONE_TRIANGLE_COUNT] = ii + vert_start_index

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
