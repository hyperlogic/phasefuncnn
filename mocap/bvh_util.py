import bvh
import copy
import glm
import math
import numpy as np
from .skeleton import Skeleton
from tqdm import trange, tqdm


def glm_to_np(glm_mat, np_mat):
    for i in range(4):
        np_mat[:, i] = glm_mat[i]


def build_xforms_at_frame(xforms, bvh, skeleton, bvh_frame, frame):

    for i in range(skeleton.num_joints):
        joint_name = skeleton.get_joint_name(i)
        offset = skeleton.get_joint_offset(joint_name)

        pos = glm.vec3(offset[0], offset[1], offset[2])
        if skeleton.has_pos(joint_name):
            pos += glm.vec3(
                bvh.frame_joint_channel(bvh_frame, joint_name, "Xposition"),
                bvh.frame_joint_channel(bvh_frame, joint_name, "Yposition"),
                bvh.frame_joint_channel(bvh_frame, joint_name, "Zposition"),
            )

        rot = glm.quat()
        if skeleton.has_rot(joint_name):
            x_rot = glm.angleAxis(
                glm.radians(
                    bvh.frame_joint_channel(bvh_frame, joint_name, "Xrotation")
                ),
                glm.vec3(1, 0, 0),
            )
            y_rot = glm.angleAxis(
                glm.radians(
                    bvh.frame_joint_channel(bvh_frame, joint_name, "Yrotation")
                ),
                glm.vec3(0, 1, 0),
            )
            z_rot = glm.angleAxis(
                glm.radians(
                    bvh.frame_joint_channel(bvh_frame, joint_name, "Zrotation")
                ),
                glm.vec3(0, 0, 1),
            )
            rot = z_rot * (y_rot * x_rot)

        m = glm.mat4(rot)
        m[3] = glm.vec4(pos, 1)
        np_m = np.zeros((4, 4))
        glm_to_np(m, np_m)

        parent_index = skeleton.get_parent_index(joint_name)
        if parent_index >= 0:
            xforms[frame, i] = xforms[frame, parent_index] @ m
        else:
            xforms[frame, i] = m


def build_xforms_from_bvh(bvh, skeleton, sample_rate):

    # sample the mocap data at the desired sample rate
    bvh_sample_rate = round(1 / bvh.frame_time)
    assert (
        bvh_sample_rate % sample_rate
    ) == 0  # mocap_sample_rate must be a multiple of DESIRED_SAMPLE_RATE
    sample_step = int(bvh_sample_rate / sample_rate)

    # Create an array of identity matrices with shape (num_frames, num_joints, 4, 4)
    xforms = np.eye(4, dtype=np.float32)[np.newaxis, np.newaxis, :, :] * np.ones(
        (bvh.nframes // sample_step, skeleton.num_joints, 1, 1)
    )

    frame = 0
    for bvh_frame in trange(0, bvh.nframes, sample_step):
        build_xforms_at_frame(xforms, bvh, skeleton, bvh_frame, frame)
        frame = frame + 1

    glm_xforms = []
    for frame in range(bvh.nframes // sample_step):
        joints = []
        for joint in range(skeleton.num_joints):
            m = glm.mat4(*np.ravel(xforms[frame, joint], order="F"))
            joints.append(m)
        glm_xforms.append(joints)

    return glm_xforms
