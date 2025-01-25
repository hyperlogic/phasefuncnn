import bvh
import copy
import math
import numpy as np
import skeleton
import math_util as mu
from tqdm import trange, tqdm


def build_xforms_at_frame(xforms, bvh, skeleton, bvh_frame, frame):
    m = np.eye(4)
    pi_180 = math.pi / 180

    for i in range(skeleton.num_joints):
        joint_name = skeleton.get_joint_name(i)
        o = skeleton.get_joint_offset(joint_name)
        if skeleton.has_pos(joint_name):
            posx = bvh.frame_joint_channel(bvh_frame, joint_name, "Xposition") + o[0]
            posy = bvh.frame_joint_channel(bvh_frame, joint_name, "Yposition") + o[1]
            posz = bvh.frame_joint_channel(bvh_frame, joint_name, "Zposition") + o[2]
        else:
            posx, posy, posz = o

        rotx, roty, rotz = 0, 0, 0
        if skeleton.has_rot(joint_name):
            rotx = bvh.frame_joint_channel(bvh_frame, joint_name, "Xrotation") * pi_180
            roty = bvh.frame_joint_channel(bvh_frame, joint_name, "Yrotation") * pi_180
            rotz = bvh.frame_joint_channel(bvh_frame, joint_name, "Zrotation") * pi_180

        mu.build_mat_from_euler(m, rotx, roty, rotz)
        m[0:3, 3] = [posx, posy, posz]

        parent_index = skeleton.get_parent_index(joint_name)
        if parent_index >= 0:
            xforms[frame, i] = xforms[frame, parent_index] @ m
        else:
            xforms[frame, i] = m


def build_xforms_from_bvh(bvh, skeleton, sample_rate):
    # sample the mocap data at the desired sample rate
    bvh_sample_rate = round(1 / bvh.frame_time)
    assert (bvh_sample_rate % sample_rate) == 0  # mocap_sample_rate must be a multiple of DESIRED_SAMPLE_RATE
    sample_step = int(bvh_sample_rate / sample_rate)

    # Create an array of identity matrices with shape (num_frames, num_joints, 4, 4)
    xforms = np.eye(4, dtype=np.float32)[np.newaxis, np.newaxis, :, :] * np.ones(
        (bvh.nframes // sample_step, skeleton.num_joints, 1, 1)
    )

    frame = 0
    for bvh_frame in range(0, bvh.nframes - 1, sample_step):
        build_xforms_at_frame(xforms, bvh, skeleton, bvh_frame, frame)
        frame = frame + 1

    return xforms


def mirror_xforms(skeleton, xforms):
    mxforms = np.zeros(xforms.shape)
    for i in range(xforms.shape[0]):
        for j in range(xforms.shape[1]):
            mxforms[i, skeleton.mirror_map[j]] = mu.mat_mirror(xforms[i, j])
    return mxforms
