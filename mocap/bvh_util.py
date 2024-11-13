import bvh
import copy
import glm
import math
import numpy as np
from .skeleton import Skeleton
from tqdm import trange, tqdm


# alpha - rotation about x axis
# beta - rotaiton about y axis
# gamma - rotaiton about z axis
# mat = rotz @ roty @ rotx
def build_mat_from_euler_pos(mat, alpha, beta, gamma, x, y, z):
    cosa, sina = math.cos(alpha), math.sin(alpha)
    cosb, sinb = math.cos(beta), math.sin(beta)
    cosg, sing = math.cos(gamma), math.sin(gamma)

    mat[0] = [
        cosb * cosg,
        cosg * sina * sinb - cosa * sing,
        cosa * cosg * sinb + sina * sing,
        x,
    ]
    mat[1] = [
        cosb * sing,
        cosa * cosg + sina * sinb * sing,
        -cosg * sina + cosa * sinb * sing,
        y,
    ]
    mat[2] = [-sinb, cosb * sina, cosa * cosb, z]
    mat[3] = [0, 0, 0, 1]


def build_mat_rotx(mat, alpha):
    cosa, sina = math.cos(alpha), math.sin(alpha)
    mat[0] = [1, 0, 0, 0]
    mat[1] = [0, cosa, -sina, 0]
    mat[2] = [0, sina, cosa, 0]
    mat[3] = [0, 0, 0, 1]


def build_mat_roty(mat, beta):
    cosb, sinb = math.cos(beta), math.sin(beta)
    mat[0] = [cosb, 0, sinb, 0]
    mat[1] = [0, 1, 0, 0]
    mat[2] = [-sinb, 0, cosb, 0]
    mat[3] = [0, 0, 0, 1]


def build_mat_rotz(mat, gamma):
    cosg, sing = math.cos(gamma), math.sin(gamma)
    mat[0] = [cosg, -sing, 0, 0]
    mat[1] = [sing, cosg, 0, 0]
    mat[2] = [0, 0, 1, 0]
    mat[3] = [0, 0, 0, 1]


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

        build_mat_from_euler_pos(m, rotx, roty, rotz, posx, posy, posz)

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

    return xforms


def xforms_numpy_to_glm(nparray):
    ii, jj, _, _ = nparray.shape
    return [
        [glm.mat4(*np.ravel(nparray[i, j], order="F")) for j in range(jj)]
        for i in range(ii)
    ]

def root_numpy_to_glm(nparray):
    return [glm.mat4(*np.ravel(nparray[i], order="F")) for i in range(nparray.shape[0])]
