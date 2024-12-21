#
# Build world space matrices for each joint at the given SAMPLE_RATE
# Also, build the root motion matrices for the character.
# And output the skeleton for the character (Skeleton)
#

import math
import os
import pickle
import sys
from typing import Any

import numpy as np
from bvh import Bvh

import bvh_util
import math_util as mu
from skeleton import Skeleton

OUTPUT_DIR = "output"
SAMPLE_RATE = 60


def pickle_obj(filename: str, obj: Any):
    with open(filename, "wb") as f:
        pickle.dump(obj, f)


def build_root_at_frame(skeleton: Skeleton, xforms: np.ndarray, root: np.ndarray, frame: int) -> np.ndarray:
    lhip_i = skeleton.get_joint_index("LeftUpLeg")
    rhip_i = skeleton.get_joint_index("RightUpLeg")
    lsho_i = skeleton.get_joint_index("LeftArm")
    rsho_i = skeleton.get_joint_index("RightArm")
    assert lhip_i != -1 and rhip_i != -1 and lsho_i != -1 and rsho_i != -1

    lhip = xforms[frame, lhip_i, 0:3, 3]
    rhip = xforms[frame, rhip_i, 0:3, 3]
    lsho = xforms[frame, lsho_i, 0:3, 3]
    rsho = xforms[frame, rsho_i, 0:3, 3]

    hip = lhip - rhip
    hip_len = np.linalg.norm(hip)
    sho = lsho - rsho
    sho_len = np.linalg.norm(sho)

    assert hip_len > 0 and sho_len > 0  # hip joints or shoulder joints are coincident!

    # compute root facing dir by averaging the vectors between the two hip joints and the two shoulder joints
    # then take that avg and cross it with the up vector.
    avg = (hip / hip_len + sho / sho_len) / 2
    up = np.array([0, 1, 0])
    facing = np.cross(avg, up)

    # compute root position by averging the two hip joints
    hip_center = (rhip + lhip) / 2

    # project both onto the ground plane
    root_pos = np.array([hip_center[0], 0, hip_center[2]])
    assert np.linalg.norm(np.array([facing[0], facing[2]])) > 0  # bad facing dir

    # the negative is because cross(x, z) is the negative y axis and we want to rot around the y axis
    root_theta = -math.atan2(facing[2], facing[0])

    # build matrix
    mu.build_mat_roty(root[frame], root_theta)
    root[frame, 0:3, 3] = root_pos

    return root


def build_root_motion(skeleton: Skeleton, xforms: np.ndarray) -> np.ndarray:
    num_frames = xforms.shape[0]

    # create array of 4x4 identity matrices, shape = (num_frames, 4, 4)
    root = np.eye(4, dtype=np.float32)[np.newaxis, :, :] * np.ones((num_frames, 1, 1))

    for frame in range(num_frames):
        build_root_at_frame(skeleton, xforms, root, frame)

    # AJT: TODO: filter rotations
    return root


if __name__ == "__main__":
    if len(sys.argv) < 2 or len(sys.argv) > 3:
        print("Error: expected bvh file argument, or -m flag")
        exit(1)

    if sys.argv[1] == "-m":
        mirror = True
        mocap_filename = sys.argv[2]
    else:
        mirror = False
        mocap_filename = sys.argv[1]

    mocap_basename = os.path.splitext(os.path.basename(mocap_filename))[0]
    outbasepath = os.path.join(OUTPUT_DIR, mocap_basename)

    print(f"Loading {mocap_filename}")
    with open(mocap_filename) as f:
        bvh = Bvh(f.read())

    skeleton = Skeleton(bvh)
    print(skeleton.joint_names)

    xforms = bvh_util.build_xforms_from_bvh(bvh, skeleton, SAMPLE_RATE)

    if mirror:
        xforms = bvh_util.mirror_xforms(skeleton, xforms)

    root = build_root_motion(skeleton, xforms)

    # create output dir
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # pickle skeleton, xforms
    pickle_obj(outbasepath + "_skeleton.pkl", skeleton)
    np.save(outbasepath + "_xforms.npy", xforms)
    np.save(outbasepath + "_root.npy", root)
