#
# Build root-space position, velocity and angles (pva) for each joint
# They are packed into a np.ndarray with shape (num_frames, num_joints, 9)
#   Indices [0:3] are position x, y, z
#   Indices [3:6] are velocity x, y, z
#   Indices [6:12] are the first two columns of 3x3 matrix.

import math_util as mu
import numpy as np
from skeleton import Skeleton
import os
import sys
import pickle

OUTPUT_DIR = "output"
SAMPLE_RATE = 60


def unpickle_obj(filename: str):
    with open(filename, "rb") as f:
        return pickle.load(f)


def build_jointpa_at_frame(skeleton: Skeleton, xforms: np.ndarray, inv_root: np.ndarray, frame: int, jointpva_array: np.ndarray):
    num_joints = skeleton.num_joints
    for i in range(num_joints):
        root_relative_xform = inv_root[frame] @ xforms[frame][i]

        # position
        jointpva_array[frame, i, 0:3] = root_relative_xform[0:3, 3]

        # angle (rotation in expmap format)
        x_axis = root_relative_xform[0:3, 0]
        y_axis = root_relative_xform[0:3, 1]
        jointpva_array[frame, i, 6:9] = x_axis / np.linalg.norm(x_axis)
        jointpva_array[frame, i, 9:12] = y_axis / np.linalg.norm(y_axis)


def build_jointv_at_frame(skeleton: Skeleton, frame: int, jointpva_array: np.ndarray):
    num_joints = skeleton.num_joints
    num_frames = len(jointpva_array)
    t = (1 / SAMPLE_RATE) * 2
    for i in range(num_joints):
        # velocties
        if frame > 0 and frame < num_frames - 1:
            dist = jointpva_array[frame + 1, i, 0:3] - jointpva_array[frame - 1, i, 0:3]
            jointpva_array[frame, i, 3:6] = dist / t
        else:
            jointpva_array[frame, i, 3:6] = [0, 0, 0]


def build_jointpva(skeleton: Skeleton, xforms: np.ndarray, root: np.ndarray) -> np.ndarray:
    num_joints = skeleton.num_joints
    num_frames = len(xforms)
    inv_root = np.linalg.inv(root)
    jointpva_array = np.zeros((num_frames, num_joints, 12))
    for frame in range(num_frames):
        build_jointpa_at_frame(skeleton, xforms, inv_root, frame, jointpva_array)
    for frame in range(num_frames):
        build_jointv_at_frame(skeleton, frame, jointpva_array)
    return jointpva_array


if __name__ == "__main__":
    """
    if len(sys.argv) != 2:
        print("Error: expected mocap filename (without .bvh extension)")
        exit(1)

    mocap_basename = sys.argv[1]
    outbasepath = os.path.join(OUTPUT_DIR, mocap_basename)
    """

    # unpickle skeleton, xforms
    skeleton = unpickle_obj(snakemake.input.skeleton)
    xforms = np.load(snakemake.input.xforms)
    root = np.load(snakemake.input.root)

    # build joint pos, vel, and angle
    jointpva_array = build_jointpva(skeleton, xforms, root)

    # save jointpva_array
    np.save(snakemake.output.jointpva, jointpva_array)
