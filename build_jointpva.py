#
# Build root-space position, velocity and angles (pva) for each joint
# They are packed into a np.ndarray with shape (num_frames, num_joints, 9)
#   Indices [0:3] are position x, y, z
#   Indices [3:6] are velocity x, y, z
#   Indices [6:9] are the joint angles in R^3 form (can use mocap.expmap to convert into a quaternion)

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
        local_xform = inv_root[frame] @ xforms[frame][i]

        # position
        jointpva_array[frame, i, 0:3] = local_xform[0:3, 3]

        # angle (rotation in expmap format)
        jointpva_array[frame, i, 6:9] = mu.logmap(mu.quat_from_mat(local_xform))


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
    jointpva_array = np.zeros((num_frames, num_joints, 9))
    for frame in range(num_frames):
        build_jointpa_at_frame(skeleton, xforms, inv_root, frame, jointpva_array)
    for frame in range(num_frames):
        build_jointv_at_frame(skeleton, frame, jointpva_array)
    return jointpva_array


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Error: expected mocap filename (without .bvh extension)")
        exit(1)

    mocap_basename = sys.argv[1]
    outbasepath = os.path.join(OUTPUT_DIR, mocap_basename)

    # unpickle skeleton, xforms
    skeleton = unpickle_obj(outbasepath + "_skeleton.pkl")
    xforms = np.load(outbasepath + "_xforms.npy")
    root = np.load(outbasepath + "_root.npy")

    # build joint pos, vel, and angle
    jointpva_array = build_jointpva(skeleton, xforms, root)

    # save jointpva_array
    np.save(outbasepath + "_jointpva.npy", jointpva_array)
