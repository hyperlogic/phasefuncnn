#
# build root-space position, velocity and angles (pva) for each joint
#

import glm
import math
import mocap
import numpy as np
import os
import sys
from tqdm import trange, tqdm

OUTPUT_DIR = "output"


def build_jointpa_at_frame(skeleton, xforms, inv_root, frame, jointpva_array):
    num_joints = skeleton.num_joints
    for i in range(num_joints):
        local_xform = inv_root[frame] * xforms[frame][i]

        # position
        jointpva_array[frame, i, 0] = local_xform[3][0]
        jointpva_array[frame, i, 1] = local_xform[3][1]
        jointpva_array[frame, i, 2] = local_xform[3][2]

        # angle (rotation in expmap format)
        exp = mocap.logmap(glm.quat(local_xform))
        jointpva_array[frame][i][6] = exp[0]
        jointpva_array[frame][i][7] = exp[1]
        jointpva_array[frame][i][8] = exp[2]


def build_jointv_at_frame(skeleton, frame, jointpva_array):
    num_joints = skeleton.num_joints
    num_frames = skeleton.num_frames
    t = skeleton.frame_time * 2
    for i in range(num_joints):
        # velocties
        if frame > 0 and frame < num_frames - 1:
            jointpva_array[frame][i][3] = (
                jointpva_array[frame + 1][i][0] - jointpva_array[frame - 1][i][0]
            ) / t
            jointpva_array[frame][i][4] = (
                jointpva_array[frame + 1][i][1] - jointpva_array[frame - 1][i][1]
            ) / t
            jointpva_array[frame][i][5] = (
                jointpva_array[frame + 1][i][2] - jointpva_array[frame - 1][i][2]
            ) / t
        else:
            jointpva_array[frame][i][3] = 0
            jointpva_array[frame][i][4] = 0
            jointpva_array[frame][i][5] = 0


def build_jointpva(skeleton, xforms, root):
    num_joints = skeleton.num_joints
    num_frames = skeleton.num_frames
    inv_root = [glm.inverse(m) for m in root]
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
    skeleton = mocap.unpickle_obj(outbasepath + "_skeleton.pkl")
    xforms = mocap.unpickle_obj(outbasepath + "_xforms.pkl")
    root = mocap.unpickle_obj(outbasepath + "_root.pkl")

    # invert root xforms
    jointpva_array = build_jointpva(skeleton, xforms, root)

    # save jointpva_array
    np.save(outbasepath + "_jointpva.npy", jointpva_array)
