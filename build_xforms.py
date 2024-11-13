#
# Build world space matrices (glm.mat4) for each joint at the given SAMPLE_RATE
# Also, build the root motion matrices  (glm.mat4) for the character.
# And output the skeleton for the character (mocap.Skeleton)
#

from bvh import Bvh
import glm
import math
import mocap
import numpy as np
import os
import sys
from tqdm import trange, tqdm


OUTPUT_DIR = "output"
SAMPLE_RATE = 60


def build_root_at_frame(skeleton, xforms, root, frame):
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
    mocap.build_mat_roty(root[frame], root_theta)
    root[frame, 0:3, 3] = root_pos

    return root


def build_root_motion(skeleton, xforms):
    num_frames = xforms.shape[0]

    # create array of 4x4 identity matrices, shape = (num_frames, 4, 4)
    root = np.eye(4, dtype=np.float32)[np.newaxis, :, :] * np.ones((num_frames, 1, 1))

    for frame in range(num_frames):
        build_root_at_frame(skeleton, xforms, root, frame)

    # AJT: TODO: filter rotations
    return root


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Error: expected bvh file argument")
        exit(1)

    mocap_filename = sys.argv[1]
    mocap_basename = os.path.splitext(os.path.basename(mocap_filename))[0]
    outbasepath = os.path.join(OUTPUT_DIR, mocap_basename)

    print(f"Loading {mocap_filename}")
    with open(mocap_filename) as f:
        bvh = Bvh(f.read())

    skeleton = mocap.Skeleton(bvh)
    print(skeleton.joint_names)

    xforms = mocap.build_xforms_from_bvh(bvh, skeleton, SAMPLE_RATE)
    glm_xforms = mocap.xforms_numpy_to_glm(xforms) # AJT: TODO REMOVE
    root = build_root_motion(skeleton, xforms)
    glm_root = mocap.root_numpy_to_glm(root)

    # create output dir
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # pickle skeleton, xforms
    mocap.pickle_obj(outbasepath + "_skeleton.pkl", skeleton)
    mocap.pickle_obj(outbasepath + "_xforms.pkl", glm_xforms)    # AJT: TODO REMOVE
    mocap.pickle_obj(outbasepath + "_root.pkl", glm_root)  # AJT: TODO REMOVER
    np.save(outbasepath + "_xforms.npy", xforms)
    np.save(outbasepath + "_root.npy", root)
