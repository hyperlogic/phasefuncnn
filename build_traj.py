#
# Build trajectory for root character motion
# Each frame contains TRAJ_WINDOW_SIZE * TRAJ_ELEMENT_SIZE floats
# Each element contains pos (x, z) and direction (x, z)
# Also output the forward and rightward velocity and the angular velocity of the root motion.
#

import cmath
import os
import sys

import numpy as np

import math_util as mu

OUTPUT_DIR = "output"
SAMPLE_RATE = 60
TRAJ_SAMPLE_RATE = 6
TRAJ_STEP = SAMPLE_RATE / TRAJ_SAMPLE_RATE
TRAJ_WINDOW_SIZE = 12
TRAJ_ELEMENT_SIZE = 4  # 4 for px, pz, dx, dz


def build_traj_at_frame(root: np.ndarray, frame: int, traj_array: np.ndarray):
    num_frames = len(root)
    half_window = TRAJ_WINDOW_SIZE / 2
    start = int(frame - half_window * TRAJ_STEP)
    end = int(frame + half_window * TRAJ_STEP)
    step = int(TRAJ_STEP)
    base_inv = np.linalg.inv(root[frame])
    i = 0
    for f in range(start, end, step):
        f = np.clip(f, 0, num_frames - 1)
        xform = base_inv @ root[f]
        o = i * TRAJ_ELEMENT_SIZE
        pos = xform[0:3, 3]
        traj_array[frame, o : o + 2] = [pos[0], pos[2]]
        dir = xform[:3, :3] @ [1, 0, 0]
        traj_array[frame, o + 2 : o + 4] = [dir[0], dir[2]]
        i = i + 1


def build_traj(root: np.ndarray) -> np.ndarray:
    num_frames = root.shape[0]
    traj_array = np.zeros((num_frames, TRAJ_WINDOW_SIZE * TRAJ_ELEMENT_SIZE))
    for frame in range(num_frames):
        build_traj_at_frame(root, frame, traj_array)
    return traj_array


def build_rootvel(root: np.ndarray) -> np.ndarray:
    num_frames = root.shape[0]
    t = (1 / SAMPLE_RATE) * 2
    rootvel = np.zeros((num_frames, 3))
    for frame in range(num_frames):
        if frame > 0 and frame < num_frames - 1:
            # calculate vel in world frame.
            dist = root[frame + 1, 0:3, 3] - root[frame - 1, 0:3, 3]
            world_vel = dist / t

            # rotate world_vel back into root frame
            inv_root_rot = mu.quat_inverse(mu.quat_from_mat(root[frame]))
            local_vel = mu.quat_rotate(inv_root_rot, world_vel)

            rootvel[frame, 0:2] = [local_vel[0], local_vel[2]]

            # calculate angular vel
            v0 = root[frame - 1, :3, :3] @ [1, 0, 0]
            v1 = root[frame + 1, :3, :3] @ [1, 0, 0]

            c0 = complex(v0[0], v0[2])
            c1 = complex(v1[0], v1[2])
            delta_c = c1 * c0.conjugate()
            # the negative is because cross(x, z) is the negative y axis and we want to rot around the y axis
            angvel = -cmath.phase(delta_c) / t
            rootvel[frame, 2] = angvel
        else:
            rootvel[frame, 0:3] = [0, 0, 0]

    return rootvel


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Error: expected mocap filename (without .bvh extension)")
        exit(1)

    mocap_basename = sys.argv[1]
    outbasepath = os.path.join(OUTPUT_DIR, mocap_basename)

    # load root xforms
    root = np.load(outbasepath + "_root.npy")

    traj_array = build_traj(root)
    rootvel = build_rootvel(root)

    # save traj
    np.save(outbasepath + "_traj.npy", traj_array)

    # save rootvel
    np.save(outbasepath + "_rootvel.npy", rootvel)
