#
# Build phase and foot contact for character
#

import math
import math_util as mu
from skeleton import Skeleton
import numpy as np
import os
import pickle
import sys


def unpickle_obj(filename):
    with open(filename, "rb") as f:
        return pickle.load(f)


OUTPUT_DIR = "output"
SAMPLE_RATE = 60
VEL_THRESH = 15


def build_contacts(skeleton, xforms):
    num_frames = xforms.shape[0]
    contacts = np.zeros((num_frames, 4))
    t = (1 / SAMPLE_RATE) * 2

    feet = [
        skeleton.get_joint_index("LeftFoot"),
        skeleton.get_joint_index("RightFoot"),
        skeleton.get_joint_index("LeftToeBase"),
        skeleton.get_joint_index("RightToeBase"),
    ]
    for frame in range(num_frames):
        for i, joint in enumerate(feet):
            # TODO: should also check angular vel it's probalby more accruate...
            if frame > 0 and frame < num_frames - 1:
                vel = xforms[frame + 1, joint, 0:3, 3] - xforms[frame - 1, joint, 0:3, 3]
                vel = vel / t
            else:
                vel = np.array([0, 0, 0])
            speed = np.linalg.norm(vel)
            contacts[frame, i] = 1 if speed < VEL_THRESH else 0

    return contacts


# 0 = no feet down, 1 = right foot down, 2 = left foot down, 3 = both feet down
def make_state(contact):
    state = 0
    # right foot
    if contact[1] > 0.5 or contact[3] > 0.5:
        state += 1
    # left foot
    if contact[0] > 0.5 or contact[2] > 0.5:
        state += 2
    return state


state_phase = [
    [None, 0, math.pi, 0],
    [None, None, math.pi, math.pi],
    [None, 0, None, 0],
    [None, 0, math.pi, None],
]


def build_phase(contacts):
    num_frames = len(contacts)

    keyframes = []

    curr_state = make_state(contacts[0])
    keyframes.append([0, 0, curr_state])
    for frame in range(1, num_frames):
        next_state = make_state(contacts[frame])
        if curr_state != next_state:
            phase = state_phase[curr_state][next_state]
            if phase != None:
                keyframes.append([frame, phase, next_state])
            curr_state = next_state

    phase = np.zeros(num_frames, dtype=np.float32)

    MIN_STANDING_PERIOD = 0.25 * SAMPLE_RATE

    for i in range(len(keyframes) - 1):
        start_frame, start_phase, start_state = keyframes[i]
        end_frame, end_phase, _ = keyframes[i + 1]

        interval_frames = end_frame - start_frame
        if start_state == 3:  # both feet down
            extra_cycle = 0.5 if start_phase != end_phase else 0
            cycle_count = (interval_frames // MIN_STANDING_PERIOD) + extra_cycle
        else:
            cycle_count = 0.5 if start_phase != end_phase else 1
        d_phase = (2 * math.pi * cycle_count) / interval_frames
        for j in range(interval_frames):
            phase[start_frame + j] = (j * d_phase + start_phase) % (2 * math.pi)
    end_phase = keyframes[-1][1]
    phase[-1] = end_phase

    return phase


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Error: expected mocap filename (without .bvh extension)")
        exit(1)

    mocap_basename = sys.argv[1]
    outbasepath = os.path.join(OUTPUT_DIR, mocap_basename)

    # unpickle skeleton and xforms
    skeleton = unpickle_obj(outbasepath + "_skeleton.pkl")
    xforms = np.load(outbasepath + "_xforms.npy")

    contacts = build_contacts(skeleton, xforms)
    phase = build_phase(contacts)

    # save contacts, phase
    np.save(outbasepath + "_contacts.npy", contacts)
    np.save(outbasepath + "_phase.npy", phase)
