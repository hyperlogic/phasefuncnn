#
# Build phase and foot contact for character
#

import glm
import mocap
import numpy as np
import os
import sys


OUTPUT_DIR = "output"
SAMPLE_RATE = 60
VEL_THRESH = 20


def build_contacts(skeleton, xforms):
    num_frames = len(xforms)
    contacts = np.zeros((num_frames, 4))
    t = (1 / SAMPLE_RATE) * 2

    feet = [skeleton.get_joint_index("LeftFoot"), skeleton.get_joint_index("RightFoot"), skeleton.get_joint_index("LeftToeBase"), skeleton.get_joint_index("RightToeBase")]
    for frame in range(num_frames):
        for i, joint in enumerate(feet):
            # TODO: should also check angular vel it's probalby more accruate...
            if frame > 0 and frame < num_frames - 1:
                vel = glm.vec3(xforms[frame + 1][joint][3]) - glm.vec3(xforms[frame - 1][joint][3])
                vel = vel / t
            else:
                vel = glm.vec3(0, 0, 0)
            speed = glm.length(vel)
            contacts[frame][i] = 1 if speed < VEL_THRESH else 0

    return contacts


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Error: expected mocap filename (without .bvh extension)")
        exit(1)

    mocap_basename = sys.argv[1]
    outbasepath = os.path.join(OUTPUT_DIR, mocap_basename)

    # unpickle skeleton and xforms
    skeleton = mocap.unpickle_obj(outbasepath + "_skeleton.pkl")
    xforms = mocap.unpickle_obj(outbasepath + "_xforms.pkl")

    contacts = build_contacts(skeleton, xforms)

    # save contacts
    np.save(outbasepath + "_contacts.npy", contacts)
