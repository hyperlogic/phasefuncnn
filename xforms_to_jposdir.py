import glm
import mocap
import os
import sys
from tqdm import trange, tqdm

OUTPUT_DIR = "output"


def build_jpos_and_jdir_at_frame(skeleton, xforms, frame):

    jpos = [glm.vec3(0, 0, 0) for i in range(skeleton.num_joints)]
    jdir = [glm.vec3(1, 0, 0) for i in range(skeleton.num_joints)]

    # AJT TODO
    """
    if frame > 0 and frame < skeleton.num_frames - 1:
        for i in range(skeleton.num_joints):
            dist = glm.vec3(xforms[frame - 1][i][3]) - glm.vec3(xforms[frame + 1][i][3])
            v[i] = dist / t
    """

    return jpos, jdir


def build_jposdir(skeleton, xforms):
    num_frames = skeleton.num_frames
    frame_time = skeleton.frame_time

    jpos_list = []
    jdir_list = []
    for frame in trange(num_frames):
        jpos, jdir = build_jpos_and_jdir_at_frame(skeleton, xforms, frame)
        jpos_list.append(jpos_list)
        jdir_list.append(jdir_list)

    return jpos_list, jdir_list


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Error: expected mocap filename (without .bvh extension)")
        exit(1)

    mocap_basename = sys.argv[1]

    # unpickle
    skeleton = mocap.unpickle_obj(
        os.path.join(OUTPUT_DIR, mocap_basename + "_skeleton.pkl")
    )
    xforms = mocap.unpickle_obj(
        os.path.join(OUTPUT_DIR, mocap_basename + "_xforms.pkl")
    )

    jpos, jdir = build_jposdir(skeleton, xforms)

    # pickle jpos & jdir
    mocap.pickle_obj(os.path.join(OUTPUT_DIR, mocap_basename + "_jpos.pkl"), jpos)
    mocap.pickle_obj(os.path.join(OUTPUT_DIR, mocap_basename + "_jdir.pkl"), jdir)
