import glm
import mocap
import os
import sys
from tqdm import trange, tqdm

OUTPUT_DIR = "output"


def build_vels_at_frame(skeleton, xforms, frame):

    v = [glm.vec3(0, 0, 0) for i in range(skeleton.num_joints)]

    t = skeleton.frame_time * 2

    if frame > 0 and frame < skeleton.num_frames - 1:
        for i in range(skeleton.num_joints):
            dist = glm.vec3(xforms[frame - 1][i][3]) - glm.vec3(xforms[frame + 1][i][3])
            v[i] = dist / t

    return v


def build_vels(skeleton, xforms):
    num_frames = skeleton.num_frames
    frame_time = skeleton.frame_time

    vels = []
    for frame in trange(num_frames):
        vels.append(build_vels_at_frame(skeleton, xforms, frame))

    return vels


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Error: expected mocap filename (without .bvh extension)")
        exit(1)

    mocap_basename = sys.argv[1]

    # unpickle skeleton, xforms
    skeleton = mocap.unpickle_obj(
        os.path.join(OUTPUT_DIR, mocap_basename + "_skeleton.pkl")
    )
    xforms = mocap.unpickle_obj(
        os.path.join(OUTPUT_DIR, mocap_basename + "_xforms.pkl")
    )

    vels = build_vels(skeleton, xforms)

    # pickle vels
    mocap.pickle_obj(os.path.join(OUTPUT_DIR, mocap_basename + "_vels.pkl"), vels)
