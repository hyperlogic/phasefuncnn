import glm
from tqdm import trange, tqdm


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
