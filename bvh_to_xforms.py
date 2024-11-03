from bvh import Bvh
import glm
import mocap
import os
import sys
from tqdm import trange, tqdm

OUTPUT_DIR = "output"


def build_xforms_at_frame(bvh, skeleton, frame):

    xforms = [glm.mat4() for i in range(skeleton.num_joints)]

    for i in range(skeleton.num_joints):
        joint_name = skeleton.get_joint_name(i)
        offset = skeleton.get_joint_offset(joint_name)

        pos = glm.vec3(offset[0], offset[1], offset[2])
        if skeleton.has_pos(joint_name):
            pos += glm.vec3(
                bvh.frame_joint_channel(frame, joint_name, "Xposition"),
                bvh.frame_joint_channel(frame, joint_name, "Yposition"),
                bvh.frame_joint_channel(frame, joint_name, "Zposition"),
            )

        rot = glm.quat()
        if skeleton.has_rot(joint_name):
            x_rot = glm.angleAxis(
                glm.radians(bvh.frame_joint_channel(frame, joint_name, "Xrotation")),
                glm.vec3(1, 0, 0),
            )
            y_rot = glm.angleAxis(
                glm.radians(bvh.frame_joint_channel(frame, joint_name, "Yrotation")),
                glm.vec3(0, 1, 0),
            )
            z_rot = glm.angleAxis(
                glm.radians(bvh.frame_joint_channel(frame, joint_name, "Zrotation")),
                glm.vec3(0, 0, 1),
            )
            rot = z_rot * (y_rot * x_rot)

        m = glm.mat4(rot)
        m[3] = glm.vec4(pos, 1)
        parent_index = skeleton.get_parent_index(joint_name)
        if parent_index >= 0:
            xforms[i] = xforms[parent_index] * m
        else:
            xforms[i] = m

    return xforms


def build_xforms(bvh, skeleton):

    num_frames = bvh.nframes
    frame_time = bvh.frame_time

    xforms = []
    for frame in trange(num_frames):
        xforms.append(build_xforms_at_frame(bvh, skeleton, frame))

    return xforms


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Error: expected bvh file argument")
        exit(1)

    mocap_filename = sys.argv[1]
    mocap_basename = os.path.splitext(os.path.basename(mocap_filename))[0]

    print(f"Loading {mocap_filename}")
    with open(mocap_filename) as f:
        bvh = Bvh(f.read())

    skeleton = mocap.Skeleton(bvh)
    print(skeleton.joint_names)

    xforms = build_xforms(bvh, skeleton)

    # create output dir
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # pickle skeleton, xforms
    mocap.pickle_obj(
        os.path.join(OUTPUT_DIR, mocap_basename + "_skeleton.pkl"), skeleton
    )
    mocap.pickle_obj(os.path.join(OUTPUT_DIR, mocap_basename + "_xforms.pkl"), xforms)
