import glm
import math
import numpy as np
from tqdm import trange, tqdm


def build_xforms_at_frame(bvh, skeleton, frame, debug=False):

    xforms = [glm.mat4() for i in range(skeleton.num_joints)]

    if debug:
        print(f"frame = {frame}")

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

        if debug:
            print(f"    {joint_name}")
            if skeleton.has_rot(joint_name):
                x_rot = bvh.frame_joint_channel(frame, joint_name, "Xrotation")
                y_rot = bvh.frame_joint_channel(frame, joint_name, "Yrotation")
                z_rot = bvh.frame_joint_channel(frame, joint_name, "Zrotation")
                print(f"        (local) euler (deg) = {(x_rot, y_rot, z_rot)}")
                print(
                    f"        (local) euler (rad) = {(glm.radians(x_rot), glm.radians(y_rot), glm.radians(z_rot))}"
                )
            print(f"        (local) pos = {pos}, rot = {rot}")

        m = glm.mat4(rot)
        m[3] = glm.vec4(pos, 1)
        parent_index = skeleton.get_parent_index(joint_name)
        if parent_index >= 0:
            xforms[i] = xforms[parent_index] * m
        else:
            xforms[i] = m

        if debug:
            global_pos = glm.vec3(xforms[i][3][0], xforms[i][3][1], xforms[i][3][2])
            global_rot = glm.normalize(glm.quat(xforms[i]))
            print(f"        (global) pos = {global_pos}, rot = {global_rot}")

    return xforms


def build_xforms(bvh, skeleton):

    num_frames = bvh.nframes
    frame_time = bvh.frame_time

    xforms = []
    for frame in trange(num_frames):
        xforms.append(build_xforms_at_frame(bvh, skeleton, frame))

    return xforms
