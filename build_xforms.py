#
# Build world space transform matrices for each joint and root motion, from a bvh file
#

from bvh import Bvh
import glm
import math
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


def build_root_at_frame(skeleton, xforms, frame):
    lhip_i = skeleton.get_joint_index("LeftUpLeg")
    rhip_i = skeleton.get_joint_index("RightUpLeg")
    lsho_i = skeleton.get_joint_index("LeftArm")
    rsho_i = skeleton.get_joint_index("RightArm")
    assert lhip_i != -1 and rhip_i != -1 and lsho_i != -1 and rsho_i != -1

    lhip = glm.vec3(xforms[frame][lhip_i][3])
    rhip = glm.vec3(xforms[frame][rhip_i][3])
    lsho = glm.vec3(xforms[frame][lsho_i][3])
    rsho = glm.vec3(xforms[frame][rsho_i][3])

    hip = lhip - rhip
    hip_len = glm.length(hip)
    sho = lsho - rsho
    sho_len = glm.length(sho)

    assert hip_len > 0 and sho_len > 0  # hip joints or shoulder joints are coincident!

    # compute root facing dir by averaging the vectors between the two hip joints and the two shoulder joints
    # then take that avg and cross it with the up vector.
    avg = (hip / hip_len + sho / sho_len) / 2
    up = glm.vec3(0, 1, 0)
    facing = glm.cross(avg, up)

    # compute root position by averging the two hip joints
    hip_center = (rhip + lhip) / 2

    # project both onto the ground plane
    root_pos = glm.vec3(hip_center.x, 0, hip_center.z)
    assert glm.length(glm.vec2(facing.x, facing.z)) > 0  # bad facing dir

    # the negative is because cross(x, z) is the negative y axis and we want to rot around the y axis
    root_theta = -math.atan2(facing.z, facing.x)

    # build matrix
    root = glm.mat4(glm.angleAxis(root_theta, glm.vec3(0, 1, 0)))
    root[3] = glm.vec4(root_pos, 1)

    return root


def build_root_motion(skeleton, xforms):
    root_list = []
    for frame in range(skeleton.num_frames):
        root = build_root_at_frame(skeleton, xforms, frame)
        root_list.append(root)

    # AJT: TODO: filter rotations
    return root_list


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

    xforms = build_xforms(bvh, skeleton)
    root_list = build_root_motion(skeleton, xforms)

    # create output dir
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # pickle skeleton, xforms
    mocap.pickle_obj(outbasepath + "_skeleton.pkl", skeleton)
    mocap.pickle_obj(outbasepath + "_xforms.pkl", xforms)
    mocap.pickle_obj(outbasepath + "_root.pkl", root_list)
