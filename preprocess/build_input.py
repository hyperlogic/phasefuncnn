import glm
import math
import numpy as np
from tqdm import trange, tqdm

TRAJ_WINDOW = 12


def axis_angle_to_quat(axis, angle):
    n = glm.normalize(axis) * math.sin(angle / 2)
    w = math.cos(angle / 2)
    return glm.quat(w, n.x, n.y, n.z)


def quat_to_mat4(quat):
    x_axis = quat * glm.vec3(1, 0, 0)
    y_axis = quat * glm.vec3(0, 1, 0)
    z_axis = quat * glm.vec3(0, 0, 1)
    return glm.mat4(
        glm.vec4(x_axis, 0),
        glm.vec4(y_axis, 0),
        glm.vec4(z_axis, 0),
        glm.vec4(0, 0, 0, 1),
    )


class Input:
    def __init__(self, traj_window, num_joints):
        self.t_pos = np.zeros(shape=(traj_window, 2), dtype=float)
        self.t_dir = np.zeros(shape=(traj_window, 2), dtype=float)
        self.j_pos = np.zeros(shape=(num_joints, 3), dtype=float)
        self.j_vel = np.zeros(shape=(num_joints, 3), dtype=float)


class Skeleton:
    def __init__(self, bvh):
        self.joint_names = bvh.get_joints_names()
        self.root_name = self.joint_names[0]
        self.joint_index_map = {
            self.joint_names[i]: i for i in range(len(self.joint_names))
        }
        self.has_pos_map = {
            j: "Xposition" in bvh.joint_channels(j) for j in self.joint_names
        }
        self.has_rot_map = {
            j: "Xrotation" in bvh.joint_channels(j) for j in self.joint_names
        }
        self.parent_map = {j: bvh.joint_parent_index(j) for j in self.joint_names}
        self.joint_offset_map = {j: bvh.joint_offset(j) for j in self.joint_names}
        self.num_joints = len(self.joint_names)

    def is_root(self, joint_name):
        return joint_name == self.root_name

    def get_joint_name(self, joint_index):
        return self.joint_names[joint_index]

    def get_joint_index(self, joint_name):
        return self.joint_index_map[joint_name]

    def get_parent_index(self, joint_name):
        return self.parent_map[joint_name]

    def has_pos(self, joint_name):
        return self.has_pos_map[joint_name]

    def has_rot(self, joint_name):
        return self.has_rot_map[joint_name]

    def get_joint_offset(self, joint_name):
        return self.joint_offset_map[joint_name]


def build_input(bvh):
    num_frames = bvh.nframes
    frame_time = bvh.frame_time

    s = Skeleton(bvh)

    debug = False

    print(s.joint_names)

    inputs = []

    for frame in trange(num_frames):

        xforms = [glm.mat4() for i in range(s.num_joints)]

        root_trans = glm.vec3(0, 0, 0)

        # build xforms
        for i in range(s.num_joints):
            joint_name = s.get_joint_name(i)
            offset = s.get_joint_offset(joint_name)

            if s.has_pos(joint_name):
                x_pos = (
                    bvh.frame_joint_channel(frame, joint_name, "Xposition") + offset[0]
                )
                y_pos = (
                    bvh.frame_joint_channel(frame, joint_name, "Yposition") + offset[1]
                )
                z_pos = (
                    bvh.frame_joint_channel(frame, joint_name, "Zposition") + offset[2]
                )
            else:
                x_pos, y_pos, z_pos = offset[0], offset[1], offset[2]

            if s.has_rot(joint_name):
                x_rot = glm.radians(
                    bvh.frame_joint_channel(frame, joint_name, "Xrotation")
                )
                y_rot = glm.radians(
                    bvh.frame_joint_channel(frame, joint_name, "Yrotation")
                )
                z_rot = glm.radians(
                    bvh.frame_joint_channel(frame, joint_name, "Zrotation")
                )
            else:
                x_rot, y_rot, z_rot = 0, 0, 0

            m = quat_to_mat4(
                axis_angle_to_quat(glm.vec3(0, 0, 1), z_rot)
                * axis_angle_to_quat(glm.vec3(1, 0, 0), x_rot)
                * axis_angle_to_quat(glm.vec3(0, 1, 0), y_rot)
            )
            m[3] = glm.vec4(x_pos, y_pos, z_pos, 1)
            parent_index = s.get_parent_index(joint_name)
            if parent_index >= 0:
                xforms[i] = xforms[parent_index] * m
            else:
                xforms[i] = m

            if debug:
                print(f"{joint_name} {i} =")
                print(f"    local_pos = {glm.vec3(x_pos, y_pos, z_pos)}")
                print(f"    local_euler = {glm.vec3(x_rot, y_rot, z_rot)}")
                print(f"    global_xform = {xforms[i]}")

        # build j_pos
        input = Input(TRAJ_WINDOW, s.num_joints)
        for i in range(len(s.joint_names)):
            input.j_pos[i] = glm.vec3(xforms[i][3][0], xforms[i][3][1], xforms[i][3][2])

        inputs.append(input)

    if debug:
        print(inputs)

    return s, inputs
