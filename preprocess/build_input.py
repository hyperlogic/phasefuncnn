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
            x_rot = axis_angle_to_quat(
                glm.vec3(1, 0, 0),
                glm.radians(bvh.frame_joint_channel(frame, joint_name, "Xrotation")),
            )
            y_rot = axis_angle_to_quat(
                glm.vec3(0, 1, 0),
                glm.radians(bvh.frame_joint_channel(frame, joint_name, "Yrotation")),
            )
            z_rot = axis_angle_to_quat(
                glm.vec3(0, 0, 1),
                glm.radians(bvh.frame_joint_channel(frame, joint_name, "Zrotation")),
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

        m = quat_to_mat4(rot)
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


def build_input(bvh):
    num_frames = bvh.nframes
    frame_time = bvh.frame_time

    skeleton = Skeleton(bvh)

    pop_check = False

    print(skeleton.joint_names)

    inputs = []

    # first pass build j_pos
    for frame in trange(num_frames):
        xforms = build_xforms_at_frame(bvh, skeleton, frame)

        # build j_pos
        input = Input(TRAJ_WINDOW, skeleton.num_joints)
        for i in range(skeleton.num_joints):
            input.j_pos[i] = glm.vec3(xforms[i][3][0], xforms[i][3][1], xforms[i][3][2])

        inputs.append(input)

        if pop_check and frame > 0:
            POP_THRESH = 1
            for i in range(skeleton.num_joints):
                prev_pos = inputs[frame - 1].j_pos[i]
                curr_pos = inputs[frame].j_pos[i]
                prev = glm.vec3(prev_pos[0], prev_pos[1], prev_pos[2])
                curr = glm.vec3(curr_pos[0], curr_pos[1], curr_pos[2])
                delta = curr - prev
                if math.sqrt(glm.dot(delta, delta)) > POP_THRESH:
                    print(f"POP frame = {frame}, joint = {skeleton.get_joint_name(i)}")
                    print(f"    prev = {prev}")
                    print(f"    curr = {curr}")
                    # build_xforms_at_frame(bvh, skeleton, frame - 1, True)
                    # build_xforms_at_frame(bvh, skeleton, frame, True)

    # second pass build j_vel
    for frame in trange(num_frames):
        for i in range(skeleton.num_joints):
            if frame > 0 and frame < num_frames - 2:
                inputs[frame].j_vel[i] = (glm.vec3(inputs[frame + 1].j_pos[i]) - glm.vec3(inputs[frame - 1].j_pos[i])) / (frame_time * 2)
            else:
                inputs[frame].j_vel[i] = [0, 0, 0]

    # AJT HACK REMOVE
    lookx = glm.quatLookAt(glm.vec3(1, 0, 0), glm.vec3(0, 1, 0))
    lookz = glm.quatLookAt(glm.vec3(0, 0, -1), glm.vec3(0, 1, 0))
    print(f"lookx = {lookx}")
    print(f"lookz = {lookz}")

    return skeleton, inputs
