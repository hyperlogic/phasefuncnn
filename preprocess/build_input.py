import glm
import math
import numpy as np
from tqdm import trange, tqdm

TRAJ_WINDOW = 12
PI_OVER_180 = math.pi / 180

X_AXIS = np.array([1, 0, 0])
Y_AXIS = np.array([0, 1, 0])
Z_AXIS = np.array([0, 0, 1])


def axis_angle_to_quat(axis, angle):
    n = axis * (math.sin(angle / 2) / np.linalg.norm(axis))
    w = math.cos(angle / 2)
    return np.array([n[0], n[1], n[2], w])


def quat_conj(quat):
    return np.array([-quat[0], -quat[1], -quat[2], quat[3]])


def quat_rotate(quat, vec):
    r = quat_mul(
        quat,
        quat_mul(
            np.array([vec[0], vec[1], vec[2], 0]),
            quat_conj(quat),
        ),
    )
    return np.array([r[0], r[1], r[2]])


def quat_rot4(quat, vec):
    return quat_mul(
        quat,
        quat_mul(
            np.array([vec[0], vec[1], vec[2], 0]),
            quat_conj(quat),
        ),
    )


def quat_to_mat4(quat, pos):
    col0 = quat_rot4(quat, X_AXIS)
    col1 = quat_rot4(quat, Y_AXIS)
    col2 = quat_rot4(quat, Z_AXIS)
    col3 = np.array([pos[0], pos[1], pos[2], 1])
    return np.column_stack((col0, col1, col2, col3))


def deg_to_rad(deg):
    return deg * PI_OVER_180


def quat_mul(lhs, rhs):
    return np.array(
        [
            lhs[0] * rhs[3] + lhs[1] * rhs[2] - lhs[2] * rhs[1] + lhs[3] * rhs[0],
            -lhs[0] * rhs[2] + lhs[1] * rhs[3] + lhs[2] * rhs[0] + lhs[3] * rhs[1],
            lhs[0] * rhs[1] - lhs[1] * rhs[0] + lhs[2] * rhs[3] + lhs[3] * rhs[2],
            -lhs[0] * rhs[0] - lhs[1] * rhs[1] - lhs[2] * rhs[2] + lhs[3] * rhs[3],
        ]
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

        pos = np.array(offset)
        if skeleton.has_pos(joint_name):
            pos += np.array(
                [
                    bvh.frame_joint_channel(frame, joint_name, "Xposition"),
                    bvh.frame_joint_channel(frame, joint_name, "Yposition"),
                    bvh.frame_joint_channel(frame, joint_name, "Zposition"),
                ]
            )

        rot = np.array([0, 0, 0, 1])
        if skeleton.has_rot(joint_name):
            x_rot = axis_angle_to_quat(
                X_AXIS,
                deg_to_rad(bvh.frame_joint_channel(frame, joint_name, "Xrotation")),
            )
            y_rot = axis_angle_to_quat(
                Y_AXIS,
                deg_to_rad(bvh.frame_joint_channel(frame, joint_name, "Yrotation")),
            )
            z_rot = axis_angle_to_quat(
                Z_AXIS,
                deg_to_rad(bvh.frame_joint_channel(frame, joint_name, "Zrotation")),
            )
            rot = quat_mul(z_rot, quat_mul(y_rot, x_rot))

        if debug:
            print(f"    {joint_name}")
            if skeleton.has_rot(joint_name):
                x_rot = bvh.frame_joint_channel(frame, joint_name, "Xrotation")
                y_rot = bvh.frame_joint_channel(frame, joint_name, "Yrotation")
                z_rot = bvh.frame_joint_channel(frame, joint_name, "Zrotation")
                print(f"        (local) euler (deg) = {(x_rot, y_rot, z_rot)}")
                print(
                    f"        (local) euler (rad) = {(deg_to_rad(x_rot), deg_to_rad(y_rot), deg_to_rad(z_rot))}"
                )
            print(f"        (local) pos = {pos}, rot = {rot}")

        m = quat_to_mat4(rot, pos)
        parent_index = skeleton.get_parent_index(joint_name)
        if parent_index >= 0:
            xforms[i] = xforms[parent_index] @ m
        else:
            xforms[i] = m

        if debug:
            global_pos = np.array([xforms[i][3][0], xforms[i][3][1], xforms[i][3][2]])
            print(f"        (global) pos = {global_pos}")

    return xforms


def build_input(bvh):
    num_frames = bvh.nframes
    frame_time = bvh.frame_time

    skeleton = Skeleton(bvh)

    pop_check = False

    print(skeleton.joint_names)

    inputs = []

    for frame in trange(num_frames):

        xforms = build_xforms_at_frame(bvh, skeleton, frame)

        # build j_pos
        input = Input(TRAJ_WINDOW, skeleton.num_joints)
        for i in range(skeleton.num_joints):
            # input.j_pos[i] = glm.vec3(xforms[i][3][0], xforms[i][3][1], xforms[i][3][2])
            input.j_pos[i] = glm.vec3(xforms[i][0][3], xforms[i][1][3], xforms[i][2][3])

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

    return skeleton, inputs
