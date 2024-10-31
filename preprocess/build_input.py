import numpy as np
import glm

TRAJ_WINDOW = 12


class Input:
    def __init__(self, traj_window, num_joints):
        self.t_pos = np.zeros(shape=(traj_window, 2), dtype=float)
        self.t_dir = np.zeros(shape=(traj_window, 2), dtype=float)
        self.j_pos = np.zeros(shape=(num_joints, 3), dtype=float)
        self.j_vel = np.zeros(shape=(num_joints, 3), dtype=float)


def build_input(bvh):
    num_frames = 120 # bvh.nframes
    frame_time = bvh.frame_time
    joint_names = bvh.get_joints_names()
    num_joints = len(joint_names)

    debug = False

    print(joint_names)

    inputs = []

    has_pos_map = {j: "Xposition" in bvh.joint_channels(j) for j in joint_names}
    has_rot_map = {j: "Xrotation" in bvh.joint_channels(j) for j in joint_names}
    parent_map = {j: bvh.joint_parent_index(j) for j in joint_names}

    for frame in range(num_frames):

        xforms = [glm.mat4() for i in range(num_joints)]

        # build xforms
        for i in range(len(joint_names)):
            joint_name = joint_names[i]
            offset = bvh.joint_offset(joint_name)
            if (has_pos_map[joint_name]):
                x_pos = bvh.frame_joint_channel(frame, joint_name, "Xposition") + offset[0]
                y_pos = bvh.frame_joint_channel(frame, joint_name, "Yposition") + offset[1]
                z_pos = bvh.frame_joint_channel(frame, joint_name, "Zposition") + offset[2]
            else:
                x_pos, y_pos, z_pos = offset[0], offset[1], offset[2]

            if (has_rot_map[joint_name]):
                x_rot = glm.radians(bvh.frame_joint_channel(frame, joint_name, "Xrotation"))
                y_rot = glm.radians(bvh.frame_joint_channel(frame, joint_name, "Yrotation"))
                z_rot = glm.radians(bvh.frame_joint_channel(frame, joint_name, "Zrotation"))
            else:
                x_rot, y_rot, z_rot = 0, 0, 0

            xrot = glm.mat4(glm.quat(x_rot, glm.vec3(1, 0, 0)))
            yrot = glm.mat4(glm.quat(y_rot, glm.vec3(0, 1, 0)))
            zrot = glm.mat4(glm.quat(z_rot, glm.vec3(0, 0, 1)))
            rot = zrot * xrot * yrot
            rot[3][0] = x_pos
            rot[3][1] = y_pos
            rot[3][2] = z_pos
            parent_index = parent_map[joint_name]
            if parent_index >= 0:
                xforms[i] = xforms[parent_index] * rot
            else:
                xforms[i] = rot

            if debug:
                print(f"{joint_name} {i} =")
                print(f"    local_pos = {glm.vec3(x_pos, y_pos, z_pos)}")
                print(f"    local_euler = {glm.vec3(x_rot, y_rot, z_rot)}")
                print(f"    global_xform = {xforms[i]}")

        # build j_pos
        input = Input(TRAJ_WINDOW, num_joints)
        for i in range(len(joint_names)):
            input.j_pos[i] = glm.vec3(xforms[i][3][0], xforms[i][3][1], xforms[i][3][2])

        inputs.append(input)

    if debug:
        print(inputs)

    return inputs
