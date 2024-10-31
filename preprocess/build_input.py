import numpy as np

TRAJ_WINDOW = 12


class Input:
    def __init__(self, traj_window, num_frames, num_joints):
        self.t_pos = np.zeros(shape=(traj_window, 2), dtype=float)
        self.t_dir = np.zeros(shape=(traj_window, 2), dtype=float)
        self.j_pos = np.zeros(shape=(num_frames, 3), dtype=float)
        self.j_vel = np.zeros(shape=(num_frames, 3), dtype=float)


def build_input(bvh):
    num_frames = bvh.nframes
    frame_time = bvh.frame_time
    joint_names = bvh.get_joints_names()
    num_joints = 1  # len(joint_names) AJT HACK

    input = Input(TRAJ_WINDOW, num_frames, num_joints)

    for joint_name in ["Hips"]:  # AJT HACK
        for i in range(bvh.nframes):

            x_pos = bvh.frame_joint_channel(i, joint_name, "Xposition")
            y_pos = bvh.frame_joint_channel(i, joint_name, "Yposition")
            z_pos = bvh.frame_joint_channel(i, joint_name, "Zposition")

            input.j_pos[i] = [x_pos, y_pos, z_pos]

    return input
