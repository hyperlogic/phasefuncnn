import cmath
import glob
import math
import os
import pickle

import numpy as np
from flycam import FlyCamInterface
import pygfx as gfx
import torch
import torch.nn as nn
from wgpu.gui.auto import run

from build_traj import TRAJ_SAMPLE_RATE
import datalens
import followcam
import math_util as mu
import skeleton_mesh
from pfnn import PFNN
from renderbuddy import RenderBuddy
from skeleton import Skeleton

OUTPUT_DIR = "output"
TRAJ_WINDOW_SIZE = 12
TRAJ_SAMPLE_RATE = 6
SAMPLE_RATE = 60

class CharacterMovement(FlyCamInterface):
    speed: float  # units per second
    rot_speed: float  # radians per second
    worldUp: np.ndarray  # vec3
    pos: np.ndarray  # vec3
    vel: np.ndarray  # vec3
    rot: np.ndarray  # quat xyzw
    camera_mat: np.ndarray  # 4x4

    def __init__(self, world_up: np.ndarray, pos: np.ndarray, rot: np.ndarray, speed: float, rot_speed: float):
        self.world_up = world_up
        self.pos = pos
        self.rot = rot
        self.vel = np.array([0, 0, 0], dtype=np.float32)
        self.speed = speed
        self.rot_speed = rot_speed
        self.camera_mat = mu.build_mat_from_quat(np.eye(4), self.rot)
        self.camera_mat[:3, 3] = self.pos

    def process(self, dt: float, left_stick: np.ndarray, right_stick: np.ndarray, roll_amount: float, up_amount: float):
        STIFF = 200.0
        K = STIFF / self.speed

        # left_stick controls position, in world space!
        stick = mu.limit(np.array([left_stick[0], 0, left_stick[1]]))
        s_over_k = (stick * STIFF) / K
        s_over_k_sq = (stick * STIFF) / (K * K)
        e_neg_kt = np.exp(-K * dt)
        v = s_over_k + e_neg_kt * (self.vel - s_over_k)
        self.pos = s_over_k * dt + (s_over_k_sq - self.vel / K) * e_neg_kt + self.pos - s_over_k_sq + (self.vel / K)
        self.vel = v

        # left_stick also determines direction.  rotate towards left_stick diration
        forward = mu.quat_rotate(self.rot, np.array([0, 0, -1], dtype=np.float32))
        right = mu.quat_rotate(self.rot, np.array([1, 0, 0], dtype=np.float32))

        stick_norm = np.linalg.norm(stick)
        if stick_norm > 0:
            theta = np.arccos(min(1, np.dot(forward, stick)))
            if theta / self.rot_speed < dt:
                forward = stick
            else:
                sign = -np.sign(min(1, np.dot(stick, right)))
                yaw = mu.quat_from_angle_axis(self.rot_speed * dt * theta * sign, self.world_up)
                forward = mu.quat_rotate(yaw, forward)

        # make sure that cameraMat will be orthogonal, and aligned with world up.
        self.camera_mat = mu.orthogonalize_camera_mat(-forward, self.world_up, self.pos)
        self.rot = mu.quat_from_mat(self.camera_mat)


def unpickle_obj(filename):
    with open(filename, "rb") as f:
        return pickle.load(f)


def nograd_tensor(array: list[float]):
    return torch.tensor(array, dtype=torch.float32, requires_grad=False)


def build_idle_input(x_lens: datalens.InputLens) -> torch.Tensor:
    x = torch.zeros(x_lens.num_cols)
    traj_pos_i = [
        [0.782, 0.189],
        [0.847, 0.161],
        [0.901, 0.128],
        [0.946, 0.088],
        [0.977, 0.039],
        [0.990, -0.022],
        [0.000, 0.000],
        [-0.964, 0.232],
        [-0.935, 0.347],
        [-0.865, 0.462],
        [-0.783, 0.484],
        [-0.634, 0.444],
    ]
    traj_dir_i = [
        [0.980, -0.254],
        [0.905, -0.240],
        [0.828, -0.225],
        [0.754, -0.210],
        [0.687, -0.195],
        [0.632, -0.180],
        [0.000, -0.000],
        [0.631, 0.145],
        [0.687, 0.192],
        [0.754, 0.213],
        [0.827, 0.186],
        [0.902, 0.156],
    ]
    joint_pos_im1 = [
        [0.065, 0.104, -0.029],
        [0.065, 0.104, -0.029],
        [-0.032, 0.105, -0.104],
        [-0.115, 0.032, -0.007],
        [-0.020, -0.093, -0.037],
        [-0.002, -0.064, -0.039],
        [0.065, 0.104, -0.029],
        [0.032, 0.107, 0.104],
        [-0.081, 0.030, -0.015],
        [-0.012, -0.089, 0.010],
        [0.005, -0.060, 0.016],
        [0.065, 0.104, -0.029],
        [-0.086, 0.096, 0.006],
        [-0.105, 0.099, 0.017],
        [-0.105, 0.099, 0.017],
        [-0.097, 0.089, 0.009],
        [-0.105, 0.083, 0.011],
        [-0.105, 0.099, 0.017],
        [-0.084, 0.087, 0.040],
        [-0.079, 0.268, -0.402],
        [-0.128, 0.266, -0.567],
        [-0.128, 0.266, -0.567],
        [-0.139, 0.254, -0.535],
        [-0.128, 0.266, -0.567],
        [-0.105, 0.099, 0.017],
        [-0.089, 0.089, -0.052],
        [-0.072, 0.264, 0.389],
        [-0.109, 0.323, 0.544],
        [-0.109, 0.323, 0.544],
        [-0.116, 0.313, 0.535],
        [-0.109, 0.323, 0.544],
    ]
    joint_vel_im1 = [
        [-0.003, -0.002, 0.003],
        [-0.003, -0.002, 0.003],
        [-0.010, -0.002, 0.004],
        [-0.001, -0.002, -0.004],
        [-0.003, 0.000, -0.002],
        [-0.003, -0.000, -0.002],
        [-0.003, -0.002, 0.003],
        [0.010, -0.002, -0.004],
        [0.000, -0.002, -0.002],
        [-0.003, 0.001, -0.003],
        [-0.002, 0.001, -0.002],
        [-0.003, -0.002, 0.003],
        [-0.006, -0.002, 0.005],
        [-0.002, -0.003, 0.011],
        [-0.002, -0.003, 0.011],
        [-0.002, -0.002, 0.012],
        [0.007, -0.001, 0.011],
        [-0.002, -0.003, 0.011],
        [0.008, 0.000, -0.002],
        [0.007, -0.017, 0.006],
        [0.009, -0.026, 0.008],
        [0.009, -0.026, 0.008],
        [0.009, -0.025, 0.006],
        [0.009, -0.026, 0.008],
        [-0.002, -0.003, 0.011],
        [0.003, -0.003, 0.020],
        [-0.007, -0.029, 0.001],
        [0.022, -0.036, -0.012],
        [0.022, -0.036, -0.012],
        [0.027, -0.034, -0.012],
        [0.022, -0.036, -0.012],
    ]
    for i, v in enumerate(traj_pos_i):
        x_lens.traj_pos_i.set(x, i, nograd_tensor(v))
    for i, v in enumerate(traj_dir_i):
        x_lens.traj_dir_i.set(x, i, nograd_tensor(v))
    for i, v in enumerate(joint_pos_im1):
        x_lens.joint_pos_im1.set(x, i, nograd_tensor(v))
    for i, v in enumerate(joint_vel_im1):
        x_lens.joint_vel_im1.set(x, i, nograd_tensor(v))
    return x


def build_idle_output(y_lens: datalens.OutputLens) -> torch.Tensor:
    y = torch.zeros(y_lens.num_cols)
    traj_pos_ip1 = [
        [0.782, 0.189],
        [0.846, 0.160],
        [0.900, 0.127],
        [0.944, 0.086],
        [0.975, 0.037],
        [0.985, -0.026],
        [-0.000, 0.000],
        [-0.959, 0.242],
        [-0.926, 0.360],
        [-0.858, 0.475],
        [-0.772, 0.485],
        [-0.617, 0.442],
    ]
    traj_dir_ip1 = [
        [0.980, -0.253],
        [0.905, -0.239],
        [0.828, -0.224],
        [0.754, -0.209],
        [0.687, -0.194],
        [0.632, -0.178],
        [0.000, 0.000],
        [0.632, 0.147],
        [0.687, 0.197],
        [0.754, 0.208],
        [0.827, 0.184],
        [0.902, 0.158],
    ]
    joint_pos_i = [
        [0.651, 1.035, -0.288],
        [0.651, 1.035, -0.288],
        [-0.328, 1.045, -1.031],
        [-1.147, 0.314, -0.077],
        [-0.209, -0.932, -0.369],
        [-0.022, -0.639, -0.390],
        [0.651, 1.035, -0.288],
        [0.328, 1.062, 1.031],
        [-0.813, 0.295, -0.149],
        [-0.119, -0.888, 0.093],
        [0.050, -0.595, 0.155],
        [0.651, 1.035, -0.288],
        [-0.860, 0.962, 0.065],
        [-1.047, 0.984, 0.178],
        [-1.047, 0.984, 0.178],
        [-0.970, 0.885, 0.099],
        [-1.048, 0.833, 0.113],
        [-1.047, 0.984, 0.178],
        [-0.840, 0.872, 0.397],
        [-0.782, 2.668, -4.017],
        [-1.271, 2.640, -5.675],
        [-1.271, 2.640, -5.675],
        [-1.384, 2.516, -5.357],
        [-1.271, 2.640, -5.675],
        [-1.047, 0.984, 0.178],
        [-0.885, 0.891, -0.503],
        [-0.728, 2.619, 3.883],
        [-1.068, 3.203, 5.432],
        [-1.068, 3.203, 5.432],
        [-1.133, 3.096, 5.347],
        [-1.068, 3.203, 5.432],
    ]
    joint_vel_i = [
        [-0.029, -0.020, 0.010],
        [-0.029, -0.020, 0.010],
        [-0.070, -0.021, 0.046],
        [-0.026, -0.018, -0.036],
        [-0.034, -0.002, -0.034],
        [-0.028, 0.001, -0.025],
        [-0.029, -0.020, 0.010],
        [0.070, -0.022, -0.046],
        [0.005, -0.019, -0.019],
        [-0.031, 0.011, -0.038],
        [-0.027, 0.010, -0.027],
        [-0.029, -0.020, 0.010],
        [-0.052, -0.017, 0.030],
        [-0.055, -0.018, 0.106],
        [-0.055, -0.018, 0.106],
        [-0.050, -0.019, 0.122],
        [0.054, -0.004, 0.099],
        [-0.055, -0.018, 0.106],
        [0.068, 0.006, -0.035],
        [0.116, -0.218, 0.063],
        [0.115, -0.334, 0.085],
        [0.115, -0.334, 0.085],
        [0.113, -0.323, 0.072],
        [0.115, -0.334, 0.085],
        [-0.055, -0.018, 0.106],
        [0.027, -0.030, 0.232],
        [-0.081, -0.366, -0.004],
        [0.256, -0.448, -0.157],
        [0.256, -0.448, -0.157],
        [0.311, -0.430, -0.157],
        [0.256, -0.448, -0.157],
    ]
    joint_rot_i = [
        [-0.075, -0.347, 0.010],
        [0.971, -0.763, -0.714],
        [1.022, 0.031, -0.811],
        [-0.488, -0.255, 0.873],
        [-0.937, 0.507, 0.632],
        [-0.864, 0.230, 0.563],
        [-0.181, 0.400, -0.545],
        [0.771, -0.330, -0.692],
        [-0.649, 0.274, 0.590],
        [-0.371, 0.114, 0.214],
        [-0.578, 0.203, 0.395],
        [-0.574, 0.228, 0.683],
        [-0.935, 0.460, 1.003],
        [-0.530, 0.067, 0.440],
        [-0.721, -0.132, 0.726],
        [-0.907, -0.051, 1.236],
        [-1.133, -0.139, 1.556],
        [-0.750, 0.376, 1.497],
        [0.434, 0.721, 0.341],
        [2.611, 2.197, 0.998],
        [2.100, 2.410, 0.447],
        [1.703, 2.255, 0.400],
        [1.703, 2.255, 0.400],
        [2.100, 2.410, 0.447],
        [-0.692, 0.057, 0.033],
        [-5.218, 0.362, -1.023],
        [-2.411, -1.065, -2.593],
        [-0.554, -0.864, -1.661],
        [-0.660, -0.995, -1.453],
        [-0.660, -0.995, -1.453],
        [-0.554, -0.864, -1.661],
    ]
    root_vel_i = [[-0.931, 0.105]]
    root_angvel_i = [[-0.166]]
    phase_vel_i = [[1.269]]
    contacts_i = [[0.926, 0.963, 0.816, 0.845]]
    for i, v in enumerate(traj_pos_ip1):
        y_lens.traj_pos_ip1.set(y, i, nograd_tensor(v))
    for i, v, in enumerate(traj_dir_ip1):
        y_lens.traj_dir_ip1.set(y, i, nograd_tensor(v))
    for i, v, in enumerate(joint_pos_i):
        y_lens.joint_pos_i.set(y, i, nograd_tensor(v))
    for i, v, in enumerate(joint_vel_i):
        y_lens.joint_vel_i.set(y, i, nograd_tensor(v))
    for i, v, in enumerate(joint_rot_i):
        y_lens.joint_rot_i.set(y, i, nograd_tensor(v))
    for i, v, in enumerate(root_vel_i):
        y_lens.root_vel_i.set(y, i, nograd_tensor(v))
    for i, v, in enumerate(root_angvel_i):
        y_lens.root_angvel_i.set(y, i, nograd_tensor(v))
    for i, v, in enumerate(phase_vel_i):
        y_lens.phase_vel_i.set(y, i, nograd_tensor(v))
    for i, v, in enumerate(contacts_i):
        y_lens.contacts_i.set(y, i, nograd_tensor(v))
    return y


class VisOutputRenderBuddy(RenderBuddy):
    skeleton: Skeleton
    x_lens: datalens.InputLens
    y_lens: datalens.OutputLens
    model: nn.Module
    x: torch.Tensor
    p: float
    y_mean: torch.Tensor
    y_std: torch.Tensor
    x_mean: torch.Tensor
    x_std: torch.Tensor
    x_w: torch.Tensor
    y: torch.Tensor
    skeleton_group: gfx.Group
    playing: bool
    tick_once: bool
    t: float
    history_xforms: torch.Tensor
    history_cursor: int
    draw_output_trajectory: bool
    draw_input_trajectory: bool
    root_xform: np.ndarray
    input_traj_line: gfx.Group
    output_traj_line: gfx.Group

    def __init__(
        self,
        skeleton: Skeleton,
        x_lens: datalens.InputLens,
        y_lens: datalens.OutputLens,
        model: nn.Module,
        y_mean: torch.Tensor,
        y_std: torch.Tensor,
        x_mean: torch.Tensor,
        x_std: torch.Tensor,
        x_w: torch.Tensor,
    ):
        super().__init__()

        # override flycam with follow cam
        ORBIT_SPEED = 1.15
        MOVE_SPEED = 22.5
        RADIUS = 50
        target_y = 15

        self.flycam = followcam.FollowCam(
            np.array([0, 1, 0]), np.array([0, target_y, 0]), RADIUS, MOVE_SPEED, ORBIT_SPEED
        )

        self.skeleton = skeleton
        self.x_lens = x_lens
        self.y_lens = y_lens
        self.model = model
        self.y_mean = y_mean
        self.y_std = y_std
        self.x_mean = x_mean
        self.x_std = x_std
        self.x_w = x_w

        # setup initial state
        self.x = build_idle_input(self.x_lens)
        self.p = nograd_tensor(0.0)
        self.y = build_idle_output(self.y_lens)
        self.y = y_lens.unnormalize(self.y, self.y_mean, self.y_std)

        # two second of history
        self.history_xforms = np.tile(np.eye(4), (SAMPLE_RATE * 2, 1, 1))
        self.history_cursor = 0
        self.root_xform = np.eye(4)

        axes = gfx.helpers.AxesHelper(3.0, 1)
        self.scene.add(axes)

        self.group = gfx.Group()
        self.scene.add(self.group)

        self.skeleton_group = gfx.Group()
        self.input_traj_line = gfx.Group()
        self.skeleton_group.add(self.input_traj_line)
        self.output_traj_line = gfx.Group()
        self.skeleton_group.add(self.output_traj_line)
        self.bones = skeleton_mesh.add_skeleton_mesh(self.skeleton, self.skeleton_group)
        self.scene.add(self.skeleton_group)

        self.draw_output_trajectory = True
        self.draw_input_trajectory = True

        self.animate_skeleton()

        self.playing = False
        self.tick_once = False
        self.t = 0.0

        # used to render joystick dir in world space
        self.stick_line = gfx.Line(
            gfx.Geometry(positions=[[0, 0, 0], [0, 0, 0]], colors=[[0, 1, 0, 1], [0, 1, 0, 1]]),
            gfx.LineSegmentMaterial(thickness=2, color_mode="vertex"),
        )
        self.scene.add(self.stick_line)


    def animate_skeleton(self):

        # rotate the bones!
        global_rots = np.array([0, 0, 0, 1]) * np.ones((self.skeleton.num_joints, 4))
        for i in range(self.skeleton.num_joints):
            # extract rotation exponent from output
            exp = self.y_lens.joint_rot_i.get(self.y, i)

            # convert into a quaternion
            global_rots[i] = mu.expmap(exp)

            # transform rotations from global_rot to local_rot
            joint_name = self.skeleton.get_joint_name(i)
            parent_index = self.skeleton.get_parent_index(joint_name)
            if parent_index >= 0:
                local_rot = mu.quat_mul(mu.quat_inverse(global_rots[parent_index]), global_rots[i])
            else:
                local_rot = global_rots[i]

            self.bones[i].local.rotation = local_rot

        # update pelvis pos
        pelvis_pos = self.y_lens.joint_pos_i.get(self.y, 0).tolist()
        self.bones[0].local.position = pelvis_pos

        # apply root motion!
        root_vel = np.array([y_lens.root_vel_i.get(self.y, 0)[0], 0, y_lens.root_vel_i.get(self.y, 0)[1]])
        root_angvel = y_lens.root_angvel_i.get(self.y, 0).item()
        dt = 1 / SAMPLE_RATE
        delta_xform = np.eye(4)
        mu.build_mat_from_quat_pos(
            delta_xform, mu.quat_from_angle_axis(root_angvel * dt, np.array([0, 1, 0])), root_vel * dt
        )
        self.root_xform = self.root_xform @ delta_xform
        self.skeleton_group.local.position = self.root_xform[0:3, 3]
        self.skeleton_group.local.rotation = mu.quat_from_mat(self.root_xform)

        if self.draw_output_trajectory:
            # create lines for the trajectory
            positions = []
            colors = []
            for i in range(TRAJ_WINDOW_SIZE - 1):
                p0 = y_lens.traj_pos_ip1.get(self.y, i)
                p1 = y_lens.traj_pos_ip1.get(self.y, i + 1)
                positions += [[p0[0], 0.0, p0[1]], [p1[0], 0.0, p1[1]]]
                if i % 2 == 0:
                    colors += [[1, 1, 1, 1], [1, 1, 1, 1]]
                else:
                    colors += [[1, 0, 0, 1], [1, 0, 0, 1]]

            # draw the directions in green
            for i in range(TRAJ_WINDOW_SIZE):
                p0 = y_lens.traj_pos_ip1.get(self.y, i)
                p1 = p0 + y_lens.traj_dir_ip1.get(self.y, i)
                positions += [[p0[0], 0.0, p0[1]], [p1[0], 0.0, p1[1]]]
                colors += [[0, 1, 0, 1], [0, 1, 0, 1]]

            output_traj_line = gfx.Line(
                gfx.Geometry(positions=positions, colors=colors), gfx.LineSegmentMaterial(thickness=2, color_mode="vertex")
            )
            self.skeleton_group.remove(self.output_traj_line)
            self.skeleton_group.add(output_traj_line)
            self.output_traj_line = output_traj_line

        if self.draw_input_trajectory:

            # unnormalize x so we can draw it.
            x = x_lens.unnormalize(self.x, self.x_mean, self.x_std, self.x_w)

            # create lines for the trajectory
            positions = []
            colors = []
            for i in range(TRAJ_WINDOW_SIZE - 1):
                p0 = x_lens.traj_pos_i.get(x, i)
                p1 = x_lens.traj_pos_i.get(x, i + 1)
                positions += [[p0[0], 0.0, p0[1]], [p1[0], 0.0, p1[1]]]
                if i % 2 == 0:
                    colors += [[1, 1, 1, 1], [1, 1, 1, 1]]
                else:
                    colors += [[0.5, 0.5, 1, 1], [0.5, 0.5, 1, 1]]

            # draw the directions in green
            for i in range(TRAJ_WINDOW_SIZE):
                p0 = x_lens.traj_pos_i.get(x, i)
                p1 = p0 + x_lens.traj_dir_i.get(x, i)
                positions += [[p0[0], 0.05, p0[1]], [p1[0], 0.05, p1[1]]]
                colors += [[0, 1, 0, 1], [0, 1, 0, 1]]

            input_traj_line = gfx.Line(
                gfx.Geometry(positions=positions, colors=colors), gfx.LineSegmentMaterial(thickness=2, color_mode="vertex")
            )
            self.skeleton_group.remove(self.input_traj_line)
            self.skeleton_group.add(input_traj_line)
            self.input_traj_line = input_traj_line

        # record traj history.
        self.history_cursor = (self.history_cursor + 1) % (SAMPLE_RATE * 2)
        self.history_xforms[self.history_cursor] = self.root_xform


    def on_animate(self, dt: float):
        super().on_animate(dt)

        joystick_left_stick = np.array([0, 0], dtype=np.float32)
        if self.joystick:
            joystick_left_stick[0] = mu.deadspot(self.joystick.get_axis(0))
            joystick_left_stick[1] = -mu.deadspot(self.joystick.get_axis(1))
        left_stick = left_stick = np.clip(self.left_stick + joystick_left_stick, -1, 1)

        # project the stick onto the ground plane, such that up on the stick points towards the camera forward direction.
        cam_forward = mu.quat_rotate(self.flycam.rot, np.array([0, 0, -1]))
        theta = math.atan2(cam_forward[2], cam_forward[0])
        c_stick = cmath.rect(1.0, theta) * complex(left_stick[1], left_stick[0])
        world_stick = np.array([c_stick.real, 0, c_stick.imag])

        self.scene.remove(self.stick_line)
        self.stick_line = gfx.Line(
            gfx.Geometry(positions=[[0, 0, 0], world_stick * 10], colors=[[0, 1, 0, 1], [0, 1, 0, 1]]),
            gfx.LineSegmentMaterial(thickness=2, color_mode="vertex"),
        )
        self.scene.add(self.stick_line)

        if self.playing or self.tick_once:
            self.t += dt
            if self.t > (1 / SAMPLE_RATE):
                self.x = self.build_input()
                self.tick_model()
                self.animate_skeleton()
            self.tick_once = False

    def build_input(self) -> torch.Tensor:
        # NOTE: self.y is already unnormalized
        root_vel = np.array([y_lens.root_vel_i.get(self.y, 0)[0], 0, y_lens.root_vel_i.get(self.y, 0)[1]])

        x = torch.zeros(x_lens.num_cols)
        x = x_lens.unnormalize(x, self.x_mean, self.x_std, self.x_w)

        # initialize past part of traj from history_xforms
        inv_root_xform = np.linalg.inv(self.root_xform)
        history_step = SAMPLE_RATE // TRAJ_SAMPLE_RATE
        N = TRAJ_WINDOW_SIZE // 2
        for i in range(N):
            cursor = (self.history_cursor - (history_step * (i + 1))) % (SAMPLE_RATE * 2)

            # rotate xform into root space
            xform = inv_root_xform @ self.history_xforms[cursor]
            traj_pos = xform[:3, 3]
            traj_dir = mu.quat_rotate(mu.quat_from_mat(xform), np.array([1, 0, 0]))

            self.x_lens.traj_pos_i.set(x, (N - 1) - i, nograd_tensor([traj_pos[0], traj_pos[2]]))
            self.x_lens.traj_dir_i.set(x, (N - 1) - i, nograd_tensor([traj_dir[0], traj_dir[2]]))

        MOVE_SPEED = 42.5
        ROT_SPEED = 3.15
        up = np.array([0, 1, 0])
        init_rot = mu.quat_from_angle_axis(-np.pi / 2, up)
        mover = CharacterMovement(up, np.array([0, 0, 0]), init_rot, MOVE_SPEED, ROT_SPEED)
        mover.vel = root_vel

        joystick_left_stick = np.array([0, 0], dtype=np.float32)
        if self.joystick:
            joystick_left_stick[0] = mu.deadspot(self.joystick.get_axis(0))
            joystick_left_stick[1] = -mu.deadspot(self.joystick.get_axis(1))
        left_stick = left_stick = np.clip(self.left_stick + joystick_left_stick, -1, 1)

        # project the stick onto the ground plane, such that up on the stick points towards the camera forward direction.
        cam_forward = mu.quat_rotate(self.flycam.rot, np.array([0, 0, -1]))
        theta = math.atan2(cam_forward[2], cam_forward[0])
        c_stick = cmath.rect(1.0, theta) * complex(left_stick[1], left_stick[0])
        world_stick = np.array([c_stick.real, 0, c_stick.imag])

        stick = np.array([world_stick[0], world_stick[2]])

        # initialize future part of traj from joystick
        for i in range(TRAJ_WINDOW_SIZE // 2, TRAJ_WINDOW_SIZE):

            pos = mover.pos
            dir = mu.quat_rotate(mover.rot, np.array([0, 0, -1]))

            traj_pos = nograd_tensor([pos[0], pos[2]])
            traj_dir = nograd_tensor([dir[0], dir[2]])

            self.x_lens.traj_pos_i.set(x, i, traj_pos)
            self.x_lens.traj_dir_i.set(x, i, traj_dir)

            NUM_SUBSTEPS = 10
            for i in range(NUM_SUBSTEPS):
                mover.process(1 / (NUM_SUBSTEPS * TRAJ_SAMPLE_RATE), stick, np.array([0, 0]), 0, 0)

        # TODO: blend between output trajectory and desired trajectory

        for i in range(self.skeleton.num_joints):
            # copy joints over from output to next input.
            joint_pos = y_lens.joint_pos_i.get(self.y, i)
            joint_vel = y_lens.joint_vel_i.get(self.y, i)
            x_lens.joint_pos_im1.set(self.x, i, joint_pos)
            x_lens.joint_vel_im1.set(self.x, i, joint_vel)

        x = x_lens.normalize(x, self.x_mean, self.x_std, self.x_w)
        return x



    def tick_model(self):

        # integrate phase
        phase_vel = y_lens.phase_vel_i.get(self.y, 0).item()
        MIN_PHASE_VEL = 0.0
        MAX_PHASE_VEL = 100000.0
        phase_vel = min(max(MIN_PHASE_VEL, phase_vel), MAX_PHASE_VEL)
        self.p += phase_vel * (1 / SAMPLE_RATE)
        self.p = self.p % (2 * math.pi)

        print(f"phase = {self.p}, phase_vel = {phase_vel}")

        self.y = self.model(self.x, self.p).detach()
        self.y = y_lens.unnormalize(self.y, self.y_mean, self.y_std)

    def on_key_down(self, event):
        super().on_key_down(event)
        if event.key == " ":
            self.playing = not self.playing

    def on_dpad_left(self):
        super().on_dpad_left()

    def on_dpad_right(self):
        super().on_dpad_right()
        self.tick_once = True


if __name__ == "__main__":

    # unpickle skeleton
    # pick ANY skeleton in the output dir, they should all be the same.
    skeleton_files = glob.glob(os.path.join(OUTPUT_DIR, "*_skeleton.pkl"))
    assert len(skeleton_files) > 0, "could not find any pickled skeletons in output folder"
    skeleton = unpickle_obj(skeleton_files[0])

    x_lens = datalens.InputLens(TRAJ_WINDOW_SIZE, skeleton.num_joints)
    y_lens = datalens.OutputLens(TRAJ_WINDOW_SIZE, skeleton.num_joints)

    print(f"skeleton.num_joints = {skeleton.num_joints}")

    device = torch.device("cpu")
    print(f"cuda.is_available() = {torch.cuda.is_available()}")
    print(f"device = {device}")

    torch.no_grad()

    # load model
    in_features = x_lens.num_cols
    out_features = y_lens.num_cols
    print(f"PFNN(in_features = {in_features}, out_features = {out_features}, device = {device}")
    model = PFNN(in_features, out_features, device=device)
    model.eval()  # deactivate dropout
    state_dict = torch.load(os.path.join(OUTPUT_DIR, "final_checkpoint.pth"), weights_only=False)
    model.load_state_dict(state_dict)

    # load input, and phase
    X_mean = torch.load(os.path.join(OUTPUT_DIR, "X_mean.pth"), weights_only=True)
    X_std = torch.load(os.path.join(OUTPUT_DIR, "X_std.pth"), weights_only=True)
    X_w = torch.load(os.path.join(OUTPUT_DIR, "X_w.pth"), weights_only=True)

    # load output
    Y_mean = torch.load(os.path.join(OUTPUT_DIR, "Y_mean.pth"), weights_only=True)
    Y_std = torch.load(os.path.join(OUTPUT_DIR, "Y_std.pth"), weights_only=True)

    render_buddy = VisOutputRenderBuddy(
        skeleton, x_lens, y_lens, model, Y_mean, Y_std, X_mean, X_std, X_w
    )
    run()
