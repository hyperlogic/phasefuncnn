from abc import ABC, abstractmethod

import math_util as mu
import numpy as np


class FlyCamInterface(ABC):
    pos: np.ndarray
    rot: np.ndarray
    camera_mat: np.ndarray

    @abstractmethod
    def process(self, dt: float, left_stick: np.ndarray, right_stick: np.ndarray, roll_amount: float, up_amount: float):
        pass


class FlyCam(FlyCamInterface):
    """sphere in fluid motion model"""

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
        STIFF = 500.0
        K = STIFF / self.speed

        # left_stick and up_amount control position
        stick = mu.quat_rotate(self.rot, np.array([left_stick[0], up_amount, -left_stick[1]]))
        s_over_k = (stick * STIFF) / K
        s_over_k_sq = (stick * STIFF) / (K * K)
        e_neg_kt = np.exp(-K * dt)
        v = s_over_k + e_neg_kt * (self.vel - s_over_k)
        self.pos = s_over_k * dt + (s_over_k_sq - self.vel / K) * e_neg_kt + self.pos - s_over_k_sq + (self.vel / K)
        self.vel = v

        # right_stick and roll_amount control rotation
        right = mu.quat_rotate(self.rot, np.array([1, 0, 0], dtype=np.float32))
        forward = mu.quat_rotate(self.rot, np.array([0, 0, -1], dtype=np.float32))
        yaw = mu.quat_from_angle_axis(self.rot_speed * dt * -right_stick[0], self.world_up)
        pitch = mu.quat_from_angle_axis(self.rot_speed * dt * right_stick[1], right)
        rot = mu.quat_mul(mu.quat_mul(yaw, pitch), self.rot)

        # apply roll to worldUp
        if np.fabs(roll_amount) > 0.1:
            self.world_up = self.camera_mat[:3, 1]
            roll = mu.quat_from_angle_axis(self.rot_speed * dt * roll_amount, forward)
            self.world_up = mu.quat_rotate(roll, self.world_up)

        # make sure that cameraMat will be orthogonal, and aligned with world up.
        z = mu.quat_rotate(rot, np.array([0, 0, 1], dtype=np.float32))
        self.camera_mat = mu.orthogonalize_camera_mat(z, self.world_up, self.pos)

        self.rot = mu.quat_from_mat(self.camera_mat)
