#
# Copyright (c) 2025 Anthony J. Thibault
# This software is licensed under the MIT License. See LICENSE for more details.
#

import math_util as mu
import numpy as np

from flycam import FlyCamInterface

class FollowCam(FlyCamInterface):
    """orbit about a moving target"""

    world_up: np.ndarray  # vec3
    target: np.ndarray # vec3
    radius: float
    move_speed: float
    orbit_speed: float  # radians per second
    rot: np.ndarray  # quat
    pos: np.ndarray  # vec3
    camera_mat: np.ndarray # 4x4

    def __init__(self, world_up: np.ndarray, target: np.ndarray, radius: float, move_speed: float, orbit_speed: float):
        self.world_up = world_up
        self.target = target
        self.radius = radius
        self.move_speed = move_speed
        self.orbit_speed = orbit_speed

        self.camera_mat = np.eye(4)
        self.rot = mu.quat_from_mat(self.camera_mat)
        self.pos = mu.quat_rotate(self.rot, np.array([0, 0, radius], dtype=np.float32))

    def process(self, dt: float, left_stick: np.ndarray, right_stick: np.ndarray, roll_amount: float, up_amount: float):
        _, _ = up_amount, left_stick

        # use roll to zoom in and out.
        if np.fabs(roll_amount) > 0.1:
            self.radius += self.move_speed * dt * roll_amount
            self.radius = np.clip(0.1, self.radius, 1000)

        # right_stick control rotation about target
        right = mu.quat_rotate(self.rot, np.array([1, 0, 0], dtype=np.float32))
        forward = mu.quat_rotate(self.rot, np.array([0, 0, -1], dtype=np.float32))

        # limit up and down rotaiton
        pitch_rot_angle = self.orbit_speed * dt * right_stick[1]
        dot = np.dot(forward, self.world_up)
        COS_TEN_DEGREES = np.cos(0.17453292519943295)
        if dot > COS_TEN_DEGREES:
            pitch_rot_angle = min(0, pitch_rot_angle)
        elif dot < -COS_TEN_DEGREES:
            pitch_rot_angle = max(0, pitch_rot_angle)

        pitch = mu.quat_from_angle_axis(pitch_rot_angle, right)
        yaw = mu.quat_from_angle_axis(self.orbit_speed * dt * -right_stick[0], self.world_up)

        rot = mu.quat_mul(mu.quat_mul(yaw, pitch), self.rot)

        self.pos = mu.quat_rotate(rot, np.array([0, 0, self.radius], dtype=np.float32)) + self.target

        # make sure that cameraMat will be orthogonal, and aligned with world up.
        self.camera_mat = mu.build_look_at_mat(self.pos, self.target, self.world_up)
        self.rot = mu.quat_from_mat(self.camera_mat)
