#
# Copyright (c) 2025 Anthony J. Thibault
# This software is licensed under the MIT License. See LICENSE for more details.
#
#
# loads all the intermediate numpy arrays from build_xforms, build_jointpva, build_traj, and build_contacts
# and outputs the input (X) output (Y) and phase (P) appropriate pytorch tensors ready for training.
#

import cmath
import os
import pickle
import sys

import numpy as np
import torch
from typing import Tuple

from skeleton import Skeleton
import datalens

OUTPUT_DIR = "output"
SAMPLE_RATE = 60
TRAJ_WINDOW_SIZE = 12
TRAJ_ELEMENT_SIZE = 4  # (px, pz, vx, vz)
NUM_GAITS = 8
JOINT_IMPORTANCE_SCALE = 0.3


def unpickle_obj(filename: str):
    with open(filename, "rb") as f:
        return pickle.load(f)


def build_tensors(
    skeleton: Skeleton,
    root: torch.Tensor,
    jointpva: torch.Tensor,
    traj: torch.Tensor,
    rootvel: torch.Tensor,
    contacts: torch.Tensor,
    phase: torch.Tensor,
    gait: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    num_joints = skeleton.num_joints
    num_rows = root.shape[0] - 2  # skip the first and last frame
    t = (1 / SAMPLE_RATE) * 2

    x_lens = datalens.InputLens(TRAJ_WINDOW_SIZE, num_joints)
    y_lens = datalens.OutputLens(TRAJ_WINDOW_SIZE, num_joints)

    x_shape = (num_rows, x_lens.num_cols)
    x = torch.zeros(x_shape)

    y_shape = (num_rows, y_lens.num_cols)
    y = torch.zeros(y_shape)

    # skip the first frames, and the last frame
    for i in range(1, root.shape[0] - 1):
        x_row = x[i - 1]

        for j in range(TRAJ_WINDOW_SIZE):
            traj_start = j * TRAJ_ELEMENT_SIZE
            x_lens.traj_pos_i.set(x_row, j, traj[i, traj_start : traj_start + 2])
            x_lens.traj_dir_i.set(x_row, j, traj[i, traj_start + 2 : traj_start + 4])

        for j in range(num_joints):
            x_lens.joint_pos_im1.set(x_row, j, jointpva[i - 1, j, 0:3])
            x_lens.joint_vel_im1.set(x_row, j, jointpva[i - 1, j, 3:6])

        x_lens.gait_i.set(x_row, 0, gait[i])

        # compute phase_vel
        # Represent angles as unit complex numbers
        z1 = cmath.rect(1, phase[i - 1].item())
        z2 = cmath.rect(1, phase[i + 1].item())
        diff = cmath.phase(z2 / z1)
        # ensure that phase_vel is always positive
        if diff < 0:
            diff = 2 * np.pi + diff
        phase_vel = diff / t
        assert (
            phase_vel >= 0
        ), f"p1 = {phase[i - 1]}, p2 = {phase[i + 1]}, diff = {diff}, diff2 = {phase[i + 1] - phase[i - 1]}"

        y_row = y[i - 1]

        for j in range(TRAJ_WINDOW_SIZE):
            traj_start = j * TRAJ_ELEMENT_SIZE
            y_lens.traj_pos_ip1.set(y_row, j, traj[i + 1, traj_start : traj_start + 2])
            y_lens.traj_dir_ip1.set(y_row, j, traj[i + 1, traj_start + 2 : traj_start + 4])

        for j in range(num_joints):
            y_lens.joint_pos_i.set(y_row, j, jointpva[i, j, 0:3])
            y_lens.joint_vel_i.set(y_row, j, jointpva[i, j, 3:6])
            y_lens.joint_rot_i.set(y_row, j, jointpva[i, j, 6:12])

        y_lens.root_vel_i.set(y_row, 0, rootvel[i, 0:2])
        y_lens.root_angvel_i.set(y_row, 0, rootvel[i, 2:3])
        y_lens.phase_vel_i.set(y_row, 0, phase_vel)
        y_lens.contacts_i.set(y_row, 0, contacts[i])

    return x, y


if __name__ == "__main__":

    skeleton = unpickle_obj(snakemake.input.skeleton)
    num_joints = skeleton.num_joints

    x_lens = datalens.InputLens(TRAJ_WINDOW_SIZE, num_joints)
    y_lens = datalens.OutputLens(TRAJ_WINDOW_SIZE, num_joints)

    root = np.load(snakemake.input.root)
    jointpva = np.load(snakemake.input.jointpva)
    traj = np.load(snakemake.input.traj)
    rootvel = np.load(snakemake.input.rootvel)
    contacts = np.load(snakemake.input.contacts)
    phase = np.load(snakemake.input.phase)
    gait = np.load(snakemake.input.gait)

    num_frames = phase.shape[0]

    # verify data shapes
    assert root.shape == (num_frames, 4, 4)
    assert jointpva.shape == (num_frames, num_joints, 12)  # (px, py, pz, vx, vy, vz, m[0,0], m[1,0], m[2,0], m[0,1], m[1,1], m[2,1])
    assert traj.shape == (num_frames, TRAJ_WINDOW_SIZE * TRAJ_ELEMENT_SIZE)
    assert rootvel.shape == (num_frames, 3)  # (vx, vz, angvel)
    assert contacts.shape == (num_frames, 4)  # (lfoot, rfoot, ltoe, rtoe)
    assert phase.shape == (num_frames,)
    assert gait.shape == (num_frames, NUM_GAITS)  # (idle, walk, jog, run, crouch, jump, crawl, unknown)

    # convert into tensors
    root = torch.from_numpy(root)
    jointpva = torch.from_numpy(jointpva)
    traj = torch.from_numpy(traj)
    rootvel = torch.from_numpy(rootvel)
    contacts = torch.from_numpy(contacts)
    phase = torch.from_numpy(phase)
    gait = torch.from_numpy(gait)

    x, y = build_tensors(skeleton, root, jointpva, traj, rootvel, contacts, phase, gait)

    assert x.shape == (num_frames - 2, x_lens.num_cols)
    assert y.shape == (num_frames - 2, y_lens.num_cols)

    p = phase[1:-1]  # also skip the first and last frame, to match x, y
    assert p.shape == (num_frames - 2,)

    print(f"x.shape = {x.shape}, y.shape = {y.shape}, p.shape = {p.shape}")

    torch.save(x, snakemake.output.x)
    torch.save(y, snakemake.output.y)
    torch.save(p, snakemake.output.p)
