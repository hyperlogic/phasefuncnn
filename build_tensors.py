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
            x_lens.traj_pos_i.set(x_row, j, traj[i, traj_start:traj_start+2])
            x_lens.traj_dir_i.set(x_row, j, traj[i, traj_start+2:traj_start+4])

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
        assert phase_vel >= 0, f"p1 = {phase[i - 1]}, p2 = {phase[i + 1]}, diff = {diff}, diff2 = {phase[i + 1] - phase[i - 1]}"

        y_row = y[i - 1]

        for j in range(TRAJ_WINDOW_SIZE):
            traj_start = j * TRAJ_ELEMENT_SIZE
            y_lens.traj_pos_ip1.set(y_row, j, traj[i + 1, traj_start:traj_start+2])
            y_lens.traj_dir_ip1.set(y_row, j, traj[i + 1, traj_start+2:traj_start+4])

        for j in range(num_joints):
            y_lens.joint_pos_i.set(y_row, j, jointpva[i, j, 0:3])
            y_lens.joint_vel_i.set(y_row, j, jointpva[i, j, 3:6])
            y_lens.joint_rot_i.set(y_row, j, jointpva[i, j, 6:9])

        y_lens.root_vel_i.set(y_row, 0, rootvel[i, 0:2])
        y_lens.root_angvel_i.set(y_row, 0, rootvel[i, 2:3])
        y_lens.phase_vel_i.set(y_row, 0, phase_vel)
        y_lens.contacts_i.set(y_row, 0, contacts[i])

    return x, y


if __name__ == "__main__":
    X = torch.tensor([], dtype=torch.float32, requires_grad=False)
    Y = torch.tensor([], dtype=torch.float32, requires_grad=False)
    P = torch.tensor([], dtype=torch.float32, requires_grad=False)

    NUM_JOINTS = 31
    x_lens = datalens.InputLens(TRAJ_WINDOW_SIZE, 31)
    y_lens = datalens.OutputLens(TRAJ_WINDOW_SIZE, 31)

    num_joints = NUM_JOINTS
    num_anims = len(snakemake.input.skeleton_list)
    assert num_anims > 0
    for i in range(num_anims):

        skeleton = unpickle_obj(snakemake.input.skeleton_list[i])
        root = np.load(snakemake.input.root_list[i])
        jointpva = np.load(snakemake.input.jointpva_list[i])
        traj = np.load(snakemake.input.traj_list[i])
        rootvel = np.load(snakemake.input.rootvel_list[i])
        contacts = np.load(snakemake.input.contacts_list[i])
        phase = np.load(snakemake.input.phase_list[i])
        gait = np.load(snakemake.input.gait_list[i])

        num_frames = root.shape[0]
        num_joints = skeleton.num_joints
        x_lens = datalens.InputLens(TRAJ_WINDOW_SIZE, num_joints)
        y_lens = datalens.OutputLens(TRAJ_WINDOW_SIZE, num_joints)

        print(f"    num_frames = {num_frames}")
        print(f"    num_joints = {num_joints}")
        print(f"    root.shape = {root.shape}")
        print(f"    jointpva.shape = {jointpva.shape}")
        print(f"    traj.shape = {traj.shape}")
        print(f"    rootvel.shape = {rootvel.shape}")
        print(f"    contacts.shape = {contacts.shape}")
        print(f"    phase.shape = {phase.shape}")
        print(f"    gait.shape = {gait.shape}")

        # verify data shapes
        assert root.shape[1] == 4 and root.shape[2] == 4
        assert jointpva.shape[0] == num_frames
        assert jointpva.shape[1] == num_joints
        assert jointpva.shape[2] == 9  # (px, py, pz, vx, vy, vz, ax, ay, az)
        assert traj.shape[0] == num_frames
        assert traj.shape[1] == TRAJ_WINDOW_SIZE * TRAJ_ELEMENT_SIZE
        assert rootvel.shape[0] == num_frames
        assert rootvel.shape[1] == 3  # (vx, vz, angvel)
        assert contacts.shape[0] == num_frames
        assert contacts.shape[1] == 4  # (lfoot, rfoot, ltoe, rtoe)
        assert phase.shape[0] == num_frames
        assert gait.shape[0] == num_frames # (idle, walk, jog, run, crouch, jump, crawl, unknown)
        assert gait.shape[1] == NUM_GAITS

        # convert into tensors
        root = torch.from_numpy(root)
        jointpva = torch.from_numpy(jointpva)
        traj = torch.from_numpy(traj)
        rootvel = torch.from_numpy(rootvel)
        contacts = torch.from_numpy(contacts)
        phase = torch.from_numpy(phase)
        gait = torch.from_numpy(gait)

        x, y = build_tensors(skeleton, root, jointpva, traj, rootvel, contacts, phase, gait)
        p = phase[1:-1]  # also skip the first and last frame, to match x, y

        print(f"    x.shape = {x.shape}")
        print(f"    y.shape = {y.shape}")
        print(f"    p.shape = {p.shape}")

        assert x.shape[0] == y.shape[0]
        assert y.shape[0] == p.shape[0]

        assert x.shape[1] == x_lens.num_cols
        assert y.shape[1] == y_lens.num_cols
        assert skeleton.num_joints == NUM_JOINTS

        if i == 0:
            X = x
            Y = y
            P = p
            num_joints = skeleton.num_joints
        else:
            X = torch.cat((X, x), dim=0)
            Y = torch.cat((Y, y), dim=0)
            P = torch.cat((P, p), dim=0)

            assert X.shape[1] == x_lens.num_cols
            assert Y.shape[1] == y_lens.num_cols

    # use weights to reduce the importance of input joint features by 10 percent
    X_w = torch.ones((X.shape[1],))
    #X_w[traj_size : traj_size + jointpv_size] = 0.1

    zero2 = torch.zeros((2,))
    for i in num_joints:
        x_lens.joint_pos_im1.set(X_w, i, zero2)
        x_lens.joint_vel_im1.set(X_w, i, zero2)

    X_mean, Y_mean = X.mean(dim=0), Y.mean(dim=0)

    # Add a small epsilon to std deviation to avoid division by zero
    epsilon = 1e-8
    X_std, Y_std = X.std(dim=0) + epsilon, Y.std(dim=0) + epsilon

    # don't apply normalization to the one hot gait vectors.
    x_lens.gait_i.set(X_mean, 0, torch.zeros((NUM_GAITS,)))
    x_lens.gait_i.set(X_std, 0, torch.ones((NUM_GAITS,)))

    # normalize and weight the importance of each feature
    X = (X - X_mean) * (X_w / X_std)
    Y = (Y - Y_mean) / Y_std

    print(f"X.shape = {X.shape}, X_mean.shape = {X_mean.shape}, X_std.shape = {X_std.shape}")
    print(f"Y.shape = {Y.shape}, Y_mean.shape = {Y_mean.shape}, Y_std.shape = {Y_std.shape}")
    print(f"P.shape = {P.shape}")
    print(f"X_mean = {X_mean}")
    print(f"X_std = {X_std}")
    print(f"Y.mean = {Y_mean}")
    print(f"Y.std = {Y_std}")

    assert torch.isnan(X).sum().item() == 0
    assert torch.isnan(Y).sum().item() == 0

    torch.save(X, snakemake.output.x)
    torch.save(X_mean, snakemake.output.x_mean)
    torch.save(X_std, snakemake.output.x_std)
    torch.save(X_w, snakemake.output.x_w)

    torch.save(Y, snakemake.output.y)
    torch.save(Y_mean, snakemake.output.y_mean)
    torch.save(Y_std, snakemake.output.y_std)

    torch.save(P, snakemake.output.p)
