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

from skeleton import Skeleton
import datalens

TRAJ_WINDOW_SIZE = 12
NUM_GAITS = 8
JOINT_IMPORTANCE_SCALE = 0.05


def unpickle_obj(filename: str):
    with open(filename, "rb") as f:
        return pickle.load(f)


if __name__ == "__main__":

    num_anims = len(snakemake.input.p_list)
    assert num_anims > 0

    skeleton = unpickle_obj(snakemake.input.skeleton_list[0])
    num_joints = skeleton.num_joints

    x_lens = datalens.InputLens(TRAJ_WINDOW_SIZE, num_joints)
    y_lens = datalens.OutputLens(TRAJ_WINDOW_SIZE, num_joints)

    total_frames = 0
    for i in range(num_anims):
        phase = torch.load(snakemake.input.p_list[i], weights_only=True)
        total_frames += phase.shape[0]

    X = torch.empty((total_frames, x_lens.num_cols), dtype=torch.float32)
    Y = torch.empty((total_frames, y_lens.num_cols), dtype=torch.float32)
    P = torch.empty((total_frames,), dtype=torch.float32)

    print(f"X.shape = {X.shape}")
    print(f"Y.shape = {Y.shape}")
    print(f"P.shape = {P.shape}")

    # concatenate tensors of all animations together.
    curr_start = 0
    for i in range(num_anims):

        x = torch.load(snakemake.input.x_list[i], weights_only=True)
        y = torch.load(snakemake.input.y_list[i], weights_only=True)
        p = torch.load(snakemake.input.p_list[i], weights_only=True)

        num_frames = p.shape[0]

        # verify data shapes
        assert x.shape == (num_frames, x_lens.num_cols)
        assert y.shape == (num_frames, y_lens.num_cols)
        assert p.shape == (num_frames,)

        next_start = curr_start + num_frames

        X[curr_start:next_start] = x
        Y[curr_start:next_start] = y
        P[curr_start:next_start] = p

        curr_start = next_start

    assert curr_start == total_frames

    # use weights to reduce the importance of input joint features by a scale factor
    X_w = torch.ones((X.shape[1],))
    jweight = torch.ones((3,)) * JOINT_IMPORTANCE_SCALE
    for i in range(num_joints):
        x_lens.joint_pos_im1.set(X_w, i, jweight)
        x_lens.joint_vel_im1.set(X_w, i, jweight)

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
    print(f"X_mean = {X_mean}")
    print(f"X_std = {X_std}")
    print(f"X_w = {X_w}")
    print(f"Y.shape = {Y.shape}, Y_mean.shape = {Y_mean.shape}, Y_std.shape = {Y_std.shape}")
    print(f"Y.mean = {Y_mean}")
    print(f"Y.std = {Y_std}")
    print(f"P.shape = {P.shape}")

    assert not torch.isnan(X).any()
    assert not torch.isnan(X_mean).any()
    assert not torch.isnan(X_std).any()
    assert not torch.isnan(X_w).any()

    assert not torch.isnan(Y).any()
    assert not torch.isnan(Y_mean).any()
    assert not torch.isnan(Y_std).any()

    assert not torch.isnan(P).any()

    torch.save(X, snakemake.output.x)
    torch.save(X_mean, snakemake.output.x_mean)
    torch.save(X_std, snakemake.output.x_std)
    torch.save(X_w, snakemake.output.x_w)

    torch.save(Y, snakemake.output.y)
    torch.save(Y_mean, snakemake.output.y_mean)
    torch.save(Y_std, snakemake.output.y_std)

    torch.save(P, snakemake.output.p)
