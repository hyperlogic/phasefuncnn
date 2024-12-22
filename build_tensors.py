#
# loads all the intermediate numpy arrays from build_xforms, build_jointpva, build_traj, and build_contacts
# and outputs the input (X) output (Y) and phase (P) appropriate pytorch tensors ready for training.
#

import os
import pickle
import sys

import numpy as np
import torch
from typing import Tuple

from skeleton import Skeleton

OUTPUT_DIR = "output"
SAMPLE_RATE = 60
TRAJ_WINDOW_SIZE = 12
TRAJ_ELEMENT_SIZE = 4  # (px, pz, vx, vz)


def unpickle_obj(filename: str):
    with open(filename, "rb") as f:
        return pickle.load(f)


def build_tensors(
    skeleton: Skeleton,
    root: torch.Tensor,
    jointva: torch.Tensor,
    traj: torch.Tensor,
    rootvel: torch.Tensor,
    contacts: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    num_joints = skeleton.num_joints
    num_rows = root.shape[0] - 2  # skip the first and last frame
    t = (1 / SAMPLE_RATE) * 2

    traj_size = TRAJ_WINDOW_SIZE * 4
    jointpv_size = num_joints * 6
    x_shape = (num_rows, traj_size + jointpv_size)
    x = torch.zeros(x_shape)

    jointpva_size = num_joints * 9
    rootvel_size = 3
    phasevel_size = 1
    contacts_size = 4
    y_shape = (
        num_rows,
        traj_size + jointpva_size + rootvel_size + phasevel_size + contacts_size,
    )
    y = torch.zeros(y_shape)

    for row in range(num_rows):
        frame = row + 1

        col = 0
        x[row, col : col + traj_size] = traj[frame]
        col += traj_size
        x[row, col : col + jointpv_size] = jointpva[frame - 1, :, 0:6].flatten()
        col += jointpv_size
        assert col == x.shape[1]

        col = 0
        y[row, col : col + traj_size] = traj[frame + 1]
        col += traj_size
        y[row, col : col + jointpva_size] = jointpva[frame, :, :].flatten()
        col += jointpva_size
        y[row, col : col + rootvel_size] = rootvel[frame]
        col += rootvel_size
        y[row, col] = (phase[frame + 1] - phase[frame - 1]) / t
        col += 1
        y[row, col : col + contacts_size] = contacts[frame]
        col += contacts_size
        assert col == y.shape[1]

    return x, y


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Error: expected list of mocap filenames (without .bvh extension)")
        exit(1)

    for i in range(len(sys.argv) - 1):
        outbasepath = os.path.join(OUTPUT_DIR, sys.argv[i + 1])
        skeleton = unpickle_obj(outbasepath + "_skeleton.pkl")
        root = np.load(outbasepath + "_root.npy")
        jointpva = np.load(outbasepath + "_jointpva.npy")
        traj = np.load(outbasepath + "_traj.npy")
        contacts = np.load(outbasepath + "_contacts.npy")
        phase = np.load(outbasepath + "_phase.npy")
        rootvel = np.load(outbasepath + "_rootvel.npy")

        num_frames = root.shape[0]
        num_joints = skeleton.num_joints

        print(outbasepath)
        print(f"    num_frames = {num_frames}")
        print(f"    num_joints = {num_joints}")
        print(f"    root.shape = {root.shape}")
        print(f"    jointpva.shape = {jointpva.shape}")
        print(f"    traj.shape = {traj.shape}")
        print(f"    rootvel.shape = {rootvel.shape}")
        print(f"    contacts.shape = {contacts.shape}")
        print(f"    phase.shape = {phase.shape}")

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

        # convert into tensors
        root = torch.from_numpy(root)
        jointpva = torch.from_numpy(jointpva)
        traj = torch.from_numpy(traj)
        rootvel = torch.from_numpy(rootvel)
        contacts = torch.from_numpy(contacts)
        phase = torch.from_numpy(phase)

        x, y = build_tensors(skeleton, root, jointpva, traj, rootvel, contacts)
        p = phase[1:-1]

        print(f"    x.shape = {x.shape}")
        print(f"    y.shape = {y.shape}")
        print(f"    p.shape = {p.shape}")

        assert x.shape[0] == y.shape[0]
        assert y.shape[0] == p.shape[0]

        if i == 0:
            X = x
            Y = y
            P = p
            num_joints = skeleton.num_joints
        else:
            X = torch.cat((X, x), dim=0)
            Y = torch.cat((Y, y), dim=0)
            P = torch.cat((P, p), dim=0)

    traj_size = TRAJ_WINDOW_SIZE * 4
    jointpv_size = num_joints * 6
    jointpva_size = num_joints * 9

    # use weights to reduce the importance of input joint features by 10 percent
    X_w = torch.ones((X.shape[1],))
    X_w[traj_size : traj_size + jointpv_size] = 0.1

    X_mean, Y_mean = X.mean(dim=0), Y.mean(dim=0)

    # Add a small epsilon to std deviation to avoid division by zero
    epsilon = 1e-8
    X_std, Y_std = X.std(dim=0) + epsilon, Y.std(dim=0) + epsilon

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

    torch.save(X, os.path.join(OUTPUT_DIR, "X.pth"))
    torch.save(X_mean, os.path.join(OUTPUT_DIR, "X_mean.pth"))
    torch.save(X_std, os.path.join(OUTPUT_DIR, "X_std.pth"))
    torch.save(X_w, os.path.join(OUTPUT_DIR, "X_w.pth"))

    torch.save(Y, os.path.join(OUTPUT_DIR, "Y.pth"))
    torch.save(Y_mean, os.path.join(OUTPUT_DIR, "Y_mean.pth"))
    torch.save(Y_std, os.path.join(OUTPUT_DIR, "Y_std.pth"))

    torch.save(P, os.path.join(OUTPUT_DIR, "P.pth"))
