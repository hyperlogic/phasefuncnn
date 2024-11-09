import glm
import mocap
import numpy as np
import os
import pickle
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F


OUTPUT_DIR = "output"
TRAJ_WINDOW_SIZE = 12
TRAJ_ELEMENT_SIZE = 4


class MocapDataset(torch.utils.data.Dataset):
    def __init__(self, mocap_basename):

        outbasepath = os.path.join(OUTPUT_DIR, mocap_basename)

        # unpickle skeleton, xforms, jointpva
        self.skeleton = mocap.unpickle_obj(outbasepath + "_skeleton.pkl")
        # self.xforms = mocap.unpickle_obj(outbasepath + "_xforms.pkl")
        self.root = mocap.unpickle_obj(outbasepath + "_root.pkl")
        self.jointpva = np.load(outbasepath + "_jointpva.npy")
        self.traj = np.load(outbasepath + "_traj.npy")
        self.contacts = np.load(outbasepath + "_contacts.npy")
        self.phase = np.load(outbasepath + "_phase.npy")
        self.rootvel = np.load(outbasepath + "_rootvel.npy")

    def __len__(self):
        # skip the first and last frames valid items
        return len(self.root) - 2

    def __getitem__(self, idx):
        """
        x = { trajpd_i trajd_i jointp_i-1 jointv_i-1f, phase_i-1 }

        trajpd_i - TRAJ_WINDOW_SIZE * 4 floats total
        jointpv_i-1 - NUM_JOINTS * 6 floats total
        phase_i-1 - 1 float
        """
        num_joints = self.skeleton.num_joints
        curr = idx + 1
        trajpd_curr = self.traj[curr]
        jointpv_prev = self.jointpva[curr - 1, 0:num_joints, 0:6].flatten()
        phase_prev = np.array([self.phase[curr - 1]])
        x = np.concatenate((trajpd_curr, jointpv_prev, phase_prev))

        """
        y = { trajp_i+1 trajd_i+1 jointp_i jointv_i jointa_i rootvel_i, phasevel_i, contacts_i }

        trajpd_i+1 - TRAJ_WINDOW_SIZE * 4 floats total
        jointpva_i - NUM_JOINTS * 9 floats total
        rootvel_i - 3 floats total
        phasevel_i - 1 float
        contacts_i - 4 floats
        """
        trajpd_next = self.traj[curr + 1]
        jointpva_curr = self.jointpva[curr, 0:num_joints, 0:9].flatten()
        rootvel_curr = self.rootvel[curr]
        contacts_curr = self.contacts[curr]
        y = np.concatenate((trajpd_next, jointpva_curr, rootvel_curr, contacts_curr))

        return torch.Tensor(x), torch.Tensor(y)


class PFNN(nn.Module):
    def __init__(self, x_len, y_len):
        super(PFNN, self).__init__()
        self.fc1 = nn.Linear(x_len, 512)
        self.fc2 = nn.Linear(512, 512)
        self.fc3 = nn.Linear(512, y_len)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        x = self.fc3(x)
        return x


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Error: expected mocap filename (without .bvh extension)")
        exit(1)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"cuda.is_available() = {torch.cuda.is_available()}")
    print(f"device = {device}")

    mocap_basename = sys.argv[1]

    # load dataset
    dataset = MocapDataset(mocap_basename)

    # instantiate model
    x, y = dataset[0]
    model = PFNN(x.shape[0], y.shape[0]).to(device)

    print(f"len(dataset) = {len(dataset)}")
    print(f"x.shape = {x.shape}")
    print(f"y.shape = {y.shape}")

    print("done! (not really)")
