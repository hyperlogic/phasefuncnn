import glm
import mocap
import numpy as np
import os
import pickle
import sys
import time
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


DEBUG_COUNT = 500
MAX_EPOCHS = 500
BATCH_SIZE = 100
VAL_DATASET_FACTOR = 0.1

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Error: expected mocap filename (without .bvh extension)")
        exit(1)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"cuda.is_available() = {torch.cuda.is_available()}")
    print(f"device = {device}")

    # load dataset
    mocap_basename = sys.argv[1]
    full_dataset = MocapDataset(mocap_basename)

    VAL_DATASET_SIZE = int(len(full_dataset) * VAL_DATASET_FACTOR)

    # instantiate model
    x, y = full_dataset[0]
    model = PFNN(x.shape[0], y.shape[0]).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)

    # split dataset into train and validation sets
    torch.manual_seed(42)
    train_dataset, val_dataset = torch.utils.data.random_split(
        full_dataset,
        [
            len(full_dataset) - VAL_DATASET_SIZE,
            VAL_DATASET_SIZE,
        ],
    )
    torch.manual_seed(torch.initial_seed())

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0
    )

    print(f"len(full_dataset) = {len(full_dataset)}")
    print(f"x.shape = {x.shape}")
    print(f"y.shape = {y.shape}")

    #
    # train
    #

    best_val_loss = float("inf")
    epochs_without_improvement = 0
    max_epochs_without_improvement = 10
    train_start_time = time.time()

    for epoch in range(MAX_EPOCHS):
        # train the model
        train_loss = 0.0
        train_count = 0

        for x, y in train_loader:
            x, y = x.to(device), y.to(device)

            optimizer.zero_grad()
            output = model(x)
            loss = criterion(output, y)

            train_loss += loss.item()
            train_count += 1

            loss.backward()
            optimizer.step()

        avg_train_loss = train_loss / train_count

        # validate the model
        val_loss = 0.0
        val_count = 0

        with torch.no_grad():
            for image, target in val_loader:
                # transfer tensors to gpu
                x, y = x.to(device), y.to(device)

                output = model(x)
                loss = criterion(output, y)
                val_loss += loss.item()
                val_count += 1

        avg_val_loss = val_loss / val_count
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        print(
            f"Epoch {epoch+1}: Training Loss = {avg_train_loss}, Validation Loss = {avg_val_loss}"
        )

        if epochs_without_improvement >= max_epochs_without_improvement:
            print("Early stopping triggered. Stopping training.")
            break

    train_end_time = time.time()
    print(f"Training took {train_end_time - train_start_time} sec")
