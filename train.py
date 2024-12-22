import math
import numpy as np
import os
import pathlib
import sys
import time
import torch
import torch.nn as nn
import torch.nn.functional as F


OUTPUT_DIR = pathlib.Path("output")
TRAJ_WINDOW_SIZE = 12
TRAJ_ELEMENT_SIZE = 4
NUM_CONTROL_POINTS = 4


class MocapDataset(torch.utils.data.Dataset):
    def __init__(self):
        # unpickle skeleton, xforms, jointpva
        self.X = torch.load(OUTPUT_DIR / "X.pth", weights_only=True)
        self.Y = torch.load(OUTPUT_DIR / "Y.pth", weights_only=True)
        self.P = torch.load(OUTPUT_DIR / "P.pth", weights_only=True)

        print(f"X.shape {self.X.shape}")
        print(f"Y.shape {self.Y.shape}")
        print(f"P.shape {self.P.shape}")
        assert self.X.shape[0] == self.Y.shape[0]
        assert self.Y.shape[0] == self.P.shape[0]

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx], self.P[idx]


def phase_function(phase: torch.Tensor, weights: torch.Tensor, biases: torch.Tensor) -> torch.Tensor:
    assert biases.shape[0] == NUM_CONTROL_POINTS
    assert weights.shape[0] == NUM_CONTROL_POINTS

    batch_size = phase.shape[0]

    """
    print(f"phase.shape = {phase.shape}")
    print(f"weights.shape = {weights.shape}")
    print(f"biases.shape = {biases.shape}")
    """

    global CATMULL_ROM_BASIS, device

    mask = phase < (1.5 * torch.pi)

    t0 = phase / (1.5 * torch.pi)
    t1 = (phase - 0.5 * torch.pi) / (1.5 * torch.pi)

    tt0 = torch.zeros((batch_size, 4), device=device)
    tt0[:, 0] = 1.0
    tt0[:, 1] = t0
    tt0[:, 1] = t0**2
    tt0[:, 2] = t0**3

    w0 = tt0 @ CATMULL_ROM_BASIS @ weights.view(4, -1)
    b0 = tt0 @ CATMULL_ROM_BASIS @ biases

    tt1 = torch.zeros((batch_size, 4), device=device)
    tt1[:, 0] = 1.0
    tt1[:, 1] = t1
    tt1[:, 1] = t1**2
    tt1[:, 2] = t1**3

    w1 = tt1 @ CATMULL_ROM_BASIS @ weights.view(4, -1)
    b1 = tt1 @ CATMULL_ROM_BASIS @ biases

    """
    print(f"w0.shape = {w0.shape}")
    print(f"mask.shape = {mask.shape}")
    """

    w = torch.where(mask.unsqueeze(1), w0, w1)
    w = w.view(batch_size, weights.shape[1], weights.shape[2])
    b = torch.where(mask.unsqueeze(1), b0, b1)

    """
    print(f"w.shape = {w.shape}")
    print(f"b.shape = {b.shape}")
    """

    return w, b


class PhaseLinear(nn.Module):
    in_features: int
    out_features: int
    weights: torch.Tensor
    biases: torch.Tensor

    def __init__(self, in_features: int, out_features: int, device=None, dtype=None):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()

        self.in_features = in_features
        self.out_features = out_features

        # allocate NUM_CONTROL_POINTS sets of weights and biases
        self.weights = nn.Parameter(torch.empty((NUM_CONTROL_POINTS, out_features, in_features), **factory_kwargs))
        self.biases = nn.Parameter(torch.empty((NUM_CONTROL_POINTS, out_features), **factory_kwargs))

        # Initialize weights and biases
        # taken from torch.nn.Linear.reset_parameters
        for i in range(NUM_CONTROL_POINTS):
            nn.init.kaiming_uniform_(self.weights[i], a=math.sqrt(5))
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weights[i])
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.biases[i], -bound, bound)

    def forward(self, input: torch.Tensor, phase: torch.Tensor) -> torch.Tensor:
        w, b = phase_function(phase, self.weights, self.biases)

        # F.linear(input, w, b)
        result = w @ input.unsqueeze(-1) + b.unsqueeze(-1)
        result = result.squeeze()

        assert result.shape[0] == input.shape[0], "batch size must be the same"
        assert result.shape[1] == self.out_features, "result must have same size as out_features"

        return result


class PFNN(nn.Module):
    def __init__(self, in_features: int, out_features: int):
        super(PFNN, self).__init__()
        self.fc1 = PhaseLinear(in_features, 512)
        self.fc2 = PhaseLinear(512, 512)
        self.fc3 = PhaseLinear(512, out_features)

    def forward(self, x: torch.Tensor, phase: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.fc1(x, phase))
        x = F.relu(self.fc2(x, phase))
        x = self.fc3(x, phase)
        return x


MAX_EPOCHS = 10000
BATCH_SIZE = 1024
VAL_DATASET_FACTOR = 0.1

if __name__ == "__main__":
    if len(sys.argv) != 1:
        print("Error: no arguments necessary")
        exit(1)

    global device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"cuda.is_available() = {torch.cuda.is_available()}")
    print(f"device = {device}")

    global CATMULL_ROM_BASIS
    CATMULL_ROM_BASIS = torch.Tensor(
        [[ 0.0, 2.0, 0.0, 0.0],
         [-1.0, 0.0, 1.0, 0.0],
         [2.0, -5.0, 4.0, -1.0],
         [-1.0, 3.0, -3.0, 1.0]]
    ).to(device)

    # load dataset
    full_dataset = MocapDataset()

    VAL_DATASET_SIZE = int(len(full_dataset) * VAL_DATASET_FACTOR)

    model = PFNN(full_dataset.X.shape[1], full_dataset.Y.shape[1]).to(device)

    # print model
    print("model =")
    print(model)
    print("parameters =")
    for name, param in model.named_parameters():
        print(f"    {name}, size = {param.size()}")

    criterion = nn.L1Loss()
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

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)

    print(f"len(full_dataset) = {len(full_dataset)}")
    print(f"x.shape = {full_dataset.X.shape}, dtype={full_dataset.X.type()}")
    print(f"y.shape = {full_dataset.Y.shape}, dtype={full_dataset.Y.type()}")
    print(f"p.shape = {full_dataset.P.shape}, dtype={full_dataset.P.type()}")

    #
    # train
    #

    best_val_loss = float("inf")
    epochs_without_improvement = 0
    MAX_EPOCHS_WITHOUT_IMPROVEMENT = 10
    CHECKPOINT_CADENCE = 10
    train_start_time = time.time()

    for epoch in range(MAX_EPOCHS):
        # train the model
        train_loss = 0.0
        train_count = 0

        for x, y, p in train_loader:
            x, y, p = x.to(device), y.to(device), p.to(device)

            optimizer.zero_grad()
            output = model(x, p)
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
            for x, y, p in val_loader:
                # transfer tensors to gpu
                x, y, p = x.to(device), y.to(device), p.to(device)

                output = model(x, p)
                loss = criterion(output, y)
                val_loss += loss.item()
                val_count += 1

        avg_val_loss = val_loss / val_count
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        print(f"Epoch {epoch+1}: Training Loss = {avg_train_loss}, Validation Loss = {avg_val_loss}")

        if epochs_without_improvement >= MAX_EPOCHS_WITHOUT_IMPROVEMENT:
            print("Early stopping triggered. Stopping training.")
            break

        if epoch > 0 and (epoch % CHECKPOINT_CADENCE) == 0:
            # save checkpoint
            print(f"   saving checkpoint_{epoch}.pth")
            torch.save(model.state_dict(), OUTPUT_DIR / f"checkpoint_{epoch}.pth")

    train_end_time = time.time()
    print(f"Training took {train_end_time - train_start_time} sec")

    # output model
    torch.save(model.state_dict(), OUTPUT_DIR / "final_checkpoint.pth")
