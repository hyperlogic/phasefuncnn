import mocap
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


class MocapDataset(torch.utils.data.Dataset):
    def __init__(self):
        # unpickle skeleton, xforms, jointpva
        self.X = torch.load(OUTPUT_DIR / "X.pth", weights_only=True)
        self.Y = torch.load(OUTPUT_DIR / "Y.pth", weights_only=True)
        self.P = torch.load(OUTPUT_DIR / "P.pth", weights_only=True)

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx], self.P[idx]


def phase_function(phase, weights, biases):
    assert biases.shape[0] == 4
    assert weights.shape[0] == 4
    t = (4 * phase) / (2 * np.pi) % 1.0
    tt = torch.zeros([t.shape[0], 4]).to(device)
    tt[:, 0] = t**3
    tt[:, 1] = t**2
    tt[:, 2] = t
    tt[:, 3] = 1
    print(f"weights.shape = {weights.shape}")
    w = tt @ catmull_rom_basis @ weights.view(4, -1)
    b = tt @ catmull_rom_basis @ biases
    print(f"tt.shape = {tt.shape}")
    print(f"basis.shape = {catmull_rom_basis.shape}")
    print(f"weights.view(4, -1).shape = {weights.view(4, -1).shape}")
    print(f"w.shape = {w.shape}")
    return w.view(phase.shape[0], weights.shape[1], weights.shape[2]), b


class PhaseLinear(nn.Module):
    def __init__(self, input_len, output_len):
        super(PhaseLinear, self).__init__()

        # allocate control points
        self.ws = nn.Parameter(torch.Tensor(4, output_len, input_len))
        self.bs = nn.Parameter(torch.Tensor(4, output_len))

        # Initialize control points
        [nn.init.kaiming_uniform_(w, nonlinearity="relu") for w in self.ws]
        [nn.init.zeros_(b) for b in self.bs]

    def forward(self, input, p):
        # w, b = phase_function(p, self.ws, self.bs)
        result = F.linear(input, self.ws[0], self.bs[0])
        # print(f"result {result.shape} = F.linear(input {input.shape}, w {self.ws[0].shape}, b {self.bs[0].shape}")

        return result


class InterpolatedLinear(nn.Module):
    def __init__(self, in_features, out_features):
        super(InterpolatedLinear, self).__init__()
        # Two sets of weights and biases
        self.weight1 = nn.Parameter(torch.Tensor(out_features, in_features))
        self.weight2 = nn.Parameter(torch.Tensor(out_features, in_features))
        self.bias1 = nn.Parameter(torch.Tensor(out_features))
        self.bias2 = nn.Parameter(torch.Tensor(out_features))
        self.reset_parameters()

    def reset_parameters(self):
        # Initialize weights and biases for both parameter sets
        nn.init.kaiming_uniform_(self.weight1, a=np.sqrt(5))
        nn.init.kaiming_uniform_(self.weight2, a=np.sqrt(5))
        nn.init.zeros_(self.bias1)
        nn.init.zeros_(self.bias2)

    def forward(self, input, alpha):
        """
        Forward pass with interpolation.

        Parameters:
        - input: Input tensor (batch_size, in_features)
        - alpha: Interpolation tensor (batch_size,) or scalar

        Returns:
        - Interpolated output tensor
        """
        # Ensure alpha is broadcastable
        if len(alpha.shape) == 1:
            alpha = alpha.view(-1, 1, 1)  # Reshape to (batch_size, 1, 1)

        # Interpolate weights and biases
        weight = alpha * self.weight1 + (1 - alpha) * self.weight2  # Shape: (batch_size, out_features, in_features)
        bias = (
            alpha.squeeze(-1) * self.bias1 + (1 - alpha.squeeze(-1)) * self.bias2
        )  # Shape: (batch_size, out_features)

        # Apply linear transformation
        output = torch.bmm(input.unsqueeze(1), weight.transpose(1, 2)).squeeze(1) + bias
        return output


class PFNN(nn.Module):
    def __init__(self, input_len, output_len):
        super(PFNN, self).__init__()
        self.fc1 = PhaseLinear(input_len, 512)
        self.fc2 = PhaseLinear(512, 512)
        self.fc3 = PhaseLinear(512, output_len)

    def forward(self, x, p):
        x = F.relu(self.fc1(x, p))
        x = F.relu(self.fc2(x, p))
        x = self.fc3(x, p)
        return x


MAX_EPOCHS = 500
BATCH_SIZE = 6
VAL_DATASET_FACTOR = 0.1

if __name__ == "__main__":
    if len(sys.argv) != 1:
        print("Error: no arguments necessary")
        exit(1)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"cuda.is_available() = {torch.cuda.is_available()}")
    print(f"device = {device}")

    catmull_rom_basis = torch.Tensor(
        [
            [-0.5, 1.5, -1.5, 0.5],
            [1.0, -2.5, 2.0, -0.5],
            [-0.5, 0.0, 0.5, 0.0],
            [0.0, 1.0, 0.0, 0.0],
        ]
    ).to(device)

    # load dataset
    full_dataset = MocapDataset()

    VAL_DATASET_SIZE = int(len(full_dataset) * VAL_DATASET_FACTOR)

    model = PFNN(full_dataset.X.shape[1], full_dataset.Y.shape[1]).to(device).to(device)

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
    max_epochs_without_improvement = 10
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

        if epochs_without_improvement >= max_epochs_without_improvement:
            print("Early stopping triggered. Stopping training.")
            break

    train_end_time = time.time()
    print(f"Training took {train_end_time - train_start_time} sec")
