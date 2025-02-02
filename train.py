import math
import os
import pathlib
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

from pfnn import PFNN

OUTPUT_DIR = pathlib.Path("output")
TRAJ_WINDOW_SIZE = 12
TRAJ_ELEMENT_SIZE = 4


class MocapDataset(torch.utils.data.Dataset):
    def __init__(self):
        # unpickle skeleton, xforms, jointpva
        self.X = torch.load(OUTPUT_DIR / "tensors/X.pth", weights_only=True)
        self.Y = torch.load(OUTPUT_DIR / "tensors/Y.pth", weights_only=True)
        self.P = torch.load(OUTPUT_DIR / "tensors/P.pth", weights_only=True)

        print(f"X.shape {self.X.shape}")
        print(f"Y.shape {self.Y.shape}")
        print(f"P.shape {self.P.shape}")
        assert self.X.shape[0] == self.Y.shape[0]
        assert self.Y.shape[0] == self.P.shape[0]

    def __len__(self):
        return self.X.shape[0] - 1

    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx], self.P[idx]

# hyperparameters
MAX_EPOCHS = 10000
BATCH_SIZE = 512
VAL_DATASET_FACTOR = 0.3  # percentage of data that is reserved for validation set
L1_LAMBDA = 0.000001  # regularization weight
MAX_EPOCHS_WITHOUT_IMPROVEMENT = 10  # early termination
CHECKPOINT_CADENCE = 100
DROPOUT_RATE = 0.3
LEARNING_RATE = 0.001

if __name__ == "__main__":
    if len(sys.argv) != 1:
        print("Error: no arguments necessary")
        exit(1)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"cuda.is_available() = {torch.cuda.is_available()}")
    print(f"device = {device}")

    # load dataset
    full_dataset = MocapDataset()

    VAL_DATASET_SIZE = int(len(full_dataset) * VAL_DATASET_FACTOR)

    model = PFNN(full_dataset.X.shape[1], full_dataset.Y.shape[1], dropout_rate=DROPOUT_RATE, device=device)

    # print model
    print("model =")
    print(model)
    print("parameters =")
    for name, param in model.named_parameters():
        print(f"    {name}, size = {param.size()}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)

    criterion = nn.MSELoss()

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
    train_start_time = time.time()

    writer = SummaryWriter()

    for epoch in range(MAX_EPOCHS):
        # train the model
        train_loss = 0.0
        train_count = 0

        for x, y, p in train_loader:
            x, y, p = x.to(device), y.to(device), p.to(device)

            output = model(x, p)
            l1_reg = sum(param.abs().sum() for param in model.parameters())
            loss = criterion(output, y) + L1_LAMBDA * l1_reg

            writer.add_scalar("Loss/train", loss, epoch)

            train_loss += loss.item()
            train_count += 1

            optimizer.zero_grad()
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
                l1_reg = sum(param.abs().sum() for param in model.parameters())
                loss = criterion(output, y) + L1_LAMBDA * l1_reg
                writer.add_scalar("Loss/validation", loss, epoch)
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

    writer.flush()
    writer.close()
    train_end_time = time.time()
    print(f"Training took {train_end_time - train_start_time} sec")

    # output model
    torch.save(model.state_dict(), OUTPUT_DIR / "final_checkpoint.pth")

