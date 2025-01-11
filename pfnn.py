import math
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

NUM_CONTROL_POINTS = 4
NUM_QUADRANTS = 4
NUM_BASIS_FUNCTIONS = 4

class PhaseLinear(nn.Module):
    in_features: int
    out_features: int
    basis: torch.Tensor
    weights: torch.Tensor
    biases: torch.Tensor

    def __init__(self, in_features: int, out_features: int, basis: torch.Tensor, device=None, dtype=None):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()

        self.basis = basis
        self.in_features = in_features
        self.out_features = out_features

        # allocate NUM_CONTROL_POINTS sets of weights and biases
        self.weights = nn.Parameter(torch.empty((NUM_CONTROL_POINTS, out_features, in_features), **factory_kwargs))
        self.biases = nn.Parameter(torch.empty((NUM_CONTROL_POINTS, out_features), **factory_kwargs))

        # Initialize weights and biases
        self.reset_parameters()

    def reset_parameters(self):
        for i in range(NUM_CONTROL_POINTS):
            # taken from torch.nn.Linear.reset_parameters
            nn.init.kaiming_uniform_(self.weights[i], a=math.sqrt(5))
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weights[i])
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.biases[i], -bound, bound)

    def phase_function(self, phase: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        assert self.biases.shape[0] == NUM_CONTROL_POINTS
        assert self.weights.shape[0] == NUM_CONTROL_POINTS
        assert NUM_CONTROL_POINTS == 4, "This code only works for 4 control points"

        batch_size = phase.shape[0]

        # Use a Catmull-Rom splines to interpolate between the control points in a circle.
        # phase is is between 0 and 2pi.
        # Catmull-Rom expects 4 control points, (P_i-1, P_i, P_i+1, P_i+2) and a t between [0, 1]
        # depending on which quadrant of the circle we are on, we choose different control points accordingly.
        # In order to do this in batches we use do all 4 quadrants simultaniously then use buckets to mask the result.

        bounds = torch.tensor([0, 0.5, 1, 1.5, 2]) * torch.pi
        buckets = torch.bucketize(phase, bounds, right=False)
        control_point_indices = torch.tensor([[3, 0, 1, 2], [0, 1, 2, 3], [1, 2, 3, 0], [2, 3, 0, 1]], device=phase.device)

        ws, bs = [], []
        for i in range(NUM_QUADRANTS):
            t = (phase - bounds[i]) / (bounds[i+1] - bounds[i])
            tt = torch.stack([t**3, t**2, t, torch.ones_like(t)], dim=-1)

            indices = control_point_indices[i]

            # Use weights.view to convert weights from (4, N, M) into (4, N * M)
            w = tt @ self.basis @ self.weights[indices].view(4, -1)
            b = tt @ self.basis @ self.biases[indices]

            ws.append(w)
            bs.append(b)

        # Stack results and index the correct one for each phase
        ws = torch.stack(ws, dim=0)
        w = ws[buckets, torch.arange(batch_size)]  # Select the correct result for each phase

        bs = torch.stack(bs, dim=0)  # Shape: (NUM_QUADRANTS, batch_size, ...)
        b = bs[buckets, torch.arange(batch_size)]  # Select the correct result for each phase

        # Use w.view to convert from (batch_size, N * M) into (batch_size, N, M)
        w = w.view(batch_size, self.out_features, self.in_features)
        return w, b

    def forward(self, input: torch.Tensor, phase: torch.Tensor) -> torch.Tensor:

        single_sample = input.ndim == 1
        if single_sample:
            input = input.unsqueeze(0)  # Add batch dimension!
            phase = phase.unsqueeze(0)

        w, b = self.phase_function(phase)

        # F.linear(input, w, b)
        result = w @ input.unsqueeze(-1) + b.unsqueeze(-1)
        result = result.squeeze(-1)

        assert result.shape[0] == input.shape[0], "batch size must be the same"
        assert result.shape[1] == self.out_features, "result must have same size as out_features"

        # If input was non-batched, remove the batch dimension from the output
        if single_sample:
            result = result.squeeze(0)  # Remove batch dimension!

        return result


class PFNN(nn.Module):
    def __init__(self, in_features: int, out_features: int, dropout_rate=0.5, device=None):
        super(PFNN, self).__init__()

        # catmull rom basis
        self.basis = torch.tensor(
            [[0.0, 2.0, 0.0, 0.0], [-1.0, 0.0, 1.0, 0.0], [2.0, -5.0, 4.0, -1.0], [-1.0, 3.0, -3.0, 1.0]], device=device
        )

        self.fc1 = PhaseLinear(in_features, 512, self.basis, device=device)
        self.dropout1 = nn.Dropout(p=dropout_rate)
        self.fc2 = PhaseLinear(512, 512, self.basis, device=device)
        self.dropout2 = nn.Dropout(p=dropout_rate)
        self.fc3 = PhaseLinear(512, out_features, self.basis, device=device)

    def forward(self, x: torch.Tensor, phase: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.fc1(x, phase))
        x = self.dropout1(x)
        x = F.relu(self.fc2(x, phase))
        x = self.dropout2(x)
        x = self.fc3(x, phase)
        return x


def nograd_tensor(array: list):
    return torch.tensor(array, dtype=torch.float32, requires_grad=False)


if __name__ == "__main__":
    # test PhaseLinear module

    torch.no_grad()

    # catmull rom basis
    basis = 0.5 * torch.tensor(
        [[-1.0, 3.0, -3.0, 1.0], [2.0, -5.0, 4.0, -1.0], [-1.0, 0.0, 1.0, 0.0], [0.0, 2.0, 0.0, 0.0]],
        device="cpu",
        requires_grad=False,
    )

    pl = PhaseLinear(2, 2, basis, device="cpu")
    ll = nn.Linear(2, 2, device="cpu")

    with torch.no_grad():

        weights = nograd_tensor([[[0, 0], [1, 0]], [[-1, 0], [0, 0]], [[0, 0], [-1, 0]], [[1, 0], [0, 0]]])
        biases = torch.zeros(4, 2)
        pl.weights.copy_(weights)
        pl.biases.copy_(biases)

        ll.weight.copy_(nograd_tensor([[0, 0], [1, 0]]))
        ll.bias.copy_(torch.zeros(2))

        x = nograd_tensor([[1, 0]])
        phase = nograd_tensor([0])
        y = pl(x, phase)
        print(f"pl({x}, {phase}) = {y}")

        y = ll(x)
        print(f"ll({x}) = {y}")

        phase = nograd_tensor([torch.pi / 2])
        y = pl(x, phase)
        print(f"pl({x}, {phase}) = {y}")

        phase = nograd_tensor([torch.pi])
        y = pl(x, phase)
        print(f"pl({x}, {phase}) = {y}")

        phase = nograd_tensor([3 * torch.pi / 2])
        y = pl(x, phase)
        print(f"pl({x}, {phase}) = {y}")
