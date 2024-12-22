import math

import torch
import torch.nn as nn
import torch.nn.functional as F

NUM_CONTROL_POINTS = 4


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

    def phase_function(self, phase: torch.Tensor) -> torch.Tensor:
        assert self.biases.shape[0] == NUM_CONTROL_POINTS
        assert self.weights.shape[0] == NUM_CONTROL_POINTS
        assert NUM_CONTROL_POINTS == 4, "This code only works for 4 control points"

        batch_size = phase.shape[0]

        # Use a catmull rom basis to interpolate between the control points in a circle.
        # Each phase is is between 0 and 2pi.
        # If 0 <= phase < 3pi/2 we interp between the 0, 1, 2, 3 weights
        # otherwise we interp between 1, 2, 3, 0 weights
        # in order to batch this operation this we do BOTH at once.
        # then use torch.where() to mask the result we want.

        mask = phase < (1.5 * torch.pi)

        # 0 <= phase < 3pi/2
        t0 = phase / (1.5 * torch.pi)
        tt0 = torch.zeros((batch_size, 4), device=self.weights.device)
        tt0[:, 0] = 1.0
        tt0[:, 1] = t0
        tt0[:, 1] = t0**2
        tt0[:, 2] = t0**3
        # Use weights.view to convert weights from (4, N, M) into (4, N * M)
        w0 = tt0 @ self.basis @ self.weights.view(4, -1)
        b0 = tt0 @ self.basis @ self.biases

        # 3pi/2 <= phase <= 2pi
        swizzled_indices = torch.tensor([1, 2, 3, 0])
        t1 = (phase - 0.5 * torch.pi) / (1.5 * torch.pi)
        tt1 = torch.zeros((batch_size, 4), device=self.weights.device)
        tt1[:, 0] = 1.0
        tt1[:, 1] = t1
        tt1[:, 1] = t1**2
        tt1[:, 2] = t1**3
        # Use weights.view to convert weights from (4, N, M) into (4, N * M)
        w1 = tt1 @ self.basis @ self.weights[swizzled_indices].view(4, -1)
        b1 = tt1 @ self.basis @ self.biases[swizzled_indices]

        # pick the appropriate result based on the mask
        w = torch.where(mask.unsqueeze(1), w0, w1)
        b = torch.where(mask.unsqueeze(1), b0, b1)

        # Use w.view to convert from (batch_size, N * M) into (batch_size, N, M)
        w = w.view(batch_size, self.out_features, self.in_features)

        return w, b

    def forward(self, input: torch.Tensor, phase: torch.Tensor) -> torch.Tensor:
        w, b = self.phase_function(phase)

        # F.linear(input, w, b)
        result = w @ input.unsqueeze(-1) + b.unsqueeze(-1)
        result = result.squeeze()

        assert result.shape[0] == input.shape[0], "batch size must be the same"
        assert result.shape[1] == self.out_features, "result must have same size as out_features"

        return result


class PFNN(nn.Module):
    def __init__(self, in_features: int, out_features: int, device=None):
        super(PFNN, self).__init__()

        # catmull rom basis
        self.basis = torch.tensor(
            [[0.0, 2.0, 0.0, 0.0], [-1.0, 0.0, 1.0, 0.0], [2.0, -5.0, 4.0, -1.0], [-1.0, 3.0, -3.0, 1.0]], device=device
        )

        self.fc1 = PhaseLinear(in_features, 512, self.basis, device=device)
        self.fc2 = PhaseLinear(512, 512, self.basis, device=device)
        self.fc3 = PhaseLinear(512, out_features, self.basis, device=device)

    def forward(self, x: torch.Tensor, phase: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.fc1(x, phase))
        x = F.relu(self.fc2(x, phase))
        x = self.fc3(x, phase)
        return x
