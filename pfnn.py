#
# Copyright (c) 2025 Anthony J. Thibault
# This software is licensed under the MIT License. See LICENSE for more details.
#

import math
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


NUM_QUADRANTS = 4
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

    def phase_function(self, phase: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        assert self.biases.shape[0] == NUM_CONTROL_POINTS
        assert self.weights.shape[0] == NUM_CONTROL_POINTS
        assert NUM_CONTROL_POINTS == 4, "This code only works for 4 control points"

        B, M, N = phase.shape[0], self.in_features, self.out_features

        assert self.weights.shape == (NUM_CONTROL_POINTS, N, M)
        assert self.biases.shape == (NUM_CONTROL_POINTS, N)
        assert B > 0

        # Use a Catmull-Rom splines to interpolate between the control points in a circle.
        # phase is is between 0 and 2pi.
        # Catmull-Rom expects 4 control points, (P_i-1, P_i, P_i+1, P_i+2) and a t between [0, 1]
        # depending on which quadrant of the circle we are on, we choose different control points accordingly.
        bounds = torch.tensor([0, 0.5, 1, 1.5, 2], device=phase.device) * torch.pi

        control_point_indices = torch.tensor(
            [[3, 0, 1, 2], [0, 1, 2, 3], [1, 2, 3, 0], [2, 3, 0, 1]], device=phase.device
        )

        if B > 1:
            # In order to do this in batches we use do all 4 quadrants simultaniously then mask the result.

            control_point_indices = torch.tensor(
                [[3, 0, 1, 2], [0, 1, 2, 3], [1, 2, 3, 0], [2, 3, 0, 1]], device=phase.device
            )

            masks, ws, bs = [], [], []
            for i in range(NUM_QUADRANTS):
                masks.append(((phase >= bounds[i]) & (phase < bounds[i + 1])).unsqueeze(-1))

                t = (phase - bounds[i]) / (bounds[i + 1] - bounds[i])
                tt = torch.stack([t**3, t**2, t, torch.ones_like(t)], dim=-1)

                indices = control_point_indices[i]

                # Use weights.view to convert weights from (4, N, M) into (4, N * M)
                w = tt @ self.basis @ self.weights[indices].view(4, -1)
                b = tt @ self.basis @ self.biases[indices]

                assert w.shape == (B, M * N), w.shape
                assert b.shape == (B, N), b.shape

                ws.append(w)
                bs.append(b)

            w = torch.where(masks[0], ws[0], torch.where(masks[1], ws[1], torch.where(masks[2], ws[2], ws[3])))
            b = torch.where(masks[0], bs[0], torch.where(masks[1], bs[1], torch.where(masks[2], bs[2], bs[3])))

            # Use w.view to convert from (B, N * M) into (B, N, M)
            w = w.view(B, N, M)

            assert w.shape == (B, N, M), w.shape
            assert b.shape == (B, N), b.shape

        else:
            # as an optimization, we don't have to evaluate all 4 quadrants at once
            i = torch.bucketize(phase, bounds, right=True)[0] - 1
            t = (phase - bounds[i]) / (bounds[i + 1] - bounds[i])
            tt = torch.stack([t**3, t**2, t, torch.ones_like(t)], dim=-1)

            indices = control_point_indices[i]

            # Use weights.view to convert weights from (4, N, M) into (4, N * M)
            w = tt @ self.basis @ self.weights[indices].view(4, -1)
            b = tt @ self.basis @ self.biases[indices]

            # Use w.view to convert from (B, N * M) into (B, M, N)
            w = w.view(B, N, M)

            assert w.shape == (B, N, M), w.shape
            assert b.shape == (B, N), b.shape

        return w, b

    def forward(self, input: torch.Tensor, phase: torch.Tensor) -> torch.Tensor:

        w, b = self.phase_function(phase)

        B = input.shape[0]
        N = self.in_features
        M = self.out_features

        XX = input.unsqueeze(1)
        AA = torch.transpose(w, 1, 2)
        BB = b.unsqueeze(1)

        # print(f"input = {input.shape}")
        # print(f"w = {w.shape}, b = {b.shape}")
        # print(f"XX = {XX.shape} AA = {AA.shape}, BB = {BB.shape}")

        assert XX.shape == (B, 1, N)
        assert AA.shape == (B, N, M)

        # batched version of input @ w^T + b
        result = torch.bmm(XX, AA) + BB

        # print(f"result = {result.shape}")
        assert result.shape == (B, 1, M)

        return result.squeeze(1)


class PFNN(nn.Module):
    def __init__(self, in_features: int, out_features: int, dropout_rate=0.5, device=None):
        super(PFNN, self).__init__()

        # catmull rom basis
        self.basis = 0.5 * torch.tensor(
            [[-1.0, 3.0, -3.0, 1.0], [2.0, -5.0, 4.0, -1.0], [-1.0, 0.0, 1.0, 0.0], [0.0, 2.0, 0.0, 0.0]],
            device=device,
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
    #
    # unit test for PhaseLinear module
    #

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    torch.no_grad()

    # catmull rom basis
    basis = 0.5 * torch.tensor(
        [[-1.0, 3.0, -3.0, 1.0], [2.0, -5.0, 4.0, -1.0], [-1.0, 0.0, 1.0, 0.0], [0.0, 2.0, 0.0, 0.0]],
        device=device,
        requires_grad=False,
    )

    pl = PhaseLinear(2, 2, basis, device=device)

    with torch.no_grad():

        # these weights are carfully constructed so that each weight will return e^(i*phase) when evaulated, with [1, 0] as input.
        weights = nograd_tensor([[[1, 0], [0, 0]], [[0, 0], [1, 0]], [[-1, 0], [0, 0]], [[0, 0], [-1, 0]]]).to(device)
        biases = torch.zeros(4, 2).to(device)
        pl.weights.copy_(weights).to(device)
        pl.biases.copy_(biases).to(device)

        # test N samples in a batch, along the circumference of a cirlce
        N = 32
        x = nograd_tensor([1, 0]).unsqueeze(0).tile(N, 1).to(device)
        assert x.shape == (N, 2), x.shape
        d_theta = (2 * torch.pi) / N
        phase = nograd_tensor([i * d_theta for i in range(N)]).to(device)
        assert phase.shape == (N,)
        y = pl(x, phase).to("cpu")
        assert y.shape == (N, 2)
        epsilon = 0.1
        for i in range(N):
            theta = i * d_theta
            point = nograd_tensor([math.cos(theta), math.sin(theta)])
            assert torch.allclose(y[i], point, atol=epsilon), f"theta = {theta}, y[{i}] = {y[i]}, point = {point}"

        print(f"x =\n{x}")
        print(f"phase =\n{phase}")
        print(f"y =\n{y}")

        # test N samples each in it's own batch of 1, along the circumference of a circle
        for i in range(N):
            theta = i * d_theta
            point = nograd_tensor([math.cos(theta), math.sin(theta)])

            x = nograd_tensor([[1, 0]]).to(device)
            assert x.shape == (1, 2)
            phase = nograd_tensor([theta]).to(device)
            assert phase.shape == (1,)
            y = pl(x, phase).to("cpu")
            assert y.shape == (1, 2)
            assert torch.allclose(y[0], point, atol=epsilon), f"theta = {theta}, y = {y[0]}, point = {point}"
