from __future__ import annotations

import torch
import torch.nn as nn

from .action_space import ACTION_DIM
from .state_encoder import NUM_CHANNELS


class ResidualBlock(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.act = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.act(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = out + residual
        return self.act(out)


class DuelingDDQN(nn.Module):
    def __init__(
        self,
        in_channels: int = NUM_CHANNELS,
        action_dim: int = ACTION_DIM,
        channels: int = 128,
        num_blocks: int = 4,
        hidden_dim: int = 512,
    ):
        super().__init__()

        trunk = [
            nn.Conv2d(in_channels, channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
        ]
        trunk.extend(ResidualBlock(channels) for _ in range(num_blocks))
        self.trunk = nn.Sequential(*trunk)

        flat_dim = channels * 10 * 9

        self.value_head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(flat_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, 1),
        )
        self.advantage_head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(flat_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, action_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.trunk(x)
        value = self.value_head(features)
        advantage = self.advantage_head(features)
        return value + advantage - advantage.mean(dim=1, keepdim=True)
