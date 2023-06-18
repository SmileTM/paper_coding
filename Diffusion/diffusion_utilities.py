# -*- coding: utf-8 -*-
#
# File: diffusion_utilities.py
# Author: SmileTM
# Site: s-tm.cn
# Github: https://github.com/SmileTM
# Time: 06.11.2023
#

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class ResidualConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, is_res: bool = False) -> None:
        super().__init__()
        self.is_res = is_res
        self.out_channels = out_channels
        self.in_channels = in_channels
        self.same_channels = in_channels == out_channels

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, 1, 1),
            nn.BatchNorm2d(out_channels),
            nn.GELU(),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, 3, 1, 1),
            nn.BatchNorm2d(out_channels),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1 = self.conv1(x)
        x2 = self.conv2(x1)
        if self.is_res:
            if self.same_channels:
                out = x + x2
            else:
                shortcut = nn.Conv2d(x.shape[1], x2.shape[1], kernel_size=1, stride=1, padding=1).to(x.device)
                out = shortcut(x) + x2
            return out / 1.414
        else:
            return x2

    def get_out_channels(self):
        return self.conv2

    def set_out_channels(self, out_channels):
        self.conv1[0].out_channels = out_channels
        self.conv2[0].in_channels = out_channels
        self.conv2[0].out_channels = out_channels


class UnetUp(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.layers = [
            nn.ConvTranspose2d(in_channels, out_channels, 2, 2),
            ResidualConvBlock(out_channels, out_channels),
            ResidualConvBlock(out_channels, out_channels),
        ]

        self.model = nn.Sequential(*self.layers)

    def forward(self, x, skip):
        x = torch.cat((x, skip), 1)
        x = self.model(x)
        return x


class UnetDown(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.layers = [
            ResidualConvBlock(in_channels, out_channels),
            ResidualConvBlock(out_channels, out_channels),
            nn.MaxPool2d(2)
        ]
        self.model = nn.Sequential(*self.layers)

    def forward(self, x):
        return self.model(x)


class EmbedFC(nn.Module):
    def __init__(self, input_dim, emb_dim):
        super().__init__()
        self.input_dim = input_dim
        self.emb_dim = emb_dim

        self.layers = [
            nn.Linear(self.input_dim, self.emb_dim),
            nn.GELU(),
            nn.Linear(self.emb_dim, self.emb_dim)
        ]
        self.model = nn.Sequential(*self.layers)

    def forward(self, x):
        x = x.view(-1, self.input_dim)
        return self.model(x)


if __name__ == '__main__':
    rcb = ResidualConvBlock(3, 3, True)
    data = torch.rand(10, 3, 16, 16)
    print(rcb(data))
    print(rcb.get_out_channels())
