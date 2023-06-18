# -*- coding: utf-8 -*-
#
# File: modeling.py
# Author: SmileTM
# Site: s-tm.cn
# Github: https://github.com/SmileTM
# Time: 06.10.2023
#

import numpy as np
import torch
import torch.nn as nn

import torch.nn.functional as F
from diffusion_utilities import ResidualConvBlock, UnetDown, UnetUp, EmbedFC


class ContextUnet(nn.Module):
    def __init__(self, in_channels, n_feat=256, n_cfeat=10, height=28):
        super().__init__()
        self.in_channels = in_channels
        self.n_feat = n_feat
        self.n_cfeat = n_cfeat
        self.h = height

        # x [b, 3, 16, 16]
        # self.init_conv  [b,n_feat,16,16]
        self.init_conv = ResidualConvBlock(in_channels, n_feat, is_res=True)
        # self.down1 [b,n_feat,8,8]
        self.down1 = UnetDown(n_feat, n_feat)
        # self.down2 [b,n_feat*2,4,4]
        self.down2 = UnetDown(n_feat, 2 * n_feat)

        # self.to_vec [b,n_feat*2,1,1]
        self.to_vec = nn.Sequential(nn.AvgPool2d(4),
                                    nn.GELU())

        self.timeembed1 = EmbedFC(1, 2 * n_feat)  # [b, n_feat*2]
        self.timeembed2 = EmbedFC(1, 1 * n_feat)  # [b, n_feat*2]

        self.contextembed1 = EmbedFC(n_cfeat, 2 * n_feat)  # [b, n_feat*2]
        self.contextembed2 = EmbedFC(n_cfeat, 1 * n_feat)  # [b, n_feat*2]
        self.up0 = nn.Sequential(
            nn.ConvTranspose2d(2 * n_feat, 2 * n_feat, self.h // 4, self.h // 4),  # (b, n_feat*2, 4, 4)
            nn.GroupNorm(8, 2 * n_feat),
            nn.ReLU()
        )
        self.up1 = UnetUp(4 * n_feat, n_feat)  # [b, n_feat, 8, 8]
        self.up2 = UnetUp(2 * n_feat, n_feat)  # [b, n_feat, 16, 16]

        self.out = nn.Sequential(  # [b, n_feat, 16, 16]
            nn.Conv2d(2 * n_feat, n_feat, 3, 1, 1),
            nn.GroupNorm(8, n_feat),
            nn.ReLU(),
            nn.Conv2d(n_feat, self.in_channels, 3, 1, 1)
        )

        def forward(self, x, t, c=None):
            # x [b,3,16,16]
            x = self.init_conv(x)
            # down1 [b, n_feat, 8, 8]
            down1 = self.down1(x)
            # down2 [b, n_feat*2, 4, 4]
            down2 = self.down2(down1)

            # hiddenvec [b, n_feat*2, 1, 1]
            hiddenvec = self.to_vec(down2)

            if c is None:
                c = torch.zeros(x.shape[0], self.n_cfeat).to(x)

            cemb1 = self.contextembed1(c).view(-1, self.n_feat * 2, 1, 1)
            temb1 = self.timembed1(t).view(-1, self.n_feat * 2, 1, 1)

            cemb2 = self.contextembed2(c).view(-1, self.n_feat, 1, 1)
            temb2 = self.timembed2(t).view(-1, self.n_feat, 1, 1)

            # up1 [b, n_feat*2, 4, 4]
            up1 = self.up0(hiddenvec)
            # up2 [b, n_feat, 8, 8]
            up2 = self.up1(cemb1 * up1 + temb1, down2)
            # up3 [b, n_feat, 16, 16]
            up3 = self.up2(cemb2 * up2 + temb2, down1)
            # out [b, 3, 18, 18], in_channels=3
            out = self.out(torch.cat((up3, x), 1))

            return out
