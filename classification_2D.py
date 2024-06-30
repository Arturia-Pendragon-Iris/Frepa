# -*- coding: utf-8 -*-
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Type
from einops import rearrange
from encoder import *


class conv_block(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(conv_block, self).__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=(3, 3), stride=(1, 1), padding=1, bias=True),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True))
        self.conv2 = nn.Sequential(
            nn.Conv2d(out_ch, out_ch, kernel_size=(3, 3), stride=(1, 1), padding=1, bias=True),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True))

    def forward(self, x):
        x1 = self.conv1(x)
        x2 = x1 + self.conv2(x1)
        return x2

class ViT_2Dclassifer(nn.Module):
    def __init__(self, num_class=2):
        super(ViT_2Dclassifer, self).__init__()
        self.encoder = ViT_encoder()

        self.decoder = nn.Sequential(
            conv_block(in_ch=128, out_ch=64),
            nn.Conv2d(64, 64, kernel_size=(4, 4), stride=(4, 4)))

        self.header = nn.Sequential(
            # nn.AdaptiveAvgPool2d(output_size=(1, 1)),
            nn.Flatten(),
            nn.Linear(in_features=64 * 8 * 8, out_features=256, bias=True),
            nn.LeakyReLU(),
            nn.Linear(in_features=256, out_features=num_class, bias=True),
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)

        return self.header(x)

class SwinT_2Dclassifer(nn.Module):
    def __init__(self, num_class=2):
        super(SwinT_2Dclassifer, self).__init__()
        self.encoder = SwinT_encoder()

        self.decoder = nn.Sequential(
            conv_block(in_ch=1024, out_ch=512),
            nn.Conv2d(512, 512, kernel_size=(4, 4), stride=(4, 4)))

        self.header = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=512 * 4 * 4, out_features=256, bias=True),
            nn.LeakyReLU(),
            nn.Linear(in_features=256, out_features=num_class, bias=True),
        )

    def forward(self, x):
        x = self.encoder(x)
        x = rearrange(x, 'b h w c -> b c h w')
        x = self.decoder(x)
        return self.header(x)