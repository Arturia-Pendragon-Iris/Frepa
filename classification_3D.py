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
from attn_module import I2Transformer
from encoder import *

class conv_block(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(conv_block, self).__init__()

        self.conv1 = nn.Sequential(
            nn.Conv3d(in_ch, out_ch, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=1, bias=True),
            nn.BatchNorm3d(out_ch),
            nn.ReLU(inplace=True))
        self.conv2 = nn.Sequential(
            nn.Conv3d(in_ch, out_ch, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=1, bias=True),
            nn.BatchNorm3d(out_ch),
            nn.ReLU(inplace=True))

    def forward(self, x):
        x1 = self.conv1(x)
        x2 = x1 + self.conv2(x1)
        return x2


class up_conv(nn.Module):
    """
    Up Convolution Block
    """

    def __init__(self, in_ch, out_ch):
        super(up_conv, self).__init__()
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.up(x)
        return x


class ViT_3Dclassifer(nn.Module):
    def __init__(self,
                 mid_chans=128,
                 num_class=2,
                 patch=2,
                 frame=16):
        super(ViT_3Dclassifer, self).__init__()
        self.encoder = ViT_encoder()

        self.transformer = I2Transformer(in_channels=mid_chans,
                                         num_frames=frame,
                                         patch_size=patch)
        self.decoder_1 = nn.Conv3d(128, 48, kernel_size=(2, 2, 2), stride=(2, 2, 2))
        self.decoder_2 = nn.Conv3d(48, 16, kernel_size=(2, 2, 2), stride=(2, 2, 2))

        self.header = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=16 * 8 * 8 * (frame // 4), out_features=256, bias=True),
            nn.LeakyReLU(),
            nn.Linear(in_features=256, out_features=num_class, bias=True),
        )

    def forward(self, x, batch):
        x = rearrange(x, 'b f c h w -> (b f) c h w')
        x = self.encoder(x)
        x = rearrange(x, '(b f) c h w -> b f c h w', b=batch)
        x = self.transformer(x)

        x = rearrange(x, 'b f c h w -> b c h w f')

        x = self.decoder_1(x)
        x = self.decoder_2(x)
        return self.header(x)


class SwinT_3Dclassifer(nn.Module):
    def __init__(self,
                 mid_chans=128,
                 num_class=2,
                 patch=2,
                 frame=16):
        super(SwinT_3Dclassifer, self).__init__()
        self.encoder = SwinT_encoder()
        self.neck = up_conv(1024, 128)

        self.transformer = I2Transformer(in_channels=mid_chans,
                                         num_frames=frame,
                                         patch_size=patch)
        self.decoder_1 = nn.Conv3d(128, 48, kernel_size=(2, 2, 2), stride=(2, 2, 2))
        self.decoder_2 = nn.Conv3d(48, 16, kernel_size=(2, 2, 2), stride=(2, 2, 2))

        self.header = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=16 * 8 * 8 * (frame // 4), out_features=256, bias=True),
            nn.LeakyReLU(),
            nn.Linear(in_features=256, out_features=num_class, bias=True),
        )

    def forward(self, x, batch):
        x = rearrange(x, 'b f c h w -> (b f) c h w')
        x = self.encoder(x)
        x = self.neck(x)

        x = rearrange(x, '(b f) c h w -> b f c h w', b=batch)
        x = self.transformer(x)

        x = rearrange(x, 'b f c h w -> b c h w f')

        x = self.decoder_1(x)
        x = self.decoder_2(x)
        return self.header(x)





