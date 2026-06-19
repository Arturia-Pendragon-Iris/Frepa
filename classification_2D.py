import torch
import torch.nn as nn
from einops import rearrange
from encoder import ViT_encoder, SwinT_encoder


class ConvBlock2D(nn.Module):
    """Two-branch 2D residual conv block."""

    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        x1 = self.conv1(x)
        return x1 + self.conv2(x1)


class ViT_2Dclassifer(nn.Module):
    def __init__(self, num_class=2):
        super().__init__()
        self.encoder = ViT_encoder()
        self.decoder = nn.Sequential(
            ConvBlock2D(in_ch=128, out_ch=64),
            nn.Conv2d(64, 64, kernel_size=4, stride=4),
        )
        self.header = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 8 * 8, 256, bias=True),
            nn.LeakyReLU(),
            nn.Linear(256, num_class, bias=True),
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return self.header(x)


class SwinT_2Dclassifer(nn.Module):
    def __init__(self, num_class=2):
        super().__init__()
        self.encoder = SwinT_encoder()
        self.decoder = nn.Sequential(
            ConvBlock2D(in_ch=1024, out_ch=512),
            nn.Conv2d(512, 512, kernel_size=4, stride=4),
        )
        self.header = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512 * 4 * 4, 256, bias=True),
            nn.LeakyReLU(),
            nn.Linear(256, num_class, bias=True),
        )

    def forward(self, x):
        x = self.encoder(x)
        x = rearrange(x, 'b h w c -> b c h w')
        x = self.decoder(x)
        return self.header(x)
