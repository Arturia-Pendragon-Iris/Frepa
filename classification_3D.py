import torch
import torch.nn as nn
from einops import rearrange
from attn_module import I2Transformer
from encoder import ViT_encoder, SwinT_encoder


class ConvBlock3D(nn.Module):
    """Two-branch 3D residual conv block."""

    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv3d(in_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm3d(out_ch),
            nn.ReLU(inplace=True),
        )
        # conv2 takes out_ch (output of conv1), not in_ch
        self.conv2 = nn.Sequential(
            nn.Conv3d(out_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm3d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        x1 = self.conv1(x)
        return x1 + self.conv2(x1)


class UpConv2D(nn.Module):
    """2D upsample neck (applied frame-by-frame to reduce SwinT channel count)."""

    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.up(x)


class ViT_3Dclassifer(nn.Module):
    def __init__(self, mid_chans=128, num_class=2, patch=2, frame=16):
        super().__init__()
        self.encoder = ViT_encoder()

        self.transformer = I2Transformer(
            in_channels=mid_chans,
            num_frames=frame,
            patch_size=patch,
        )
        self.decoder_1 = nn.Conv3d(128, 48, kernel_size=2, stride=2)
        self.decoder_2 = nn.Conv3d(48, 16, kernel_size=2, stride=2)

        self.header = nn.Sequential(
            nn.Flatten(),
            nn.Linear(16 * 8 * 8 * (frame // 4), 256, bias=True),
            nn.LeakyReLU(),
            nn.Linear(256, num_class, bias=True),
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
    def __init__(self, mid_chans=128, num_class=2, patch=2, frame=16):
        super().__init__()
        self.encoder = SwinT_encoder()
        # neck reduces SwinT output (1024ch, channels_last) to mid_chans (channels_first)
        self.neck = UpConv2D(1024, mid_chans)

        self.transformer = I2Transformer(
            in_channels=mid_chans,
            num_frames=frame,
            patch_size=patch,
        )
        self.decoder_1 = nn.Conv3d(mid_chans, 48, kernel_size=2, stride=2)
        self.decoder_2 = nn.Conv3d(48, 16, kernel_size=2, stride=2)

        self.header = nn.Sequential(
            nn.Flatten(),
            nn.Linear(16 * 8 * 8 * (frame // 4), 256, bias=True),
            nn.LeakyReLU(),
            nn.Linear(256, num_class, bias=True),
        )

    def forward(self, x, batch):
        x = rearrange(x, 'b f c h w -> (b f) c h w')
        x = self.encoder(x)
        # SwinT features output is channels_last (BF, H, W, C); convert before neck
        x = rearrange(x, 'bf h w c -> bf c h w')
        x = self.neck(x)
        x = rearrange(x, '(b f) c h w -> b f c h w', b=batch)
        x = self.transformer(x)
        x = rearrange(x, 'b f c h w -> b c h w f')
        x = self.decoder_1(x)
        x = self.decoder_2(x)
        return self.header(x)
