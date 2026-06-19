import torch
import torch.nn as nn
from einops import rearrange
from encoder import ViT_encoder, SwinT_encoder
from model import ImageDecoder


class UpConv2D(nn.Module):
    """Upsample (x2) -> Conv -> BN -> ReLU."""

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


class HAME_2Dimg2img(nn.Module):
    def __init__(self, num_channs=2):
        super().__init__()
        self.encoder = ViT_encoder()
        self.decoder = ImageDecoder(in_chans=128, out_chans=num_channs)

    def forward(self, x):
        return self.decoder(self.encoder(x))


class SwinT_2Dimg2img(nn.Module):
    def __init__(self, num_channs=1):
        super().__init__()
        self.encoder = SwinT_encoder()
        self.neck = UpConv2D(1024, 128)
        self.decoder = ImageDecoder(in_chans=128, out_chans=num_channs)

    def forward(self, x):
        x = self.encoder(x)
        x = rearrange(x, 'b h w c -> b c h w')
        x = self.neck(x)
        return self.decoder(x)
