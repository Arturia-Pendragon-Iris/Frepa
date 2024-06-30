import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from encoder import *
from model import ImageDecoder


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

class HAME_2Dimg2img(nn.Module):
    def __init__(self, num_channs=2):
        super(HAME_2Dimg2img, self).__init__()
        self.encoder = ViT_encoder()
        self.decoder = ImageDecoder(in_chans=128, out_chans=num_channs)

    def forward(self, x):
        x = self.encoder(x)
        return self.decoder(x)


class SwinT_2Dimg2img(nn.Module):
    def __init__(self, num_channs=1):
        super(SwinT_2Dimg2img, self).__init__()
        self.encoder = SwinT_encoder()
        self.neck = up_conv(1024, 128)
        self.decoder = ImageDecoder(in_chans=128, out_chans=num_channs)

    def forward(self, x):
        x = self.encoder(x)
        x = rearrange(x, 'b h w c -> b c h w')
        x = self.neck(x)
        x = self.decoder(x)
        return x