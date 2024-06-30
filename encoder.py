from model import ImageEncoderViT, Frepa_ViT, Frepa_SwinT
import torch
from visualization.view_2D import plot_parallel
from medclip import MedCLIPVisionModelViT, MedCLIPModel
import torch.nn as nn
import torch
import numpy as np
from einops import rearrange
from skimage.filters import frangi


def SwinT_encoder():
    model = Frepa_SwinT(in_chans=4)
    model.load_state_dict(torch.load("/home/checkpoint/Frepa_SwinT_pretrained.pth"))
    model = model.encoder
    return model


def ViT_encoder():
    model = Frepa_ViT(in_chans=4, mid_chans=128)
    model.load_state_dict(torch.load("/home/checkpoint/Frepa_ViT_pretrained.pth"))
    model.cuda()

    model = model.encoder
    return model