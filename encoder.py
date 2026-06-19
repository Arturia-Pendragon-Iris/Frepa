import torch
from model import Frepa_ViT, Frepa_SwinT


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
