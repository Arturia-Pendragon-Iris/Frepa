import numpy as np
from visualization.view_2D import *
import torch
from pretrain.losses import Exponential_filter

filter = Exponential_filter(d0=50).cuda()
img = np.load("/data/Train_and_Test/denoise_data/V2303/CT_slice/D_1/000_000009_04_03.npz")["arr_0"]
img = np.clip((img + 1000) / 1400, 0, 1)
img = torch.tensor(img).cuda()
filtered = filter(img)
filtered = filtered.cpu().numpy()
plot_parallel(
    a=filtered
)
