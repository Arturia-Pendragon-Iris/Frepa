import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List


class Exponential_filter(nn.Module):
    """High-pass filter in the frequency domain using an exponential mask."""

    def __init__(self, shape=(512, 512), d0=10):
        super().__init__()
        rows, cols = shape[-2], shape[-1]
        crow, ccol = rows // 2, cols // 2

        # Use indexing='ij' so Y varies with rows and X varies with cols
        y = torch.linspace(0, rows, rows)
        x = torch.linspace(0, cols, cols)
        Y, X = torch.meshgrid(y, x, indexing='ij')
        distance = (X - ccol) ** 2 + (Y - crow) ** 2

        self.register_buffer('filter', 1 - torch.exp(-distance / (d0 ** 2)))

    def forward(self, image):
        dft = torch.fft.fftshift(torch.fft.fft2(image))
        dft_filtered = dft * self.filter
        return torch.fft.ifft2(torch.fft.ifftshift(dft_filtered)).real


class HFL_loss(nn.Module):
    """Hierarchical Frequency Loss: L1/L2 on filtered outputs at multiple frequency bands."""

    def __init__(self, max_levels=5, base_d0=10, loss_type='L1', shape=(512, 512)):
        super().__init__()
        self.max_levels = max_levels

        if loss_type == 'L1':
            self.loss = nn.L1Loss()
        elif loss_type == 'L2':
            self.loss = nn.MSELoss()
        else:
            raise ValueError(f"Invalid loss_type '{loss_type}', expected one of ['L1', 'L2']")

        self.filters = nn.ModuleList([
            Exponential_filter(shape=shape, d0=base_d0 * (i + 1))
            for i in range(max_levels)
        ])

    def forward(self, input, target):
        return sum(self.loss(f(input), f(target)) for f in self.filters)


class grad_loss(nn.Module):
    """Image gradient loss (finite differences in x and y)."""

    def __init__(self, loss_type='L1', clip=False, clipmin=0., clipmax=1.):
        super().__init__()
        if loss_type == 'L1':
            self.loss = nn.L1Loss()
        elif loss_type == 'L2':
            self.loss = nn.MSELoss()
        else:
            raise ValueError(f"Invalid loss_type '{loss_type}', expected one of ['L1', 'L2']")
        self.clip = clip
        self.clipmin = clipmin
        self.clipmax = clipmax

    def forward(self, input, target):
        if self.clip:
            input = torch.clamp(input, self.clipmin, self.clipmax)
        diff_x = lambda t: t[:, :, :-1, :] - t[:, :, 1:, :]
        diff_y = lambda t: t[:, :, :, :-1] - t[:, :, :, 1:]
        return self.loss(diff_x(input), diff_x(target)) + self.loss(diff_y(input), diff_y(target))


class image_loss(nn.Module):
    """Root mean squared error loss."""

    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss()

    def forward(self, input, target):
        return torch.sqrt(self.mse(input, target))


class DiceLoss(nn.Module):
    def __init__(self, smooth=1e-5):
        super().__init__()
        self.smooth = smooth

    def forward(self, logits, targets):
        probs = torch.sigmoid(logits)
        inter = torch.sum(probs * targets)
        norm = torch.sum(probs * probs) + torch.sum(targets * targets)
        return 1 - 2 * (inter + self.smooth) / (norm + self.smooth)


class MultiClassDiceLoss(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.num_classes = num_classes

    def forward(self, preds, labels):
        labels = labels.squeeze(1).long()
        preds = F.softmax(preds, dim=1)
        smooth = 1e-5
        dice_loss = 0.0

        for c in range(self.num_classes):
            pred_c = preds[:, c]
            label_c = (labels == c).float()
            inter = torch.sum(pred_c * label_c, dim=(1, 2))
            norm = torch.sum(pred_c, dim=(1, 2)) + torch.sum(label_c, dim=(1, 2))
            dice_loss += 1 - (2 * inter + smooth) / (norm + smooth)

        return torch.mean(dice_loss)
