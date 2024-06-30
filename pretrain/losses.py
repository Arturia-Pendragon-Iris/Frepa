import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, List
import torch.fft as fft


class Exponential_filter(nn.Module):
    def __init__(self, shape=(512, 512), d0=10, device="cuda"):
        super(Exponential_filter, self).__init__()

        rows, cols = shape[-2], shape[-1]
        crow, ccol = rows // 2, cols // 2

        x = torch.linspace(0, cols, cols)
        y = torch.linspace(0, rows, rows)

        X, Y = torch.meshgrid(x, y)
        distance = (X - crow) ** 2 + (Y - ccol) ** 2

        self.filter = 1 - torch.exp(-distance / (d0 ** 2))
        if device == "cuda":
            self.filter = self.filter.cuda()

    def forward(self, image):
        dft = torch.fft.fftshift(torch.fft.fft2(image))
        dft_filtered = dft * self.filter
        filtered_image = torch.fft.ifft2(torch.fft.ifftshift(dft_filtered))
        return filtered_image


class HFL_loss(nn.Module):
    def __init__(self, max_levels=5, base_d0=10, loss_type='L1'):
        super(HFL_loss, self).__init__()
        self.max_levels = max_levels
        self.base_d0 = base_d0

        loss_list: List[str] = ['L1', 'L2']
        if loss_type not in loss_list:
            raise ValueError("Invalid loss_type, we expect the following: {0}."
                             "Got: {1}".format(loss_list, loss_type))
        self.loss_type = loss_type
        if self.loss_type == 'L1':
            self.loss = nn.L1Loss()
        else:
            self.loss = nn.MSELoss()

    def forward(self, input, target):
        losses = []
        for x in range(self.max_levels):
            O_filter = Exponential_filter(d0=self.base_d0 * (x + 1))
            losses.append(self.loss(O_filter(input), O_filter(target)))
        return sum(losses)


class grad_loss(nn.Module):
    def __init__(self, loss_type='L1',clip=False,clipmin=0.,clipmax=1.):
        super(grad_loss, self).__init__()
        loss_list: List[str] = ['L1', 'L2']
        if loss_type not in loss_list:
            raise ValueError("Invalid loss_type, we expect the following: {0}."
                             "Got: {1}".format(loss_list, loss_type))
        self.loss_type = loss_type
        if self.loss_type == 'L1':
            self.loss = nn.L1Loss()
        elif self.loss_type == 'L2':
            self.loss = nn.MSELoss()
        self.clip=clip
        self.clipmin = clipmin
        self.clipmax = clipmax

    def forward(self, input, target):
        if self.clip:
            input=torch.clamp(input,self.clipmin,self.clipmax)
        inputx_ = input[:, :, 0:-1, :]-input[:, :, 1:, :]
        inputy_ = input[:, :, :, 0:-1]-input[:, :, :, 1:]
        targetx_ = target[:, :, 0:-1, :]-target[:, :, 1:, :]
        targety_ = target[:, :, :, 0:-1]-target[:, :, :, 1:]

        loss = self.loss(inputx_,targetx_)+self.loss(inputy_,targety_)
        return loss


class image_loss(nn.Module):
    def __init__(self, lamb=1):
        super(image_loss, self).__init__()
        self.lamb = lamb

    def forward(self, input, target):
        loss_mse = nn.MSELoss()
        return torch.sqrt(loss_mse(input, target))

class DiceLoss(nn.Module):
    def __init__(self, smooth=1e-5):
        super(DiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, logits, targets):
        # Apply sigmoid to logits to get probabilities
        probs = torch.sigmoid(logits)
        inter = torch.sum(probs * targets)
        norm = torch.sum(probs * probs) + torch.sum(targets * targets)

        return 1 - 2 * (inter + self.smooth) / (norm + self.smooth)


class MultiClassDiceLoss(nn.Module):
    def __init__(self, num_classes):
        super(MultiClassDiceLoss, self).__init__()
        self.num_classes = num_classes

    def forward(self, preds, labels):
        labels = labels.squeeze(1).long()
        preds = F.softmax(preds, dim=1)
        smooth = 1e-5
        dice_loss = 0.0

        # for c in range(self.num_classes):
        #     pred = preds[:, c]
        #     label = (labels == c).float()
        #
        #     inter = torch.sum(pred * label)
        #     norm = torch.sum(pred) + torch.sum(label)
        #
        #     dice_loss += 1 - (2 * inter + smooth) / (norm + smooth)
        for c in range(self.num_classes):
            pred_c = preds[:, c]
            label_c = (labels == c).float()

            inter = torch.sum(pred_c * label_c, dim=(1, 2))
            norm = torch.sum(pred_c, dim=(1, 2)) + torch.sum(label_c, dim=(1, 2))

            dice_loss += 1 - (2 * inter + smooth) / (norm + smooth)

        return torch.mean(dice_loss)


