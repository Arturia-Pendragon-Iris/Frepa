import os
import random
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data.dataset import Dataset
from skimage.filters import frangi
from monai.transforms import (
    Compose,
    RandFlip,
    RandRotate90,
    RandHistogramShift,
    RandRotate,
    Rand2DElastic,
    RandGaussianSmooth,
    RandGaussianSharpen,
    RandSpatialCrop,
    RandGaussianNoise,
    RandAffine,
)

from file_to_numpy import read_h5_file


train_transforms = Compose([
    RandAffine(
        prob=0.5,
        padding_mode="zeros",
        spatial_size=(512, 512),
        translate_range=(64, 64),
        rotate_range=(-0.1, 0.1),
        scale_range=(-0.4, 0.5),
    ),
    RandFlip(prob=0.5),
    RandRotate(prob=0.8, range_z=0.5),
    RandHistogramShift(prob=0.8, num_control_points=[8, 12]),
])


def refine_ct(ct_array):
    k = np.random.randint(low=-1000, high=0)
    mu = np.random.randint(low=400, high=1800)
    return np.clip((ct_array - k) / mu, 0, 1)


def img_degrade(ct_array):
    if np.random.uniform() < 0.5:
        ratio = np.random.uniform(low=0.5, high=0.75)
        return img_mask(ct_array, masked_ratio=ratio)
    else:
        ratio = np.random.uniform(low=0.5, high=0.75)
        sigma = abs(np.random.normal(loc=0.0005, scale=0.0001))
        d0 = np.random.uniform(low=0.15, high=0.25)
        return fre_mask(ct_array, masked_ratio=ratio, sigma=sigma, d=d0)


def img_mask(ct_array, masked_ratio=0.6):
    """Randomly replace 32x32 spatial patches with Gaussian noise."""
    damaged_ct = ct_array.copy()
    for i in range(16):
        for j in range(16):
            if np.random.uniform() < masked_ratio:
                patch = ct_array[:, 32 * i: 32 * (i + 1), 32 * j: 32 * (j + 1)]
                for chan in range(3):
                    mu = np.mean(patch[chan])
                    sigma = np.std(patch[chan])
                    damaged_ct[chan, 32 * i: 32 * (i + 1), 32 * j: 32 * (j + 1)] = (
                        np.random.normal(loc=mu, scale=sigma, size=(32, 32))
                    )
    return damaged_ct


# The four center patches (i, j) ∈ {15,16}×{15,16} straddle the DC component
# of the DFT and are preserved to avoid corrupting the image mean.
_FREQ_CENTER_PATCHES = {(15, 15), (15, 16), (16, 15), (16, 16)}


def fre_mask(ct_array, masked_ratio=0.8, sigma=0.001, d=0.2):
    """Randomly corrupt 16x16 frequency-domain patches with Gaussian noise."""
    damaged_fre = np.stack(
        [np.fft.fftshift(np.fft.fft2(ct_array[c])) for c in range(3)], axis=0
    )

    mask = np.exp(
        -((np.arange(512)[:, np.newaxis] - 256) ** 2 +
          (np.arange(512)[np.newaxis, :] - 256) ** 2) / (2 * (d * 512) ** 2)
    )

    for i in range(32):
        for j in range(32):
            if (i, j) in _FREQ_CENTER_PATCHES:
                continue
            if np.random.uniform() < masked_ratio:
                patch = ct_array[:, 16 * i: 16 * (i + 1), 16 * j: 16 * (j + 1)]
                for chan in range(3):
                    mu = np.mean(patch[chan])
                    std = np.std(patch[chan])
                    damaged_fre[chan, 16 * i: 16 * (i + 1), 16 * j: 16 * (j + 1)] = (
                        np.random.normal(loc=mu, scale=std, size=(16, 16))
                    )

    max_signal = np.max(np.abs(damaged_fre))
    disturb = np.random.normal(loc=0, scale=sigma * max_signal, size=[3, 512, 512])
    disturb[:, 256, 256] = 0
    fft_filtered = damaged_fre + mask * disturb

    return np.stack(
        [np.fft.ifft2(np.fft.ifftshift(fft_filtered[c])).real for c in range(3)], axis=0
    )


class TrainSetLoader(Dataset):
    def __init__(self, file_list):
        super().__init__()
        self.file_list = file_list
        print(f"Dataset size: {len(self.file_list)}")

    def __getitem__(self, index):
        img = read_h5_file(self.file_list[index], key="data")
        img = np.clip((img - np.min(img)) / (np.max(img) - np.min(img)), 0, 1)

        if img.ndim == 2:
            img = np.stack((img,) * 3, axis=-1)
        img = np.transpose(img, (2, 0, 1))[:3]

        # train_transforms expects (C, H, W); [0] removes the leading batch dim MONAI adds
        img = np.array(train_transforms(img[np.newaxis]))[0]

        damaged_img = img_degrade(img)

        gt_tensor = torch.tensor(img, dtype=torch.float32)
        damaged_tensor = torch.tensor(damaged_img, dtype=torch.float32)

        return damaged_tensor, gt_tensor

    def __len__(self):
        return len(self.file_list)


class TrainSetLoader_withE(Dataset):
    def __init__(self, total_list):
        super().__init__()
        self.file_list = total_list
        self.sample_list = {key: len(val) for key, val in total_list.items()}
        self.img_number = sum(self.sample_list.values())
        self.modality = list(self.sample_list.keys())
        print(f"Dataset size: {self.img_number}")
        print(self.sample_list)

    def __getitem__(self, index):
        modality = random.choices(self.modality)[0]
        sample_index = np.random.randint(low=0, high=len(self.file_list[modality]))

        try:
            img = read_h5_file(self.file_list[modality][sample_index], key="data")
        except Exception as e:
            print(f"Failed to read {self.file_list[modality][sample_index]}: {e}")
            img = np.zeros((512, 512), dtype=np.float32)

        if np.max(img) != np.min(img):
            img = np.clip((img - np.min(img)) / (np.max(img) - np.min(img)), 0, 1)

        if img.ndim == 2:
            img = np.stack((img,) * 3, axis=-1)
        img = np.transpose(img, (2, 0, 1))[:3]
        img = np.array(train_transforms(img))

        edge = frangi(0.299 * img[0] + 0.587 * img[1] + 0.114 * img[2], sigmas=[0.5, 1, 1.5])
        damaged_img = img_degrade(img)

        gt_tensor = torch.tensor(img, dtype=torch.float32)
        damaged_tensor = torch.tensor(
            np.concatenate((damaged_img, edge[np.newaxis]), axis=0), dtype=torch.float32
        )
        return damaged_tensor, gt_tensor

    def __len__(self):
        return self.img_number


def do_datalist(root_path):
    file_list = {}
    for folder in np.sort(os.listdir(root_path)):
        file_list[folder] = []
        path_1 = os.path.join(root_path, folder)
        for sub_file in np.sort(os.listdir(path_1)):
            path_2 = os.path.join(path_1, sub_file)
            for sub_sub_file in np.sort(os.listdir(path_2)):
                path_3 = os.path.join(path_2, sub_sub_file)
                if not os.path.isdir(path_3) and path_3.endswith(".h5"):
                    file_list[folder].append(path_3)
    return file_list
