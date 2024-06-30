from file_to_numpy import read_h5_file
import torch.nn.functional as F
# from visualization.view_2D import plot_parallel
# from skimage.transform import iradon, radon
import random
import os
from skimage.filters import frangi
import numpy as np
import torch.nn as nn
from torch.utils.data.dataset import Dataset
import torch
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
    RandAffine)


train_transforms = Compose(
    [RandAffine(
        prob=0.5,
        padding_mode="zeros",
        spatial_size=(512, 512),
        translate_range=(64, 64),
        rotate_range=(-0.1, 0.1),
        scale_range=(-0.4, 0.5)),
     RandFlip(prob=0.5),
     RandRotate(prob=0.8, range_z=0.5),
     RandHistogramShift(prob=0.8, num_control_points=[8, 12]),
    ]
)


def refine_ct(ct_array):
    k = np.random.randint(low=-1000, high=0)
    mu = np.random.randint(low=400, high=1800)
    ct_array = np.clip((ct_array - k) / mu, 0, 1)

    return ct_array


def img_degrade(ct_array):
    k = np.random.uniform(low=0, high=1)
    if k < 0.5:
        ratio = np.random.uniform(low=0.5, high=0.75)
        degrade_ct = img_mask(ct_array, masked_ratio=ratio)
    else:
        ratio = np.random.uniform(low=0.5, high=0.75)
        sigma = abs(np.random.normal(loc=0.0005, scale=0.0001))
        d0 = np.random.uniform(low=0.15, high=0.25)
        degrade_ct = fre_mask(ct_array, masked_ratio=ratio, sigma=sigma, d=d0)
    return degrade_ct


def img_mask(ct_array, masked_ratio=0.6):
    damaged_ct = ct_array.copy()
    for i in range(16):
        for j in range(16):
            k = np.random.uniform(low=0, high=1)
            if k < masked_ratio:
                patch = ct_array[:, 32 * i: 32 * (i + 1), 32 * j: 32 * (j + 1)]
                for chan in range(3):
                    mu = np.mean(patch[chan])
                    sigma = np.std(patch[chan])
                    damaged_ct[chan, 32 * i: 32 * (i + 1), 32 * j: 32 * (j + 1)] = np.random.normal(
                        loc=mu, scale=sigma, size=(32, 32)
                    )

    return damaged_ct


def fre_mask(ct_array, masked_ratio=0.8, sigma=0.001, d=0.2):
    damaged_fre = np.stack((np.fft.fftshift(np.fft.fft2(ct_array[0])),
                            np.fft.fftshift(np.fft.fft2(ct_array[1])),
                            np.fft.fftshift(np.fft.fft2(ct_array[2]))), axis=0)

    mask = np.exp(-((np.arange(512)[:, np.newaxis] - 256) ** 2 +
                    (np.arange(512)[np.newaxis, :] - 256) ** 2) / (2 * (d * 512) ** 2))

    for i in range(32):
        for j in range(32):
            if i == 15 and j == 15:
                continue
            if i == 15 and j == 16:
                continue
            if i == 16 and j == 15:
                continue
            if i == 16 and j == 16:
                continue

            k = np.random.uniform(low=0, high=1)
            if k < masked_ratio:
                patch = ct_array[:, 16 * i: 16 * (i + 1), 16 * j: 16 * (j + 1)]
                for chan in range(3):
                    mu = np.mean(patch[chan])
                    std = np.std(patch[chan])
                    damaged_fre[chan, 16 * i: 16 * (i + 1), 16 * j: 16 * (j + 1)] = np.random.normal(
                        loc=mu, scale=std, size=(16, 16)
                    )
    # plot_parallel(
    #     a=np.abs(np.log(damaged_fre[0] + 1)),
    #     b=np.abs(np.log(damaged_fre[1] + 1)),
    #     c=np.abs(np.log(damaged_fre[2] + 1)),
    # )
    max_signal = np.max(np.abs(damaged_fre))
    disturb = np.random.normal(loc=0, scale=sigma * max_signal, size=[3, 512, 512])
    disturb[:, 256, 256] = 0
    fft_image_filtered = damaged_fre + mask * disturb

    return np.stack((np.fft.ifft2(np.fft.ifftshift(fft_image_filtered[0])).real,
                     np.fft.ifft2(np.fft.ifftshift(fft_image_filtered[1])).real,
                     np.fft.ifft2(np.fft.ifftshift(fft_image_filtered[2])).real), axis=0)


class TrainSetLoader(Dataset):
    def __init__(self, file_list):
        super(TrainSetLoader, self).__init__()
        self.file_list = file_list
        print("HDCT_slice number is", len(self.file_list))

    def __getitem__(self, index):
        img = read_h5_file(self.file_list[index], key="data")
        img = np.clip((img - np.min(img)) / (np.max(img) - np.min(img)), 0, 1)

        if not len(img.shape) == 3:
            img = np.stack((img, ) * 3, axis=-1)

        img = np.transpose(img, (2, 0, 1))[:3]

        img = train_transforms(img[np.newaxis])
        damaged_img = img_degrade(img[0])
        # plot_parallel(
        #     a=ct_slice,
        #     b=damaged_array
        # )
        # print(raw_array.shape, damaged_array.shape)
        raw_tensor = torch.tensor(img).to(torch.float).to("cuda")
        damaged_tensor = torch.tensor(damaged_img[np.newaxis]).to(torch.float).to("cuda")

        return raw_tensor, damaged_tensor

    def __len__(self):
        return len(self.file_list)


class TrainSetLoader_withE(Dataset):
    def __init__(self, total_list):
        super(TrainSetLoader_withE, self).__init__()
        self.file_list = total_list
        self.sample_list = {}
        for key in self.file_list.keys():
            self.sample_list[key] = len(self.file_list[key])
        self.img_number = sum(self.sample_list.values())
        print("Image number is", self.img_number)
        print(self.sample_list)

        # for key in self.file_list.keys():
        #     self.sample_list[key] = 1 / np.sqrt(self.sample_list[key])
        #
        self.modality = list(self.sample_list.keys())
        # self.weight = list(self.sample_list.values())

    def __getitem__(self, index):
        # modality = random.choices(self.modality,
        #                           weights=self.weight,
        #                           k=1)[0]
        # print(self.file_list[modality])
        modality = random.choices(self.modality)[0]
        sample_index = np.random.randint(low=0, high=len(self.file_list[modality]))
        try:
            img = read_h5_file(self.file_list[modality][sample_index], key="data")
        except:
            print(self.file_list[modality][sample_index])

        if not np.max(img) == np.min(img):
            img = np.clip((img - np.min(img)) / (np.max(img) - np.min(img)), 0, 1)

        if not len(img.shape) == 3:
            img = np.stack((img,) * 3, axis=-1)
            # k_color = np.random.uniform(low=0, high=1)
            # if k_color < 0.5:
            #     img = np.stack((img,) * 3, axis=-1)
            # else:
            #     img = np.stack(([img * np.random.uniform(low=0, high=1) for color_i in range(3)]), axis=-1)

        img = np.transpose(img, (2, 0, 1))[:3]

        img = np.array(train_transforms(img))

        edge = frangi(0.299 * img[0] + 0.587 * img[1] + 0.114 * img[2], sigmas=[0.5, 1, 1.5])

        damaged_img = img_degrade(img)
        # damaged_img = img
        # else:
        #     damaged_img = img
        # plot_parallel(
        #     a=np.transpose(img, (1, 2, 0)),
        #     b=np.transpose(damaged_img, (1, 2, 0)),
        #     c=edge
        # )
        # print(raw_array.shape, damaged_array.shape)
        gt_tensor = torch.tensor(img).to(torch.float).to("cuda")
        damaged_tensor = torch.tensor(np.concatenate((damaged_img, edge[np.newaxis]), axis=0)).to(torch.float).to("cuda")
        # print(raw_tensor.shape)
        return damaged_tensor, gt_tensor

    def __len__(self):
        return self.img_number


def do_datalist(root_path):
    file_list = {}
    for file in np.sort(os.listdir(root_path)):
        file_list[file] = []

    for file in np.sort(os.listdir(root_path)):
        # if file == "AIMI":
        #     continue
        path_1 = os.path.join(root_path, file)
        for sub_file in np.sort(os.listdir(path_1)):
            path_2 = os.path.join(path_1, sub_file)

            # print(path_2)
            # if (not os.path.isdir(path_2)) and path_2[-2:] == "h5":
            #     file_list[file].append(path_2)
            # else:
            for sub_sub_file in np.sort(os.listdir(path_2)):
                path_3 = os.path.join(path_2, sub_sub_file)
                if (not os.path.isdir(path_3)) and path_3[-2:] == "h5":
                    file_list[file].append(path_3)


    return file_list



