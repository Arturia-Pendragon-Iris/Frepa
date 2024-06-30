import nibabel as nib
# import SimpleITK as sitk
import numpy as np
import cv2
import os
# import pydicom
import h5py
# import pandas as pd
import pickle


def load_nii(path, return_resolution=False):
    file_load = nib.load(path)
    resolution = file_load.header["pixdim"]
    ct_array = file_load.get_fdata()
    # print(np.max(ct_array))
    # for key in file_load.header.keys():
    #     print(key, file_load.header[key])
    # exit()
    if return_resolution:
        return ct_array, [resolution[1], resolution[2], resolution[3]]
    else:
        return ct_array


def read_in_mha(path, return_resolution=False):

    ar = sitk.ReadImage(path)
    img = sitk.GetArrayFromImage(ar)  # z y x
    spacing = ar.GetSpacing()
    # mask = np.swapaxes(mask, 0, 2)
    # mask = np.swapaxes(mask, 0, 1)
    # mask = np.array(mask > 0, 'float32')
    img = np.transpose(img, (1, 2, 0))
    if return_resolution:
        return img, spacing
    else:
        return img


def simple_stack_dcm_files(dcm_dict):
    reader = sitk.ImageSeriesReader()
    dcm_series = reader.GetGDCMSeriesFileNames(dcm_dict)
    reader.SetFileNames(dcm_series)

    img = reader.Execute()
    img_array = sitk.GetArrayFromImage(img)  # z y x
    img_array = np.swapaxes(img_array, 0, 2)
    img_array = np.swapaxes(img_array, 0, 1)

    resolution = get_resolution_from_dcm(dcm_dict)
    example_file = os.path.join(dcm_dict, os.listdir(dcm_dict)[0])
    example_dcm = pydicom.dcmread(example_file)
    pixel_space = example_dcm.PixelSpacing
    slice_space = example_dcm.SliceThickness
    # print(pixel_space, slice_space)
    try:
        resolution = [float(pixel_space[0]), float(pixel_space[1]), float(slice_space)]
    except:
        print("default z-resolution")
        resolution = [float(pixel_space[0]), float(pixel_space[1]), 1.0]

    return img_array, resolution


def simple_stack_dcm_files_2(dcm_path):
    img_array = np.zeros([512, 512, len(os.listdir(dcm_path))])

    file_list = np.sort(os.listdir(dcm_path))
    for i in range(1, len(os.listdir(dcm_path))):
        print(os.path.join(dcm_path, file_list[i]))
        sub_dcm = pydicom.dcmread(os.path.join(dcm_path, file_list[i]))
        pixel_space = sub_dcm.PixelSpacing
        slice_space = sub_dcm.SliceThickness
        # print(np.shape(sub_dcm.pixel_array))
        img_array[:, :, i] = np.array(sub_dcm.pixel_array, "int16") - 1000
        # print(pixel_space, slice_space)
        try:
            resolution = [float(pixel_space[0]), float(pixel_space[1]), float(slice_space)]
        except:
            print("default z-resolution")
            resolution = [float(pixel_space[0]), float(pixel_space[1]), 1.0]

    return img_array, resolution

def get_resolution_from_dcm(dcm_dict):
    example_0 = os.path.join(dcm_dict, np.sort(os.listdir(dcm_dict))[1])
    example_1 = os.path.join(dcm_dict, np.sort(os.listdir(dcm_dict))[2])
    # print(example_0)
    example_dcm_0 = pydicom.dcmread(example_0)
    example_dcm_1 = pydicom.dcmread(example_1)
    modality = example_dcm_0.Modality
    # print(modality)
    assert modality == "CT"

    pixel_space = example_dcm_0.PixelSpacing
    try:
        thickness = example_dcm_1.SliceThickness
        resolution = [float(pixel_space[0]), float(pixel_space[1]), float(thickness)]
    except:
        slice_0 = example_dcm_0.ImagePositionPatient[-1]
        slice_1 = example_dcm_1.ImagePositionPatient[-1]
        # print(slice_0, slice_1)
        # print(pixel_space, slice_space)
        resolution = [float(pixel_space[0]), float(pixel_space[1]),
                  abs(float(slice_1) - float(slice_0))]

    return resolution


def read_h5_file(path, key):
    with h5py.File(path, 'r') as f:
        array = np.array(f[key])

    return array


def read_tif(path):
    tif_img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    return np.array(tif_img, "float32")


def read_csv(csv_path=".csv"):
    df = pd.read_csv(csv_path)
    return np.array(df, "float32")


def read_pickle(fanme):
    with open(fanme, "rb") as f:
        return pickle.load(f)
