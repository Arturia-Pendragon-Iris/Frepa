# Frepa

Official repository for the paper **"Improving Representation of High-Frequency Components for Medical Visual Foundation Models"**
by Yuetan Chu, Yilan Zhang, Zhongyi Han, Changchun Yang, Longxi Zhou, Gongning Luo, Chao Huang, Xin Gao.

Published in *IEEE Transactions on Medical Imaging* (2025). [[Paper]](https://ieeexplore.ieee.org/document/10960415)

---

## Introduction

Medical foundation models pretrained with Masked Autoencoders (MAE) tend to learn low-frequency image features while underrepresenting fine-grained, high-frequency details — which are clinically critical (e.g., vessel boundaries, micro-nodule edges).

**Frepa** (Frequency-domain Pre-training Augmentation) addresses this through two complementary masking strategies:

1. **Frequency dual-component masking** — corrupts images in the frequency domain by masking frequency-domain patches. This forces the encoder to learn to reconstruct high-frequency details that are selectively removed from the spectrum.

2. **Equal-histogram image-domain masking** — replaces spatial patches with Gaussian noise sampled to match the local histogram of each patch. Unlike uniform masking (which is trivially recoverable from context), equal-histogram masking preserves local statistics, creating a harder reconstruction task that generalises MAE-style pretraining to non-ViT architectures (Swin Transformer, ConvNeXt) without modifying their pipelines.

Pretrained on **17 million images** across 7 modalities and validated on **32 downstream tasks** (2D and 3D, zero-shot), Frepa consistently outperforms prior medical foundation models — especially on tasks requiring fine-grained information such as vessel segmentation, small nodule detection, and image restoration.

<img src="https://github.com/Arturia-Pendragon-Iris/Frepa/blob/main/fig/main_figure_2.png" alt="Frepa overview">

---

## Key Results

### Image Restoration (additional experiments)

|                | Metric | MAE          | MFM          | CLIP         | SAM          | Frepa+ViT        | Frepa+SwinT      |
|----------------|--------|--------------|--------------|--------------|--------------|------------------|------------------|
| **CT denoise** | PSNR   | 30.412±2.501 | 28.152±2.752 | 14.102±1.478 | 21.230±3.093 | 31.625±2.539     | **32.700±1.722** |
|                | SSIM   | 0.801±0.153  | 0.907±0.024  | 0.370±0.158  | 0.787±0.154  | 0.936±0.032      | **0.954±0.015**  |
|                | RMSE   | 0.077±0.071  | 0.109±0.126  | 0.485±0.144  | 0.207±0.169  | 0.058±0.064      | **0.055±0.039**  |
| **CT SR ×2**   | PSNR   | 30.232±2.704 | 28.065±2.265 | 13.073±2.013 | 21.723±2.469 | **32.800±2.674** | 32.720±2.646     |
|                | SSIM   | 0.877±0.045  | 0.867±0.047  | 0.488±0.157  | 0.807±0.077  | 0.908±0.092      | **0.915±0.031**  |
|                | RMSE   | 0.073±0.036  | 0.091±0.041  | 0.558±0.146  | 0.192±0.062  | **0.062±0.034**  | 0.061±0.032      |

---

## Pretrained Checkpoints

| Backbone              | Download | Parameters |
|-----------------------|----------|------------|
| ViT-B                 | [Google Drive](https://drive.google.com/file/d/1DFTxIE3XSOQCdUIGcBP8HPMQsDRDwo9W/view?usp=sharing) | 86.9 M |
| Swin Transformer-B    | [Google Drive](https://drive.google.com/file/d/1-7y1yU9pwrl9W2deVx0iY2e2PiqH115c/view?usp=sharing) | 86.9 M |
| ConvNeXt-B            | [Google Drive](https://drive.google.com/file/d/1if2oL5EyDP82g6SR9dU7_TZtpi8Qzx6T/view?usp=sharing) | 87.5 M |

Place downloaded checkpoints in a local `checkpoint/` directory and update the paths in `encoder.py`.

---

## Repository Structure

```
Frepa/
├── model.py              # Core encoder/decoder architectures
│                         #   ImageEncoderViT  – ViT encoder (standard or hierarchical embed)
│                         #   ImageDecoder     – 5-stage upsampling decoder
│                         #   Frepa_ViT / Frepa_SwinT – full pretraining models
├── encoder.py            # Factory functions to load pretrained encoders
├── attn_module.py        # I2Transformer: spatial-temporal transformer for 3D volumes
├── ConvNeXt.py           # ConvNeXt backbone (for direct ConvNeXt pretraining)
│
├── pretrain/
│   ├── train.py          # Pretraining entry point
│   ├── losses.py         # HFL_loss, grad_loss, image_loss
│   └── utils.py          # Dataset loaders & masking strategies (img_mask, fre_mask)
│
├── classification_2D.py  # 2D image classification heads (ViT / SwinT)
├── classification_3D.py  # 3D volume classification via I2Transformer
├── img2img_2D.py         # 2D image-to-image tasks (denoising, super-resolution)
├── evaluation.py         # Metrics: PSNR, SSIM, Dice, sensitivity, specificity
├── file_to_numpy.py      # I/O utilities for NIfTI, DICOM, H5, TIF formats
└── analysis/
    └── filter_2D.py      # Frequency filter analysis tools (Frangi, Jerman, etc.)
```

---

## Installation

**Python 3.9+ and CUDA 11.8+ are recommended.**

```bash
# Clone the repository
git clone https://github.com/Arturia-Pendragon-Iris/Frepa.git
cd Frepa

# Install dependencies
pip install torch torchvision einops monai scikit-image nibabel h5py scipy numpy
```

The full list of tested packages is recorded in `environment.yml`.
---

## Usage

### 1. Load a pretrained encoder

```python
from encoder import ViT_encoder, SwinT_encoder

# ViT-B encoder (outputs feature map: B x 128 x H/16 x W/16)
encoder = ViT_encoder()   # loads from checkpoint/Frepa_ViT_pretrained.pth

# Swin Transformer-B encoder (outputs channels-last: B x H/32 x W/32 x 1024)
encoder = SwinT_encoder() # loads from checkpoint/Frepa_SwinT_pretrained.pth
```

### 2. 2D image classification

```python
from classification_2D import ViT_2Dclassifer, SwinT_2Dclassifer

model = ViT_2Dclassifer(num_class=2)
# Input:  (B, 4, 512, 512)  — 3 image channels + 1 Frangi edge channel
# Output: (B, num_class)
```

### 3. 3D volume classification

```python
from classification_3D import ViT_3Dclassifer, SwinT_3Dclassifer

model = ViT_3Dclassifer(num_class=2, frame=16)
# Input:  x of shape (B, F, C, H, W)  — F frames per volume
# Output: (B, num_class)
logits = model(x, batch=B)
```

The 3D classifier applies the pretrained 2D encoder slice-by-slice, then uses `I2Transformer` to aggregate temporal (slice) context.

### 4. 2D image-to-image (denoising / super-resolution)

```python
from img2img_2D import HAME_2Dimg2img, SwinT_2Dimg2img

model = HAME_2Dimg2img(num_channs=1)  # e.g. single-channel output for CT denoising
# Input:  (B, 4, 512, 512)
# Output: (B, 1, 512, 512)
```

### 5. Pretraining from scratch

```bash
python -m pretrain.train \
    --batchSize 4 \
    --nEpochs 100 \
    --lr 1e-4 \
    --threads 8
```

Edit the data root path in `pretrain/train.py` (`do_datalist(...)` call) to point to your dataset.

---

## Pretraining Loss

Frepa combines three losses during pretraining:

| Loss | Description |
|------|-------------|
| `image_loss` | Root MSE between prediction and clean image |
| `grad_loss` | L1 on spatial image gradients (sharpness) |
| `HFL_loss` | Hierarchical Frequency Loss: L1 in the frequency domain at 5 progressively wider high-pass bands, computed via exponential filters |

```
total = image_loss + grad_loss + 0.2 × HFL_loss
```

The `HFL_loss` is the core supervision signal that drives the encoder to attend to high-frequency components across multiple frequency bands.

---

## Pretraining Data

Pretrained on 17 million images across 7 imaging modalities. Public datasets used:

| Modality    | Datasets |
|-------------|----------|
| CT          | [DeepLesion](https://nihcc.app.box.com/v/DeepLesion) · [Colorectal-Liver-Metastases](https://www.cancerimagingarchive.net/collection/colorectal-liver-metastases/) · [RSNA ICH](https://www.kaggle.com/c/rsna-intracranial-hemorrhage-detection/data) · [CQ500](http://headctstudy.qure.ai/dataset) · [TCIA HCC-TACE](https://wiki.cancerimagingarchive.net/pages/viewpage.action?pageId=70230229) |
| MRI         | [UCSF-PDGM](https://www.cancerimagingarchive.net/collection/ucsf-pdgm/) · [BraTS2021](http://www.braintumorsegmentation.org/) · [Duke Breast MRI](https://www.cancerimagingarchive.net/collection/duke-breast-cancer-mri/) · [ACRIN](https://www.cancerimagingarchive.net/collection/acrin-6698/) |
| X-ray       | [CheXpert](https://stanfordmlgroup.github.io/competitions/chexpert/) · [MIMIC-CXR](https://physionet.org/content/mimic-cxr/2.0.0/) · [NIH Chest](https://www.kaggle.com/datasets/nih-chest-xrays/data) · [CRASS12](https://crass.grand-challenge.org/Details/) |
| Ultrasound  | [BrEaST](https://www.cancerimagingarchive.net/collection/breast-lesions-usg/) · [Fetal Planes](https://zenodo.org/records/3904280) · [TN3K](https://www.sciencedirect.com/science/article/pii/S0010482522010976) · [EBUS](https://zenodo.org/records/4991954) |
| OCT         | [OCT2017](https://www.kaggle.com/datasets/paultimothymooney/kermany2018) · [OCTDL](https://www.nature.com/articles/s41597-024-03182-7) · [CVI-OCT](https://zenodo.org/records/7618624) · [IVOCT](https://zenodo.org/records/3554935) |
| Retina      | [ODIR-5K](https://odir2019.grand-challenge.org/dataset/) · [AREDS](https://areds.org/) · [Diabetic Retinopathy Detection](https://www.kaggle.com/c/diabetic-retinopathy-detection/data) |
| Dermoscopy  | [ISIC](https://challenge.isic-archive.com/data/) |

---

## Visualization

<img src="https://github.com/Arturia-Pendragon-Iris/Frepa/blob/main/fig/restore.png">

Reconstructed images on external (unseen) datasets. Images are corrupted by random spatial masking (top) and low-frequency filtering (bottom). Frepa successfully recovers fine structural details in both cases. RMSE is shown in the upper-left corner of each reconstruction.

<img src="https://github.com/Arturia-Pendragon-Iris/Frepa/blob/main/fig/dataset.png">

---

## Acknowledgements

We thank the authors of [MedSAM](https://github.com/bowang-lab/MedSAM) and [MAE](https://github.com/pengzhiliang/MAE-pytorch) for providing reference implementations.

For questions or issues, contact:
- Yuetan Chu: yuetan.chu@kaust.edu.sa
- Yilan Zhang: yilan.zhang@kaust.edu.sa

---

## Citation

If you find this work useful, please cite:

```bibtex
@article{10960415,
  author  = {Chu, Yuetan and Zhang, Yilan and Han, Zhongyi and Yang, Changchun and
             Zhou, Longxi and Luo, Gongning and Huang, Chao and Gao, Xin},
  journal = {IEEE Transactions on Medical Imaging},
  title   = {Improving Representation of High-frequency Components for Medical Visual Foundation Models},
  year    = {2025},
  pages   = {1--1},
  doi     = {10.1109/TMI.2025.3559402}
}
```
