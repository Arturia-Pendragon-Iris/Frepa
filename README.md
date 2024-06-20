# Frepa

## Introduction
Frepa is proposed to address the limitations of previous medical foundation models in representing fine-grained information and high-frequency components. Frepa demonstrates that even when using a vanilla Vision Transformer (ViT), foundation models can learn more fine-grained representations through a frequency dual-component masking strategy. Additionally, the equal-histogram image-domain masking extends the Masked Autoencoder (MAE) beyond ViT to other architectures, such as scale-feature networks (e.g., Swin Transformer) and convolutional networks, without requiring modifications to the pretraining pipeline.

Frepa achieves substantial success in various tasks, including vessel segmentation, small lung nodule detection, and image restoration, which were previously considered highly challenging or even inapplicable for foundation models.

* We introduce Frepa, a novel approach that significantly enhances the high-frequency capturing capability of foundation models without compromising their low-frequency representation.
  
* We extend the MAE beyond vanilla ViT to improve its generalizability, as well as enable pretrained 2D encoders directly deployed on volume data.
  
* We develop Frepa on 17 million images and validated it on 32 downstream tasks for both 2D and 3D data, without requiring fine-tuning.
  
* Extensive experiments demonstrate that Frepa outperforms SOTA methods on most downstream tasks, particularly those involving fine-grained information. Frepa also exhibits exceptional performance on previously unseen modalities.

<img src="https://github.com/Arturia-Pendragon-Iris/Frepa/blob/main/fig/main_figure.png" alt="image">

## Pretrained models
| Model      | Pretrained checkpoints | Parameters|
| -----------| -----------------------|-----------|
| ViT-B      | [download](https://drive.google.com/file/d/184NT0mM_dNjU2euQ1vzqukGN7YDcqlLd/view?usp=sharing)  |86.9M      |
| Swin Transformer-B      |[download](https://drive.google.com/file/d/1-7y1yU9pwrl9W2deVx0iY2e2PiqH115c/view?usp=sharing)  |86.9M      |

## Extra experiments on image restoration
|         | MAE              | MFM              | CLIP             | SAM              | Frepa±ViT        | Frepa±SwinT      |
|---------|------------------|------------------|------------------|------------------|------------------|------------------|
| **CT denoise** |                  |                  |                  |                  |                  |                  |
| PSNR    | 30.412±2.501     | 28.152±2.752     | 14.102±1.478     | 21.230±3.093     | 31.625±2.539     | 32.700±1.722     |
| SSIM    | 0.801±0.153      | 0.907±0.024      | 0.370±0.158      | 0.787±0.154      | 0.936±0.032      | 0.954±0.015      |
| RMSE    | 0.077±0.071      | 0.109±0.126      | 0.485±0.144      | 0.207±0.169      | 0.058±0.064      | 0.055±0.039      |
| **CT SR*2**     |                  |                  |                  |                  |                  |                  |
| PSNR    | 30.232±2.704     | 28.065±2.265     | 13.073±2.013     | 21.723±2.469     | 29.800±2.674     | 30.720±2.646     |
| SSIM    | 0.877±0.0454     | 0.867±0.047      | 0.488±0.157      | 0.807±0.077      | 0.898±0.092      | 0.905±0.031      |
| RMSE    | 0.073±0.036      | 0.091±0.041      | 0.558±0.146      | 0.192±0.062      | 0.068±0.034      | 0.068±0.032      |


## Datasets 
<img src="https://github.com/Arturia-Pendragon-Iris/Frepa/blob/main/fig/dataset.png" alt="image" style="width: 50%; height: auto;">
This is the summary of the datasets we employed for model pretraining, and their distribution of involved modalities.
Public datasets involved in the pretraining are listed as follows:

| Modality      | Datasets  |
| -----------   | ----------|
| CT     | [DeepLesion](https://nihcc.app.box.com/v/DeepLesion); [Colorectal-Liver-Metastases](https://www.cancerimagingarchive.net/collection/colorectal-liver-metastases/); [RSNA Intracranial Hemorrhage](https://www.kaggle.com/c/rsna-intracranial-hemorrhage-detection/data); [CQ500](http://headctstudy.qure.ai/dataset); [TCIA HCC-TACE](https://wiki.cancerimagingarchive.net/pages/viewpage.action?pageId=70230229)       |
| MRI    | [UCSF-PDGM](https://www.cancerimagingarchive.net/collection/ucsf-pdgm/); [BraTS2021](http://www.braintumorsegmentation.org/); [Duke-Breast-Cancer-MRI](https://www.cancerimagingarchive.net/collection/duke-breast-cancer-mri/); [ACRIN](https://www.cancerimagingarchive.net/collection/acrin-6698/); [Colorectal-Liver-Metastases](https://www.cancerimagingarchive.net/collection/colorectal-liver-metastases/)        |
|X-ray   |[CheXpert](https://stanfordmlgroup.github.io/competitions/chexpert/); [MIMIC-CXR](https://physionet.org/content/mimic-cxr/2.0.0/); [NIH Chest](https://www.kaggle.com/datasets/nih-chest-xrays/data); [CRASS12](https://crass.grand-challenge.org/Details/); [COVID-19 Radiography Database](https://drive.google.com/file/d/1xt7g5LkZuX09e1a8rK9sRXIrGFN6rjzl/view); [COVIDGR](https://github.com/ari-dasci/OD-covidgr)|
|Ultrasound|[BrEaST](https://www.cancerimagingarchive.net/collection/breast-lesions-usg/); [FETAL PLANES](https://zenodo.org/records/3904280); [Ultrasound Nerve Segmentation](https://github.com/piyushagade/Ultrasound-Nerve-Segmentation); [TN3K](https://www.sciencedirect.com/science/article/pii/S0010482522010976); [EBUS](https://zenodo.org/records/4991954); [Micro-Ultrasound Prostate](https://zenodo.org/records/10475293)|
|Retina  |[ODIR-5K](https://odir2019.grand-challenge.org/dataset/); [AREDS](https://areds.org/); [Diabetic Retinopathy Detection](https://www.kaggle.com/c/diabetic-retinopathy-detection/data)  |
|Dermoscopy|[ISIC](https://challenge.isic-archive.com/data/)|

## Visualization
<img src="https://github.com/Arturia-Pendragon-Iris/Frepa/blob/main/fig/restore.png">
Example results of reconstructed images on external datasets. The images are corrupted by random masking and low-frequency filtering, respectively. Notably, such low-frequency filtered images are not seen during the training phases of Frepa. We visualize both the images and their frequency spectrum. RMAE is shown in the upper left corner of each image. Zoom in to an appropriate size for better viewing of the images.

