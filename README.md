# Frepa
## Visualization
<img src="https://github.com/Arturia-Pendragon-Iris/Frepa/blob/main/fig/restore.png">
Example results of reconstructed images on external datasets. The images are corrupted by random masking and low-frequency filtering, respectively. Notably, such low-frequency filtered images are not seen during the training phases of Frepa. We visualize both the images and their frequency spectrum. RMAE is shown in the upper left corner of each image. Zoom in to an appropriate size for better viewing of the images.

## Datasets 
<img src="https://github.com/Arturia-Pendragon-Iris/Frepa/blob/main/fig/dataset.png" alt="image" style="width: 50%; height: auto;">
This is the summary of the datasets we employed for model pretraining, and their distribution of involved modalities.
Public datasets involved in the pretraining are listed as follows:

**CT:**
[DeepLesion](https://nihcc.app.box.com/v/DeepLesion); [Colorectal-Liver-Metastases](https://www.cancerimagingarchive.net/collection/colorectal-liver-metastases/); [RSNA Intracranial Hemorrhage](https://www.kaggle.com/c/rsna-intracranial-hemorrhage-detection/data); [CQ500](http://headctstudy.qure.ai/dataset); [TCIA HCC-TACE](https://wiki.cancerimagingarchive.net/pages/viewpage.action?pageId=70230229)

**MRI:**
[UCSF-PDGM](https://www.cancerimagingarchive.net/collection/ucsf-pdgm/); [BraTS2021](http://www.braintumorsegmentation.org/); [Duke-Breast-Cancer-MRI](https://www.cancerimagingarchive.net/collection/duke-breast-cancer-mri/); [ACRIN](https://www.cancerimagingarchive.net/collection/acrin-6698/); [Colorectal-Liver-Metastases](https://www.cancerimagingarchive.net/collection/colorectal-liver-metastases/)

**X-ray:**
[CheXpert](https://stanfordmlgroup.github.io/competitions/chexpert/); [MIMIC-CXR](https://physionet.org/content/mimic-cxr/2.0.0/); [NIH Chest](https://www.kaggle.com/datasets/nih-chest-xrays/data); [CRASS12](https://crass.grand-challenge.org/Details/); [COVID-19 Radiography Database](https://drive.google.com/file/d/1xt7g5LkZuX09e1a8rK9sRXIrGFN6rjzl/view); [COVIDGR](https://github.com/ari-dasci/OD-covidgr)

**Ultrasound:**
[BrEaST](https://www.cancerimagingarchive.net/collection/breast-lesions-usg/); [FETAL PLANES](https://zenodo.org/records/3904280); [Ultrasound Nerve Segmentation](https://github.com/piyushagade/Ultrasound-Nerve-Segmentation); [TN3K](https://www.sciencedirect.com/science/article/pii/S0010482522010976); [EBUS](https://zenodo.org/records/4991954); [Micro-Ultrasound Prostate](https://zenodo.org/records/10475293)

**Retina:**:
[ODIR-5K](https://odir2019.grand-challenge.org/dataset/); [AREDS](https://areds.org/); [Diabetic Retinopathy Detection](https://www.kaggle.com/c/diabetic-retinopathy-detection/data)

**Dermoscopy:**
[ISIC](https://challenge.isic-archive.com/data/)

