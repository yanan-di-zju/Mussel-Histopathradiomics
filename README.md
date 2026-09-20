# Mussel-Histopathradiomics

**Automated, high-throughput quantitative analysis of mussel tissue pathology using U-Net deep learning segmentation.**

Mussels are widely used as model organisms for marine environmental monitoring. Their gill filament epithelium and digestive gland pathology can indicate the health of marine life — but manual pathological analysis is slow, costly, and subjective. This project applies U-Net-based deep learning to segment pathological images of mussel gill tissue sections and output binary masks for downstream quantitative assessment.

![License](https://img.shields.io/badge/license-CC%20BY%204.0-lightgrey) ![Python](https://img.shields.io/badge/python-3.8%2B-blue) ![TensorFlow](https://img.shields.io/badge/tensorflow-2.8%2B-orange)

## Features

- **Two tissue targets**: gill filament epithelium (`G`) and digestive tubule wall area (`DG`)
- **U-Net variants evaluated with the baseline encoder configuration and VGG16, ResNet34, or EfficientNetB0 backbones
- **512×512 RGB patches** with binary segmentation masks, train/test split
- **Reported evaluation metrics: Accuracy, Dice, Jaccard/IoU, and HD95 (via `medpy`)
- **TensorBoard** logging and best-weight checkpointing

## Dataset (MMHID)

The **M**ussel **M**arine **H**istopathology **I**mage **D**ataset (MMHID) contains paired 512×512 pathological image patches and binary segmentation masks, organized as:

```
MMHID/
├── MMHID-train/   # training split (DG + G, imgs + labels)
└── MMHID-test/    # test split (DG + G, imgs + labels)
```

The dataset is publicly available on **Zenodo** under the CC BY 4.0 license

## Repository Structure

```
.
├── Models for DG/                     # digestive gland U-Net models
│   ├── DG_Unet1_none.ipynb            # plain U-Net
│   ├── DG_Unet2_vgg.ipynb             # VGG16 encoder
│   ├── DG_Unet3_resnet34.ipynb        # ResNet34 encoder ★
│   └── DG_Unet4_efficientnetb0.ipynb  # EfficientNet-B0 encoder
├── Models for G/                      # gill filament U-Net models
│   ├── G_Unet1_none.ipynb             # plain U-Net
│   ├── G_Unet2_vgg.ipynb              # VGG16 encoder
│   ├── G_Unet3_resnet34.ipynb         # ResNet34 encoder ★
│   └── G_Unet4_efficientnetb0.ipynb   # EfficientNet-B0 encoder
├── MMHID/                             # dataset (MMHID-train/ + MMHID-test/)
├── ModelWeights/                      # best-weight checkpoints
└── TECHNICAL_DOCUMENTATION.ipynb         # full technical documentation
└── README.md
```

★ = primary implementations (ResNet34 backbone). The `DG_` prefix = digestive gland; the `G_` prefix = gill filament.

## Quick Start

### 1. Environment

```bash
conda create -n mussel_seg python=3.9
conda activate mussel_seg
pip install tensorflow-gpu==2.8.0 segmentation-models keras-unet-collection numpy matplotlib medpy
```

Requirements: NVIDIA GPU with ≥8 GB VRAM (CUDA 11+), ≥16 GB RAM. 

### 2. Prepare data

Download the dataset from Zenodo and unzip so that the `MMHID/` folder sits in the project root.

### 3. Train

Open `DG_Unet3_resnet34.ipynb` (or `G_Unet3_resnet34.ipynb`) and run all cells sequentially. Best weights are saved to `ModelWeights/`.

```python
# Load data
x_train = np.load('MMHID/MMHID-train/MMHID-train-DG-imgs.npy')
y_train = np.load('MMHID/MMHID-train/MMHID-train-DG-labels.npy')
```

### 4. Predict & evaluate

```python
preds = model.predict(x_test, batch_size=1, verbose=1)
preds[preds > 0.5] = 1
preds[preds < 0.5] = 0

from medpy import metric
dice = metric.binary.dc(preds, y_test)
jc   = metric.binary.jc(preds, y_test)
hd95 = metric.binary.hd95(preds, y_test)
asd  = metric.binary.asd(preds, y_test)
```

## Model

U-Net from `segmentation_models` with a ResNet34 encoder, input `(512, 512, 3)`, binary cross-entropy loss, Adam optimizer, batch size 50, 100 epochs, 40% validation split (~24.5 M parameters). Full architecture and training details: [`TECHNICAL_DOCUMENTATION.ipynb`](TECHNICAL_DOCUMENTATION.ipynb).

## License

This project is released under the [Creative Commons Attribution 4.0 International (CC BY 4.0)](LICENSE) license.
