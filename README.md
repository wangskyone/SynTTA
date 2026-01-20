# Robust Test-time Adaptation by Unifying Principled Priors and Adaptive Feature Regularization

![Status](https://img.shields.io/badge/Status-Accepted%20at%20ICASSP2026-success)
![License](https://img.shields.io/badge/License-MIT-green)

## 📜 Paper Information

**Title**: Robust Test-time Adaptation by Unifying Principled Priors and Adaptive Feature Regularization

**Authors**: Tianyi Wang, Xu Gu, Yangge Qian, Jingyang Shan, Xiaolin Qin

**Conference**: ICASSP 2026 (Accepted) - [Official Accepted Papers List](https://cmsworkshops.com/ICASSP2026/papers/accepted_papers.php)

This is the official PyTorch implementation of SynTTA, a robust test-time adaptation method that unifies principled priors and adaptive feature regularization.

---

## 🔑 Key Features

SynTTA introduces two novel components:
- **Online Bayesian Prior Correction (OBPC)**: Dynamically corrects prior distributions during test-time adaptation
- **Gradient-Modulated Batch Normalization (GMBN)**: Adapts feature statistics with gradient-based modulation

---

## 🚀 Quick Start

### Basic Usage

Run SynTTA on different datasets with the following command format:

```bash
python test_time.py --cfg cfgs/[DATASET]/syntta.yaml SETTING [SETTING_TYPE]
```

### Examples

**ImageNet-C:**
```bash
python test_time.py --cfg cfgs/imagenet_c/syntta.yaml SETTING correlated
```

**CIFAR10-C:**
```bash
python test_time.py --cfg cfgs/cifar10_c/syntta.yaml SETTING correlated
```

**CIFAR100-C:**
```bash
python test_time.py --cfg cfgs/cifar100_c/syntta.yaml SETTING correlated
```

**ImageNet-R:**
```bash
python test_time.py --cfg cfgs/imagenet_r/syntta.yaml SETTING correlated
```

### Settings Explanation

The `SETTING` parameter controls how the model adapts to domain shifts:

- `correlated`: Practical test-time adaptation with sorted class labels
- `continual`: Train on sequence of domain shifts without knowing when shifts occur
- `reset_each_shift`: Reset model state after adaptation to each domain
- `gradual`: Sequence of gradually increasing/decreasing domain shifts
- `mixed_domains`: Consecutive test samples from different domains

---

## 📋 Prerequisites

```bash
# Python 3.8+
pip install torch torchvision
pip install timm
pip install loguru
pip install yacs
pip install iopath
```

See `requirements.txt` for complete dependency list.

---

## ⚙️ Configuration

### Step 1: Set Up Data Directory

Edit `conf.py` to configure your dataset paths:

```python
# Line ~44 - Update to your dataset directory
_C.DATA_DIR = "/path/to/your/datasets"

# Line ~47 - Update to your checkpoint directory  
_C.CKPT_DIR = "/path/to/your/checkpoints"
```

**Important**: Replace the placeholder paths with your actual directory paths!

### Step 2: Download Datasets

**For CIFAR-10-C and CIFAR-100-C**:  
These datasets will be **automatically downloaded** by RobustBench when you first run the code. No manual download required!

**For ImageNet-C**:  
Download manually from **[Zenodo](https://zenodo.org/)**:
- Search for "ImageNet-C" on Zenodo
- Or visit: [Hendrycks & Dietterich (2019)](https://zenodo.org/record/2535967)
- Official repository: https://github.com/hendrycks/robustness

Place the downloaded datasets in your configured `DATA_DIR` following the structure below.

### Step 3: Dataset Organization

Organize your datasets according to the following structure:

```
YOUR_DATASET_PATH/
├── CIFAR/
│   ├── cifar10/
│   │   ├── cifar-10-batches-py/
│   │   └── CIFAR-10-C/
│   └── cifar100/
│       ├── cifar-100-python/
│       └── CIFAR-100-C/
├── ImageNet/
│   ├── imagenet-c/
│   ├── imagenet-r/
│   └── imagenet2012/
└── [other datasets]
```

### Step 4: Download Pre-trained Models

**Checkpoint download [link](https://drive.google.com/drive/folders/1xt4N9Y6NV_u-5QLfJYUZx-mIN2kv0lb6?usp=sharing)**

**Note**: You can also download pre-trained models from [RobustBench](https://github.com/RobustBench/robustbench).
After downloading, organize checkpoints in this structure:


```
YOUR_CKPT_PATH/
├── cifar10/corruptions/
│   └── natural.pt.tar
├── cifar100/corruptions/
│   └── Hendrycks2020AugMix_ResNeXt.pt
└── [other datasets]
```

**Note**: If you use different checkpoint names, update the `CKPT_PATH` parameter in the config files accordingly.

---

## 📁 Project Structure

```
SynTTA/
├── cfgs/                    # Configuration files for different datasets
│   ├── imagenet_c/
│   ├── cifar10_c/
│   ├── cifar100_c/
│   └── imagenet_r/
├── datasets/                 # Data loading utilities
├── methods/                  # Test-time adaptation methods
│   └── syntta.py           # SynTTA implementation
├── models/                   # Model architectures
│   ├── model.py            # Model loading utilities
│   └── resnet26.py         # CIFAR models
├── robustbench/            # RobustBench integration
├── test_time.py            # Main evaluation script
├── conf.py                 # Configuration management
└── README.md
```

---

## 🔬 Supported Datasets

- **CIFAR-10-C**: 15 corruptions × 5 severity levels
- **CIFAR-100-C**: 15 corruptions × 5 severity levels  
- **ImageNet-C**: 15 corruptions × 5 severity levels
- **ImageNet-R**: Renditions and artistic images

### Dataset Download Sources

**CIFAR-10-C and CIFAR-100-C**:  
These datasets can be automatically downloaded from RobustBench. The code will download them automatically when first run if they are not found in the specified directory.

**ImageNet-C**:  
Download from [Zenodo](https://zenodo.org/). Search for "ImageNet-C" or use the following common sources:
- [Hendrycks & Dietterich (2019)](https://zenodo.org/records/2235448)
- Official repository: https://github.com/hendrycks/robustness

**Note**: Ensure the downloaded datasets are placed in the correct directory structure as shown in the Configuration section.
---

## 📝 Citation

```bibtex
@inproceedings{wang2026syntta,
  title={Robust Test-time Adaptation by Unifying Principled Priors and Adaptive Feature Regularization},
  author={Wang, Tianyi and Gu, Xu and Qian, Yangge and Shan, Jingyang and Qin, Xiaolin},
  booktitle={ICASSP 2026 - 2026 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)},
  year={2026}
}
```

---

## 🤝 Acknowledgments

This codebase builds upon several open-source projects:
- [Tent](https://github.com/Deeplearning-SciTokyo/Tent)
- [RobustBench](https://github.com/RobustBench/robustbench)
- [PALM](https://github.com/sarthaxxxxx/PALM)

We thank the authors for their valuable contributions.

---

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

---

## 📧 Contact

For questions and suggestions, please contact:
- Tianyi Wang (heuwty@gmail.com)
- Or open an issue on GitHub

---

## ⚠️ Important Notes

1. **Path Configuration**: You must update `DATA_DIR` and `CKPT_DIR` in `conf.py` before running the code
2. **Checkpoint Availability**: Pre-trained checkpoints are not included in this repository and must be downloaded separately
3. **GPU Requirement**: This implementation requires CUDA-compatible GPU for efficient execution
4. **Memory Usage**: Batch size may need adjustment based on available GPU memory

---

## 🔄 Troubleshooting

**Issue**: `FileNotFoundError: [Errno 2] No such file or directory`

**Solution**: Ensure your `DATA_DIR` and `CKPT_DIR` paths are correctly set in `conf.py` and the dataset/checkpoint files exist at those locations.

**Issue**: `RuntimeError: CUDA out of memory`

**Solution**: Reduce `TEST.BATCH_SIZE` in the config file (e.g., from 64 to 32 or 16).

**Issue**: Model loading fails

**Solution**: Verify checkpoint paths match the expected structure and filenames in `conf.py`.
