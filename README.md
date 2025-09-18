# SynTTA

![Status](https://img.shields.io/badge/Status-Under%20Review-orange)
![License](https://img.shields.io/badge/License-MIT-green)

This is the official PyTorch implementation for our paper, **"ROBUST TEST-TIME ADAPTATION BY UNIFYING PRINCIPLED PRIORS AND ADAPTIVE FEATURE REGULARIZATION"**.

---

## 📢 Important Notice: Code Status

Thank you for your interest in our work!

Our paper is currently under peer review. To maintain the integrity of the review process and prepare for the official release, we are providing a **partial release of the core code** at this time.

**This repository currently includes:**
* The core **model architecture** for `SynTTA`, including our proposed `Online Bayesian Prior Correction (OBPC)` and `Gradient-Modulated Batch Normalization (GMBN)` modules.

**This repository does <u>not</u> currently include:**
* Full training and evaluation scripts
* Data loading and preprocessing pipelines
* Pre-trained model weights

We are committed to releasing the full, reproducible codebase, including all necessary scripts and model weights, **as soon as the paper is accepted**.

Thank you for your understanding and patience!

---
<!-- 
## 🛠️ Setup & Installation

1.  Clone this repository:
    ```bash
    git clone [your_repository_ssh_or_https_link]
    cd [your_repository_name]
    ```

2.  We recommend creating a virtual environment using Conda:
    ```bash
    conda create -n syntta python=3.8
    conda activate syntta
    ```

3.  Install the required dependencies:
    ```bash
    pip install torch torchvision numpy
    # Add other dependencies as needed
    ```

---

## 🚀 Quick Start

As we currently provide only the model structure, you can import and instantiate our model as shown below for understanding and integration.

```python
import torch
from models.syntta import SynTTA  # Assuming your model is defined here

# TODO: Update parameters according to your model definition
# e.g., backbone, num_classes, etc.
model_config = {
    "num_classes": 10,
    "backbone": "resnet50" 
}

# Instantiate the model
model = SynTTA(**model_config)
model.eval()

# Create a dummy input tensor for testing
dummy_input = torch.randn(64, 3, 224, 224) 

# Forward pass
output = model(dummy_input)

print("Model instantiated successfully!")
print("Output tensor shape:", output.shape)
```

---

## Future Plans

-   [ ] Release the full code for training and evaluation upon paper acceptance.
-   [ ] Provide pre-trained model weights for benchmarks like ImageNet-C, ImageNet-R, etc.
-   [ ] Provide detailed instructions to reproduce all experimental results from the paper.

---

## Citation

If you find our work helpful for your research, please consider citing our paper. We will update the BibTeX entry here upon the paper's official publication.

```bibtex
@article{YourLastName2025SynTTA,
  title={[Title of Your Paper]},
  author={[Author One] and [Author Two] and [Author Three]},
  journal={[Journal or Conference Name]},
  year={2025}
}
```

---

## Contact

For any questions, please feel free to contact us via email at `[your.email@institution.edu]` or by opening a GitHub Issue. -->
