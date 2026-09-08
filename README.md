
# VAE / CVAE for Medical Image Reconstruction

A PyTorch implementation of a **Variational Autoencoder (VAE)** and **Conditional Variational Autoencoder (CVAE)** for reconstructing and generating medical images from the [MedMNIST v2](https://medmnist.com/) collection. The project targets stable training, a well-behaved latent space, and class-conditioned generation across multiple imaging modalities.

<img width="4731" height="2036" alt="Reconstruction and generation results" src="https://github.com/user-attachments/assets/1f96d6d5-395a-4a65-8155-8746b4289229" />

---

## 📌 Overview

The notebook trains one unconditional VAE and four class-conditional CVAEs, one per modality:

| Model | Dataset (MedMNIST) | Modality |
|---|---|---|
| VAE | TissueMNIST | Tissue microscopy |
| CVAE | BreastMNIST | Breast ultrasound |
| CVAE | ChestMNIST | Chest X-ray |
| CVAE | PathMNIST | Colorectal histopathology |
| CVAE | DermaMNIST | Dermatoscopy |

All images are resized to 28×28 and normalized to 3 channels. Grayscale sources (breast, chest) are converted to grayscale then replicated across 3 channels so a single architecture handles every dataset.

**Focus areas:**
- Stable training dynamics (no KL collapse)
- Proper KL-divergence weighting (β-VAE style)
- Dataset-specific model choice (plain VAE vs. class-conditional CVAE)
- Latent space inspection via t-SNE

---

## 🏗️ Architecture

Both models share a convolutional encoder/decoder backbone:

**Encoder:** 3 conv blocks (32 → 64 → 128 channels, stride-2/2/1) with BatchNorm + LeakyReLU, flattened and projected to `mu` and `logvar` (latent dim = 128).

**Decoder:** linear projection back to a 128×7×7 feature map, followed by 2 transposed-conv upsampling blocks (128 → 64 → 32) and a final conv + Tanh to 3 channels.

**CVAE** extends the VAE by embedding the class label (embedding dim = 16) and concatenating it with the image (at the encoder input) and with the latent vector `z` (at the decoder input), enabling class-conditioned reconstruction and sampling.

Loss: mean-reduced MSE reconstruction term, scaled by batch size, plus a β-weighted KL-divergence term:

loss = MSE(recon, x) * batch_size + β * KL(mu, logvar)


**Training hyperparameters** (from `config` in the notebook):

| Parameter | Value |
|---|---|
| Batch size | 128 |
| Latent dim (`z_dim`) | 128 |
| Learning rate | 3e-4 (Adam, cosine annealing) |
| Epochs | 20 |
| β (KL weight) | 0.5 |
| Gradient clipping | max norm 1.0 |

---

## 📂 Repository contents

.
├── Copy_of_alter_DGM_trial (1).ipynb # main notebook: data loading, models, training, evaluation
├── cvae_breast.pth # trained CVAE weights — BreastMNIST
├── cvae_chest.pth # trained CVAE weights — ChestMNIST
├── cvae_derma.pth # trained CVAE weights — DermaMNIST
├── cvae_path.pth # trained CVAE weights — PathMNIST
└── README.md


Each `.pth` checkpoint (~11 MB) is a dict with `model_state_dict`, `model_type`, and `dataset` keys.

---

## ⚙️ Installation & Dependencies

Requires **Python 3.9+**.

```bash
git clone https://github.com/nandajana/VAE-for-Medical-Image-Reconstruction-.git
cd VAE-for-Medical-Image-Reconstruction-
pip install -r requirements.txt
```

If you don't have a `requirements.txt` yet, create one with:

torch
torchvision
medmnist
scikit-learn
numpy
matplotlib
tqdm


MedMNIST datasets are downloaded automatically on first run via the `medmnist` package (no manual download needed).

---

## 🖥️ Infrastructure

- **Developed/trained on:** Google Colab (the notebook auto-detects a Colab runtime and adjusts file-saving/download behavior accordingly; it also runs fine in a local Jupyter environment).
- **Hardware:** GPU recommended (CUDA-enabled), but the code falls back to CPU automatically via `torch.device('cuda' if torch.cuda.is_available() else 'cpu')`.
- **Training cost:** ~20 epochs per model; a single epoch over any one MedMNIST split completes in a few minutes on a T4-class GPU.
- **Storage:** each checkpoint is ~11 MB; generated sample images and models are written to `saved_models/` and `generated_images/` (or `models/`, `images/` in the `__main__` block) and can optionally be zipped and downloaded automatically when running in Colab.

---

## 🚀 Usage

Open and run the notebook top to bottom:

```bash
jupyter notebook "Copy_of_alter_DGM_trial (1).ipynb"
```

The `__main__` cell will, in order:
1. Train a standard VAE on **TissueMNIST**.
2. Train a CVAE on each of **BreastMNIST, ChestMNIST, PathMNIST, DermaMNIST**.
3. Save model weights and generated/reconstructed sample images.
4. Plot training/validation loss curves and t-SNE latent-space visualizations.

To reuse a pretrained checkpoint instead of retraining:

```python
import torch
from model import CVAE  # or wherever CVAE is defined

state = torch.load('cvae_chest.pth', map_location='cpu')
model = CVAE(num_classes=<n_classes_for_dataset>, z_dim=128)
model.load_state_dict(state['model_state_dict'])
model.eval()
```

---

## 📊 Results

- Stable convergence within ~20 epochs across all five datasets.
- No KL collapse after tuning the β weight and MSE scaling.
- Class-conditional CVAE samples show visibly better class separation in latent space (verified via t-SNE) than the unconditional VAE.

---

## 🚀 Future Extensions

- Perceptual/LPIPS-based losses for sharper reconstructions
- Higher-resolution MedMNIST variants (e.g. 64×64, 224×224 via `medmnist`'s `size` parameter)
- Alternative conditioning mechanisms (FiLM, cross-attention)
- Comparison against diffusion-based generative baselines

---

## 📖 Citation

This project builds on the **MedMNIST v2** dataset collection. If you use this code or its outputs, please cite the MedMNIST papers:

```bibtex
@article{medmnistv2,
    title={MedMNIST v2-A large-scale lightweight benchmark for 2D and 3D biomedical image classification},
    author={Yang, Jiancheng and Shi, Rui and Wei, Donglai and Liu, Zequan and Zhao, Lin and Ke, Bilian and Pfister, Hanspeter and Ni, Bingbing},
    journal={Scientific Data},
    volume={10},
    number={1},
    pages={41},
    year={2023},
    publisher={Nature Publishing Group UK London}
}

@inproceedings{medmnistv1,
    title={MedMNIST Classification Decathlon: A Lightweight AutoML Benchmark for Medical Image Analysis},
    author={Yang, Jiancheng and Shi, Rui and Ni, Bingbing},
    booktitle={IEEE 18th International Symposium on Biomedical Imaging (ISBI)},
    pages={191--195},
    year={2021}
}
```

Note: DermaMNIST is distributed under CC BY-NC 4.0; the other MedMNIST subsets used here are CC BY 4.0. See the [MedMNIST project page](https://medmnist.com/) for the citation of each underlying source dataset (e.g. the original breast ultrasound, chest X-ray, histopathology, and dermatoscopy datasets).

If you use this repository itself, you can cite it as:

```bibtex
@misc{nandajana_vae_medical_2024,
  title  = {VAE / CVAE for Medical Image Reconstruction},
  author = {nandajana},
  year   = {2024},
  url    = {https://github.com/nandajana/VAE-for-Medical-Image-Reconstruction-}
}
```

---

## License

Add a license (e.g. MIT) if you intend others to reuse this code — none is currently specified in the repository.
