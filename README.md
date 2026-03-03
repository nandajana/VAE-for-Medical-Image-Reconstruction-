# VAE-for-Medical-Image-Reconstruction-

MedMNIST datasets based Variational Autoencoder (VAE) and Conditional Variational Autoencoder (CVAE) framework for medical image synthesis. This work concentrates on enhancing reconstruction quality, latent space organization and training stability by architectural tuning and loss scaling optimization.

📌 Project Overview
This project explores deep generative modeling for medical imaging using VAE and CVAE architectures. The objective is to generate high-quality synthetic medical images while maintaining meaningful latent representations.

We experimented with multiple medical datasets including:

TissueMNIST

BreastMNIST (MRI)

ChestMNIST (X-ray)

PathMNIST (Pathology)

DermaMNIST (Dermatology)

The project emphasizes:

Stable training dynamics

Proper KL divergence regularization

Dataset-specific model selection (VAE vs CVAE)

Latent space visualization and interpretability

🏗️ Model Architecture
Variational Autoencoder (VAE)
Used for unconditional datasets to capture general structural patterns.

Conditional Variational Autoencoder (CVAE)
Used for labeled datasets to enable class-conditioned generation.

Architectural Enhancements
Deeper encoder–decoder networks

Batch Normalization for stability

LeakyReLU activations

Expanded latent space (128 dimensions)

Structured and modular training pipeline

These refinements significantly improved representation capacity and feature sensitivity.

⚙️ Training & Optimization Improvements
The final model includes several key optimizations:

Properly scaled MSE reconstruction loss

Balanced KL divergence regularization (β-VAE style weighting)

Cosine annealing learning rate scheduler

Gradient clipping for stable updates

Deterministic training with fixed random seeds

Unified training loop across all datasets

This resolved earlier instability issues and prevented KL collapse, leading to smoother convergence and better generalization.

📊 Results & Performance
Stable convergence within ~20 epochs

Strong reconstruction quality across datasets

Improved class separation in CVAE latent spaces

Reduced training variance across experiments

Better structured latent distributions

Latent spaces were visualized using t-SNE, confirming meaningful clustering and coverage for most datasets.

 Key Contributions
Designed a robust VAE/CVAE training framework

Fixed improper loss scaling that caused unstable optimization

Improved latent representation quality

Standardized training for reproducibility

Built a modular and extensible generative pipeline

🚀 Future Extensions
Improve reconstruction sharpness with perceptual losses

Explore advanced conditional mechanisms

Enhance latent space continuity

Apply diffusion-based generative comparisons

💡 Final Outcome
This project demonstrates a strong understanding of:

Variational inference

Generative modeling

Loss balancing in VAEs

Latent space analysis

Experimental reproducibility

The final system provides a stable and extensible foundation for medical image generation research.
