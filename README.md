# Human Face Generation Using DCGAN

This repository contains a Deep Learning project that implements a **Deep Convolutional Generative Adversarial Network (DCGAN)** to generate realistic human face images. The model is trained on the **CelebA dataset** and learns to synthesize new, unseen human faces from random noise.

---

## 📌 Project Overview

Generating realistic human faces is a challenging task due to complex facial structures, textures, and variations.  
This project uses **Generative Adversarial Networks (GANs)**—specifically **DCGANs**—to model the distribution of human facial features and generate high-quality synthetic face images.

### Objectives
- Learn the underlying distribution of human facial features
- Generate realistic and unseen human face images
- Maintain stable adversarial training

---

## 🧠 Model Architecture

### Generator
- Input: 100-dimensional random noise vector
- Dense layer reshaped to `8×8×256`
- Multiple `Conv2DTranspose` layers for upsampling
- Batch Normalization + LeakyReLU
- Final activation: **Tanh**
- Output: `64×64×3` RGB image

### Discriminator
- Multi-layer CNN with increasing filters (64 → 128 → 256 → 512)
- Strided convolutions for downsampling
- LeakyReLU activation
- Dropout for regularization
- Sigmoid output for real/fake classification

---

## 📂 Dataset

- **Dataset:** CelebFaces Attributes (CelebA)
- **Total Images:** ~202,599
- **Image Type:** RGB, aligned celebrity faces
- **Resolution Used:** 64×64

### Preprocessing
- Resize images to 64×64
- Normalize pixel values to range **[-1, 1]**
- Shuffle and batch the dataset
- Prefetch for efficient training

---

## ⚙️ Training Details

- **Loss Function:** Binary Cross-Entropy
- **Optimizer:** Adam  
  - Learning Rate: `0.0002`  
  - Beta1 (β₁): `0.5`
- **Training Tricks:**
  - Label smoothing (real labels = 0.9)
  - Dropout in discriminator
  - Batch normalization in generator
- **Epochs:** ~60
- **Output:** Generated images saved after each epoch

---

## 📊 Results

- Early epochs produced noisy, unstructured images
- Progressive learning of:
  - Facial boundaries and symmetry
  - Skin tone consistency
  - Hair texture and structure
- Later epochs generated recognizable, realistic face-like images
- Stable training with reduced risk of mode collapse

---

## 🚧 Challenges Faced

- Training instability due to adversarial learning
- Risk of mode collapse
- Long training time and GPU dependency
- Sensitivity to hyperparameters


