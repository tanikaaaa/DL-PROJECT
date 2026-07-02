# 🧠 Human Face Generation using DCGAN

A Deep Convolutional Generative Adversarial Network (DCGAN) built using TensorFlow/Keras to generate realistic human face images from random noise. The model is trained on the CelebA dataset and learns the underlying distribution of facial features to synthesize high-quality, previously unseen faces.

---

## 🚀 Project Overview

This project implements a complete DCGAN training pipeline, including data preprocessing, adversarial training, training stabilization techniques, and image generation.

The model progressively learns facial structures, textures, lighting patterns, and expressions through adversarial learning, enabling it to generate realistic synthetic face images from a 100-dimensional latent vector.

---

## 📈 Key Highlights

- Trained a DCGAN on 200K+ CelebA face images
- Implemented an end-to-end TensorFlow training pipeline
- Applied label smoothing, batch normalization, and dropout for stable adversarial training
- Generated realistic 64×64 RGB face images from a 100-dimensional latent space
- Tuned hyperparameters to improve image quality and reduce mode collapse

--- 

## ✨ Features

- Deep Convolutional GAN (DCGAN) architecture
- Generator and Discriminator implemented using TensorFlow/Keras
- Training on the CelebA dataset (200K+ aligned face images)
- Efficient TensorFlow data pipeline with shuffling, batching, and prefetching
- Stable adversarial training using:
  - Label smoothing
  - Batch normalization
  - Dropout
  - Adam optimizer
- Automatic visualization of generated images during training
- Hyperparameter tuning to improve image quality and reduce mode collapse

---

## 🧠 Model Architecture

### Generator

- Input: 100-dimensional latent vector
- Dense projection to 8×8×256 feature map
- Progressive upsampling using Conv2DTranspose layers
- Batch Normalization + LeakyReLU activations
- Tanh output layer
- Output: 64×64×3 RGB image

### Discriminator

- Deep CNN with filters: 64 → 128 → 256 → 512
- Strided convolutions for spatial downsampling
- LeakyReLU activations
- Dropout regularization
- Sigmoid classifier for real/fake prediction

---

## 📂 Dataset

**CelebFaces Attributes Dataset (CelebA)**

- 202,599 aligned celebrity face images
- RGB images resized to 64×64
- Pixel normalization to [-1, 1]
- Shuffle, batching, and prefetching for efficient GPU training

---

## 🛠 Training Stabilization

To improve convergence and reduce GAN instability, the training pipeline incorporates:

- Label smoothing
- Batch normalization
- Dropout regularization
- Adam optimization
- Learning rate tuning

---

## 📊 Results

The model successfully learned meaningful facial feature distributions throughout training.

Observed improvements across epochs include:

- Better facial symmetry
- Clearer facial boundaries
- Improved hair and skin textures
- Higher visual realism
- Reduced mode collapse
- Stable adversarial convergence

---

## 🚧 Challenges

- Adversarial training instability
- Mode collapse prevention
- Hyperparameter sensitivity
- High computational cost
- Long training duration

---

## 🛠 Tech Stack

- Python
- TensorFlow / Keras
- NumPy
- Matplotlib

---

## 📌 Applications

- Synthetic dataset generation
- Data augmentation
- Generative AI research
- Deep learning experimentation
