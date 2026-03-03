# 🧠 Human Face Generation using DCGAN

This project implements a **Deep Convolutional Generative Adversarial Network (DCGAN)** to generate realistic human face images from random noise.

The model is trained on the **CelebA dataset** and learns the underlying distribution of facial structures, textures, and features to synthesize high-quality, unseen human faces.



## 🚀 Project Overview

Generating realistic human faces is a complex generative modeling problem due to variations in:

- Facial symmetry  
- Skin tone and lighting  
- Hair texture and structure  
- Pose and expression  

This project leverages **Generative Adversarial Networks (GANs)** - specifically DCGAN - to learn the data distribution of aligned celebrity faces and generate synthetic images that resemble real human faces.



## 🎯 Objectives

- Learn deep feature representations of human facial structures  
- Generate high-quality synthetic faces from random noise  
- Achieve stable adversarial training  
- Minimize mode collapse  
- Improve visual realism across epochs  

---

## 🧠 Model Architecture

### 🔷 Generator Network

- Input: 100-dimensional random noise vector (latent space)
- Fully connected layer reshaped to `8×8×256`
- Series of `Conv2DTranspose` layers for progressive upsampling
- Batch Normalization for stable gradients
- LeakyReLU activations
- Final activation: **Tanh**
- Output: `64×64×3` RGB synthetic image

The generator learns to transform random noise into structured facial features.



### 🔶 Discriminator Network

- Deep CNN with increasing filters: `64 → 128 → 256 → 512`
- Strided convolutions for spatial downsampling
- LeakyReLU activations
- Dropout for regularization
- Final sigmoid layer for real/fake classification

The discriminator learns to distinguish between real CelebA images and generated images.


## 📂 Dataset

- **Dataset:** CelebFaces Attributes Dataset (CelebA)
- **Total Images:** ~202,599 aligned face images
- **Image Type:** RGB
- **Resolution Used:** 64×64

### 🔄 Preprocessing Pipeline

- Resize images to 64×64
- Normalize pixel values to range **[-1, 1]**
- Shuffle dataset
- Batch loading for efficient training
- Prefetching to optimize GPU utilization


### 🛠 Training Stabilization Techniques

- Label smoothing (real labels set to 0.9)
- Dropout in discriminator
- Batch normalization in generator
- Careful learning rate tuning

## 📊 Results & Observations

- Early epochs generated noisy and unstructured outputs
- Gradual emergence of:
  - Facial symmetry
  - Defined facial boundaries
  - Hair texture patterns
  - Skin tone consistency
- Later epochs produced visually recognizable face-like images
- Stable adversarial learning with reduced mode collapse

The model successfully captured high-level facial feature distributions despite adversarial training challenges.



## 🚧 Challenges

- GAN training instability  
- Risk of mode collapse  
- Sensitivity to hyperparameters  
- Long training duration  
- High GPU computational requirement  



## 📌 Applications

- Synthetic dataset generation  
- Image augmentation  
- Creative AI applications  
- Deep learning research experiments  
