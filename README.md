🎨 Human Face Generation using Deep Convolutional GAN (DCGAN)

This project implements a Deep Convolutional Generative Adversarial Network (DCGAN) to generate realistic human face images using the CelebA dataset.
The GAN architecture includes a Generator network that synthesizes fake images from random noise and a Discriminator network that distinguishes real images from fake ones. Both networks are trained adversarially until the generator produces visually realistic outputs.

🚀 Project Features

Fully implemented DCGAN architecture using TensorFlow/Keras

Custom training loop with tf.GradientTape()

Image normalization & denormalization pipeline

Intermediate image generation after each epoch

Works with CelebA or any directory-structured image dataset

Trains on GPU for high performance

📁 Project Structure
human_face_generation_using_gan.ipynb
generate_face/       -> generated samples saved here
img_align_celeba/     -> dataset folder (CelebA)

📌 Code Notebook

Click below to view the complete notebook on GitHub:
👉 https://github.com/tanikaaaa/DL-PROJECT/blob/main/human_face_generation_using_gan.ipynb

📦 Dataset

This project uses the CelebA Dataset (Aligned & Cropped Faces).

🔗 CelebA Dataset Link:
https://mmlab.ie.cuhk.edu.hk/projects/CelebA.html

Place the dataset folder as:

DATA_DIR = "img_align_celeba"

⚙️ How It Works
1️⃣ Data Loading

Images are loaded using image_dataset_from_directory

Resized to 64×64

Normalized to the range [-1, 1]

2️⃣ Discriminator

4 convolutional blocks (64 → 128 → 256 → 512 filters)

LeakyReLU + Dropout

Final dense layer → real/fake probability

3️⃣ Generator

Dense layer reshaped to (8×8×256)

3 transposed convolution layers to upscale to 64×64

Final activation → tanh

4️⃣ Training Loop

For each batch:

Generator creates fake images

Discriminator classifies real/fake

Losses computed using Binary Cross-Entropy

Both networks updated using Adam optimizer

5️⃣ Image Generation

After every epoch, 16 sample images are saved in:

generate_face/samples_epoch_xxx.png

▶️ How to Run

Install dependencies:

pip install tensorflow matplotlib numpy


Place the dataset folder:

img_align_celeba/


Run the script:

python human_face_generation_using_gan.ipynb   # if using Jupyter


Or run cell-by-cell inside Jupyter Notebook.

📊 Results

The GAN learns to generate increasingly realistic human faces with each epoch.

Early images are noisy; later epochs show:

Defined facial structure

Skin consistency

Hair outlines

Symmetry

🧠 Key Concepts Used

Deep Convolutional GAN

Adversarial training

Binary cross-entropy

Batch normalization

LeakyReLU

GradientTape

✨ Future Improvements

Use Wasserstein GAN (WGAN-GP) for more stable training

Train on higher-resolution CelebA-HQ dataset

Add hyperparameter tuning

Experiment with StyleGAN architecture
