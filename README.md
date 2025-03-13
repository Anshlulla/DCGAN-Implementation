# Deep Convolutional Generative Adversarial Network (DCGAN)

This repository provides an implementation of a **Deep Convolutional Generative Adversarial Network (DCGAN)** designed for generating realistic facial images using the **CelebA** dataset. The project is developed in **PyTorch** and supports both model training from scratch and fine-tuning of pre-trained models.

## Table of Contents
- [Installation](#installation)
- [Dataset](#dataset)
- [Model Architecture](#model-architecture)
- [Training](#training)
- [Fine-Tuning](#fine-tuning)
- [Generating Images](#generating-images)
- [Results](#results)
- [Acknowledgments](#acknowledgments)

## Installation
Before getting started, ensure that **Python** and **PyTorch** are installed, preferably with **CUDA** support for GPU acceleration. Then, install the necessary dependencies by running:

```sh
pip install torch torchvision tqdm matplotlib
```

## Dataset
The model is trained on the CelebA dataset. Ensure that the dataset is available as a ZIP file named celeba.zip inside the base_dir directory. If the dataset is not already extracted, the script will handle the extraction process automatically.

## Model Architecture
DCGAN consists of two primary components:

### Generator: 
A deep neural network utilizing transposed convolution layers to synthesize realistic facial images from randomly generated noise.
### Discriminator: 
A convolutional neural network responsible for distinguishing real images from artificially generated ones.

## Training
To train the DCGAN model from scratch, execute the following command:

```sh
python train.py
```
The training process includes:

* Initializing the generator and discriminator models
* Employing Binary Cross Entropy (BCE) as the loss function
* Optimizing the networks using the Adam optimizer
* Training over multiple epochs (default: 3 epochs)
* During training, generated images and model checkpoints (generator.pth, discriminator.pth) will be stored in the output directory.

## Fine-Tuning
To further improve a pre-trained model, execute:

```sh
python fine_tune.py
```
This script loads existing model weights and continues training for an additional 10 epochs by default.

## Generating Images
To create new face images using a trained generator, run:

```sh
python generate.py
```
The script will generate and save images in the generated directory, while also displaying them via Matplotlib.

## Results
Generated images are stored in the output and generated directories. The quality of synthesized images improves with additional training epochs.

## Acknowledgments
This implementation is inspired by the DCGAN paper:

* Radford et al., "Unsupervised Representation Learning with Deep Convolutional Generative Adversarial Networks"
* The dataset used: CelebA - Large-scale CelebFaces Attributes Dataset
