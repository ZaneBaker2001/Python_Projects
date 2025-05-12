# Patch-based Image Compression using Autoencoders and Autoregressive Models

This project implements a deep learning pipeline for image compression using a combination of AutoEncoders and Autoregressive models. It is designed for research and educational purposes, showcasing how modern neural architectures can be applied to the task of efficient image representation and reconstruction.

## Features

- Patch-based autoencoding for compressing images.
- Autoregressive modeling to improve entropy estimation.
- Training scripts for building models from scratch.
- Pre-trained model support for quick compression demos.
- Modular design for extending components.

## Project Structure

src/
├── ae.py # Patch-based autoencoder architecture
├── autoregressive.py # Autoregressive entropy model
├── bsq.py # BSQ (Band Sequential) compression support
├── compress.py # Script for compressing input images
├── data.py # Data utilities for preprocessing
├── generation.py # Generation utilities (sampling etc.)
├── tokenize.py # Tokenizer for patches
├── train.py # Training script
├── init.py

## Installation

To train the models, run the following command:

python3 src/train.py

After training, to compress images, run the following command:
python3 src/compress.py --input path-to-image --output path-to-compressed-output

Be sure to replace path-to-image and path-to-compressed-output with your specific file paths.




