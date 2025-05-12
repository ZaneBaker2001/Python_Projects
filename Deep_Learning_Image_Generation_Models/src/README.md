# Patch-based Image Compression using Autoencoders and Autoregressive Models

This project implements a deep learning pipeline for image compression using a combination of AutoEncoders and Autoregressive models. It is designed for research and educational purposes, showcasing how modern neural architectures can be applied to the task of efficient image representation and reconstruction.

## Features

- Patch-based autoencoding for compressing images.
- Autoregressive modeling to improve entropy estimation.
- Training scripts for building models from scratch.
- Pre-trained model support for quick compression demos.
- Modular design for extending components.

To download the required data for this project, type in the following link: https://drive.google.com/file/d/1p2NrOQrAOoeukF-V3VSPBTVOynFpsU4z/view?usp=drive_link

This takes you to the zip file containing the data. Download it to your local machine and unzip it by typing in the following command in your terminal:
unzip file_directory supertux_data.zip

Replace file_directory with the actual directory containing your file. 

Once unzipped, a folder called data will be outputted. Locate it and drag it into your project's folder. The data has now been correctly inserted into your project. 


## Project Structure

src/
- ae.py # Patch-based autoencoder architecture
- autoregressive.py # Autoregressive entropy model
- bsq.py # BSQ (Band Sequential) compression support
- compress.py # Script for compressing input images
- data.py # Data utilities for preprocessing
- generation.py # Generation utilities (sampling, etc.)
- tokenize.py # Tokenizer for patches
- train.py # Training script
- init.py # Package initialization

## Training and Compression

To train the models, run the following command:

python3 src/train.py

After training, to compress images, run the following command:
python3 src/compress.py --input path-to-image --output path-to-compressed-output

Be sure to replace path-to-image and path-to-compressed-output with your specific file paths.




