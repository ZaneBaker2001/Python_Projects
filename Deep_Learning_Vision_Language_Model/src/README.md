# Vision-Language Model Fine-Tuning & QA Generation

## Overview

This project is designed for generating visual question-answer (QA) datasets from structured images and fine-tuning a Vision-Language Model (VLM) to improve its understanding of complex scenes. The dataset appears to involve labeled visual elements commonly found in a kart racing environment, such as karts, track boundaries, and special objects.

To download the required data for this project, visit the following link in your web browser: https://drive.google.com/file/d/1p2NrOQrAOoeukF-V3VSPBTVOynFpsU4z/view?usp=drive_link

This takes you to the zip file containing the data. Download it to your local machine and unzip it by typing in the following command in your terminal:
unzip supertux_data.zip -d file_directory

Replace file_directory with the actual directory containing your file. 

Once unzipped, a folder called data will be outputted. Locate it and drag it into your project's folder. The data has now been correctly inserted into your project. 

## Features

- Generation of image-based QA pairs from object detection metadata.
- Custom base VLM architecture for fine-tuning.
- Dataset preparation and transformation utilities.
- Color-coded object annotation rendering.
- CLI support for automation and scripting via `fire`.

## Directory Structure

src/
- generate_qa.py # Script to create QA pairs from annotated images
- base_vlm.py # Base class and model definition for the VLM
- finetune.py # Fine-tuning and training logic for the VLM
- data.py # Data preprocessing and loading utilities
- init.py # Package initializer

## Generating the Dataset

To generate the dataset, create a directory that will store the data for training using the following command:
mkdir data/train

Then, generate the dataset using the follwoing command:
python3 src/generate_qa.py generate --input_dir data/train --output_file data/train/qa_pairs.json 

## Training

Then, conduct training using the following command:
python3 -m src/finetune train
