# Vision-Language Model Fine-Tuning & QA Generation

## Overview

This project is designed for generating visual question-answer (QA) datasets from structured images and fine-tuning a Vision-Language Model (VLM) to improve its understanding of complex scenes. The dataset appears to involve labeled visual elements commonly found in a kart racing environment, such as karts, track boundaries, and special objects.

The workflow includes generating QA data from annotated images, preparing the dataset, and fine-tuning a base VLM using this data.

## Features

- Generation of image-based QA pairs from object detection metadata.
- Custom base VLM architecture for fine-tuning.
- Dataset preparation and transformation utilities.
- Color-coded object annotation rendering.
- CLI support for automation and scripting via `fire`.

## Directory Structure

homework/
- generate_qa.py # Script to create QA pairs from annotated images
- base_vlm.py # Base class and model definition for the VLM
- finetune.py # Fine-tuning and training logic for the VLM
- data.py # Data preprocessing and loading utilities
- init.py # Package initializer

## Generating the Dataset

To generate the dataset create a directory called data using the following command:
mkdir data

Then generate the dataset using the follwoing command:
python3 src/generate_qa.py generate --input_dir data/train --output_file data/train/qa_pairs.json 

## Training

Then, conduct training using the following command:
python3 -m src/finetune train
