# LLM Training Framework using RFT and SFT 

This project provides a framework for training and evaluating large language models using various fine-tuning strategies including Chain-of-Thought prompting (CoT), Supervised Fine-Tuning (SFT), and Reinforcement Fine-Tuning (RFT). It also supports dataset generation and evaluation pipelines.

## Setup

To install the required dependencies, run the following command:
pip3 install -r requirements.txt

## Project Structure


src/
- base_llm.py           # Base class for language models
- cot.py                # CoT model definition and prompt formatting
- datagen.py            # Script for generating dataset using CoT
- data.py               # Dataset loading and iteration logic
- sft.py                # Supervised fine-tuning implementation
- rft.py                # Reinforcement fine-tuning implementation













To test the cot model (must work in order to train the sft model), type in the following command:
python3 -m src.cot test


To train the sft model, type in the following command: 
python3 -m src.sft train

To generate the dataset to train the rft model with, type in the following command:
python3 -m src.datagen

Once the dataset has been generated, type in the following command to train the rft model:
python3 -m src.rft train
