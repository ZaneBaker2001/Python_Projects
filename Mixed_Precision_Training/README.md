# Mixed-Precision Training with BIGNET Model

This project explores multiple approaches to compress and fine-tune a large neural network model (BIGNET) using various techniques like quantization, LoRA, and half-precision formats. It also includes tools for evaluation and comparison.

## Requirements

To install the required dependencies, use the following command:
pip3 install -r requirements.txt

## Project Structure

src/
- bignet.py           - Base model definition
- fit.py              - Training logic for binary classifier
- compare.py          - Compare two models' forward passes
- half_precision.py   - LayerNorm module with half-precision handling
- low_precision.py    - 4-bit quantization/dequantization routines
- lora.py             - LoRA-based layer definition
- qlora.py            - Quantized LoRA layer using 4-bit weights
- stats.py            - Model loader and registry
- __init__.py         - Module initializer

## Viewing Statistics

To view the statistics for bignet, type in the following command:
python3 -m src.stats bignet

To view the statistics for half-precision, type in the following command:

python3 -m src.stats bignet half_precision

To compare bignet's statistics with half-precisions', type in the following command:
python3 -m src.compare bignet half_precision

## Lora Adapter

To see if the lora adapter trains, type in the following command:
python3 -m src.fit lora

## Low Precision
After verifying that the lora adapter trains, type in the following command:
python3 -m solution.stats bignet low_precision

This should reveal massive memory savings if everything is working correctly. 
