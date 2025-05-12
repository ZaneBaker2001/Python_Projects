import json
import re
import os
from tqdm import tqdm
import torch

from .cot import CoTModel
from .data import Dataset


def generate_dataset(output_json: str, oversample: int = 20, temperature: float = 0.8):
    oversample = int(oversample)
    temperature = float(temperature)
    cot_model = CoTModel()
    dataset = Dataset("train")
    results = []

    for question, correct_answer in tqdm(dataset, desc="Generating CoT data"):
        try:
            # PROPERLY format the prompt using the CoTModel's formatter
            formatted_prompt = cot_model.format_prompt(question)
            
            # Generate using the batched_generate method which handles tokenization internally
            completions = cot_model.batched_generate(
                [formatted_prompt],  # Pass as list of formatted prompts
                num_return_sequences=oversample,
                temperature=temperature,
            )
            completions = completions[0]  # Get completions for our single question

        except Exception as e:
            print(f"Failed generation for: {question[:50]}... ({e})")
            continue

        found = False
        for completion in completions:
            # Clean the completion by removing the original question
            completion = completion.replace(question, "").strip()
            
            # Extract answer from completion
            match = re.search(r"<answer>(.*?)</answer>", completion)
            if not match:
                continue  # Skip if no answer tag

            try:
                predicted = float(match.group(1).strip())
                
                # Check if answer is correct (with tolerance for floating point)
                if abs(predicted - correct_answer) < 1e-6:
                    # Format the output as specified in instructions
                    formatted_entry = [
                        question,
                        correct_answer,
                        completion
                    ]
                    results.append(formatted_entry)
                    found = True
                    break  # Only keep first correct completion
            except ValueError:
                continue  # Skip if answer isn't a valid float

    # Save results in required format
    os.makedirs(os.path.dirname(output_json), exist_ok=True)
    with open(output_json, "w") as f:
        json.dump(results, f, indent=2)

    success_rate = len(results) / len(dataset)
    print(f"Saved {len(results)} high-quality CoT examples to {output_json}")
    print(f"Success rate: {success_rate:.2%}")
    return success_rate

if __name__ == "__main__":
    from fire import Fire

    Fire(generate_dataset)



