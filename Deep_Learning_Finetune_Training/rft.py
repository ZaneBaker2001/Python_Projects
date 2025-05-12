import json
from pathlib import Path

import torch
from transformers import TrainingArguments, Trainer
from peft import get_peft_model, LoraConfig, TaskType

from .base_llm import BaseLLM
from .sft import tokenize
from .data import Dataset, benchmark


def load() -> BaseLLM:
    model_name = "rft_model"
    model_path = Path(__file__).parent / model_name
    
    # Load base model first
    llm = BaseLLM()
    
    # Then load the adapter
    from peft import PeftModel
    llm.model = PeftModel.from_pretrained(llm.model, model_path).to(llm.device)
    llm.model.eval()
    
    return llm


class RFTDataset:
    def __init__(self, tokenizer, json_path="data/rft.json"):
        self.tokenizer = tokenizer
        with open(json_path, "r") as f:
            self.data = json.load(f)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        question, answer, cot = self.data[idx]
        # Pass the chain-of-thought as input and the answer as the target
        return tokenize(self.tokenizer, question, cot)


def train_model(output_dir: str, **kwargs):
    # Start with fresh base model
    llm = BaseLLM()

    # Load RFT dataset
    tokenizer = llm.tokenizer
    dataset = RFTDataset(tokenizer)

    # Add LoRA adapter with recommended settings from the assignment
    lora_config = LoraConfig(
        r=16,  # Slightly increased rank for better performance
        lora_alpha=64,  # Maintaining the 4x rank ratio
        target_modules="all-linear",  # Apply to all linear layers
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )
    llm.model = get_peft_model(llm.model, lora_config)
    llm.model.enable_input_require_grads()  # Required for gradient checkpointing

    # Training arguments
    training_args = TrainingArguments(
        output_dir=output_dir,
        logging_dir=output_dir,
        per_device_train_batch_size=32,
        num_train_epochs=5,
        gradient_checkpointing=True,
        learning_rate=1e-3,  # Slightly lower learning rate for more stable training
        report_to="tensorboard",
        save_strategy="epoch",
        logging_strategy="epoch",
        fp16=torch.cuda.is_available() and not torch.backends.mps.is_available(),
    )

    trainer = Trainer(
        model=llm.model,
        args=training_args,
        train_dataset=dataset,
        tokenizer=tokenizer,
    )

    # Train and save
    trainer.train()
    llm.model.save_pretrained(output_dir)
    llm.tokenizer.save_pretrained(output_dir)

    # Run benchmark after training
    print("\nRunning benchmark on validation set...")
    test_model(output_dir)


def test_model(model_path="homework/rft_model"):
    # Load base model first
    llm = BaseLLM()
    
    # Then load the adapter
    from peft import PeftModel
    llm.model = PeftModel.from_pretrained(llm.model, model_path).to(llm.device)
    llm.model.eval()

    valset = Dataset("rft")
    result = benchmark(llm, valset, max_question=100)

    print(f"benchmark_result.accuracy={result.accuracy}  benchmark_result.answer_rate={result.answer_rate}")


if __name__ == "__main__":
    from fire import Fire
    Fire({
        "train": train_model,
        "test": test_model,
        "load": load
    })



