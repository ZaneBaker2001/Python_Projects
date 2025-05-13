from .base_llm import BaseLLM
from .data import Dataset, benchmark


def load() -> BaseLLM:
    from pathlib import Path
    from peft import PeftModel

    model_name = "sft_model"
    model_path = Path(__file__).parent / model_name

    llm = BaseLLM()
    llm.model = PeftModel.from_pretrained(llm.model, model_path).to(llm.device)
    llm.model.eval()

    return llm


def tokenize(tokenizer, question: str, answer: str):
    full_text = f"{question} {answer}{tokenizer.eos_token}"

    tokenizer.padding_side = "right"
    tokenizer.pad_token = tokenizer.eos_token
    full = tokenizer(full_text, padding="max_length", truncation=True, max_length=128)

    input_ids = full["input_ids"]
    attention_mask = full["attention_mask"]

    # Determine question length based on where the answer begins
    try:
        split_idx = full_text.index(answer)
        prefix = full_text[:split_idx]
        question_len = len(tokenizer(prefix, add_special_tokens=False)["input_ids"])
    except ValueError:
        question_len = len(tokenizer(question, add_special_tokens=False)["input_ids"])

    labels = [-100] * len(input_ids)
    for i in range(question_len, len(input_ids)):
        if attention_mask[i] == 1:
            labels[i] = input_ids[i]

    full["labels"] = labels
    return full


def format_example(prompt: str, answer: float) -> dict[str, str]:
    # Round answer to a reasonable precision and format it with <answer> tags
    formatted_answer = f"<answer>{round(answer, 2)}</answer>"
    return {"question": prompt, "answer": formatted_answer}


class TokenizedDataset:
    def __init__(self, tokenizer, data: Dataset, format_fn):
        self.format_fn = format_fn
        self.tokenizer = tokenizer
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        formatted_data = self.format_fn(*self.data[idx])
        return tokenize(self.tokenizer, **formatted_data)


def train_model(output_dir: str, **kwargs):
    import torch
    from transformers import TrainingArguments, Trainer
    from peft import get_peft_model, LoraConfig, TaskType

    model = BaseLLM()
    tokenizer = model.tokenizer
    raw_trainset = Dataset("train")
    train_dataset = TokenizedDataset(tokenizer, raw_trainset, format_example)

    # Configure LoRA
    lora_config = LoraConfig(
        r=8,
        lora_alpha=32,
        target_modules="all-linear",
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )
    model.model = get_peft_model(model.model, lora_config)
    model.model.enable_input_require_grads()

    training_args = TrainingArguments(
        output_dir=output_dir,
        logging_dir=output_dir,
        per_device_train_batch_size=32,
        num_train_epochs=5,
        gradient_checkpointing=True,
        learning_rate=1e-3,
        report_to="tensorboard",
        save_strategy="epoch",
        logging_strategy="epoch",
        fp16=torch.cuda.is_available() and torch.backends.mps.is_available() is False,
    )

    trainer = Trainer(
        model=model.model,
        args=training_args,
        train_dataset=train_dataset,
        tokenizer=tokenizer,
    )

    trainer.train()
    trainer.save_model(output_dir)
    test_model(output_dir)


def test_model(ckpt_path: str):
    testset = Dataset("valid")
    llm = BaseLLM()

    from peft import PeftModel
    llm.model = PeftModel.from_pretrained(llm.model, ckpt_path).to(llm.device)

    benchmark_result = benchmark(llm, testset, max_question=100)
    print(f"{benchmark_result.accuracy=}  {benchmark_result.answer_rate=}")


if __name__ == "__main__":
    from fire import Fire
    Fire({"train": train_model, "test": test_model, "load": load})
