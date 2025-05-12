from .base_llm import BaseLLM


class CoTModel(BaseLLM):
    def format_prompt(self, question: str) -> str:
        """
        Formats the prompt using chat template with detailed instructions and multiple in-context examples.
        Aims to improve both accuracy and parseability of answers.
        """
        messages = [
            {
                "role": "system",
                "content": (
                    "You are a precise unit conversion assistant. "
                    "Solve the problem step-by-step, then wrap the final numeric result in <answer>...</answer>. "
                    "Do not round unless explicitly asked. "
                    "Do not include any text after the <answer>...</answer>."
                )
            },
            {
                "role": "user",
                "content": "How many feet are in 2 meters?"
            },
            {
                "role": "assistant",
                "content": "1 meter = 3.28084 feet. 2 * 3.28084 = <answer>6.56168</answer>"
            },
            {
                "role": "user",
                "content": "How many grams are in 3 kilograms?"
            },
            {
                "role": "assistant",
                "content": "1 kilogram = 1000 grams. 3 * 1000 = <answer>3000</answer>"
            },
            {
                "role": "user",
                "content": "How many centimeters are there in 0.4 meters?"
            },
            {
                "role": "assistant",
                "content": "1 meter = 100 centimeters. 0.4 * 100 = <answer>40.0</answer>"
            },
            {
                "role": "user",
                "content": "How many miles are in 5 kilometers?"
            },
            {
                "role": "assistant",
                "content": "1 kilometer = 0.621371 miles. 5 * 0.621371 = <answer>3.106855</answer>"
            },
            {
                "role": "user",
                "content": question
            }
        ]
        
        return self.tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=False
        )


def load() -> CoTModel:
    return CoTModel()


def test_model():
    from .data import Dataset, benchmark

    testset = Dataset("valid")
    model = CoTModel()
    benchmark_result = benchmark(model, testset, 100)
    print(f"{benchmark_result.accuracy=}  {benchmark_result.answer_rate=}")


if __name__ == "__main__":
    from fire import Fire

    Fire({"test": test_model, "load": load})

