## Project Description

This project aims to improve the accuracy of a large language model (LLM) through two fine-tuning techniques: supervised fine-tuning (SFT) and reinforcement fine-tuning (RFT). Two models were developed as part of this work: an SFT model and an RFT model.

### Supervised Fine-Tuning (SFT)

- **Script**: `sft.py`
- **Dependency**: `cot.py`
- The `cot.py` file defines the types of questions used to train the model using the "chain of thought" prompting method.
- `sft.py` loads these question types and fine-tunes the base LLM accordingly.

### Reinforcement Fine-Tuning (RFT)

- **Script**: `rft.py`
- **Dependency**: `datagen.py`
- The `datagen.py` script generates question-answer pairs used as input for the reinforcement learning process in `rft.py`.
- `rft.py` then uses these examples to train the model using reinforcement learning techniques.

---


