# 🦙 LLM Project

*In this project, I built a GPT-2 model from scratch with the help of Pytorch.
Then I finetuned the model loaded with OpenAI weights for downstream tasks such as classification and instruction, 
and deployed them with Gradio.*

![Python](https://img.shields.io/badge/python-3.12%2B-blue?logo=python)
![PyTorch](https://img.shields.io/badge/PyTorch-2.7-orange?logo=pytorch)
![License](https://img.shields.io/badge/license-MIT-lightgrey)
[![🤗 Spaces](https://img.shields.io/badge/%F0%9F%A4%97%20Spaces-ndkhoa211-yellow?logo=huggingface)](https://huggingface.co/ndkhoa211)
---

## Table of Contents

1. [Project Goals](#project-goals)
2. [Repository Structure](#repository-structure)
3. [Running the Demos](#running-the-demos)
4. [Live Demos on Hugging Face Spaces](#live-demos-on-hugging-face-spaces)

---

## Project Goals

This repository documents my exploration of language model engineering, covering the full cycle:

* **Architecture** – implement GPT‑2 in pure PyTorch, layer‑by‑layer.
* **Training** – train a 124 M parameter model from scratch on open‑text corpora.
* **Stability tricks** – experiment with cosine LR decay, warm‑up, gradient clipping and LoRA.
* **Down‑stream tasks** – finetune the base model for:
  * **Instruction following**
  * **Text classification**
* **Inference UX** – package the checkpoints into lightweight **Gradio** apps that anyone can run locally or as 🤗 **Hugging Face Spaces**.

> Directory names and app folders are taken directly from the repository listing. ([github.com](https://github.com/ndkhoa211/llm_project))


---

## Repository Structure

```text
llm_project/
│
├── modeling/                         # train & evaluation functions
├── models/                           # LLM architectures and layers
├── datasets/                         # Raw data dumps (git‑ignored)
├── data/                             # Custom Datasets & DataLoaders
├── exp_notebooks/                    # Experiments & visualisations
│
├── pretrained_gpt_app/               # Gradio app – base GPT‑2
├── instruction_finetune_gpt2_app/    # Gradio app – instruction‑tuned GPT‑2
├── classification_finetune_gpt2_app/ # Gradio app – classification GPT‑2
│
├── utils/                            # Helper functions
├── main.py                           # Entry point
├── pyproject.toml                    # Project metadata & deps
└── uv.lock                           # Exact package versions (uv) ★
```

---



## Running the Demos

The repo bundles three standalone Gradio interfaces. Launch them locally:

```bash
git clone https://github.com/ndkhoa211/llm_project && cd llm_project

# 1️⃣ Base GPT‑2 generation
gradio app: pretrained_gpt_app/app.py
python -m pretrained_gpt_app.app

# 2️⃣ Instruction‑tuned
gradio app: instruction_finetune_gpt2_app/app.py
python -m instruction_finetune_gpt2_app.app

# 3️⃣ Classification
gradio app: classification_finetune_gpt2_app/app.py
python -m classification_finetune_gpt2_app.app
```


---

## Live Demos on Hugging Face Spaces

> Prefer a *zero‑setup* experience? Click a badge below to open the running app in your browser.

| Demo                        | Model / Checkpoint                 | Hugging Face Space                                                                                                                                 |
|-----------------------------| ---------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Pretrained**              | GPT‑2 124 M (trained from scratch) | [![Open](https://img.shields.io/badge/%20-Launch-blue?logo=huggingface)](https://huggingface.co/spaces/ndk211/pretrained_gpt2_small)               |
| **Instruction Finetune**    | GPT‑2 124 M               | [![Open](https://img.shields.io/badge/%20-Launch-blue?logo=huggingface)](https://huggingface.co/spaces/ndk211/instruction_finetune_gpt2_big)    |
| **Classification Finetune** | GPT‑2 774 M                | [![Open](https://img.shields.io/badge/%20-Launch-blue?logo=huggingface)](https://huggingface.co/spaces/ndk211/classification_finetune_gpt2_small) |

*Cold‑start note*: the container spins up on first request, so the initial load can take 20‑30 s.

---

Happy experimenting! 🙌



