# LLM Playground — Project 1: Building My LLM Playground

A hands-on Jupyter notebook exploring the inner workings of Large Language Models from the ground up, from tokenization to text generation, decoding strategies, and model comparison.

---

## Overview

Instead of jumping straight into high-level frameworks, this project focuses on the **core mechanics** of LLMs. By working directly with the underlying components, you can observe how a model processes raw text, predicts tokens, and produces output.

This is **Project 1** in a series. Future projects will build on this foundation using tools like **Ollama** and **LangChain**.

---

## Learning Objectives

- Understand how text is converted into tokens before being processed by a model
- Load a pretrained GPT-2 model and explore its architecture
- Understand how **logits** are transformed into probabilities for next-token prediction
- Explore the number of **model parameters** and their significance
- Compare **greedy decoding** vs **top-p (nucleus) sampling**
- Understand the differences between GPT-2 and modern instruction-following LLMs
- Get an overview of inference engines (Ollama, vLLM, SGLang) and their role

---

## Project Structure

```
llm-playground/
├── llm_playground_project1.ipynb   # Main notebook
├── environment.yaml                 # Conda environment definition
├── requirements.txt                 # pip / uv dependencies
└── README.md
```

---

## Getting Started

### Option A — Google Colab (Recommended)

Open the notebook directly in [Google Colab](https://colab.research.google.com/). No local setup needed.

### Option B — Local Setup

#### Using Conda

```bash
conda env create -f environment.yaml && conda activate llm_playground
```

#### Using uv (fast alternative)

```bash
# Install uv if not already installed
curl -LsSf https://astral.sh/uv/install.sh | sh

# Create environment and install dependencies
uv venv .venv && source .venv/bin/activate
uv pip install -r requirements.txt
```

#### Register the Jupyter Kernel

If the environment doesn't appear in Jupyter, register it manually:

```bash
python -m ipykernel install --user --name=llm_playground --display-name "llm_playground"
```

Then select the `llm_playground` kernel from the Jupyter kernel menu.

---

## Notebook Contents

### 1 · Tokenization

Explores three tokenization strategies:

| Approach | Description |
|---|---|
| **Word-level** | Each word → unique token ID; simple but large vocab & OOV issues |
| **Character-level** | Each character → token; no OOV but long sequences |
| **Subword (BPE)** | Balances vocab size and sequence length; used by GPT-2 |

Also covers **tiktoken** — OpenAI's production tokenizer — and compares encodings across GPT model families (`gpt2`, `cl100k_base`).

### 2 · Loading GPT-2 & Exploring the Architecture

- Loads the pretrained GPT-2 model via Hugging Face `transformers`
- Inspects model architecture and layer structure
- Counts total trainable parameters (~124M)

### 3 · Logits, Probabilities & Decoding Strategies

- Examines raw logit outputs from the model
- Converts logits → probabilities via softmax
- Compares two decoding approaches:
  - **Greedy decoding** — always picks the highest-probability token
  - **Top-p (nucleus) sampling** — samples from the smallest set of tokens whose cumulative probability exceeds *p*, producing more diverse outputs

### 4 · GPT-2 vs. Modern LLMs (Qwen3-0.6B)

- **GPT-2** (124M params): text completion only; no instruction understanding
- **Qwen3-0.6B** (600M params): instruction-tuned; handles conversation, reasoning, and Q&A
- Demonstrates **chat templates** (`apply_chat_template`) used by instruction-tuned models
- Side-by-side comparison on the same prompts to observe behavioral differences

### 5 · Interactive Playground *(Optional)*

A `ipywidgets`-based interactive UI inside the notebook:
- Prompt input box
- Model selector (GPT-2 / Qwen3-0.6B)
- Decoding strategy selector (Greedy / Top-k / Top-p)
- Temperature slider
- Live text generation

### 6 · Inference Engines Overview

A conceptual overview of production-grade model serving:

| Engine | Best For |
|---|---|
| **Ollama** | Local experimentation |
| **vLLM** | High-performance production inference |
| **SGLang** | Structured generation & efficient serving |

All support the **OpenAI-compatible API**, so you can swap backends without changing application code.

---

## 🛠️ Tech Stack

| Tool | Purpose |
|---|---|
| `torch` | Deep learning backbone |
| `transformers` | Model loading & text generation (Hugging Face) |
| `tiktoken` | OpenAI's fast tokenizer library |
| `ipywidgets` | Interactive notebook widgets |
| `IPython` | Rich notebook display utilities |



## 📄 License

This project is for educational purposes.

---

> *Part of my LLMs from the Ground Up learning series — building intuition before using the abstractions.*