<div align="center">

#  LLM Playground — Project 1

### Building My LLM Playground from the Ground Up

*Exploring tokenization, text generation, decoding strategies, and model comparison — without the frameworks.*



![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)
![HuggingFace](https://img.shields.io/badge/HuggingFace-FFD21E?style=for-the-badge&logo=huggingface&logoColor=black)
![Jupyter](https://img.shields.io/badge/Jupyter-F37626?style=for-the-badge&logo=jupyter&logoColor=white)
![Colab](https://img.shields.io/badge/Google%20Colab-F9AB00?style=for-the-badge&logo=googlecolab&logoColor=black)

</div>


##  Overview

Instead of jumping straight into high-level frameworks, this project focuses on the **core mechanics** of LLMs. By working directly with the underlying components, you can observe how a model processes raw text, predicts tokens, and produces output.

> 💡 This is **Project 1** in a series. Future projects will build on this foundation using tools like **Ollama** and **LangChain**.



##  Learning Objectives

-  Understand how text is converted into **tokens** before being processed by a model
-  Load a pretrained **GPT-2** model and explore its architecture
-  Understand how **logits** are transformed into probabilities for next-token prediction
-  Explore the number of **model parameters** and their significance
-  Compare **greedy decoding** vs **top-p (nucleus) sampling**
-  Understand the differences between GPT-2 and modern **instruction-following LLMs**
-  Get an overview of **inference engines** (Ollama, vLLM, SGLang) and their role


##  Project Structure

```
llm-playground/
├──  llm_playground_project1.ipynb   -- Main notebook
├──  environment.yaml                -- Conda environment definition
├──  requirements.txt                -- pip / uv dependencies
└──  README.md
```


##  Getting Started

###  Option A — Google Colab *(Recommended)*

Open directly in your browser — no installation needed.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)]([https://colab.research.google.com/](https://colab.research.google.com/drive/1EwSYieZtmfOhGue7iEEC270sbFoWkA9L?usp=sharing))


###  Option B — Local Setup

#### 🅰 Using Conda

```bash
# Create and activate the environment in one step
conda env create -f environment.yaml && conda activate llm_playground
```

#### 🅱 Using uv *(faster alternative)*

```bash
# Install uv if not already installed
curl -LsSf https://astral.sh/uv/install.sh | sh

# Create virtual environment and install dependencies
uv venv .venv && source .venv/bin/activate
uv pip install -r requirements.txt
```

####  Register the Jupyter Kernel

If the environment doesn't appear in Jupyter, register it manually:

```bash
python -m ipykernel install --user --name=llm_playground --display-name "llm_playground"
```

Then select the **llm_playground** kernel from the Jupyter kernel menu before running cells.


##  Notebook Contents

### 1️ Tokenization

Explores three tokenization strategies from scratch:

| Approach | Description | Trade-off |
|---|---|---|
|  **Word-level** | Each word → unique token ID | Simple, but large vocab & OOV issues |
|  **Character-level** | Each character → token | No OOV, but very long sequences |
|  **Subword / BPE** | Meaningful subword units | Best balance — used by GPT-2 |

Also covers **tiktoken** — OpenAI's production-grade tokenizer — comparing encodings across GPT model families (`gpt2`, `cl100k_base`).


### 2️ Loading GPT-2 & Exploring the Architecture

- Loads the pretrained **GPT-2** model via Hugging Face `transformers`
- Inspects model architecture and layer structure
- Counts total trainable parameters — **~124M**


### 3️ Logits, Probabilities & Decoding Strategies

- Examines raw **logit** outputs from the model
- Converts logits → probabilities via **softmax**
- Compares two decoding approaches:

| Strategy | How It Works | Output Style |
|---|---|---|
|  **Greedy** | Always picks the highest-probability token | Deterministic, repetitive |
|  **Top-p Sampling** | Samples from tokens whose cumulative probability ≥ *p* | Diverse, more natural |


### 4️ GPT-2 vs. Modern LLMs (Qwen3-0.6B)

| Model | Params | Type | Capabilities |
|---|---|---|---|
| **GPT-2** | 124M | Base / Completion | Text continuation only |
| **Qwen3-0.6B** | 600M | Instruction-tuned | Q&A, reasoning, conversation |

- Demonstrates **chat templates** (`apply_chat_template`) used by instruction-tuned models
- Side-by-side comparison on identical prompts to observe behavioral differences


### 5️ Interactive Playground *(Optional)*

A `ipywidgets`-based interactive UI built directly inside the notebook:

-  Prompt input box
-  Model selector — GPT-2 / Qwen3-0.6B
-  Decoding strategy selector — Greedy / Top-k / Top-p
-  Temperature slider
-  Live text generation on button click



### 6️ Inference Engines Overview

| Engine | Best For | API Compatible |
|---|---|---|
|  **Ollama** | Local experimentation | ✅ OpenAI-compatible |
|  **vLLM** | High-performance production inference | ✅ OpenAI-compatible |
|  **SGLang** | Structured generation & efficient serving | ✅ OpenAI-compatible |

>  All support the **OpenAI-compatible API** — swap backends without changing your application code.



##  Tech Stack

| Tool | Version | Purpose |
|---|---|---|
|  `torch` | Latest stable | Deep learning backbone |
|  `transformers` | Latest stable | Model loading & text generation |
|  `tiktoken` | Latest stable | OpenAI's fast tokenizer library |
|  `ipywidgets` | Latest stable | Interactive notebook widgets |
|  `IPython` | Latest stable | Rich notebook display utilities |



##  What's Next

Upcoming projects in this series:

- [ ]  Running LLMs locally with **Ollama**
- [ ]  Building LLM-powered applications with **LangChain**
- [ ]  Prompt engineering patterns and best practices
- [ ]  RAG — Retrieval-Augmented Generation


## 📄 License

This project is for **educational purposes** only.


<div align="center">

*Part of my **LLMs from the Ground Up** learning series*
*— building intuition before using the abstractions 🚀*

</div>
