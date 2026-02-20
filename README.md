# MicroJulia

A minimal GPT built from scratch in pure Julia. No frameworks, no dependencies beyond stdlib.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/DavinciDreams/micro-julia/blob/main/microjulia.ipynb)

## What is this?

A faithful port of Karpathy's microGPT: the most atomic way to train a GPT.
Everything built from scratch — scalar autograd engine, transformer, Adam optimizer.

**Architecture:**
- Custom scalar autograd engine (`Value` type)
- Single-layer transformer with multi-head attention
- RMSNorm, ReLU activations, no biases
- KV cache for natural causal masking
- Adam optimizer with linear LR decay
- Temperature-controlled text generation
- Best-model checkpointing with validation loss tracking
- W&B logging + HuggingFace Hub push/pull

**Default config:** 1 layer, 4 heads, 16-dim embeddings, 256 block size, ~5-9K params

## Training Data

Ships with preprocessed Aristotle's Rhetoric (`data/aristotle_rhetoric.txt`), cleaned through a text processing pipeline:
- Gutenberg/MIT Classics boilerplate stripped
- Unicode normalized to ASCII
- Lowercased, chunked at sentence boundaries (max 256 chars)

Upload your own `.txt` files or use the built-in data processing pipeline cell in the notebook.

## Quick Start

1. Click "Open in Colab" above
2. Add Colab secrets: `HF_TOKEN`, `WANDB_KEY`, `HF_REPO`
3. Run the Python login cell
4. Run the Julia install cell (~3-5 min)
5. Switch runtime to **Julia 1.10**
6. Run all remaining cells

Training steps scale automatically based on dataset size (~3 epochs).

## Based on

[Karpathy's microGPT](https://gist.github.com/karpathy/8627fe009c40f57531cb18360106ce95)
