# Genetic Attention: BERT Dual-Encoder Ablation

This directory contains the ablation study comparing **Multi-Head Genetic Attention (MGA)** against standard multi-head attention in a BERT dual-encoder setup for passage retrieval on MS-MARCO.

## Overview

| | |
|---|---|
| **Task** | Passage retrieval (dual-encoder) |
| **Dataset** | MS-MARCO v2.1 |
| **Comparison** | `MultiheadGeneticAttention` vs `nn.MultiheadAttention` |
| **Loss** | InfoNCE with in-batch negatives |
| **Training steps** | 2,048 (≈ 2 epochs) |

## Directory Structure

```text
research/hybrid_ai/genetic_attention/
├── README.md                  # This file
├── ABLATION_REPORT.md         # Full ablation methodology, results, and analysis
├── run_ablation.py            # Training script (runs both configurations sequentially)
└── artefacts/                 # Saved training outputs and plots
```

## Quick Start

Install the required extras, then run the ablation:

```bash
uv sync --extra examples
python research/hybrid_ai/genetic_attention/run_ablation.py
```

CLI options:

```bash
python research/hybrid_ai/genetic_attention/run_ablation.py --max-steps 100   # quick run
python research/hybrid_ai/genetic_attention/run_ablation.py -m 500 -b 8 -lr 1e-5 -o results/
```

The script runs both configurations (standard then genetic) sequentially and saves results to `artefacts/`.

## Results Summary

Results from the full 2,048-step run (see [ABLATION_REPORT.md](ABLATION_REPORT.md) for full analysis):

| Configuration | MRR | MAP | NDCG@5 | Hit@5 | Pos Sim | Training Time |
|---|---|---|---|---|---|---|
| Standard attention | 0.492 | 0.492 | 0.500 | 0.573 | 0.368 | 7,822 s |
| Genetic attention | 0.486 | 0.486 | 0.492 | 0.568 | 0.335 | 11,947 s |

> Standard attention outperforms genetic attention across all metrics after extended training, with a 52% increase in training time for the genetic variant.

## MGA Module Parameters

`MultiheadGeneticAttention` (from `pikaia.models.nn_modules.mga`) accepts the following constructor arguments:

| Parameter | Type | Default | Description |
|---|---|---|---|
| `embed_dim` | `int` | — | Total embedding dimension |
| `n_heads` | `int` | — | Number of query attention heads |
| `n_kv_heads` | `int \| None` | `None` (= `n_heads`) | Key/value heads for GQA; `None` disables GQA |
| `in_proj_dim` | `int \| None` | `None` | Input compression dimension for MLA; `None` disables MLA |
| `dropout` | `float` | `0.0` | Attention dropout probability |
| `bias` | `bool` | `True` | Whether to include bias in projections |

## Architecture: BERT Dual-Encoder

Both configurations use BERT-base with the following settings:

| Parameter | Value |
|---|---|
| Hidden size | 384 |
| Layers | 6 |
| Attention heads | 12 (head dim = 32) |
| Intermediate size | 1,536 |
| Max sequence length | 512 |
| Batch size | 512 |
| Learning rate | 2e-5 |
| Training steps | 2,048 |
| Pooling | Mean pooling over valid tokens |

## References

- [BERT: Pre-training of Deep Bidirectional Transformers](https://arxiv.org/abs/1810.04805)
- [MS MARCO: A Human Generated MAchine Reading COmprehension Dataset](https://arxiv.org/abs/1611.09268)
- [Dense Passage Retrieval for Open-Domain Question Answering](https://arxiv.org/abs/2004.04906)
- [Genetic AI preprint](https://arxiv.org/abs/2501.19113)
