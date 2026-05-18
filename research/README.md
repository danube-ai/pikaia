# Research

This directory contains exploratory research work investigating hybrid approaches that combine **Genetic AI** principles with modern machine learning architectures.

> **Note:** All research here is exploratory and experimental. Findings should be treated as preliminary and hypothesis-generating rather than definitive.

---

## Structure

```
research/
└── hybrid_ai/                  # Hybrid Genetic AI + deep learning research
    ├── hybrid_ai.md            # Overview: where genetic sorting could replace ML components
    ├── genetic_ablation_summary.md  # Summary of all genetic attention ablation studies
    ├── genetic_layer/          # GeneticLayer neural network benchmarks
    ├── genetic_attention/      # Genetic attention mechanism experiments (BERT dual-encoder)
    ├── genetic_adapter/        # Genetic adapter for multi-task BERT (Bias-in-Bios)
    ├── references/             # Background reference notes (DeltaNet, GQA, MLA, MoE, SWA)
    └── _white_paper/           # LaTeX source for the hybrid AI white paper
```

---

## hybrid_ai/

### `hybrid_ai.md`

High-level overview of where genetic sorting could replace or augment traditional neural network components. Covers attention mechanisms (Scaled Dot-Product, MHA, GQA, MLA, DeltaNet) and how evolutionary fitness scoring can be used as an alternative to query-key similarity.

### `genetic_ablation_summary.md`

Consolidated summary of all genetic attention mechanism attempts across encoder-only and decoder-only architectures. Each section documents a distinct iteration of the concept, including architecture details, results, and lessons learned.

---

## hybrid_ai/genetic_layer/

Benchmarks comparing **GeneticLayer** (a neural network layer that computes fitness via evolutionary simulation) against classical linear layers across multiple architectures and datasets.

- [`README.md`](hybrid_ai/genetic_layer/README.md) — architecture overview, benchmark setup, results, and conclusions
- [`genetic_networks.ipynb`](hybrid_ai/genetic_layer/genetic_networks.ipynb) — interactive benchmark notebook
- [`compute_summary.py`](hybrid_ai/genetic_layer/compute_summary.py) — script to aggregate benchmark results
- [`artefacts/`](hybrid_ai/genetic_layer/artefacts/) — saved benchmark outputs and figures

---

## hybrid_ai/genetic_attention/

Experiments training BERT models with **Multi-Head Genetic Attention (MGA)** for text retrieval on MS-MARCO, comparing genetic attention against standard multi-head attention in a dual-encoder setup.

- [`README.md`](hybrid_ai/genetic_attention/README.md) — training setup, dataset, metrics
- [`ABLATION_REPORT.md`](hybrid_ai/genetic_attention/ABLATION_REPORT.md) — detailed ablation results
- [`run_ablation.py`](hybrid_ai/genetic_attention/run_ablation.py) — ablation runner script
- [`artefacts/`](hybrid_ai/genetic_attention/artefacts/) — saved training outputs

---

## hybrid_ai/genetic_adapter/

Multi-task BERT experiment using a genetic adapter on the **Bias-in-Bios** dataset. Demonstrates shared encoder with separate classification heads and counterfactual flip-rate evaluation via pronoun swapping.

- [`example.py`](hybrid_ai/genetic_adapter/example.py) — standalone training and evaluation script

---

## hybrid_ai/references/

Background notes on attention mechanisms and related architectures referenced in the hybrid AI work:

| File | Topic |
|------|-------|
| [`deltanet.md`](hybrid_ai/references/deltanet.md) | Gated DeltaNet linear recurrent attention |
| [`gqa.md`](hybrid_ai/references/gqa.md) | Grouped-Query Attention (GQA) |
| [`mla.md`](hybrid_ai/references/mla.md) | Multi-Head Latent Attention (MLA) |
| [`moe.md`](hybrid_ai/references/moe.md) | Mixture of Experts (MoE) |
| [`swa.md`](hybrid_ai/references/swa.md) | Sliding Window Attention (SWA) |

---

## hybrid_ai/_white_paper/

LaTeX source for the hybrid AI white paper, including bibliography.

- [`white_paper.tex`](hybrid_ai/_white_paper/white_paper.tex)
- [`references.bib`](hybrid_ai/_white_paper/references.bib)
