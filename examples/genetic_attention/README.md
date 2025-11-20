# SmolLM 2 with Modular Attention Mechanisms

This directory contains a modular implementation of SmolLM 2 (~135M parameters) with support for different attention mechanisms for ablation studies.

## 🚀 Quick Start - Training

### 1. Install Dependencies

From the repository root:

```bash
# Install with examples dependencies
uv sync --extra examples
```

Or with pip:

```bash
pip install -e ".[examples]"
```

### 2. Download Data

```bash
cd examples/genetic_attention
python download_pretraining_data.py
```

This downloads shards from the SmolLM corpus (fineweb-edu-dedup) to `.data/`.

### 3. Run Training Ablation

Option A - Using uv (recommended):

```bash
# From repository root
uv run --extra examples python examples/genetic_attention/train_ablation.py
```

Option B - Using bash script:

```bash
cd examples/genetic_attention
bash run_training.sh
```

Option C - With activated environment:

```bash
source .venv/bin/activate.fish  # or .venv/bin/activate for bash
cd examples/genetic_attention
python train_ablation.py
```

Or to test only the MGA model:

```bash
python train_ablation.py --only-mga
```

This will:

- Train 4 small LLM models (MHA, GQA, MLA, MGA) on the downloaded data, or only MGA if `--only-mga` is specified
- Track metrics: tokens/sec, memory usage, loss, perplexity
- Save checkpoints to `ablation_results/`
- Generate a comparison table

Results are saved to `ablation_results/ablation_results.json`.

## 🏗️ Architecture Components

### Attention Mechanisms

Four attention mechanisms are implemented:

1. **Multi-Head Attention (MHA)** - `utils/nn_components/mha.py`

   - Standard transformer attention mechanism
   - Full query, key, and value heads
   - Baseline for comparison

2. **Grouped-Query Attention (GQA)** - `utils/nn_components/gqa.py`

   - Memory-efficient variant
   - Multiple query heads share key-value heads
   - Reduces KV cache size during inference

3. **Multi-Head Latent Attention (MLA)** - `utils/nn_components/mla.py`

   - Compressed KV representation
   - Projects to latent space before computing attention
   - Reduces memory usage while maintaining expressiveness

4. **Multi-Head Genetic Attention (MGA)** - `pikaia/nn_components/mga.py`
   - Novel genetic algorithm-inspired attention
   - Uses fitness-based organism selection
   - Built-in causal masking
   - Supports GQA-style value head reduction

All standard attention mechanisms (MHA/GQA/MLA) support:

- **Causal Masking**: via `is_causal=True` parameter
- **Sliding Window Attention (SWA)**: via `use_sliding_window=True`
- **QK Normalization**: via `qk_norm=True`

MGA has causal masking built-in and supports sliding windows via `window_size` parameter.

### Normalization Strategies

The transformer blocks support three normalization strategies:

1. **Pre-Norm** (`norm_strategy='pre'`)

   - Used in Llama 3 8B
   - Normalization before attention/FFN
   - Better gradient flow, easier to train

2. **Post-Norm** (`norm_strategy='post'`)

   - Used in OLMo 2 7B
   - Normalization after attention/FFN but before residual
   - Can provide better performance with proper tuning

3. **Sandwich Norm** (`norm_strategy='sandwich'`)
   - Both pre and post normalization
   - Maximum stability but more parameters

### SmolLM Architecture

The core model is defined in `utils/smollm.py`:

- **RMSNorm**: Root mean square layer normalization
- **SwiGLU**: Gated activation function for feedforward layers
- **TransformerBlock**: Standard transformer block with pre-normalization
- **SmolLM**: Full language model with modular attention

## 🚀 Quick Start

### Basic Usage

```python
from utils.nn_components.mha import MultiHeadAttention
from utils.smollm import create_smollm_with_attention

# Create attention module with optional features
attention = MultiHeadAttention(
    embed_dim=576,
    num_heads=9,
    dropout=0.0,
    use_sliding_window=False,  # Enable sliding window attention
    qk_norm=False,  # Enable QK normalization
)

# Create model with configurable normalization strategy
model = create_smollm_with_attention(
    attention_module=attention,
    vocab_size=49152,
    embed_dim=576,
    num_layers=30,
    ffn_hidden_dim=1536,
    norm_strategy='pre',  # 'pre', 'post', or 'sandwich'
)

print(f"Total parameters: {model.count_parameters():,}")
```

### Running the Example

```bash
python example_ablation.py
```

This will:

1. Create models with MHA, GQA, and MLA attention
2. Print model statistics and parameter counts
3. Test forward passes through each model

## 🔬 Ablation Studies

### Creating Models with Different Attention

```python
from utils.nn_components import (
    MultiHeadAttention,
    GroupedQueryAttention,
    MultiHeadLatentAttention,
)
from pikaia.nn_components.mga import MultiHeadGeneticAttention
from utils.smollm import create_smollm_with_attention

embed_dim = 576
num_heads = 9

# 1. Multi-Head Attention (MHA)
mha = MultiHeadAttention(
    embed_dim=embed_dim,
    num_heads=num_heads,
    is_causal=True,  # Enable causal masking
)
model_mha = create_smollm_with_attention(attention_module=mha)

# 2. Grouped-Query Attention (GQA)
gqa = GroupedQueryAttention(
    embed_dim=embed_dim,
    num_heads=num_heads,
    num_kv_heads=3,  # 3 KV heads shared by 9 query heads
    is_causal=True,
)
model_gqa = create_smollm_with_attention(attention_module=gqa)

# 3. Multi-Head Latent Attention (MLA)
mla = MultiHeadLatentAttention(
    embed_dim=embed_dim,
    num_heads=num_heads,
    latent_dim=288,  # Compress to half the dimension
    is_causal=True,
)
model_mla = create_smollm_with_attention(attention_module=mla)

# 4. Multi-Head Genetic Attention (MGA)
mga = MultiHeadGeneticAttention(
    d_model=embed_dim,
    num_heads=num_heads,
    window_size=256,  # Required sliding window for efficiency
)
model_mga = create_smollm_with_attention(attention_module=mga)
```

### Enabling Causal Masking

```python
# Enable causal masking for autoregressive generation
attention = MultiHeadAttention(
    embed_dim=576,
    num_heads=9,
    is_causal=True,  # Prevent attending to future tokens
)
```

### Enabling Sliding Window Attention

```python
# Enable sliding window with 256-token context
attention = MultiHeadAttention(
    embed_dim=576,
    num_heads=9,
    is_causal=True,
    use_sliding_window=True,
    window_size=256,
)

# For MGA (window_size is required)
mga = MultiHeadGeneticAttention(
    d_model=576,
    num_heads=9,
    window_size=256,  # Required for MGA
)
```

### Enabling QK Normalization

```python
# Enable QK normalization for more stable attention
attention = MultiHeadAttention(
    embed_dim=576,
    num_heads=9,
    is_causal=True,
    qk_norm=True,  # Normalizes queries and keys
)
```

### Choosing Normalization Strategy

```python
# Llama 3 style (pre-norm)
model = create_smollm_with_attention(
    attention_module=attention,
    norm_strategy='pre',
)

# OLMo 2 style (post-norm)
model = create_smollm_with_attention(
    attention_module=attention,
    norm_strategy='post',
)

# Sandwich norm (both)
model = create_smollm_with_attention(
    attention_module=attention,
    norm_strategy='sandwich',
)
```

## 📊 Model Configuration

Default SmolLM 2 configuration (~135M parameters):

| Parameter        | Value | Description                            |
| ---------------- | ----- | -------------------------------------- |
| `vocab_size`     | 49152 | Vocabulary size                        |
| `embed_dim`      | 576   | Embedding dimension                    |
| `num_layers`     | 30    | Number of transformer layers           |
| `num_heads`      | 9     | Number of attention heads              |
| `ffn_hidden_dim` | 1536  | FFN hidden dimension (2.67x embed_dim) |
| `max_seq_len`    | 2048  | Maximum sequence length                |
| `dropout`        | 0.0   | Dropout probability                    |
| `use_bias`       | False | Bias in linear layers                  |
| `tie_embeddings` | True  | Tie input/output embeddings            |

## 🔧 Attention Mechanism Comparison

| Mechanism | KV Heads      | Memory | Complexity | Use Case               |
| --------- | ------------- | ------ | ---------- | ---------------------- |
| **MHA**   | Full (9)      | High   | O(n²d)     | Baseline, best quality |
| **GQA**   | Reduced (3)   | Medium | O(n²d)     | Balanced efficiency    |
| **MLA**   | Latent (288d) | Low    | O(n²d)     | Maximum efficiency     |
| **MGA**   | Variable      | Medium | O(n²d)     | Novel genetic approach |

### Additional Features

**QK Normalization**: Applies LayerNorm to queries and keys before computing attention scores. Can improve training stability and performance.

#### Sliding Window Attention

When enabled (`use_sliding_window=True`):

- Restricts attention to a local window
- Reduces memory for long sequences
- Window size controls context length
- Causal masking is automatically applied

## 📝 Training Example

```python
import torch
from torch.optim import AdamW

# Create model
model = create_smollm_with_attention(attention_module=mha)
optimizer = AdamW(model.parameters(), lr=1e-4)

# Training loop
model.train()
for batch in dataloader:
    input_ids = batch["input_ids"]
    labels = batch["labels"]

    # Forward pass
    outputs = model(input_ids, labels=labels)
    loss = outputs["loss"]

    # Backward pass
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
```

## 🔍 Evaluation Metrics

For ablation studies, compare:

1. **Perplexity**: Language modeling performance
2. **Training Time**: Tokens per second
3. **Memory Usage**: Peak GPU memory
4. **Inference Speed**: Generation throughput
5. **KV Cache Size**: Memory during generation
6. **Training Stability**: Loss curves and gradient norms

### Ablation Study Dimensions

You can systematically test combinations of:

- **Attention mechanism**: MHA, GQA, MLA
- **Normalization strategy**: pre-norm, post-norm, sandwich
- **QK normalization**: enabled/disabled
- **Sliding window**: enabled/disabled with various window sizes
- **Architecture parameters**: num_heads, latent_dim, num_kv_heads

## 📁 File Structure

```text
examples/genetic_attention/
├── download_pretraining_data.py     # Download SmolLM corpus
├── train_ablation.py                # Full training script with metrics
├── example_ablation.py              # Simple model creation example
├── run_training.sh                  # Quick start training script
├── ABLATION_GUIDE.txt              # Detailed ablation configurations
├── README.md                        # This file
├── .data/                           # Downloaded training data (gitignored)
└── utils/
    ├── smollm.py                    # SmolLM architecture
    └── nn_components/
        ├── __init__.py              # Package exports
        ├── mha.py                   # Multi-Head Attention
        ├── gqa.py                   # Grouped-Query Attention
        └── mla.py                   # Multi-Head Latent Attention
```

Note: MGA (Multi-Head Genetic Attention) is in `pikaia/nn_components/mga.py`

## 🎯 Training Configuration

The `train_ablation.py` script uses a reduced configuration for faster training:

| Parameter     | Training Config | Full SmolLM   |
| ------------- | --------------- | ------------- |
| `num_layers`  | 4               | 30            |
| `max_seq_len` | 512             | 2048          |
| `batch_size`  | 4               | varies        |
| `max_steps`   | 1000            | full dataset  |
| `num_shards`  | 2               | all available |

You can adjust these in the script to trade off training time vs model quality.

## 📊 Output Metrics

After training completes, you'll get:

```text
Model      Loss       Perplexity   Tok/s      Params       Peak Mem (MB)
--------------------------------------------------------------------------------
MHA        3.1234     22.71        1543       12,345,678   1024.56
GQA        3.0987     22.12        1687       11,234,567   896.32
MLA        3.1456     23.15        1432       10,987,654   823.45
MGA        3.0765     21.65        1598       12,456,789   1012.34

Best model by loss: MGA
Best model by throughput: GQA
Best model by memory: MLA
```

Results are saved to:

- `ablation_results/ablation_results.json` - Full metrics
- `ablation_results/{mha,gqa,mla,mga}_model.pt` - Model checkpoints

## 🎯 Next Steps

1. **Download Data**: Run `python download_pretraining_data.py`
2. **Train Models**: Run `python train_ablation.py` or `bash run_training.sh`
3. **Analyze Results**: Check `ablation_results/ablation_results.json`
4. **Experiment**: Try different configurations in `ABLATION_GUIDE.txt`

## 📚 References

- **SmolLM**: [HuggingFace SmolLM](https://huggingface.co/HuggingFaceTB/SmolLM-135M)
- **GQA**: [Grouped-Query Attention](https://arxiv.org/abs/2305.13245)
- **MLA**: [Multi-Head Latent Attention](https://arxiv.org/abs/2405.04434)
- **SWA**: [Sliding Window Attention in Mistral](https://arxiv.org/abs/2310.06825)
- **Pikaia**: [Genetic Algorithm Library](https://github.com/danube-ai/pikaia)
