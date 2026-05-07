# Genetic Attention BERT Dual-Encoder Training

This example demonstrates training BERT models with genetic attention for text retrieval tasks using the MS-MARCO dataset. It compares MultiheadGeneticAttention (MGA) against standard multi-head attention in a dual-encoder setup.

## 🎯 Overview

- **Task**: Train dual-encoder BERT models for text retrieval
- **Dataset**: MS-MARCO v2.1 (queries and passages)
- **Comparison**: Genetic attention vs. standard attention
- **Metrics**: Embedding quality, training time, resource usage
- **Training**: InfoNCE loss with in-batch negatives

## 🚀 Quick Start

### 1. Install Dependencies

```bash
pip install -e ".[examples]"
```

### 2. Run Training

```bash
cd examples/genetic_attention
python train_dual_encoder.py
```

The script will automatically run training for both configurations sequentially:

- **Standard Attention**: BERT with `nn.MultiheadAttention`
- **Genetic Attention**: BERT with `MultiheadGeneticAttention`

### 3. View Results

Results are saved in `artefacts/` as JSON files with training metrics and timing.

## 📋 Details

- Uses MS-MARCO v2.1 dataset for dual-encoder training
- Compares genetic attention vs standard multi-head attention
- Measures embedding quality, training time, and resource usage
- In-batch negatives for efficient contrastive learning

## 🏗️ Architecture Components

### BERT Backbone

The BERT model is modified to support genetic attention:

- **Standard BERT**: Uses `nn.MultiheadAttention` in encoder layers
- **Genetic BERT**: Uses `MultiheadGeneticAttention` in encoder layers
- **Shared Encoder**: Same backbone for query and passage encoding
- **Pooling**: Mean pooling for sentence embeddings

### Dual-Encoder Training

- **Query Encoder**: Encodes search queries
- **Passage Encoder**: Encodes relevant passages
- **Loss Function**: InfoNCE with in-batch negatives
- **Optimization**: AdamW with linear warmup and decay

### Genetic Attention Integration

The `MultiheadGeneticAttention` module:

- Uses fitness-based organism selection
- Built-in causal masking (adapted for bidirectional BERT)
- Supports configurable population size and generations
- Integrated seamlessly into BERT encoder layers

## 📊 Training Configuration

| Parameter          | Value     | Description                    |
| ------------------ | --------- | ------------------------------ |
| `model_name`       | bert-base-uncased | Base BERT model          |
| `max_seq_len`      | 512       | Maximum sequence length        |
| `batch_size`       | 8         | Training batch size             |
| `learning_rate`    | 2e-5      | AdamW learning rate             |
| `num_epochs`       | 1         | Number of training epochs       |
| `warmup_steps`     | 100       | Linear warmup steps             |
| `save_steps`       | 500       | Checkpoint save frequency       |
| `eval_steps`       | 500       | Evaluation frequency            |

### Genetic Attention Parameters

| Parameter          | Value     | Description                    |
| ------------------ | --------- | ------------------------------ |
| `population_size`  | 32        | Number of organisms per head   |
| `num_generations`  | 5         | Evolution generations          |
| `mutation_rate`    | 0.1       | Genetic mutation probability   |
| `crossover_rate`   | 0.8       | Genetic crossover probability  |

## 📁 File Structure

```text
examples/genetic_attention/
├── train_dual_encoder.py          # Main training script
├── README.md                      # This documentation
├── artefacts/                     # Training results and checkpoints
│   ├── standard_attention_results.json
│   ├── genetic_attention_results.json
│   └── training_curves/
└── utils/                         # Legacy LLM components (deprecated)
```

## 🔬 Key Features

### Automatic Comparison

The training script automatically runs both configurations:

```python
# In train_dual_encoder.py
for use_genetic in [False, True]:
    model = BertModel(use_genetic=use_genetic)
    trainer = Trainer(model, ...)
    trainer.train()
```

### Metrics Tracking

Tracks comprehensive metrics:

- **Training Loss**: InfoNCE loss over epochs
- **Training Time**: Time per epoch and total training time
- **Memory Usage**: Peak GPU memory consumption
- **Embedding Quality**: Cosine similarity distributions

### Checkpointing

- Saves model checkpoints every `save_steps`
- Saves training state for resuming
- Saves final embeddings for evaluation

## 🎯 Usage Examples

### Custom Training

```python
from transformers import AutoTokenizer
from pikaia.models.backbones.bert import BertModel
from examples.genetic_attention.train_dual_encoder import train_dual_encoder

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

# Create genetic-enabled BERT
model = BertModel(use_genetic=True)

# Train on MS-MARCO
results = train_dual_encoder(
    model=model,
    tokenizer=tokenizer,
    use_genetic=True,
    num_epochs=3,
    batch_size=16
)
```

### Model Comparison

```python
import json

# Load results
with open("artefacts/standard_attention_results.json") as f:
    standard_results = json.load(f)

with open("artefacts/genetic_attention_results.json") as f:
    genetic_results = json.load(f)

# Compare final losses
print(f"Standard Loss: {standard_results['final_loss']:.4f}")
print(f"Genetic Loss: {genetic_results['final_loss']:.4f}")

# Compare training times
print(f"Standard Time: {standard_results['total_time']:.2f}s")
print(f"Genetic Time: {genetic_results['total_time']:.2f}s")
```

## 📊 Expected Results

After training, you should see results like:

```text
Standard Attention Results:
- Final Loss: 2.3456
- Training Time: 1247.89s
- Peak Memory: 4.2GB

Genetic Attention Results:
- Final Loss: 2.3124
- Training Time: 1356.12s
- Peak Memory: 4.8GB
```

## 🔧 Configuration

### Modifying Training Parameters

Edit `train_dual_encoder.py` to change:

```python
# Training hyperparameters
training_args = TrainingArguments(
    output_dir="./artefacts",
    num_train_epochs=1,  # Increase for full training
    per_device_train_batch_size=8,
    learning_rate=2e-5,
    warmup_steps=100,
    save_steps=500,
    evaluation_strategy="steps",
    eval_steps=500,
)
```

### Genetic Attention Tuning

Modify genetic parameters in the model initialization:

```python
model = BertModel(
    use_genetic=True,
    population_size=64,  # Larger population
    num_generations=10,  # More generations
)
```

## 🎯 Next Steps

1. **Run Training**: Execute `python train_dual_encoder.py`
2. **Analyze Results**: Compare losses and training times
3. **Evaluate Embeddings**: Test on STS-B or BEIR benchmarks
4. **Tune Parameters**: Experiment with genetic algorithm settings
5. **Scale Up**: Train on full MS-MARCO dataset

## 📚 References

- **BERT**: [BERT: Pre-training of Deep Bidirectional Transformers](https://arxiv.org/abs/1810.04805)
- **MS-MARCO**: [MS MARCO: A Human Generated MAchine Reading COmprehension Dataset](https://arxiv.org/abs/1611.09268)
- **Dual Encoders**: [Dense Passage Retrieval for Open-Domain Question Answering](https://arxiv.org/abs/2004.04906)
- **Pikaia**: [Genetic Algorithm Library](https://github.com/danube-ai/pikaia)
