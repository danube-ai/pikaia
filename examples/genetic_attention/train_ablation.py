"""
Training script for ablation study comparing different attention mechanisms.

This script trains small LLM models with 8 different attention mechanism combinations
for systematic ablation studies:

1. Standard MHA (Multi-Head Attention)
2. MHA + Genetic sorting
3. GQA (Grouped Query Attention)
4. GQA + Genetic sorting
5. MLA (Multi-Head Latent Attention)
6. MLA + Genetic sorting
7. SLA (Single-Head Latent Attention)
8. SLA + Genetic sorting

Each combination is trained on the SmolLM corpus for a full epoch with increased batch size
and data shards to maximize training tokens. Only metrics are saved (no model checkpoints).

Usage:
    Usage:
    python train_ablation.py                    # Train all 8 attention mechanism combinations
    python train_ablation.py --only-genetic     # Train only the 4 variants with genetic sorting
    python train_ablation.py --max-steps 100    # Train for only 100 steps per model
    python train_ablation.py --quick-test       # Run 10 steps per model for quick validation
"""

import os

os.environ["TOKENIZERS_PARALLELISM"] = "false"

import argparse
import gc
import json
import time
from pathlib import Path
from typing import Dict, Optional

import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

try:
    from transformers import AutoTokenizer
except ImportError:
    AutoTokenizer = None

from utils.nn_components.ablation_attention import (
    AblationAttention,
    create_ablation_configs,
    get_config_name,
)
from utils.smollm import create_smollm_with_attention


class ParquetTextDataset(Dataset):
    """Dataset that loads text from parquet files and tokenizes on-the-fly."""

    def __init__(
        self,
        data_dir: Path,
        tokenizer,
        max_length: int = 2048,
        num_shards: Optional[int] = None,
    ):
        """
        Initialize dataset from parquet files.

        Args:
            data_dir: Directory containing parquet files
            tokenizer: HuggingFace tokenizer
            max_length: Maximum sequence length
            num_shards: Number of shards to load (None = all)
        """
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.data_dir = Path(data_dir)

        # Load parquet files
        parquet_files = sorted(self.data_dir.glob("train-*.parquet"))
        if num_shards is not None:
            parquet_files = parquet_files[:num_shards]

        print(f"Loading {len(parquet_files)} parquet files from {data_dir}")

        # Load and concatenate all dataframes
        dfs = []
        for file in tqdm(parquet_files, desc="Loading parquet files"):
            df = pd.read_parquet(file)
            dfs.append(df)

        self.df = pd.concat(dfs, ignore_index=True)
        print(f"Loaded {len(self.df)} documents")

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        """Get tokenized text sample."""
        text = self.df.iloc[idx]["text"]

        # Tokenize with padding/truncation
        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )

        input_ids = encoding["input_ids"].squeeze(0)
        attention_mask = encoding["attention_mask"].squeeze(0)

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
        }


def collate_fn(batch: list[dict]) -> dict[str, torch.Tensor]:
    """
    Collate function for DataLoader to batch tokenized samples.

    Args:
        batch: List of samples from dataset

    Returns:
        Dict with batched tensors
    """
    input_ids = torch.stack([item["input_ids"] for item in batch])
    attention_mask = torch.stack([item["attention_mask"] for item in batch])

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
    }


def measure_memory_usage(
    model: torch.nn.Module, device: str = "cuda", reset_stats: bool = True
) -> Dict[str, float]:
    """
    Measure memory usage of attention components.

    Args:
        model: The model to measure
        device: Device ('cuda', 'mps', or 'cpu')
        reset_stats: Whether to reset peak memory stats before measurement

    Returns:
        Dictionary with memory metrics in MB. Note: Only CUDA devices support
        comprehensive memory tracking. MPS and CPU return zeros.
    """
    if device == "cuda" and torch.cuda.is_available():
        if reset_stats:
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.empty_cache()

            # Run a forward pass to measure memory
            vocab_size = 50000  # Default vocab size
            if hasattr(model, "vocab_size"):
                vs = model.vocab_size
                if isinstance(vs, int):
                    vocab_size = vs
            dummy_input = torch.randint(0, vocab_size, (1, 128)).to(device)
            with torch.no_grad():
                _ = model(dummy_input)

        # Get memory stats
        allocated = torch.cuda.memory_allocated() / 1024**2  # MB
        reserved = torch.cuda.memory_reserved() / 1024**2  # MB
        max_allocated = torch.cuda.max_memory_allocated() / 1024**2  # MB

        return {
            "allocated_mb": allocated,
            "reserved_mb": reserved,
            "peak_mb": max_allocated,
        }

    elif device == "mps" and hasattr(torch, "mps") and torch.mps.is_available():
        # MPS doesn't have comprehensive memory tracking like CUDA
        # Return zeros for now
        return {
            "allocated_mb": 0.0,
            "reserved_mb": 0.0,
            "peak_mb": 0.0,
        }

    else:
        # CPU, MPS, or unsupported device - memory tracking not available
        return {
            "allocated_mb": 0.0,
            "reserved_mb": 0.0,
            "peak_mb": 0.0,
        }


def train_epoch(
    model: torch.nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: str,
    max_steps: Optional[int] = None,
    log_interval: int = 10,
) -> Dict[str, float]:
    """
    Train for one epoch.

    Args:
        model: Model to train
        dataloader: Training dataloader
        optimizer: Optimizer
        device: Device to train on
        max_steps: Maximum number of steps (None = full epoch)
        log_interval: Log metrics every N steps

    Returns:
        Dictionary with training metrics
    """
    model.train()
    total_loss = 0
    total_tokens = 0
    total_time = 0
    step = 0

    pbar = tqdm(dataloader, desc="Training", total=max_steps)

    for batch_idx, batch in enumerate(pbar):
        if max_steps is not None and batch_idx >= max_steps:
            break

        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)

        # Create labels (shift input_ids by 1)
        labels = input_ids.clone()
        labels[:, :-1] = input_ids[:, 1:]
        labels[:, -1] = -100  # Ignore last token

        # Forward pass with timing
        start_time = time.time()

        optimizer.zero_grad()
        outputs = model(input_ids, labels=labels)
        loss = outputs["loss"]

        # Backward pass
        loss.backward()
        optimizer.step()

        end_time = time.time()
        batch_time = end_time - start_time

        # Calculate tokens processed
        num_tokens = attention_mask.sum().item()

        # Accumulate metrics
        total_loss += loss.item()
        total_tokens += num_tokens
        total_time += batch_time
        step += 1

        # Update progress bar
        if step % log_interval == 0:
            avg_loss = total_loss / step
            avg_tokens_per_sec = total_tokens / total_time
            pbar.set_postfix(
                {
                    "loss": f"{avg_loss:.4f}",
                    "tok/s": f"{avg_tokens_per_sec:.0f}",
                }
            )

    # Calculate final metrics
    avg_loss = total_loss / step
    avg_tokens_per_sec = total_tokens / total_time
    perplexity = torch.exp(torch.tensor(avg_loss)).item()

    return {
        "loss": avg_loss,
        "perplexity": perplexity,
        "tokens_per_sec": avg_tokens_per_sec,
        "total_tokens": total_tokens,
        "total_time": total_time,
    }


def create_attention_variants(
    embed_dim: int = 576,
    num_heads: int = 9,
    window_size: int = 256,
    dropout: float = 0.1,
    is_causal: bool = True,
    only_genetic: bool = False,
) -> dict[str, torch.nn.Module]:
    """
    Create attention mechanism variants for ablation study.

    When only_genetic=False, creates all 8 ablation configurations using AblationAttention.
    When only_genetic=True, creates only the 4 variants with genetic sorting enabled.

    Args:
        embed_dim: Total dimension of the model
        num_heads: Number of attention heads
        window_size: Size of the sliding window for attention
        dropout: Dropout probability
        is_causal: Whether to use causal masking
        only_genetic: If True, only create variants with genetic sorting enabled

    Returns:
        Dictionary mapping variant names to attention modules
    """
    variants = {}

    # Create all 8 ablation configurations using AblationAttention
    configs = create_ablation_configs(embed_dim, num_heads, window_size)

    for i, config in enumerate(configs):
        # Skip non-genetic variants if only_genetic is True
        if only_genetic and not config.get("use_genetic", False):
            continue

        # Update config with runtime parameters
        config.update(
            {
                "dropout": dropout,
                "is_causal": is_causal,
            }
        )

        # Create attention module
        attention = AblationAttention(**config)

        # Get human-readable name
        name = get_config_name(config).lower().replace(" + ", "_").replace(" ", "_")

        variants[name] = attention

    return variants


def main(
    only_genetic: bool = False,
    max_steps: Optional[int] = None,
    quick_test: bool = False,
) -> None:
    """Main training function for ablation study."""
    # Configuration
    device = "cpu"

    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"

    # Adjust config based on quick_test flag
    if quick_test:
        batch_size = 4  # Smaller batch for quick testing
        num_shards = 1  # Use minimal data
        max_steps = max_steps or 16  # Default to 16 steps for quick test
    else:
        batch_size = 8
        num_shards = 2
        max_steps = max_steps or 512

    config = {
        "vocab_size": 49152,
        "embed_dim": 576,
        "num_layers": 4,  # Reduced for faster training
        "num_heads": 9,
        "ffn_hidden_dim": 1536,
        "max_seq_len": 512,  # Reduced for memory efficiency
        "window_size": 256,  # Sliding window size for MGA
        "dropout": 0.1,
        "use_bias": False,
        "tie_embeddings": True,
        "norm_strategy": "pre",
        # Training config
        "batch_size": batch_size,
        "learning_rate": 3e-4,
        "max_steps": max_steps,
        "num_shards": num_shards,
        "device": device,
        "log_interval": 10,
    }

    print("=" * 80)
    print("Attention Mechanism Ablation Study")
    print("=" * 80)
    print("\nConfiguration:")
    for key, value in config.items():
        print(f"  {key}: {value}")
    print()

    # Setup paths
    script_dir = Path(__file__).parent
    data_dir = script_dir / ".data" / "fineweb-edu-dedup"
    results_dir = script_dir / ".ablation_results"
    results_dir.mkdir(exist_ok=True)

    # Check if data exists
    if not data_dir.exists():
        print(f"Error: Data directory not found: {data_dir}")
        print("Please run download_pretraining_data.py first!")
        return

    # Load tokenizer (SmolLM uses GPT-2 tokenizer)
    print("Loading tokenizer...")
    if AutoTokenizer is None:
        print(
            "Error: transformers library not installed. Please run: pip install transformers"
        )
        return
    tokenizer = AutoTokenizer.from_pretrained("HuggingFaceTB/SmolLM2-135M")
    tokenizer.pad_token = tokenizer.eos_token

    # Create dataset
    print("\nCreating dataset...")
    dataset = ParquetTextDataset(
        data_dir=data_dir,
        tokenizer=tokenizer,
        max_length=config["max_seq_len"],
        num_shards=config["num_shards"],
    )

    dataloader = DataLoader(
        dataset,
        batch_size=config["batch_size"],
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=4,
    )

    # Create attention variants
    print("\nCreating attention variants...")
    attention_variants = create_attention_variants(
        embed_dim=config["embed_dim"],
        num_heads=config["num_heads"],
        window_size=config["window_size"],
        dropout=config["dropout"],
        only_genetic=only_genetic,
    )

    # Results storage
    all_results = {}

    # Train each variant
    for name, attention in attention_variants.items():
        print("\n" + "=" * 80)
        print(f"Training {name.upper()} Model")
        print("=" * 80)

        # Create model
        model = create_smollm_with_attention(
            attention_module=attention,
            norm_strategy=config["norm_strategy"],
            vocab_size=config["vocab_size"],
            embed_dim=config["embed_dim"],
            num_layers=config["num_layers"],
            ffn_hidden_dim=config["ffn_hidden_dim"],
            max_seq_len=config["max_seq_len"],
            dropout=config["dropout"],
            use_bias=config["use_bias"],
            tie_embeddings=config["tie_embeddings"],
        )

        model = model.to(config["device"])
        total_params = model.count_parameters()
        print(f"\nTotal parameters: {total_params:,}")

        # Reset memory stats before training to capture peak usage
        if config["device"] == "cuda" and torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.empty_cache()

        # Create optimizer
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=config["learning_rate"],
            weight_decay=0.01,
        )

        # Train
        print("\nStarting training...")
        train_metrics = train_epoch(
            model=model,
            dataloader=dataloader,
            optimizer=optimizer,
            device=config["device"],
            max_steps=config["max_steps"],
            log_interval=config["log_interval"],
        )

        # Measure memory usage after training (captures peak during training)
        memory_stats = measure_memory_usage(model, config["device"], reset_stats=False)
        print("Memory usage:")
        print(f"  Allocated: {memory_stats['allocated_mb']:.2f} MB")
        print(f"  Peak: {memory_stats['peak_mb']:.2f} MB")

        # Store results
        results = {
            "attention_type": name,
            "total_params": total_params,
            **memory_stats,
            **train_metrics,
        }
        all_results[name] = results

        # Print summary
        print(f"\n{name.upper()} Results:")
        print(f"  Final Loss: {train_metrics['loss']:.4f}")
        print(f"  Perplexity: {train_metrics['perplexity']:.2f}")
        print(f"  Tokens/sec: {train_metrics['tokens_per_sec']:.0f}")
        print(f"  Total tokens: {train_metrics['total_tokens']:,}")

        # Clean up (no checkpoint saving)
        del model, optimizer
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # Save all results
    results_path = results_dir / "ablation_results.json"
    with open(results_path, "w") as f:
        json.dump(all_results, f, indent=2)

    # Save results as CSV for easier analysis
    results_df = pd.DataFrame.from_dict(all_results, orient="index")
    csv_path = results_dir / "ablation_results.csv"
    results_df.to_csv(csv_path)

    print("\n" + "=" * 80)
    print("Ablation Study Complete!")
    print("=" * 80)
    print("\nResults saved to:")
    print(f"  JSON: {results_path}")
    print(f"  CSV:  {csv_path}")

    # Print comparison table
    print("\n" + "=" * 80)
    print("Results Comparison")
    print("=" * 80)
    print(
        f"\n{'Model':<12} {'Loss':<10} {'Perplexity':<12} {'Tok/s':<10} {'Params':<12} {'Peak Mem (MB)':<15}"
    )
    print("-" * 85)

    for name, results in all_results.items():
        print(
            f"{name.upper():<12} "
            f"{results['loss']:<10.4f} "
            f"{results['perplexity']:<12.2f} "
            f"{results['tokens_per_sec']:<10.0f} "
            f"{results['total_params']:<12,} "
            f"{results['peak_mb']:<15.2f}"
        )

    print(
        "\nBest model by loss:",
        min(all_results.items(), key=lambda x: x[1]["loss"])[0].upper(),
    )
    print(
        "Best model by throughput:",
        max(all_results.items(), key=lambda x: x[1]["tokens_per_sec"])[0].upper(),
    )
    print(
        "Best model by memory:",
        min(all_results.items(), key=lambda x: x[1]["peak_mb"])[0].upper(),
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train attention mechanism ablation study"
    )
    parser.add_argument(
        "--only-genetic",
        action="store_true",
        help="Train only the attention variants with genetic sorting enabled",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=512,
        help="Maximum number of training steps per model (default: 512)",
    )
    parser.add_argument(
        "--quick-test",
        action="store_true",
        help="Run a quick test with just 10 steps per model for validation",
    )
    args = parser.parse_args()

    main(
        only_genetic=args.only_genetic,
        max_steps=args.max_steps if not args.quick_test else 10,
        quick_test=args.quick_test,
    )
