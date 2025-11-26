"""
Dual-encoder training script for BERT with MS-MARCO dataset using PyTorch Lightning.

Trains two configurations sequentially:
1. BERT with genetic attention disabled (standard)
2. BERT with genetic attention enabled

Uses dual-encoder (bi-encoder) setup for retrieval tasks.
Automatically selects the best available device (CUDA > MPS > CPU).
Generates and saves training curve plots comparing both configurations using seaborn.

Usage:
    python run_ablation.py                    # Train for one full epoch
    python run_ablation.py --max-steps 100    # Train for 100 steps
    python run_ablation.py -m 500 -b 8 -lr 1e-5 -o results/
"""

import json
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import pandas as pd
import pytorch_lightning as pl
import seaborn as sns
import torch
import typer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import CSVLogger
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

# Set environment variables to avoid warnings
os.environ["TOKENIZERS_PARALLELISM"] = "false"

try:
    from transformers import AutoTokenizer
except ImportError:
    AutoTokenizer = None

from datasets import load_dataset

from pikaia.models.backbones.bert import BertModel

# Constants
TEMPERATURE = 0.1
DEFAULT_MAX_LENGTH = 128
DEFAULT_BATCH_SIZE = 512
DEFAULT_LEARNING_RATE = 2e-5

app = typer.Typer(
    help="Train dual-encoder BERT models on MS-MARCO dataset with genetic attention comparison."
)


class SimpleDataset(Dataset):
    """Dataset for MS-MARCO query-passage pairs.

    Args:
        data: List of (query, passage) tuples
        tokenizer: HuggingFace tokenizer for text tokenization
        max_length: Maximum sequence length for tokenization
    """

    def __init__(
        self,
        data: List[Tuple[str, str]],
        tokenizer,
        max_length: int = DEFAULT_MAX_LENGTH,
    ) -> None:
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(
        self, idx: int
    ) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
        """Get tokenized query-passage pair at index.

        Args:
            idx: Index of the data sample

        Returns:
            Tuple of (query_tokens, passage_tokens) dictionaries
        """
        query, passage = self.data[idx]
        query_tokens = self.tokenizer(
            query,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        passage_tokens = self.tokenizer(
            passage,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        return query_tokens, passage_tokens


class DualEncoderModule(pl.LightningModule):
    """PyTorch Lightning module for dual-encoder training with genetic attention.

    Args:
        use_genetic: Whether to use genetic attention mechanism
        vocab_size: Vocabulary size for the model
        learning_rate: Learning rate for AdamW optimizer
    """

    def __init__(
        self, use_genetic: bool, vocab_size: int = 30522, learning_rate: float = 2e-5
    ) -> None:
        super().__init__()
        self.save_hyperparameters()
        self.use_genetic = use_genetic
        self.learning_rate = learning_rate

        self.model = BertModel(
            vocab_size=vocab_size,
            hidden_size=384,
            num_layers=6,
            num_attention_heads=12,
            intermediate_size=1536,
            max_position_embeddings=512,
            type_vocab_size=2,
            layer_norm_eps=1e-12,
            dropout=0.1,
            use_genetic=use_genetic,
        )

    def forward(
        self,
        query_batch: Dict[str, torch.Tensor],
        passage_batch: Dict[str, torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass through the dual encoder.

        Args:
            query_batch: Tokenized query batch
            passage_batch: Tokenized passage batch

        Returns:
            Tuple of (query_embeddings, passage_embeddings)
        """
        # Flatten batch and move to device
        queries = {k: v.view(-1, v.size(-1)) for k, v in query_batch.items()}
        passages = {k: v.view(-1, v.size(-1)) for k, v in passage_batch.items()}

        # Get embeddings
        query_out = self.model(**queries)
        passage_out = self.model(**passages)
        query_emb = query_out["pooled_embedding"]
        passage_emb = passage_out["pooled_embedding"]

        return query_emb, passage_emb

    def training_step(
        self,
        batch: Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]],
        batch_idx: int,
    ) -> torch.Tensor:
        """Training step with in-batch negatives.

        Args:
            batch: Tuple of (query_batch, passage_batch)
            batch_idx: Batch index

        Returns:
            Loss tensor
        """
        query_batch, passage_batch = batch
        query_emb, passage_emb = self(query_batch, passage_batch)

        # In-batch negatives: each query's positive is the corresponding passage,
        # negatives are other passages in batch
        batch_size_actual = query_emb.size(0)
        # Cosine similarity matrix
        sim_matrix = torch.matmul(query_emb, passage_emb.t())  # (batch, batch)

        # Labels: diagonal is positive
        labels = torch.arange(batch_size_actual, device=sim_matrix.device)

        # InfoNCE loss
        loss = torch.nn.functional.cross_entropy(
            sim_matrix / TEMPERATURE, labels
        )  # temperature scaling

        # Compute retrieval metrics within batch (on training data)
        batch_size = sim_matrix.size(0)

        # Get rankings for each query (higher similarity = better rank)
        rankings = torch.argsort(sim_matrix, dim=1, descending=True)  # (batch, batch)

        # Find rank of correct passage for each query
        correct_ranks = []
        for i in range(batch_size):
            correct_ranks.append(
                (rankings[i] == i).nonzero(as_tuple=True)[0].item() + 1
            )  # 1-based rank

        correct_ranks = torch.tensor(
            correct_ranks, dtype=torch.float, device=sim_matrix.device
        )

        # Mean Reciprocal Rank (MRR)
        mrr = torch.mean(1.0 / correct_ranks)

        # Precision@1, @5
        precision_at_1 = torch.mean((correct_ranks <= 1).float())
        precision_at_5 = torch.mean((correct_ranks <= 5).float())

        # Recall@1, @5 (since we have 1 positive per query in batch)
        recall_at_1 = precision_at_1  # Only 1 positive, so recall = precision
        recall_at_5 = precision_at_5

        # Mean Average Precision (MAP) - simplified for single positive per query
        map_score = mrr  # For single positive, MAP = MRR

        # NDCG@5 (Normalized Discounted Cumulative Gain)
        ndcg_at_5 = self._compute_ndcg(correct_ranks, k=5)

        # Hit Rate@5 (whether correct passage is in top-5)
        hit_rate_at_5 = torch.mean((correct_ranks <= 5).float())

        # Embedding quality metrics
        query_norm = torch.norm(query_emb, dim=1).mean()
        passage_norm = torch.norm(passage_emb, dim=1).mean()
        embedding_variance = (
            torch.cat([query_emb, passage_emb], dim=0).var(dim=0).mean()
        )

        # Semantic similarity (average cosine similarity of positive pairs)
        positive_similarities = torch.diag(sim_matrix)
        avg_positive_similarity = positive_similarities.mean()

        # Log all metrics
        self.log("train_loss", loss, prog_bar=True)
        self.log("train_mrr", mrr, prog_bar=False)
        self.log("train_precision@1", precision_at_1, prog_bar=False)
        self.log("train_precision@5", precision_at_5, prog_bar=False)
        self.log("train_recall@1", recall_at_1, prog_bar=False)
        self.log("train_recall@5", recall_at_5, prog_bar=False)
        self.log("train_map", map_score, prog_bar=False)
        self.log("train_ndcg@5", ndcg_at_5, prog_bar=False)
        self.log("train_hit_rate@5", hit_rate_at_5, prog_bar=False)
        self.log(
            "train_avg_positive_similarity", avg_positive_similarity, prog_bar=False
        )
        self.log("train_embedding_variance", embedding_variance, prog_bar=False)
        self.log("train_passage_norm", passage_norm, prog_bar=False)
        self.log("train_query_norm", query_norm, prog_bar=False)

        return loss

    def validation_step(
        self,
        batch: Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]],
        batch_idx: int,
    ) -> torch.Tensor:
        """Validation step with comprehensive retrieval and embedding quality metrics.

        Args:
            batch: Tuple of (query_batch, passage_batch)
            batch_idx: Batch index

        Returns:
            Loss tensor
        """
        query_batch, passage_batch = batch
        query_emb, passage_emb = self(query_batch, passage_batch)

        # Cosine similarity matrix
        sim_matrix = torch.matmul(query_emb, passage_emb.t())
        labels = torch.arange(query_emb.size(0), device=sim_matrix.device)
        loss = torch.nn.functional.cross_entropy(sim_matrix / TEMPERATURE, labels)

        # Compute retrieval metrics within batch
        batch_size = sim_matrix.size(0)

        # Get rankings for each query (higher similarity = better rank)
        rankings = torch.argsort(sim_matrix, dim=1, descending=True)  # (batch, batch)

        # Find rank of correct passage for each query
        correct_ranks = []
        for i in range(batch_size):
            correct_ranks.append(
                (rankings[i] == i).nonzero(as_tuple=True)[0].item() + 1
            )  # 1-based rank

        correct_ranks = torch.tensor(
            correct_ranks, dtype=torch.float, device=sim_matrix.device
        )

        # Mean Reciprocal Rank (MRR)
        mrr = torch.mean(1.0 / correct_ranks)

        # Precision@1, @5
        precision_at_1 = torch.mean((correct_ranks <= 1).float())
        precision_at_5 = torch.mean((correct_ranks <= 5).float())

        # Recall@1, @5 (since we have 1 positive per query in batch)
        recall_at_1 = precision_at_1  # Only 1 positive, so recall = precision
        recall_at_5 = precision_at_5

        # Mean Average Precision (MAP) - simplified for single positive per query
        map_score = mrr  # For single positive, MAP = MRR

        # NDCG@5 (Normalized Discounted Cumulative Gain)
        ndcg_at_5 = self._compute_ndcg(correct_ranks, k=5)

        # Hit Rate@5 (whether correct passage is in top-5)
        hit_rate_at_5 = torch.mean((correct_ranks <= 5).float())

        # Embedding quality metrics
        query_norm = torch.norm(query_emb, dim=1).mean()
        passage_norm = torch.norm(passage_emb, dim=1).mean()
        embedding_variance = (
            torch.cat([query_emb, passage_emb], dim=0).var(dim=0).mean()
        )

        # Semantic similarity (average cosine similarity of positive pairs)
        positive_similarities = torch.diag(sim_matrix)
        avg_positive_similarity = positive_similarities.mean()

        # Log all metrics
        self.log("val_loss", loss, prog_bar=True)
        self.log("val_mrr", mrr, prog_bar=False)
        self.log("val_precision@1", precision_at_1, prog_bar=False)
        self.log("val_precision@5", precision_at_5, prog_bar=False)
        self.log("val_recall@1", recall_at_1, prog_bar=False)
        self.log("val_recall@5", recall_at_5, prog_bar=False)
        self.log("val_map", map_score, prog_bar=False)
        self.log("val_ndcg@5", ndcg_at_5, prog_bar=False)
        self.log("val_hit_rate@5", hit_rate_at_5, prog_bar=False)
        self.log("val_avg_positive_similarity", avg_positive_similarity, prog_bar=False)
        self.log("val_embedding_variance", embedding_variance, prog_bar=False)
        self.log("val_passage_norm", passage_norm, prog_bar=False)
        self.log("val_query_norm", query_norm, prog_bar=False)

        return loss

    def test_step(
        self,
        batch: Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]],
        batch_idx: int,
    ) -> torch.Tensor:
        """Test step with comprehensive retrieval and embedding quality metrics.

        Args:
            batch: Tuple of (query_batch, passage_batch)
            batch_idx: Batch index

        Returns:
            Loss tensor
        """
        query_batch, passage_batch = batch
        query_emb, passage_emb = self(query_batch, passage_batch)

        # Cosine similarity matrix
        sim_matrix = torch.matmul(query_emb, passage_emb.t())
        labels = torch.arange(query_emb.size(0), device=sim_matrix.device)
        loss = torch.nn.functional.cross_entropy(sim_matrix / TEMPERATURE, labels)

        # Compute retrieval metrics within batch
        batch_size = sim_matrix.size(0)

        # Get rankings for each query (higher similarity = better rank)
        rankings = torch.argsort(sim_matrix, dim=1, descending=True)  # (batch, batch)

        # Find rank of correct passage for each query
        correct_ranks = []
        for i in range(batch_size):
            correct_ranks.append(
                (rankings[i] == i).nonzero(as_tuple=True)[0].item() + 1
            )  # 1-based rank

        correct_ranks = torch.tensor(
            correct_ranks, dtype=torch.float, device=sim_matrix.device
        )

        # Mean Reciprocal Rank (MRR)
        mrr = torch.mean(1.0 / correct_ranks)

        # Precision@1, @5
        precision_at_1 = torch.mean((correct_ranks <= 1).float())
        precision_at_5 = torch.mean((correct_ranks <= 5).float())

        # Recall@1, @5 (since we have 1 positive per query in batch)
        recall_at_1 = precision_at_1  # Only 1 positive, so recall = precision
        recall_at_5 = precision_at_5

        # Mean Average Precision (MAP) - simplified for single positive per query
        map_score = mrr  # For single positive, MAP = MRR

        # NDCG@5 (Normalized Discounted Cumulative Gain)
        ndcg_at_5 = self._compute_ndcg(correct_ranks, k=5)

        # Hit Rate@5 (whether correct passage is in top-5)
        hit_rate_at_5 = torch.mean((correct_ranks <= 5).float())

        # Embedding quality metrics
        query_norm = torch.norm(query_emb, dim=1).mean()
        passage_norm = torch.norm(passage_emb, dim=1).mean()
        embedding_variance = (
            torch.cat([query_emb, passage_emb], dim=0).var(dim=0).mean()
        )

        # Semantic similarity (average cosine similarity of positive pairs)
        positive_similarities = torch.diag(sim_matrix)
        avg_positive_similarity = positive_similarities.mean()

        # Log all metrics
        self.log("test_loss", loss, prog_bar=True)
        self.log("test_mrr", mrr, prog_bar=True)
        self.log("test_precision@1", precision_at_1, prog_bar=False)
        self.log("test_precision@5", precision_at_5, prog_bar=False)
        self.log("test_recall@1", recall_at_1, prog_bar=False)
        self.log("test_recall@5", recall_at_5, prog_bar=False)
        self.log("test_map", map_score, prog_bar=False)
        self.log("test_ndcg@5", ndcg_at_5, prog_bar=False)
        self.log("test_hit_rate@5", hit_rate_at_5, prog_bar=False)
        self.log(
            "test_avg_positive_similarity", avg_positive_similarity, prog_bar=False
        )
        self.log("test_query_norm", query_norm, prog_bar=False)
        self.log("test_passage_norm", passage_norm, prog_bar=False)
        self.log("test_embedding_variance", embedding_variance, prog_bar=False)

        return loss

    def _compute_ndcg(self, ranks: torch.Tensor, k: int = 5) -> torch.Tensor:
        """Compute Normalized Discounted Cumulative Gain at k.

        Args:
            ranks: Tensor of correct passage ranks (1-based)
            k: Cutoff rank

        Returns:
            NDCG@k score
        """
        # For binary relevance (relevant=1, irrelevant=0), DCG = 1/log2(rank+1) for relevant items
        dcg = torch.zeros_like(ranks)
        relevant_mask = ranks <= k
        dcg[relevant_mask] = 1.0 / torch.log2(ranks[relevant_mask] + 1)

        # IDCG (Ideal DCG) = 1/log2(2) = 1 for the first position
        idcg = 1.0
        ndcg = dcg / idcg

        return ndcg.mean()

    def configure_optimizers(self) -> torch.optim.AdamW:
        """Configure AdamW optimizer.

        Returns:
            Configured AdamW optimizer
        """
        return torch.optim.AdamW(self.parameters(), lr=self.learning_rate)


def get_device() -> torch.device:
    """Get the best available device (CUDA > MPS > CPU).

    Returns:
        PyTorch device object
    """
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")


def load_and_process_datasets(
    tokenizer,
) -> Tuple[
    SimpleDataset,
    SimpleDataset,
    SimpleDataset,
    List[Tuple[str, str]],
    List[Tuple[str, str]],
    List[Tuple[str, str]],
]:
    """Load and process MS-MARCO datasets once for both configurations.

    Args:
        tokenizer: HuggingFace tokenizer

    Returns:
        Tuple of (train_dataset, val_dataset, test_dataset, train_data, val_data, test_data)
    """
    print("Loading MS-MARCO datasets...")
    train_ds = load_dataset("microsoft/ms_marco", "v2.1", split="train")
    val_ds = load_dataset("microsoft/ms_marco", "v2.1", split="validation")
    test_ds = load_dataset("microsoft/ms_marco", "v2.1", split="test")

    print(f"Train samples: {len(train_ds):,}")  # type: ignore
    print(f"Validation samples: {len(val_ds):,}")  # type: ignore
    print(f"Test samples: {len(test_ds):,}")  # type: ignore

    # Process datasets
    train_data = _extract_positive_pairs(train_ds, "training")
    val_data = _extract_positive_pairs(val_ds, "validation")
    test_data = _extract_positive_pairs(test_ds, "test")

    print(f"Processed {len(train_data):,} training pairs")
    print(f"Processed {len(val_data):,} validation pairs")
    print(f"Processed {len(test_data):,} test pairs")

    # Create datasets
    train_dataset = SimpleDataset(train_data, tokenizer)
    val_dataset = SimpleDataset(val_data, tokenizer)
    test_dataset = SimpleDataset(test_data, tokenizer)

    return train_dataset, val_dataset, test_dataset, train_data, val_data, test_data


def _extract_positive_pairs(dataset, dataset_name: str) -> List[Tuple[str, str]]:
    """Extract positive query-passage pairs from MS-MARCO dataset.

    Args:
        dataset: MS-MARCO dataset split
        dataset_name: Name of the dataset for progress bar

    Returns:
        List of (query, passage) tuples
    """
    data = []
    for item in tqdm(dataset, desc=f"Processing {dataset_name} data"):
        query = item["query"]  # type: ignore
        passages = item["passages"]  # type: ignore
        for i, is_selected in enumerate(passages["is_selected"]):  # type: ignore
            if is_selected:
                data.append((query, passages["passage_text"][i]))  # type: ignore
                break  # One positive per query
    return data


def train_dual_encoder(
    use_genetic: bool,
    train_dataset: SimpleDataset,
    val_dataset: SimpleDataset,
    test_dataset: SimpleDataset,
    train_data: List[Tuple[str, str]],
    val_data: List[Tuple[str, str]],
    test_data: List[Tuple[str, str]],
    tokenizer,
    max_steps: Optional[int] = None,
    batch_size: int = DEFAULT_BATCH_SIZE,
    learning_rate: float = DEFAULT_LEARNING_RATE,
    checkpoint_dir: Optional[Path] = None,
    metrics_dir: Optional[Path] = None,
) -> Dict[str, Any]:
    """Train dual-encoder with MS-MARCO using PyTorch Lightning.

    Args:
        use_genetic: Whether to use genetic attention
        train_dataset: Training dataset
        val_dataset: Validation dataset
        test_dataset: Test dataset
        train_data: Raw training data
        val_data: Raw validation data
        test_data: Raw test data
        tokenizer: HuggingFace tokenizer
        max_steps: Maximum training steps
        batch_size: Batch size for training
        learning_rate: Learning rate for optimizer
        checkpoint_dir: Directory for checkpoints and logs
        metrics_dir: Directory for metrics and results

    Returns:
        Dictionary with training results and metrics
    """
    # Set default directories relative to script location
    if checkpoint_dir is None:
        script_dir = Path(__file__).parent
        checkpoint_dir = script_dir / ".artefacts"
    if metrics_dir is None:
        script_dir = Path(__file__).parent
        metrics_dir = script_dir / "artefacts"

    # Create directories if they don't exist
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir.mkdir(parents=True, exist_ok=True)

    # Set all random seeds for reproducibility
    pl.seed_everything(8080)

    print(f"Training with genetic={'enabled' if use_genetic else 'disabled'}")

    # Create dataloaders
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        persistent_workers=True,
    )
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        persistent_workers=True,
    )
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        persistent_workers=True,
    )

    # Create model
    model = DualEncoderModule(
        use_genetic, vocab_size=tokenizer.vocab_size, learning_rate=learning_rate
    )

    # Setup callbacks and logger
    checkpoint_callback = ModelCheckpoint(
        dirpath=checkpoint_dir,
        filename=f"best_model_{'genetic' if use_genetic else 'standard'}",
        monitor="val_loss",
        mode="min",
        save_top_k=1,
    )

    logger = CSVLogger(
        checkpoint_dir,
        name=f"lightning_logs_{'genetic' if use_genetic else 'standard'}",
    )

    # Create trainer
    trainer_kwargs = {
        "max_steps": max_steps if max_steps is not None else 512,
        "accelerator": "auto",
        "devices": "auto",
        "logger": logger,
        "callbacks": [checkpoint_callback],
        "enable_progress_bar": True,
        "log_every_n_steps": 1,
    }

    # Set validation to run every 8 steps (half of max_steps for 16)
    if len(val_dataset) > 0:
        trainer_kwargs["val_check_interval"] = max(
            1, (max_steps if max_steps is not None else 128) // 8
        )
        # Limit validation batches to match training steps for fair comparison
        if max_steps is not None:
            trainer_kwargs["limit_val_batches"] = max_steps

    # Limit test batches to match training steps for fair comparison
    if max_steps is not None:
        trainer_kwargs["limit_test_batches"] = max_steps

    trainer = pl.Trainer(**trainer_kwargs)

    # Train the model
    start_time = time.time()
    trainer.fit(model, train_dataloader, val_dataloader)

    # Test the model
    test_results = trainer.test(model, test_dataloader)

    training_time = time.time() - start_time

    # Save results
    results = {
        "seed": 8080,
        "use_genetic": use_genetic,
        "max_steps": max_steps,
        "total_steps": trainer.global_step,
        "epochs_completed": trainer.current_epoch + 1,
        "final_train_loss": (
            test_results[0].get("test_loss", None) if test_results else None
        ),
        "best_val_loss": (
            checkpoint_callback.best_model_score.item()
            if checkpoint_callback.best_model_score is not None
            else None
        ),
        "final_test_loss": (
            test_results[0].get("test_loss", None) if test_results else None
        ),
        "final_test_mrr": (
            test_results[0].get("test_mrr", None) if test_results else None
        ),
        "final_test_precision@1": (
            test_results[0].get("test_precision@1", None) if test_results else None
        ),
        "final_test_precision@5": (
            test_results[0].get("test_precision@5", None) if test_results else None
        ),
        "final_test_recall@1": (
            test_results[0].get("test_recall@1", None) if test_results else None
        ),
        "final_test_recall@5": (
            test_results[0].get("test_recall@5", None) if test_results else None
        ),
        "final_test_map": (
            test_results[0].get("test_map", None) if test_results else None
        ),
        "final_test_ndcg@5": (
            test_results[0].get("test_ndcg@5", None) if test_results else None
        ),
        "final_test_hit_rate@5": (
            test_results[0].get("test_hit_rate@5", None) if test_results else None
        ),
        "final_test_avg_positive_similarity": (
            test_results[0].get("test_avg_positive_similarity", None)
            if test_results
            else None
        ),
        "final_test_query_norm": (
            test_results[0].get("test_query_norm", None) if test_results else None
        ),
        "final_test_passage_norm": (
            test_results[0].get("test_passage_norm", None) if test_results else None
        ),
        "final_test_embedding_variance": (
            test_results[0].get("test_embedding_variance", None)
            if test_results
            else None
        ),
        "training_time": training_time,
        "train_samples": len(train_data),
        "val_samples": len(val_data),
        "test_samples": len(test_data),
    }

    output_file = (
        metrics_dir
        / f"dual_encoder_{'genetic' if use_genetic else 'standard'}_results.json"
    )
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)

    print(f"Results saved to {output_file}")
    return results


def plot_training_curves(checkpoint_dir: Path, metrics_dir: Path) -> None:
    """Generate and save training curve plots for both configurations.

    Args:
        checkpoint_dir: Directory containing the lightning logs
        metrics_dir: Directory to save the plots
    """

    print("Generating training curve plots...")

    # Set seaborn style for better aesthetics
    sns.set_style("whitegrid")
    sns.set_palette("husl")

    # Find the latest log directories for both configurations
    genetic_logs_dir = checkpoint_dir / "lightning_logs_genetic"
    standard_logs_dir = checkpoint_dir / "lightning_logs_standard"

    if not genetic_logs_dir.exists() or not standard_logs_dir.exists():
        print("Warning: Log directories not found. Skipping plot generation.")
        return

    # Get the latest version for each configuration
    genetic_versions = [d for d in genetic_logs_dir.iterdir() if d.is_dir()]
    standard_versions = [d for d in standard_logs_dir.iterdir() if d.is_dir()]

    if not genetic_versions or not standard_versions:
        print("Warning: No log versions found. Skipping plot generation.")
        return

    # Sort by version number and get the latest
    genetic_latest = max(genetic_versions, key=lambda x: int(x.name.split("_")[-1]))
    standard_latest = max(standard_versions, key=lambda x: int(x.name.split("_")[-1]))

    # Read CSV files
    genetic_csv = genetic_latest / "metrics.csv"
    standard_csv = standard_latest / "metrics.csv"

    if not genetic_csv.exists() or not standard_csv.exists():
        print("Warning: Metrics CSV files not found. Skipping plot generation.")
        return

    try:
        genetic_df = pd.read_csv(genetic_csv)
        standard_df = pd.read_csv(standard_csv)
    except Exception as e:
        print(f"Warning: Error reading CSV files: {e}. Skipping plot generation.")
        return

    # Create plots directory
    plots_dir = metrics_dir
    plots_dir.mkdir(exist_ok=True)

    # Create a combined dataframe for easier plotting
    genetic_df["model"] = "Genetic Attention"
    standard_df["model"] = "Standard Attention"

    # Combined plot: Training and Validation Loss
    plt.figure(figsize=(16, 6))

    # Plot 1: Training Loss
    plt.subplot(1, 2, 1)
    if "train_loss" in genetic_df.columns and genetic_df["train_loss"].notna().any():
        genetic_steps = genetic_df["step"][genetic_df["train_loss"].notna()]
        genetic_loss = genetic_df["train_loss"][genetic_df["train_loss"].notna()]
        plt.plot(
            genetic_steps,
            genetic_loss,
            label="Genetic Attention",
            color=sns.color_palette()[0],
            linewidth=2,
            alpha=0.8,
        )

    if "train_loss" in standard_df.columns and standard_df["train_loss"].notna().any():
        standard_steps = standard_df["step"][standard_df["train_loss"].notna()]
        standard_loss = standard_df["train_loss"][standard_df["train_loss"].notna()]
        plt.plot(
            standard_steps,
            standard_loss,
            label="Standard Attention",
            color=sns.color_palette()[1],
            linewidth=2,
            alpha=0.8,
        )

    plt.xlabel("Training Step", fontsize=12)
    plt.ylabel("Training Loss", fontsize=12)
    plt.title("Training Loss Curves", fontsize=14, fontweight="bold")
    plt.legend(frameon=True, fancybox=True, shadow=True)
    plt.grid(True, alpha=0.3)

    # Plot 2: Validation Loss
    plt.subplot(1, 2, 2)
    if "val_loss" in genetic_df.columns and genetic_df["val_loss"].notna().any():
        genetic_val_steps = genetic_df["step"][genetic_df["val_loss"].notna()]
        genetic_val_loss = genetic_df["val_loss"][genetic_df["val_loss"].notna()]
        plt.plot(
            genetic_val_steps,
            genetic_val_loss,
            label="Genetic Attention",
            color=sns.color_palette()[0],
            linewidth=2,
            alpha=0.8,
            marker="o",
            markersize=4,
        )

    if "val_loss" in standard_df.columns and standard_df["val_loss"].notna().any():
        standard_val_steps = standard_df["step"][standard_df["val_loss"].notna()]
        standard_val_loss = standard_df["val_loss"][standard_df["val_loss"].notna()]
        plt.plot(
            standard_val_steps,
            standard_val_loss,
            label="Standard Attention",
            color=sns.color_palette()[1],
            linewidth=2,
            alpha=0.8,
            marker="o",
            markersize=4,
        )

    plt.xlabel("Training Step", fontsize=12)
    plt.ylabel("Validation Loss", fontsize=12)
    plt.title("Validation Loss Curves", fontsize=14, fontweight="bold")
    plt.legend(frameon=True, fancybox=True, shadow=True)
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(plots_dir / "loss_comparison.png", dpi=300, bbox_inches="tight")
    plt.close()

    # Create individual plots for each metric with training and validation subplots
    metrics_to_plot = [
        ("loss", "Loss"),
        ("mrr", "MRR"),
        ("precision@1", "Precision@1"),
        ("precision@5", "Precision@5"),
        ("recall@1", "Recall@1"),
        ("recall@5", "Recall@5"),
        ("ndcg@5", "NDCG@5"),
        ("hit_rate@5", "Hit Rate@5"),
    ]

    for metric_base, title in metrics_to_plot:
        train_col = f"train_{metric_base}"
        val_col = f"val_{metric_base}"
        test_col = f"test_{metric_base}"

        plt.figure(figsize=(16, 6))

        # Left subplot: Standard Attention
        plt.subplot(1, 2, 1)
        plt.title("Standard Attention", fontsize=14, fontweight="bold")

        # Plot standard's train_col if exists
        if train_col in standard_df.columns and standard_df[train_col].notna().any():
            steps = standard_df["step"][standard_df[train_col].notna()]
            values = standard_df[train_col][standard_df[train_col].notna()]
            plt.plot(
                steps,
                values,
                label="Training",
                color=sns.color_palette()[0],
                linewidth=2,
                alpha=0.8,
            )

        # Plot standard's val_col if exists, else test_col
        right_col = (
            val_col
            if val_col in standard_df.columns and standard_df[val_col].notna().any()
            else test_col
        )
        if (
            right_col
            and right_col in standard_df.columns
            and standard_df[right_col].notna().any()
        ):
            steps = standard_df["step"][standard_df[right_col].notna()]
            values = standard_df[right_col][standard_df[right_col].notna()]
            label = "Validation" if "val" in right_col else "Test"
            plt.plot(
                steps,
                values,
                label=label,
                color=sns.color_palette()[1],
                linewidth=2,
                alpha=0.8,
                marker="o" if "val" in right_col else "s",
                markersize=4,
            )

        plt.xlabel("Training Step", fontsize=12)
        plt.ylabel(title, fontsize=12)
        plt.legend(frameon=True, fancybox=True, shadow=True)
        plt.grid(True, alpha=0.3)

        # Right subplot: Genetic Attention
        plt.subplot(1, 2, 2)
        plt.title("Genetic Attention", fontsize=14, fontweight="bold")

        # Plot genetic's train_col if exists
        if train_col in genetic_df.columns and genetic_df[train_col].notna().any():
            steps = genetic_df["step"][genetic_df[train_col].notna()]
            values = genetic_df[train_col][genetic_df[train_col].notna()]
            plt.plot(
                steps,
                values,
                label="Training",
                color=sns.color_palette()[0],
                linewidth=2,
                alpha=0.8,
            )

        # Plot genetic's val_col if exists, else test_col
        right_col = (
            val_col
            if val_col in genetic_df.columns and genetic_df[val_col].notna().any()
            else test_col
        )
        if (
            right_col
            and right_col in genetic_df.columns
            and genetic_df[right_col].notna().any()
        ):
            steps = genetic_df["step"][genetic_df[right_col].notna()]
            values = genetic_df[right_col][genetic_df[right_col].notna()]
            label = "Validation" if "val" in right_col else "Test"
            plt.plot(
                steps,
                values,
                label=label,
                color=sns.color_palette()[1],
                linewidth=2,
                alpha=0.8,
                marker="o" if "val" in right_col else "s",
                markersize=4,
            )

        plt.xlabel("Training Step", fontsize=12)
        plt.ylabel(title, fontsize=12)
        plt.legend(frameon=True, fancybox=True, shadow=True)
        plt.grid(True, alpha=0.3)

        plt.tight_layout()
        safe_filename = (
            title.lower().replace(" ", "_").replace("@", "at") + "_comparison.png"
        )
        plt.savefig(plots_dir / safe_filename, dpi=300, bbox_inches="tight")
        plt.close()

    print(f"Training curve plots saved to {plots_dir}")


@app.command()
def run(
    max_steps: Optional[int] = typer.Option(
        512,
        "--max-steps",
        "-m",
        help="Maximum training steps per configuration (None = one full epoch)",
        min=4,
    ),
    batch_size: int = typer.Option(
        DEFAULT_BATCH_SIZE,
        "--batch-size",
        "-b",
        help="Batch size for training",
        min=1,
    ),
    learning_rate: float = typer.Option(
        DEFAULT_LEARNING_RATE,
        "--learning-rate",
        "-lr",
        help="Learning rate for AdamW optimizer",
        min=0.0,
    ),
    checkpoint_dir: Optional[str] = typer.Option(
        None,
        "--checkpoint-dir",
        "-c",
        help="Directory to save model checkpoints and logs (default: examples/genetic_attention/.artefacts)",
    ),
    metrics_dir: Optional[str] = typer.Option(
        None,
        "--metrics-dir",
        "-md",
        help="Directory to save metrics and results (default: examples/genetic_attention/artefacts)",
    ),
) -> None:
    """
    Train dual-encoder BERT models on MS-MARCO dataset.

    This command runs training for both genetic attention enabled and disabled
    configurations sequentially, comparing their performance on text retrieval tasks.
    """
    # Set default directories
    script_dir = Path(__file__).parent
    if checkpoint_dir is None:
        checkpoint_path = script_dir / ".artefacts"
    else:
        checkpoint_path = Path(checkpoint_dir)

    if metrics_dir is None:
        metrics_path = script_dir / "artefacts"
    else:
        metrics_path = Path(metrics_dir)

    # Load tokenizer
    if AutoTokenizer is None:
        raise ImportError("transformers not installed")
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

    # Load and process datasets once for both configurations
    train_dataset, val_dataset, test_dataset, train_data, val_data, test_data = (
        load_and_process_datasets(tokenizer)
    )

    # Run training for both configurations
    for use_genetic in [False, True]:
        typer.echo(f"\n{'=' * 50}")
        typer.echo(
            f"Starting training with genetic={'enabled' if use_genetic else 'disabled'}"
        )
        typer.echo(f"{'=' * 50}\n")

        train_dual_encoder(
            use_genetic=use_genetic,
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            test_dataset=test_dataset,
            train_data=train_data,
            val_data=val_data,
            test_data=test_data,
            tokenizer=tokenizer,
            max_steps=max_steps,
            batch_size=batch_size,
            learning_rate=learning_rate,
            checkpoint_dir=checkpoint_path,
            metrics_dir=metrics_path,
        )

    # Generate training curve plots
    plot_training_curves(checkpoint_path, metrics_path)

    typer.echo(f"\n{'=' * 50}")
    typer.echo("Training completed for both configurations!")
    typer.echo(f"Checkpoints and logs saved in {checkpoint_path}")
    typer.echo(f"Metrics and results saved in {metrics_path}")
    typer.echo(f"Seaborn training curve plots saved in {metrics_path}")
    typer.echo(f"{'=' * 50}")


if __name__ == "__main__":
    app()
