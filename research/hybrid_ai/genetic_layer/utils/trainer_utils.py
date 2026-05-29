import time

import torch
import torch.nn as nn
from pytorch_lightning import LightningModule, Trainer


class LitClassifier(LightningModule):
    """PyTorch Lightning module for classification."""

    def __init__(self, model, learning_rate=1e-3, genetic_lr_multiplier=0.5):
        super().__init__()
        self.model = model
        self.lr = learning_rate
        self.genetic_lr_multiplier = genetic_lr_multiplier
        self.criterion = nn.CrossEntropyLoss()
        self.train_loss_history = []
        self.val_loss_history = []
        self.val_acc_history = []
        self.val_bal_acc_history = []
        self.val_y_true = []
        self.val_y_pred = []
        self.epoch_train_losses = []

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.criterion(y_hat, y)
        self.log("train_loss", loss)
        self.epoch_train_losses.append(loss)

        # Apply gradient clipping for genetic networks to prevent exploding gradients
        if hasattr(self.model, "genetic_layers") or hasattr(self.model, "heads"):
            torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)

        return loss

    def on_train_epoch_end(self):
        avg_loss = torch.stack(self.epoch_train_losses).mean()
        self.train_loss_history.append(avg_loss.item())
        self.epoch_train_losses = []

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.criterion(y_hat, y)
        acc = (y_hat.argmax(dim=1) == y).float().mean()
        self.log("val_loss", loss)
        self.log("val_acc", acc)
        # Collect for balanced accuracy
        self.val_y_true.extend(y.cpu().numpy())
        self.val_y_pred.extend(y_hat.argmax(dim=1).cpu().numpy())

    def on_validation_epoch_end(self):
        # Compute average loss and acc from logged metrics
        val_loss = self.trainer.callback_metrics.get("val_loss")
        val_acc = self.trainer.callback_metrics.get("val_acc")
        if val_loss is not None:
            self.val_loss_history.append(val_loss.item())
        if val_acc is not None:
            self.val_acc_history.append(val_acc.item())
        # Compute balanced accuracy using torch
        if self.val_y_true and self.val_y_pred:
            y_true = torch.tensor(self.val_y_true, dtype=torch.long)
            y_pred = torch.tensor(self.val_y_pred, dtype=torch.long)
            num_classes = len(torch.unique(y_true))
            recalls = []
            for c in range(num_classes):
                tp = ((y_pred == c) & (y_true == c)).sum().float()
                total = (y_true == c).sum().float()
                if total > 0:
                    recalls.append(tp / total)
                else:
                    recalls.append(torch.tensor(1.0))
            balanced_acc = torch.stack(recalls).mean().item()
            self.val_bal_acc_history.append(balanced_acc)
        # Reset for next epoch
        self.val_y_true = []
        self.val_y_pred = []

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)


def train_and_evaluate(
    model,
    train_loader,
    val_loader,
    max_epochs=10,
    learning_rate=1e-3,
    genetic_lr_multiplier=0.5,
):
    """Train the model and return evaluation metrics."""
    # Adjust learning rate multiplier based on architecture
    if hasattr(model, "genetic_layers"):  # Feedforward genetic network
        genetic_lr_multiplier = 0.9  # Less aggressive reduction for feedforward - they need higher learning rates
    elif hasattr(model, "heads"):  # Heads genetic network
        genetic_lr_multiplier = 0.5  # Keep more aggressive reduction for heads

    lit_model = LitClassifier(model, learning_rate, genetic_lr_multiplier)
    trainer = Trainer(
        max_epochs=max_epochs,
        enable_progress_bar=False,
        logger=False,
        enable_checkpointing=False,
    )

    start_time = time.time()
    trainer.fit(lit_model, train_loader, val_loader)
    training_time = time.time() - start_time

    # Get final metrics from callback_metrics
    final_val_loss = trainer.callback_metrics.get("val_loss", float("inf"))
    final_val_acc = trainer.callback_metrics.get("val_acc", 0.0)

    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    return {
        "training_time": training_time,
        "final_val_loss": (
            final_val_loss.item() if torch.is_tensor(final_val_loss) else final_val_loss
        ),
        "final_val_acc": (
            final_val_acc.item() if torch.is_tensor(final_val_acc) else final_val_acc
        ),
        "num_params": num_params,
        "train_loss_history": lit_model.train_loss_history,
        "val_loss_history": lit_model.val_loss_history,
        "val_acc_history": lit_model.val_acc_history,
        "val_bal_acc_history": lit_model.val_bal_acc_history,
    }
