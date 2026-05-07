"""Bias-in-Bios multi-head BERT example with genetic adapter.

Demonstrates multi-task learning with a shared BERT encoder and separate
profession and gender classification heads on the Bias-in-Bios dataset.
Includes counterfactual flip-rate evaluation via pronoun swapping.

Requires: transformers, datasets, scikit-learn, torch, numpy
    pip install transformers datasets scikit-learn torch
"""
# Requires: transformers, datasets, scikit-learn, torch, numpy
# pip install transformers datasets scikit-learn torch

import numpy as np
import torch
import torch.nn as nn
import transformers
from datasets import load_dataset
from sklearn.metrics import accuracy_score, f1_score
from torch.utils.data import DataLoader
from transformers import (
    AutoTokenizer,
    BertModel,
    BertPreTrainedModel,
    default_data_collator,
)
from transformers.modeling_outputs import SequenceClassifierOutput

# 1) Load dataset (Bias-in-Bios)
ds = load_dataset("LabHC/bias_in_bios")

# Take the train/validation split if present; otherwise split
if "validation" not in ds:
    ds = ds["train"].train_test_split(test_size=0.1)  # type: ignore
    train_ds = ds["train"]
    val_ds = ds["test"]
else:
    train_ds = ds["train"]
    val_ds = ds["validation"]

# Inspect label fields - adjust keys if dataset uses different colnames
# We'll assume the dataset has e.g. 'text', 'profession', 'gender' or similar.
print("Example keys:", train_ds.column_names)  # type: ignore

# 2) Build label maps dynamically
# Adapt these key names if the dataset uses slightly different column names.
TEXT_COL = "text"
PROF_COL = "profession"  # change if different
GEND_COL = "gender"  # change if different


def build_label_maps(
    dataset, col: str
) -> tuple[dict[str, int], dict[int, str]]:
    """Build label-to-id and id-to-label mappings from a dataset column.

    Args:
        dataset: HuggingFace dataset split to scan for labels.
        col: Column name containing the categorical labels.

    Returns:
        Tuple of (label2id, id2label) dictionaries.
    """
    labels = sorted(list({ex[col] for ex in dataset}))
    label2id = {label: i for i, label in enumerate(labels)}
    id2label = {i: label for label, i in label2id.items()}
    return label2id, id2label


prof2id, prof_id2 = build_label_maps(train_ds, PROF_COL)
gender2id, gender_id2 = build_label_maps(train_ds, GEND_COL)

NUM_PROF = len(prof2id)
NUM_GENDER = len(gender2id)
print("Num professions:", NUM_PROF, "Num genders:", NUM_GENDER)

# 3) Tokenizer and model config
MODEL_NAME = "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)


# 4) Preprocess: tokenize + map labels
def preprocess(examples: dict) -> dict:
    """Tokenize text and map profession/gender labels to integer IDs.

    Args:
        examples: Batch dictionary from a HuggingFace dataset with TEXT_COL,
            PROF_COL and GEND_COL keys.

    Returns:
        Dictionary with tokenized fields plus ``profession_label`` and
        ``gender_label`` integer lists.
    """
    toks = tokenizer(
        examples[TEXT_COL], truncation=True, padding="max_length", max_length=128
    )
    toks["profession_label"] = [prof2id[p] for p in examples[PROF_COL]]
    toks["gender_label"] = [gender2id[g] for g in examples[GEND_COL]]
    return toks


train = train_ds.map(preprocess, batched=True, remove_columns=train_ds.column_names)  # type: ignore
val = val_ds.map(preprocess, batched=True, remove_columns=val_ds.column_names)  # type: ignore

train.set_format(type="torch")  # type: ignore
val.set_format(type="torch")  # type: ignore

# 5) Custom multi-head model (shared encoder + two heads)


class BertMultiHead(BertPreTrainedModel):
    """BERT model with shared encoder and separate profession/gender heads.

    Implements a multi-task classifier that shares a BERT backbone and adds
    two independent classification heads for profession and gender prediction.
    """

    def __init__(self, config, num_prof: int, num_gender: int) -> None:
        """Initialise the multi-head BERT model.

        Args:
            config: HuggingFace BERT config object.
            num_prof: Number of profession classes.
            num_gender: Number of gender classes.
        """
        super().__init__(config)
        self.bert = BertModel(config)
        hidden_size = config.hidden_size
        # Profession head
        self.prof_head = nn.Sequential(
            nn.Dropout(config.hidden_dropout_prob),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(config.hidden_dropout_prob),
            nn.Linear(hidden_size // 2, num_prof),
        )
        # Gender head
        self.gender_head = nn.Sequential(
            nn.Dropout(config.hidden_dropout_prob),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(config.hidden_dropout_prob),
            nn.Linear(hidden_size // 2, num_gender),
        )
        # Initialize
        self.post_init()

    def forward(
        self,
        input_ids: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        token_type_ids: torch.Tensor | None = None,
        profession_label: torch.Tensor | None = None,
        gender_label: torch.Tensor | None = None,
    ) -> tuple[SequenceClassifierOutput, dict[str, torch.Tensor]]:
        """Forward pass returning losses and per-head logits.

        Args:
            input_ids: Token IDs of shape (batch, seq_len).
            attention_mask: Attention mask of shape (batch, seq_len).
            token_type_ids: Token type IDs of shape (batch, seq_len).
            profession_label: Integer profession labels of shape (batch,).
            gender_label: Integer gender labels of shape (batch,).

        Returns:
            Tuple of (SequenceClassifierOutput with combined loss,
            dict with ``prof_logits`` and ``gender_logits`` tensors).
        """
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            return_dict=True,
        )
        pooled = outputs.pooler_output  # [batch, hidden]
        prof_logits = self.prof_head(pooled)
        gender_logits = self.gender_head(pooled)

        loss = None
        loss_fct = nn.CrossEntropyLoss()
        if profession_label is not None and gender_label is not None:
            loss_prof = loss_fct(prof_logits, profession_label)
            loss_gender = loss_fct(gender_logits, gender_label)
            # simple equal-weight sum; tune weights if needed
            loss = loss_prof + loss_gender

        return SequenceClassifierOutput(
            loss=loss,
            logits=None,  # combined not used
            hidden_states=None,
            attentions=None,
            # We'll return logits in a dict below by custom Trainer compute
        ), {"prof_logits": prof_logits, "gender_logits": gender_logits}


# 6) Wrap model in a small helper to work with Trainer
config = transformers.AutoConfig.from_pretrained(MODEL_NAME)
base = BertMultiHead.from_pretrained(
    MODEL_NAME, config=config, num_prof=NUM_PROF, num_gender=NUM_GENDER
)

# 7) DataCollator: default collator works since dataset is torch tensors


# 8) Metrics: compute per-task acc, macro-F1, joint accuracy, per-gender gaps
def compute_metrics(p):
    # p is an EvalPrediction with predictions from Trainer; we'll receive raw logits via predict loop hack.
    # For simplicity in this example, we'll run prediction outside Trainer using model directly (see below).
    return {}


# 9) Custom training loop with Trainer requires custom predict step to get two heads.
# Simpler: implement a small PyTorch training loop

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = base.to(device)  # type: ignore
optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)
epochs = 3
batch_size = 16

train_loader = DataLoader(
    train,  # type: ignore[arg-type]
    batch_size=batch_size,
    shuffle=True,
    collate_fn=default_data_collator,  # type: ignore
)
val_loader = DataLoader(
    val,  # type: ignore[arg-type]
    batch_size=batch_size,
    shuffle=False,
    collate_fn=default_data_collator,  # type: ignore
)


def evaluate(
    model: BertMultiHead, loader: DataLoader
) -> dict[str, object]:
    """Evaluate the multi-head model on a data loader.

    Computes profession and gender accuracy, macro-F1, joint accuracy, and
    per-gender profession accuracy gap.

    Args:
        model: Trained BertMultiHead model.
        loader: DataLoader yielding tokenised batches with ``profession_label``
            and ``gender_label`` keys.

    Returns:
        Dictionary with keys: ``prof_acc``, ``prof_f1``, ``gender_acc``,
        ``gender_f1``, ``joint_acc``, ``per_gender_prof_acc``.
    """
    model.eval()
    prof_preds = []
    prof_trues = []
    gender_preds = []
    gender_trues = []
    with torch.no_grad():
        for batch in loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            prof_label = batch["profession_label"].to(device)
            gender_label = batch["gender_label"].to(device)
            outputs, logits_dict = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                profession_label=None,
                gender_label=None,
            )
            prof_logits = logits_dict["prof_logits"]
            gender_logits = logits_dict["gender_logits"]
            prof_pred = prof_logits.argmax(dim=-1).cpu().numpy()
            gender_pred = gender_logits.argmax(dim=-1).cpu().numpy()
            prof_preds.extend(prof_pred.tolist())
            prof_trues.extend(prof_label.cpu().numpy().tolist())
            gender_preds.extend(gender_pred.tolist())
            gender_trues.extend(gender_label.cpu().numpy().tolist())
    prof_acc = accuracy_score(prof_trues, prof_preds)
    prof_f1 = f1_score(prof_trues, prof_preds, average="macro")
    gender_acc = accuracy_score(gender_trues, gender_preds)
    gender_f1 = f1_score(gender_trues, gender_preds, average="macro")
    # joint accuracy: both correct
    joint = np.mean(
        [
            (p == t1) and (g == t2)
            for p, t1, g, t2 in zip(prof_preds, prof_trues, gender_preds, gender_trues)
        ]
    )
    # per-gender accuracy gap: compute profession accuracy conditioned on gender true label
    per_gender_acc = {}
    for gid, gname in gender_id2.items():
        idxs = [i for i, gt in enumerate(gender_trues) if gt == gid]
        if len(idxs) == 0:
            per_gender_acc[gname] = None
            continue
        acc = accuracy_score(
            [prof_trues[i] for i in idxs], [prof_preds[i] for i in idxs]
        )
        per_gender_acc[gname] = acc
    return {
        "prof_acc": prof_acc,
        "prof_f1": prof_f1,
        "gender_acc": gender_acc,
        "gender_f1": gender_f1,
        "joint_acc": joint,
        "per_gender_prof_acc": per_gender_acc,
    }


# 10) Training loop
for epoch in range(epochs):
    model.train()
    running_loss = 0.0
    for step, batch in enumerate(train_loader):
        optimizer.zero_grad()
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        prof_label = batch["profession_label"].to(device)
        gender_label = batch["gender_label"].to(device)
        outputs, logits_dict = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            profession_label=prof_label,
            gender_label=gender_label,
        )
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        if (step + 1) % 200 == 0:
            print(f"Epoch {epoch + 1} Step {step + 1} Loss {running_loss / 200:.4f}")
            running_loss = 0.0
    # Eval
    metrics = evaluate(model, val_loader)
    print(f"Epoch {epoch + 1} evaluation:", metrics)


# 11) Counterfactual flip-rate measurement (pronoun swap)
def swap_pronouns_text(text: str) -> str:
    """Naively swap he/she pronouns in text for counterfactual evaluation.

    Replaces all occurrences of " he " ↔ " she " and " He " ↔ " She " using
    a temporary placeholder to avoid double-swapping.

    Args:
        text: Input sentence to transform.

    Returns:
        Text with gendered pronouns swapped.
    """
    # naive; adapt for edge cases
    t = text
    t = (
        t.replace(" she ", " __TMP__ ")
        .replace(" he ", " she ")
        .replace(" __TMP__ ", " he ")
    )
    t = (
        t.replace(" She ", " __TMP__ ")
        .replace(" He ", " She ")
        .replace(" __TMP__ ", " He ")
    )
    return t


def compute_flip_rate(
    model: BertMultiHead,
    dataset,
    max_examples: int | None = None,
) -> float:
    """Compute the profession prediction flip rate after pronoun swapping.

    Measures what fraction of profession predictions change when gendered
    pronouns in the input text are swapped (he ↔ she).

    Args:
        model: Trained BertMultiHead model to evaluate.
        dataset: Tokenised HuggingFace dataset (torch format) to evaluate on.
        max_examples: If given, evaluate only this many examples. Currently
            unused (retained for API compatibility).

    Returns:
        Flip rate in [0, 1]: fraction of examples where the profession
        prediction changed after pronoun swapping.
    """
    model.eval()
    flips = 0
    total = 0
    loader = DataLoader(dataset, batch_size=32, collate_fn=default_data_collator)
    # get original predictions
    orig_preds = []
    for batch in loader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        _, logits_dict = model(input_ids=input_ids, attention_mask=attention_mask)
        prof_pred = logits_dict["prof_logits"].argmax(dim=-1).cpu().numpy().tolist()
        orig_preds.extend(prof_pred)
    # create swapped texts and re-tokenize
    swapped_texts = [
        swap_pronouns_text(x[TEXT_COL])  # type: ignore[index]
        for x in val_ds  # type: ignore
    ]  # careful: val_ds original object
    toks = tokenizer(
        swapped_texts,
        truncation=True,
        padding=True,
        max_length=128,
        return_tensors="pt",
    )
    toks = {k: v.to(device) for k, v in toks.items()}
    # batch predict swapped
    swapped_preds = []
    with torch.no_grad():
        B = 32
        n = toks["input_ids"].size(0)
        for i in range(0, n, B):
            ids = toks["input_ids"][i : i + B]
            mask = toks["attention_mask"][i : i + B]
            _, logits_dict = model(input_ids=ids, attention_mask=mask)
            swapped_preds.extend(
                logits_dict["prof_logits"].argmax(dim=-1).cpu().numpy().tolist()
            )
    # compare
    n = len(orig_preds)
    for i in range(n):
        total += 1
        if orig_preds[i] != swapped_preds[i]:
            flips += 1
    return flips / total


flip_rate = compute_flip_rate(model, val, max_examples=1000)
print("Counterfactual flip rate (profession) after pronoun swap:", flip_rate)
