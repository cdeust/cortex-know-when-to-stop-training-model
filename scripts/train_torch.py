#!/usr/bin/env python3
"""Train the abstention classifier using PyTorch + HuggingFace Trainer.

Fine-tunes DistilBERT for binary classification (relevant/irrelevant)
on community-contributed (query, passage, label) pairs.

Usage:
    python scripts/train_torch.py --data data/ --epochs 3 --lr 2e-5
    python scripts/train_torch.py --data data/ --output checkpoints/v1 --eval
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

try:
    import torch
    from datasets import Dataset
    from transformers import (
        AutoModelForSequenceClassification,
        AutoTokenizer,
        Trainer,
        TrainingArguments,
    )
except ImportError:
    print("PyTorch/transformers not installed.")
    print("Install with: pip install cortex-abstention[torch]")
    sys.exit(1)


def load_jsonl_data(data_dir: Path) -> list[dict]:
    """Load all JSONL files from data directory."""
    records = []
    for jsonl in sorted(data_dir.rglob("*.jsonl")):
        with open(jsonl) as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        rec = json.loads(line)
                        if "query" in rec and "passage" in rec and "label" in rec:
                            records.append(rec)
                    except json.JSONDecodeError:
                        continue
    return records


def prepare_dataset(records: list[dict], tokenizer, max_length: int = 256):
    """Convert records to HuggingFace Dataset."""
    queries = [r["query"] for r in records]
    passages = [r["passage"] for r in records]
    labels = [1 if r["label"] == "relevant" else 0 for r in records]

    encodings = tokenizer(
        queries,
        passages,
        padding="max_length",
        truncation=True,
        max_length=max_length,
    )
    encodings["labels"] = labels
    return Dataset.from_dict(encodings)


def compute_metrics(eval_pred):
    """Compute precision, recall, F1 for evaluation."""
    import numpy as np

    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)

    tp = ((preds == 1) & (labels == 1)).sum()
    fp = ((preds == 1) & (labels == 0)).sum()
    fn = ((preds == 0) & (labels == 1)).sum()

    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-6)
    accuracy = (preds == labels).mean()

    return {
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
    }


def main():
    parser = argparse.ArgumentParser(description="Train abstention classifier (PyTorch)")
    parser.add_argument("--data", type=Path, default=Path("data"), help="Data directory")
    parser.add_argument("--output", type=Path, default=Path("checkpoints/best"))
    parser.add_argument("--base-model", default="distilbert-base-uncased")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--eval", action="store_true", help="Run evaluation after training")
    args = parser.parse_args()

    # Load data
    records = load_jsonl_data(args.data)
    if not records:
        print(f"No JSONL data found in {args.data}")
        sys.exit(1)

    # Split 90/10
    val_size = max(1, int(len(records) * 0.1))
    train_records = records[val_size:]
    val_records = records[:val_size]

    print(f"Training: {len(train_records)} pairs")
    print(f"Validation: {len(val_records)} pairs")
    print(f"Label distribution (train): "
          f"{sum(1 for r in train_records if r['label'] == 'relevant')} relevant, "
          f"{sum(1 for r in train_records if r['label'] == 'irrelevant')} irrelevant")

    # Load model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    model = AutoModelForSequenceClassification.from_pretrained(
        args.base_model, num_labels=2
    )

    # Prepare datasets
    train_ds = prepare_dataset(train_records, tokenizer, args.max_length)
    val_ds = prepare_dataset(val_records, tokenizer, args.max_length)

    # Training
    training_args = TrainingArguments(
        output_dir=str(args.output),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        learning_rate=args.lr,
        weight_decay=0.01,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        logging_steps=10,
        save_total_limit=2,
        fp16=torch.cuda.is_available(),
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        compute_metrics=compute_metrics,
    )

    print(f"\nTraining {args.base_model} for {args.epochs} epochs...")
    trainer.train()

    # Save best model
    trainer.save_model(str(args.output))
    tokenizer.save_pretrained(str(args.output))
    print(f"\nModel saved to {args.output}")

    # Final evaluation
    if args.eval:
        results = trainer.evaluate()
        print(f"\nFinal evaluation: {results}")


if __name__ == "__main__":
    main()
