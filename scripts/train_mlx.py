#!/usr/bin/env python3
"""Train the abstention classifier using Apple MLX.

Fine-tunes DistilBERT on (query, passage, label) pairs using LoRA
for efficient training on Apple Silicon.

Usage:
    python scripts/train_mlx.py --data data/ --epochs 3 --lr 2e-5
    python scripts/train_mlx.py --data data/ --output checkpoints/v1
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

try:
    import mlx.core as mx
    import mlx.nn as nn
    import mlx.optimizers as optim
except ImportError:
    print("MLX not installed. Install with: pip install mlx mlx-lm")
    print("Requires Apple Silicon (M1/M2/M3/M4).")
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
                        records.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue
    return records


def split_data(
    records: list[dict], val_ratio: float = 0.1
) -> tuple[list[dict], list[dict]]:
    """Split into train/val, stratified by label."""
    relevant = [r for r in records if r.get("label") == "relevant"]
    irrelevant = [r for r in records if r.get("label") == "irrelevant"]

    def split(items: list, ratio: float):
        n = max(1, int(len(items) * ratio))
        return items[n:], items[:n]

    train_r, val_r = split(relevant, val_ratio)
    train_i, val_i = split(irrelevant, val_ratio)
    return train_r + train_i, val_r + val_i


def main():
    parser = argparse.ArgumentParser(description="Train abstention classifier (MLX)")
    parser.add_argument("--data", type=Path, default=Path("data"), help="Data directory")
    parser.add_argument("--output", type=Path, default=Path("checkpoints/latest"))
    parser.add_argument("--base-model", default="distilbert-base-uncased")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--lora-rank", type=int, default=8)
    args = parser.parse_args()

    # Load data
    records = load_jsonl_data(args.data)
    if not records:
        print(f"No JSONL data found in {args.data}")
        sys.exit(1)

    train_data, val_data = split_data(records)
    print(f"Training: {len(train_data)} pairs, Validation: {len(val_data)} pairs")

    # Load tokenizer
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.base_model)

    # Tokenize
    def tokenize(batch: list[dict]):
        queries = [r["query"] for r in batch]
        passages = [r["passage"] for r in batch]
        labels = [1 if r["label"] == "relevant" else 0 for r in batch]
        encoded = tokenizer(
            queries, passages,
            padding="max_length",
            truncation=True,
            max_length=args.max_length,
            return_tensors="np",
        )
        return {
            "input_ids": mx.array(encoded["input_ids"]),
            "attention_mask": mx.array(encoded["attention_mask"]),
            "labels": mx.array(labels),
        }

    # MLX LoRA training loop
    print(f"Training with MLX LoRA (rank={args.lora_rank})")
    print(f"Base model: {args.base_model}")
    print(f"Epochs: {args.epochs}, LR: {args.lr}, Batch: {args.batch_size}")

    # Note: Full MLX DistilBERT fine-tuning requires mlx-lm model conversion.
    # This script provides the training loop structure.
    # For the initial release, use scripts/train_torch.py which works with
    # standard HuggingFace Trainer, then export to ONNX.
    print()
    print("MLX LoRA training for DistilBERT is under development.")
    print("For now, use: python scripts/train_torch.py --data data/")
    print("Then export: python scripts/export_onnx.py --checkpoint checkpoints/best")


if __name__ == "__main__":
    main()
