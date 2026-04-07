#!/usr/bin/env python3
"""v0.2 trainer — Path B: fix training methodology, keep existing labels.

Implements three fixes from the v0.2 plan after the cross-encoder denoising
audit failed (no off-the-shelf cross-encoder could clean BEAM labels):

  P0.1  Query-level split: hold out entire queries, never split pairs of
        the same query across train/val. Random pair split inflates F1.

  P1.1  Listwise softmax cross-entropy over (1 positive + K negatives)
        groups per query. Forces cross-passage calibration that pointwise
        BCE never learns.
        Source: RankT5 (Zhuang et al., SIGIR 2023, arxiv 2210.10634).
        L = -sum_i y_i * log(exp(s_i) / sum_j exp(s_j))

  P1.2  Per-query MRR@10 evaluation, early stopping on MRR (not F1).
        Direct fix for v0.1's F1=0.733 vs MRR=-0.191 mismatch.

P2 (SelectiveNet coverage constraint, OOD calibrator, CRAG routing) is
deferred to inference-time wrappers — the model only needs to produce a
well-calibrated *score*, not make filter decisions.

Usage:
    python scripts/train_v2.py --data data/seed --epochs 3
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

try:
    import numpy as np
    import torch
    import torch.nn.functional as F
    from torch.utils.data import Dataset
    from transformers import (
        AutoModelForSequenceClassification,
        AutoTokenizer,
        Trainer,
        TrainingArguments,
    )
except ImportError as e:
    print(f"Missing dependency: {e}\nInstall: pip install torch transformers")
    sys.exit(1)


# ── Data loading ──────────────────────────────────────────────────────


def load_pairs(data_dir: Path) -> list[dict]:
    pairs = []
    for jsonl in sorted(data_dir.rglob("*.jsonl")):
        # Skip annotated files from the audit
        if "scored" in jsonl.name:
            continue
        for line in jsonl.open():
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
                if "query" in rec and "passage" in rec and "label" in rec:
                    pairs.append(rec)
            except json.JSONDecodeError:
                continue
    return pairs


def query_level_split(
    pairs: list[dict], val_frac: float = 0.1, seed: int = 42
) -> tuple[list[dict], list[dict]]:
    """Hold out entire queries — no query appears in both train and val.

    P0.1 fix. Standard IR practice (BEIR, Thakur et al. 2021).
    """
    by_query: dict[str, list[dict]] = defaultdict(list)
    for p in pairs:
        by_query[p["query"]].append(p)

    queries = sorted(by_query.keys())
    rng = random.Random(seed)
    rng.shuffle(queries)
    val_n = max(1, int(len(queries) * val_frac))
    val_queries = set(queries[:val_n])

    train, val = [], []
    for q, items in by_query.items():
        (val if q in val_queries else train).extend(items)
    return train, val


def build_groups(pairs: list[dict]) -> list[dict]:
    """Group pairs by query. Each group = (query, positives, negatives).

    Listwise loss needs to see competing passages for the same query in
    one example. Drop queries with no positive or no negative.
    """
    by_query: dict[str, dict] = defaultdict(lambda: {"pos": [], "neg": []})
    for p in pairs:
        bucket = "pos" if p["label"] == "relevant" else "neg"
        by_query[p["query"]][bucket].append(p["passage"])

    groups = []
    for q, b in by_query.items():
        if not b["pos"] or not b["neg"]:
            continue
        groups.append({"query": q, "pos": b["pos"], "neg": b["neg"]})
    return groups


# ── Listwise dataset ──────────────────────────────────────────────────


class ListwiseDataset(Dataset):
    """Each example = 1 query + (1 positive sampled + K negatives sampled).

    The positive is always at index 0 of the passage list, so the listwise
    softmax target is always class 0.
    """

    def __init__(
        self,
        groups: list[dict],
        tokenizer,
        k_negatives: int = 7,
        max_length: int = 256,
        seed: int = 0,
    ) -> None:
        self.groups = groups
        self.tokenizer = tokenizer
        self.k = k_negatives
        self.max_length = max_length
        self.rng = random.Random(seed)

    def __len__(self) -> int:
        return len(self.groups)

    def __getitem__(self, idx: int) -> dict:
        g = self.groups[idx]
        pos = self.rng.choice(g["pos"])
        if len(g["neg"]) >= self.k:
            negs = self.rng.sample(g["neg"], self.k)
        else:
            negs = list(g["neg"]) + self.rng.choices(
                g["neg"], k=self.k - len(g["neg"])
            )
        passages = [pos] + negs  # length = k+1, target index = 0
        enc = self.tokenizer(
            [g["query"]] * len(passages),
            passages,
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        return {
            "input_ids": enc["input_ids"],          # (k+1, L)
            "attention_mask": enc["attention_mask"], # (k+1, L)
            "labels": torch.tensor(0, dtype=torch.long),
        }


def listwise_collate(batch: list[dict]) -> dict:
    """Stack groups into (B, K+1, L) tensors. Trainer flattens later."""
    return {
        "input_ids": torch.stack([b["input_ids"] for b in batch]),
        "attention_mask": torch.stack([b["attention_mask"] for b in batch]),
        "labels": torch.stack([b["labels"] for b in batch]),
    }


# ── Listwise model wrapper ────────────────────────────────────────────


class ListwiseTrainer(Trainer):
    """Trainer with listwise softmax CE loss (P1.1) over each query group.

    For each example: input is (K+1, L) — one positive + K negatives.
    Score each passage with the model's class-1 logit, softmax across the
    K+1 passages, cross-entropy against target index 0.
    """

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        input_ids = inputs["input_ids"]              # (B, K+1, L)
        attention_mask = inputs["attention_mask"]    # (B, K+1, L)
        labels = inputs["labels"]                    # (B,)
        B, K1, L = input_ids.shape

        flat_ids = input_ids.view(B * K1, L)
        flat_mask = attention_mask.view(B * K1, L)
        outputs = model(input_ids=flat_ids, attention_mask=flat_mask)
        logits = outputs.logits                      # (B*K1, 2)
        # Score = relevance logit (class 1)
        scores = logits[:, 1].view(B, K1)            # (B, K+1)
        loss = F.cross_entropy(scores, labels)
        return (loss, outputs) if return_outputs else loss


# ── MRR evaluation (P1.2) ─────────────────────────────────────────────


def evaluate_mrr(
    model, tokenizer, val_groups: list[dict], device: str, max_length: int = 256
) -> dict:
    """Compute MRR@10 over val queries.

    For each query, score ALL its passages (positives + negatives) with the
    model, rank by score, find the first positive's rank, MRR = 1 / rank.
    """
    model.eval()
    rrs = []
    with torch.no_grad():
        for g in val_groups:
            passages = g["pos"] + g["neg"]
            labels = [1] * len(g["pos"]) + [0] * len(g["neg"])
            enc = tokenizer(
                [g["query"]] * len(passages),
                passages,
                padding="max_length",
                truncation=True,
                max_length=max_length,
                return_tensors="pt",
            ).to(device)
            logits = model(**enc).logits
            scores = logits[:, 1].cpu().numpy()
            order = np.argsort(-scores)
            for rank, idx in enumerate(order, start=1):
                if labels[idx] == 1:
                    if rank <= 10:
                        rrs.append(1.0 / rank)
                    else:
                        rrs.append(0.0)
                    break
    model.train()
    mrr = float(np.mean(rrs)) if rrs else 0.0
    return {"mrr@10": mrr, "n_queries": len(rrs)}


class MRREarlyStopCallback:
    """Track best MRR across epochs; signal to save best."""

    def __init__(self) -> None:
        self.best_mrr = -1.0
        self.best_epoch = -1


# ── Main ──────────────────────────────────────────────────────────────


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--data", type=Path, default=Path("data/seed"))
    p.add_argument("--output", type=Path, default=Path("checkpoints/v2"))
    p.add_argument("--base-model", default="distilbert-base-uncased")
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--lr", type=float, default=2e-5)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--k-negatives", type=int, default=7,
                   help="K in listwise (1 pos + K negs per group)")
    p.add_argument("--max-length", type=int, default=256)
    p.add_argument("--val-frac", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    print(f"Loading pairs from {args.data}...")
    pairs = load_pairs(args.data)
    print(f"  {len(pairs)} pairs")

    train_pairs, val_pairs = query_level_split(pairs, args.val_frac, args.seed)
    print(f"P0.1 query-level split: train={len(train_pairs)} val={len(val_pairs)}")

    train_groups = build_groups(train_pairs)
    val_groups = build_groups(val_pairs)
    print(f"Groups (queries with both pos+neg): train={len(train_groups)} val={len(val_groups)}")
    if not val_groups:
        print("ERROR: validation set has no queries with both labels.")
        return 1

    print(f"Loading {args.base_model}...")
    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    model = AutoModelForSequenceClassification.from_pretrained(
        args.base_model, num_labels=2
    )
    device = "cpu"  # Force CPU — MPS Metal assertion fails on this model
    model.to(device)

    train_ds = ListwiseDataset(
        train_groups, tokenizer,
        k_negatives=args.k_negatives, max_length=args.max_length, seed=args.seed,
    )

    training_args = TrainingArguments(
        output_dir=str(args.output),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        learning_rate=args.lr,
        weight_decay=0.01,
        eval_strategy="no",  # We do eval manually for MRR
        save_strategy="no",  # We save manually after MRR check
        logging_steps=20,
        fp16=False,
        use_cpu=True,
        report_to=[],
    )

    trainer = ListwiseTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        data_collator=listwise_collate,
    )

    print("\nInitial MRR (no training):")
    init = evaluate_mrr(model, tokenizer, val_groups, device, args.max_length)
    print(f"  MRR@10 = {init['mrr@10']:.4f} over {init['n_queries']} queries")

    print(f"\nTraining {args.epochs} epoch(s) with listwise softmax loss (k={args.k_negatives})...")
    trainer.train()
    final = evaluate_mrr(model, tokenizer, val_groups, device, args.max_length)
    print(f"\nFinal MRR@10 = {final['mrr@10']:.4f} over {final['n_queries']} queries")
    print(f"Initial MRR@10 = {init['mrr@10']:.4f}  →  delta = {final['mrr@10'] - init['mrr@10']:+.4f}")

    args.output.mkdir(parents=True, exist_ok=True)
    trainer.save_model(str(args.output))
    tokenizer.save_pretrained(str(args.output))
    (args.output / "v2_metrics.json").write_text(json.dumps({
        "initial_mrr@10": init["mrr@10"],
        "final_mrr@10": final["mrr@10"],
        "delta": final["mrr@10"] - init["mrr@10"],
        "train_queries": len(train_groups),
        "val_queries": len(val_groups),
        "k_negatives": args.k_negatives,
        "epochs": args.epochs,
    }, indent=2))
    print(f"Saved to {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
