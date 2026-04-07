"""P0.2 v2 — Re-score with BAAI/bge-reranker-v2-m3.

ms-marco-MiniLM-L-12-v2 collapsed (76% of positives scored <0.1) due to
domain shift from web Q&A to conversational memory. BGE-reranker-v2-m3 is
multilingual, trained on more diverse data including long-context, and is
the standard recommendation for non-MSMARCO domains.

Writes data/seed/beam.bge_scored.jsonl with bge_score added per pair.
"""
from __future__ import annotations

import json
import statistics
from collections import defaultdict
from pathlib import Path

INPUT = Path(__file__).parent.parent / "data" / "seed" / "beam.jsonl"
OUTPUT = Path(__file__).parent.parent / "data" / "seed" / "beam.bge_scored.jsonl"


def main() -> None:
    from sentence_transformers import CrossEncoder

    pairs = [json.loads(l) for l in INPUT.open()]
    print(f"Loaded {len(pairs)} pairs")

    print("Loading BAAI/bge-reranker-v2-m3 (first time downloads ~600MB)...")
    model = CrossEncoder("BAAI/bge-reranker-v2-m3", max_length=512, device="cpu")

    print("Scoring in batches of 64...")
    inputs = [(p["query"], p["passage"]) for p in pairs]
    scores = model.predict(inputs, batch_size=64, show_progress_bar=True)

    with OUTPUT.open("w") as f:
        for p, s in zip(pairs, scores, strict=True):
            f.write(json.dumps({**p, "bge_score": float(s)}) + "\n")
    print(f"Wrote {OUTPUT}")

    # Per-query ranking analysis
    by_query: dict[str, list[dict]] = defaultdict(list)
    for p, s in zip(pairs, scores, strict=True):
        by_query[p["query"]].append({**p, "bge_score": float(s)})

    queries_with_both = 0
    pos_top1 = 0
    pos_above_neg_max = 0
    for q, items in by_query.items():
        pos = [i for i in items if i["label"] == "relevant"]
        neg = [i for i in items if i["label"] == "irrelevant"]
        if not pos or not neg:
            continue
        queries_with_both += 1
        ranked = sorted(items, key=lambda x: x["bge_score"], reverse=True)
        if ranked[0]["label"] == "relevant":
            pos_top1 += 1
        if max(p["bge_score"] for p in pos) > max(n["bge_score"] for n in neg):
            pos_above_neg_max += 1

    avg_pos_frac = statistics.mean(
        sum(1 for i in items if i["label"] == "relevant") / len(items)
        for items in by_query.values()
        if any(i["label"] == "relevant" for i in items) and any(i["label"] == "irrelevant" for i in items)
    )

    print(f"\n── BGE-reranker-v2-m3 agreement with labels ──")
    print(f"  Queries with both pos and neg: {queries_with_both}")
    print(f"  Top-1 is positive:   {pos_top1}/{queries_with_both} = {100*pos_top1/queries_with_both:.1f}%")
    print(f"  Best pos > best neg: {pos_above_neg_max}/{queries_with_both} = {100*pos_above_neg_max/queries_with_both:.1f}%")
    print(f"  Random baseline:     {100*avg_pos_frac:.1f}%")
    print(f"  Lift over random:    {100*(pos_top1/queries_with_both - avg_pos_frac):.1f} pts")

    # Score distribution
    print(f"\n── Score distribution ──")
    for label in ("relevant", "irrelevant"):
        ss = [float(s) for p, s in zip(pairs, scores, strict=True) if p["label"] == label]
        ss_sorted = sorted(ss)
        print(f"  {label:12s} n={len(ss):5d}  "
              f"min={ss_sorted[0]:7.3f}  "
              f"p25={ss_sorted[len(ss)//4]:7.3f}  "
              f"median={statistics.median(ss):7.3f}  "
              f"p75={ss_sorted[3*len(ss)//4]:7.3f}  "
              f"max={ss_sorted[-1]:7.3f}  "
              f"mean={statistics.mean(ss):7.3f}")


if __name__ == "__main__":
    main()
