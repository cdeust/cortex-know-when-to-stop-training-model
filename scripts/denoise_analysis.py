"""Per-query ranking analysis of cross-encoder scores from beam.scored.jsonl.

The first audit found that pos/neg median scores both = 0.000, meaning the
cross-encoder (ms-marco-MiniLM-L-12-v2) largely disagrees with the labels
in both directions. That's expected given domain shift (MS MARCO web Q&A
vs BEAM conversational memory) — but the question we actually care about
is: WITHIN A SINGLE QUERY, does the cross-encoder rank positives above
negatives? That's the signal that determines whether cross-encoder
relabeling can clean the training data.
"""
from __future__ import annotations

import json
import statistics
from collections import defaultdict
from pathlib import Path

INPUT = Path(__file__).parent.parent / "data" / "seed" / "beam.scored.jsonl"


def main() -> None:
    pairs = [json.loads(l) for l in INPUT.open()]
    by_query: dict[str, list[dict]] = defaultdict(list)
    for p in pairs:
        by_query[p["query"]].append(p)

    # Per-query: how often is the highest-scored passage a positive?
    queries_with_both = 0
    pos_above_neg_max = 0  # query has at least one pos > all negs
    pos_top1 = 0           # top-scored is positive
    pos_mean_above_neg_mean = 0
    avg_rank_of_pos = []   # rank position of positives (0 = top)

    for q, items in by_query.items():
        pos = [i for i in items if i["label"] == "relevant"]
        neg = [i for i in items if i["label"] == "irrelevant"]
        if not pos or not neg:
            continue
        queries_with_both += 1

        max_pos = max(p["ce_score"] for p in pos)
        max_neg = max(n["ce_score"] for n in neg)
        if max_pos > max_neg:
            pos_above_neg_max += 1

        ranked = sorted(items, key=lambda x: x["ce_score"], reverse=True)
        if ranked[0]["label"] == "relevant":
            pos_top1 += 1

        if statistics.mean(p["ce_score"] for p in pos) > statistics.mean(n["ce_score"] for n in neg):
            pos_mean_above_neg_mean += 1

        # Rank of best positive
        for i, item in enumerate(ranked):
            if item["label"] == "relevant":
                avg_rank_of_pos.append(i / max(len(ranked) - 1, 1))  # normalized 0..1
                break

    print(f"Queries with both pos and neg: {queries_with_both}")
    print(f"\n── Cross-encoder agreement with labels ──")
    print(f"  Top-1 is positive:           {pos_top1}/{queries_with_both} = {100*pos_top1/queries_with_both:.1f}%")
    print(f"  Best pos > best neg:         {pos_above_neg_max}/{queries_with_both} = {100*pos_above_neg_max/queries_with_both:.1f}%")
    print(f"  Mean(pos) > mean(neg):       {pos_mean_above_neg_mean}/{queries_with_both} = {100*pos_mean_above_neg_mean/queries_with_both:.1f}%")
    print(f"  Avg normalized rank of best pos: {statistics.mean(avg_rank_of_pos):.3f} (0=top, 1=bottom)")

    # Random baseline: if labels are random wrt cross-encoder, top-1 hit rate = #pos/total
    avg_pos_frac = statistics.mean(
        sum(1 for i in items if i["label"] == "relevant") / len(items)
        for items in by_query.values()
        if any(i["label"] == "relevant" for i in items) and any(i["label"] == "irrelevant" for i in items)
    )
    print(f"\n  Random baseline (top-1):     {100*avg_pos_frac:.1f}%")
    print(f"  Cross-encoder lift over random: {100*(pos_top1/queries_with_both - avg_pos_frac):.1f} pts")

    # Histogram of CE scores for positives vs negatives
    print(f"\n── Score distribution by label ──")
    for label in ("relevant", "irrelevant"):
        scores = [p["ce_score"] for p in pairs if p["label"] == label]
        buckets = [0, 0, 0, 0, 0]  # 0, 0-0.1, 0.1-0.3, 0.3-0.7, 0.7-1
        for s in scores:
            if s == 0: buckets[0] += 1
            elif s < 0.1: buckets[1] += 1
            elif s < 0.3: buckets[2] += 1
            elif s < 0.7: buckets[3] += 1
            else: buckets[4] += 1
        total = len(scores)
        print(f"  {label:12s} (n={total})")
        for name, b in zip(["==0", "0-0.1", "0.1-0.3", "0.3-0.7", "0.7-1.0"], buckets):
            print(f"    {name:8s} {b:5d}  {100*b/total:5.1f}%")


if __name__ == "__main__":
    main()
