"""P0.2 — Cross-encoder denoising audit (RocketQA, Qu et al. NAACL 2021).

Re-scores every "irrelevant" pair in data/seed/beam.jsonl with the FlashRank
ms-marco-MiniLM-L-12-v2 cross-encoder. Reports the distribution of cross-encoder
scores for the negatives, and how many negatives the cross-encoder thinks are
actually relevant relative to the positive distribution.

Hypothesis (RocketQA arxiv 2010.08191): ~30% of cosine-mined hard negatives
are actually false negatives. v0.1 filters ~32% of true-relevant passages on
BEAM — these numbers should match if the failure mode is the same.

Output:
  data/seed/beam.scored.jsonl  — original pairs + cross_encoder_score
  Console: distribution stats, suggested denoising threshold, flip count.

Usage:
  python3 scripts/denoise_audit.py
"""
from __future__ import annotations

import json
import statistics
import sys
from pathlib import Path

INPUT = Path(__file__).parent.parent / "data" / "seed" / "beam.jsonl"
OUTPUT = Path(__file__).parent.parent / "data" / "seed" / "beam.scored.jsonl"


def main() -> int:
    try:
        from flashrank import Ranker, RerankRequest
    except ImportError:
        print("ERROR: pip install flashrank", file=sys.stderr)
        return 1

    print(f"Loading {INPUT}...")
    pairs = [json.loads(l) for l in INPUT.open()]
    print(f"  {len(pairs)} pairs ({sum(1 for p in pairs if p['label']=='relevant')} relevant, "
          f"{sum(1 for p in pairs if p['label']=='irrelevant')} irrelevant)")

    print("Loading FlashRank ms-marco-MiniLM-L-12-v2...")
    ranker = Ranker(model_name="ms-marco-MiniLM-L-12-v2")

    # Group by query so we can rerank in one call per query (FlashRank API)
    from collections import defaultdict
    by_query: dict[str, list[int]] = defaultdict(list)
    for i, p in enumerate(pairs):
        by_query[p["query"]].append(i)

    print(f"  {len(by_query)} unique queries")
    print("Scoring (this may take a few minutes)...")

    scores: list[float] = [0.0] * len(pairs)
    for n, (query, idxs) in enumerate(by_query.items()):
        if n % 50 == 0:
            print(f"  {n}/{len(by_query)} queries", flush=True)
        passages = [{"id": i, "text": pairs[i]["passage"]} for i in idxs]
        req = RerankRequest(query=query, passages=passages)
        results = ranker.rerank(req)
        for r in results:
            scores[r["id"]] = float(r["score"])

    # Write annotated file
    with OUTPUT.open("w") as f:
        for p, s in zip(pairs, scores, strict=True):
            f.write(json.dumps({**p, "ce_score": s}) + "\n")
    print(f"\nWrote {OUTPUT}")

    # ── Analysis ──────────────────────────────────────────────────────
    pos_scores = [s for p, s in zip(pairs, scores, strict=True) if p["label"] == "relevant"]
    neg_scores = [s for p, s in zip(pairs, scores, strict=True) if p["label"] == "irrelevant"]

    def stats(name: str, xs: list[float]) -> None:
        xs_sorted = sorted(xs)
        print(f"  {name:12s} n={len(xs):6d}  "
              f"min={xs_sorted[0]:.3f}  "
              f"p25={xs_sorted[len(xs)//4]:.3f}  "
              f"median={statistics.median(xs):.3f}  "
              f"p75={xs_sorted[3*len(xs)//4]:.3f}  "
              f"max={xs_sorted[-1]:.3f}  "
              f"mean={statistics.mean(xs):.3f}")

    print("\n── Cross-encoder score distribution ──")
    stats("relevant", pos_scores)
    stats("irrelevant", neg_scores)

    # Denoising thresholds: if a "negative" scores above the median positive,
    # the cross-encoder thinks it's actually relevant — flag as false negative.
    pos_median = statistics.median(pos_scores)
    pos_p25 = sorted(pos_scores)[len(pos_scores) // 4]

    flips_strict = sum(1 for s in neg_scores if s > pos_median)
    flips_loose = sum(1 for s in neg_scores if s > pos_p25)

    print("\n── False negative analysis (RocketQA-style) ──")
    print(f"  Negatives above pos-median ({pos_median:.3f}): "
          f"{flips_strict}/{len(neg_scores)} = {100*flips_strict/len(neg_scores):.1f}%")
    print(f"  Negatives above pos-p25 ({pos_p25:.3f}):    "
          f"{flips_loose}/{len(neg_scores)} = {100*flips_loose/len(neg_scores):.1f}%")
    print(f"\n  RocketQA reported ~30% on MS MARCO. v0.1 filters 32% of true-relevant on BEAM.")
    print(f"  If our flip rate is in this ballpark, P0.2 (denoising) is the right intervention.")

    return 0


if __name__ == "__main__":
    sys.exit(main())
