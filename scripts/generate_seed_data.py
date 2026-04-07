#!/usr/bin/env python3
"""Generate seed training data from BEAM benchmark.

Extracts (query, passage, label) pairs from BEAM conversations:
- Abstention questions → paired with random turns as irrelevant
- Other questions with source matches → paired as relevant
- Non-matching results → paired as irrelevant (hard negatives)

Usage:
    python scripts/generate_seed_data.py --output data/seed/beam.jsonl
    python scripts/generate_seed_data.py --split 100K --limit 5
"""

from __future__ import annotations

import argparse
import ast
import json
import sys


def _parse_chat(chat) -> dict[int, str]:
    """Parse BEAM chat into {turn_id: content} dict."""
    if isinstance(chat, str):
        try:
            chat = ast.literal_eval(chat) if chat.startswith("[") else json.loads(chat)
        except (ValueError, json.JSONDecodeError):
            return {}

    turns: dict[int, str] = {}
    if isinstance(chat, list):
        # BEAM format: list of sessions, each session is list of turn dicts
        for item in chat:
            if isinstance(item, list):
                for turn in item:
                    if isinstance(turn, dict):
                        tid = turn.get("id")
                        content = turn.get("content", "")
                        if tid is not None and content and len(content) > 10:
                            turns[int(tid)] = content[:500]
            elif isinstance(item, dict):
                tid = item.get("id")
                content = item.get("content", "")
                if tid is not None and content and len(content) > 10:
                    turns[int(tid)] = content[:500]
    return turns


def _flatten_source_ids(raw) -> list[int]:
    """Extract integer source IDs from various formats."""
    ids = []
    if isinstance(raw, dict):
        for v in raw.values():
            if isinstance(v, list):
                ids.extend(v)
            elif isinstance(v, int):
                ids.append(v)
    elif isinstance(raw, list):
        for v in raw:
            if isinstance(v, int):
                ids.append(v)
            elif isinstance(v, list):
                ids.extend(x for x in v if isinstance(x, (int, float)))
    return [int(x) for x in ids if isinstance(x, (int, float))]


def _hard_negatives(
    query: str,
    turns: dict,
    exclude_ids: set,
    encoder,
    n: int,
) -> list[tuple[int, str]]:
    """Find the n most semantically similar non-source turns.

    Hard negatives are passages topically close to the query but
    not the actual answer. These force the model to learn the
    distinction between "looks relevant" and "is relevant".
    """
    import numpy as np

    candidates = [(tid, content) for tid, content in turns.items() if tid not in exclude_ids]
    if not candidates:
        return []

    q_emb = encoder.encode(query, convert_to_numpy=True, show_progress_bar=False)
    cand_texts = [c[:500] for _, c in candidates]
    cand_embs = encoder.encode(cand_texts, convert_to_numpy=True, show_progress_bar=False, batch_size=32)

    # Cosine similarity
    q_norm = np.linalg.norm(q_emb)
    c_norms = np.linalg.norm(cand_embs, axis=1)
    sims = (cand_embs @ q_emb) / (c_norms * q_norm + 1e-10)

    # Top-n most similar
    top_idx = np.argsort(-sims)[:n]
    return [candidates[i] for i in top_idx]


def main():
    parser = argparse.ArgumentParser(description="Generate seed data from BEAM")
    parser.add_argument("--output", type=str, default="data/seed/beam.jsonl")
    parser.add_argument(
        "--splits",
        default="100K,500K,1M",
        help="Comma-separated BEAM splits to use",
    )
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument(
        "--hard-negatives",
        type=int,
        default=3,
        help="Number of hard negatives per question (default 3)",
    )
    parser.add_argument(
        "--easy-negatives",
        type=int,
        default=1,
        help="Number of random easy negatives per question (default 1)",
    )
    args = parser.parse_args()

    try:
        from datasets import load_dataset
        from sentence_transformers import SentenceTransformer
    except ImportError:
        print("Install: pip install datasets sentence-transformers")
        sys.exit(1)

    print("Loading sentence-transformer for hard negative mining...")
    # Force CPU — MPS has a Metal assertion bug with this model
    encoder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device="cpu")

    # Load all requested splits and concatenate
    all_convs = []
    for split in args.splits.split(","):
        split = split.strip()
        try:
            sub = load_dataset("Mohammadta/BEAM", split=split)
            for c in sub:
                all_convs.append(c)
            print(f"  Loaded {split}: {len(sub)} conversations")
        except Exception as e:
            print(f"  Failed {split}: {e}")

    if args.limit:
        all_convs = all_convs[: args.limit]
    ds = all_convs
    print(f"Total conversations: {len(ds)}")

    records = []
    for conv_idx, conv in enumerate(ds):
        turns = _parse_chat(conv.get("chat", ""))
        if not turns:
            continue

        raw_pq = conv.get("probing_questions", "{}")
        if isinstance(raw_pq, str):
            try:
                pq = ast.literal_eval(raw_pq)
            except (ValueError, SyntaxError):
                try:
                    pq = json.loads(raw_pq)
                except (ValueError, TypeError):
                    continue
        else:
            pq = raw_pq

        turn_ids = list(turns.keys())

        for ability, qs in pq.items():
            if not isinstance(qs, list):
                qs = [qs]
            for q in qs:
                if not isinstance(q, dict):
                    continue
                query = q.get("question", "")
                if not query or len(query) < 10:
                    continue

                source_ids = _flatten_source_ids(q.get("source_chat_ids", []))

                if ability == "abstention":
                    # Abstention: hardest case — query asks about info NOT in
                    # the conversation. The closest passages are the most
                    # informative training signal (the model must learn to
                    # reject them despite high topical similarity).
                    hard = _hard_negatives(query, turns, set(), encoder, n=2)
                    for _, content in hard:
                        records.append({
                            "query": query,
                            "passage": content,
                            "label": "irrelevant",
                            "source": "beam",
                            "ability": ability,
                        })
                elif source_ids:
                    # All source turns are relevant (not just first 2)
                    rel_count = 0
                    for sid in source_ids:
                        if sid in turns:
                            records.append({
                                "query": query,
                                "passage": turns[sid],
                                "label": "relevant",
                                "source": "beam",
                                "ability": ability,
                            })
                            rel_count += 1

                    # Negatives: balanced 1:1 with positives.
                    # Mix of hard (semantically close) and easy (random).
                    # Both teach important signals: hard = boundary, easy = clear noise.
                    n_negs = max(rel_count, 2)
                    n_hard = min(args.hard_negatives, n_negs - 1)
                    n_easy = n_negs - n_hard

                    exclude = set(source_ids)
                    hard = _hard_negatives(query, turns, exclude, encoder, n=n_hard)
                    for _, content in hard:
                        records.append({
                            "query": query,
                            "passage": content,
                            "label": "irrelevant",
                            "source": "beam",
                            "ability": ability,
                        })

                    if n_easy > 0:
                        import random

                        hard_ids = {h[0] for h in hard}
                        non_source = [
                            tid for tid in turn_ids
                            if tid not in source_ids and tid not in hard_ids
                        ]
                        random.shuffle(non_source)
                        for tid in non_source[:n_easy]:
                            records.append({
                                "query": query,
                                "passage": turns[tid],
                                "label": "irrelevant",
                                "source": "beam",
                                "ability": ability,
                            })

        if (conv_idx + 1) % 5 == 0:
            print(f"  [{conv_idx + 1}/{len(ds)}] {len(records)} pairs so far")

    # Write JSONL
    with open(args.output, "w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")

    rel = sum(1 for r in records if r["label"] == "relevant")
    irr = sum(1 for r in records if r["label"] == "irrelevant")
    print(f"Generated {len(records)} pairs: {rel} relevant, {irr} irrelevant")
    print(f"Written to {args.output}")


if __name__ == "__main__":
    main()
