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


def main():
    parser = argparse.ArgumentParser(description="Generate seed data from BEAM")
    parser.add_argument("--output", type=str, default="data/seed/beam.jsonl")
    parser.add_argument("--split", default="100K")
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    try:
        from datasets import load_dataset
    except ImportError:
        print("Install: pip install datasets")
        sys.exit(1)

    ds = load_dataset("Mohammadta/BEAM", split=args.split)
    if args.limit:
        ds = ds.select(range(min(args.limit, len(ds))))

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
                    # Pair with random turns as irrelevant
                    for tid in turn_ids[:3]:
                        records.append({
                            "query": query,
                            "passage": turns[tid],
                            "label": "irrelevant",
                            "source": "beam",
                            "ability": ability,
                        })
                elif source_ids:
                    # Source turns are relevant
                    for sid in source_ids[:2]:
                        if sid in turns:
                            records.append({
                                "query": query,
                                "passage": turns[sid],
                                "label": "relevant",
                                "source": "beam",
                                "ability": ability,
                            })
                    # Non-source turns as hard negatives
                    neg_count = 0
                    for tid in turn_ids:
                        if tid not in source_ids and neg_count < 2:
                            records.append({
                                "query": query,
                                "passage": turns[tid],
                                "label": "irrelevant",
                                "source": "beam",
                                "ability": ability,
                            })
                            neg_count += 1

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
