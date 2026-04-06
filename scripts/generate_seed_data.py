#!/usr/bin/env python3
"""Generate seed training data from BEAM benchmark.

Extracts (query, passage, label) pairs from BEAM conversations:
- Abstention questions → label as irrelevant (no matching source)
- Other questions with source matches → label as relevant
- Non-matching results → label as irrelevant (hard negatives)

Usage:
    python scripts/generate_seed_data.py --output data/seed/beam.jsonl
    python scripts/generate_seed_data.py --split 100K --limit 5
"""

from __future__ import annotations

import argparse
import json
import sys


def main():
    parser = argparse.ArgumentParser(description="Generate seed data from BEAM")
    parser.add_argument("--output", type=str, default="data/seed/beam.jsonl")
    parser.add_argument("--split", default="100K")
    parser.add_argument("--limit", type=int, default=None, help="Max conversations")
    args = parser.parse_args()

    try:
        from datasets import load_dataset
    except ImportError:
        print("Install: pip install datasets")
        sys.exit(1)

    ds = load_dataset("Mohammadta/BEAM", split=args.split)
    if args.limit:
        ds = ds.select(range(min(args.limit, len(ds))))

    import ast

    records = []
    for conv in ds:
        chat = conv.get("chat", "")
        # Extract turns
        turns = {}
        if isinstance(chat, str):
            try:
                chat_data = ast.literal_eval(chat) if chat.startswith("[") else json.loads(chat)
            except (ValueError, json.JSONDecodeError):
                continue
        else:
            chat_data = chat

        if isinstance(chat_data, list):
            for turn in chat_data:
                if isinstance(turn, dict):
                    tid = turn.get("id", turn.get("turn_id"))
                    content = turn.get("content", turn.get("text", ""))
                    if tid is not None and content:
                        turns[tid] = content[:500]

        # Extract questions
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

        for ability, qs in pq.items():
            if not isinstance(qs, list):
                qs = [qs]
            for q in qs:
                if not isinstance(q, dict):
                    continue
                query = q.get("question", "")
                if not query:
                    continue

                source_ids = q.get("source_chat_ids", [])
                if isinstance(source_ids, dict):
                    flat = []
                    for v in source_ids.values():
                        if isinstance(v, list):
                            flat.extend(v)
                    source_ids = flat

                if ability == "abstention":
                    # Use random turns as irrelevant passages
                    for tid, content in list(turns.items())[:3]:
                        records.append({
                            "query": query,
                            "passage": content,
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
                    # Non-source turns are irrelevant (hard negatives)
                    for tid, content in list(turns.items())[:2]:
                        if tid not in source_ids:
                            records.append({
                                "query": query,
                                "passage": content,
                                "label": "irrelevant",
                                "source": "beam",
                                "ability": ability,
                            })

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
