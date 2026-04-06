#!/usr/bin/env python3
"""Validate contributed JSONL data files.

Checks schema compliance, duplicates, and label distribution.

Usage:
    python scripts/validate_data.py                    # Validate all data/
    python scripts/validate_data.py data/community/    # Validate specific dir
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

REQUIRED_FIELDS = {"query", "passage", "label"}
VALID_LABELS = {"relevant", "irrelevant"}
VALID_SOURCES = {"beam", "synthetic", "user", "locomo", "longmemeval"}


def validate_file(path: Path) -> list[str]:
    """Validate a single JSONL file. Returns list of errors."""
    errors = []
    seen = set()

    with open(path) as f:
        for i, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError as e:
                errors.append(f"{path}:{i}: Invalid JSON: {e}")
                continue

            # Required fields
            missing = REQUIRED_FIELDS - set(rec.keys())
            if missing:
                errors.append(f"{path}:{i}: Missing fields: {missing}")
                continue

            # Valid label
            if rec["label"] not in VALID_LABELS:
                errors.append(f"{path}:{i}: Invalid label '{rec['label']}', "
                              f"expected {VALID_LABELS}")

            # Non-empty content
            if len(rec["query"].strip()) < 5:
                errors.append(f"{path}:{i}: Query too short")
            if len(rec["passage"].strip()) < 10:
                errors.append(f"{path}:{i}: Passage too short")

            # Duplicate check
            key = (rec["query"][:100], rec["passage"][:100])
            if key in seen:
                errors.append(f"{path}:{i}: Duplicate pair")
            seen.add(key)

    return errors


def main():
    data_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("data")

    files = list(data_dir.rglob("*.jsonl"))
    if not files:
        print(f"No JSONL files found in {data_dir}")
        sys.exit(1)

    total_errors = []
    total_records = 0
    label_counts = {"relevant": 0, "irrelevant": 0}

    for f in sorted(files):
        errors = validate_file(f)
        total_errors.extend(errors)

        with open(f) as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                    total_records += 1
                    label = rec.get("label", "")
                    if label in label_counts:
                        label_counts[label] += 1
                except json.JSONDecodeError:
                    pass

    # Report
    print(f"Files: {len(files)}")
    print(f"Total records: {total_records}")
    print(f"Labels: {label_counts}")
    print(f"Errors: {len(total_errors)}")

    if total_errors:
        print("\nErrors:")
        for e in total_errors[:20]:
            print(f"  {e}")
        if len(total_errors) > 20:
            print(f"  ... and {len(total_errors) - 20} more")
        sys.exit(1)
    else:
        print("\nAll data valid.")


if __name__ == "__main__":
    main()
