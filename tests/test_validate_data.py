"""Regression checks for full-text duplicate detection and data quarantine."""

import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("validate_data", ROOT / "scripts/validate_data.py")
VALIDATOR = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(VALIDATOR)


class ValidateDataTests(unittest.TestCase):
    def validate(self, records):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "records.jsonl"
            path.write_text("".join(json.dumps(record) + "\n" for record in records))
            return VALIDATOR.validate_file(path)

    def record(self, **overrides):
        return {
            "query": "Which option was selected?",
            "passage": "The user selected the blue option.",
            "label": "relevant",
            **overrides,
        }

    def test_identical_full_pair_is_rejected(self):
        record = self.record()
        self.assertEqual(len(self.validate([record, record])), 1)
        self.assertIn("Duplicate pair", self.validate([record, record])[0])

    def test_conflicting_labels_do_not_make_pair_unique(self):
        errors = self.validate([self.record(), self.record(label="irrelevant")])
        self.assertEqual(len(errors), 1)
        self.assertIn("Duplicate pair", errors[0])

    def test_distinct_passages_with_shared_prefix_are_preserved(self):
        prefix = "Shared context. " * 10
        self.assertEqual(
            self.validate(
                [
                    self.record(passage=prefix + "Option blue."),
                    self.record(passage=prefix + "Option red."),
                ]
            ),
            [],
        )

    def test_distinct_queries_with_shared_prefix_are_preserved(self):
        prefix = "Shared question context. " * 10
        self.assertEqual(
            self.validate(
                [
                    self.record(query=prefix + "Which option?"),
                    self.record(query=prefix + "Which date?"),
                ]
            ),
            [],
        )

    def test_invalid_label_remains_rejected(self):
        errors = self.validate([self.record(label="unknown")])
        self.assertEqual(len(errors), 1)
        self.assertIn("Invalid label", errors[0])

    def test_quarantined_pair_is_preserved_and_excluded_from_training(self):
        quarantined = [
            json.loads(line)
            for line in (ROOT / "quarantine/beam-label-conflicts.jsonl").read_text().splitlines()
        ]
        self.assertEqual({entry["source_line"] for entry in quarantined}, {17251, 17255})
        self.assertEqual(
            {entry["record"]["label"] for entry in quarantined},
            {"relevant", "irrelevant"},
        )
        pairs = {(entry["record"]["query"], entry["record"]["passage"]) for entry in quarantined}
        self.assertEqual(len(pairs), 1)
        for line in (ROOT / "data/seed/beam.jsonl").read_text().splitlines():
            record = json.loads(line)
            self.assertNotIn((record["query"], record["passage"]), pairs)
        errors = self.validate([entry["record"] for entry in quarantined])
        self.assertEqual(len(errors), 1)
        self.assertIn("Duplicate pair", errors[0])


if __name__ == "__main__":
    unittest.main()
