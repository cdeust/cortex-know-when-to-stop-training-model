"""Tests for the abstention classifier."""

from cortex_beam_abstain.classifier import AbstentionClassifier


class TestAbstentionClassifier:
    def test_heuristic_mode(self):
        clf = AbstentionClassifier(use_heuristic=True)
        score = clf.predict("What color is the car?", "The car was red and shiny.")
        assert 0.0 <= score <= 1.0

    def test_heuristic_relevant(self):
        clf = AbstentionClassifier(use_heuristic=True)
        score = clf.predict(
            "What language do they use?",
            "The team decided to use TypeScript for the frontend project.",
        )
        assert score > 0.3

    def test_heuristic_irrelevant(self):
        clf = AbstentionClassifier(use_heuristic=True)
        score = clf.predict(
            "What recipe did they discuss?",
            "We fixed the authentication bug in the API middleware.",
        )
        assert score < 0.5

    def test_should_abstain_empty(self):
        clf = AbstentionClassifier(use_heuristic=True)
        assert clf.should_abstain("query", []) is True

    def test_predict_batch(self):
        clf = AbstentionClassifier(use_heuristic=True)
        scores = clf.predict_batch([
            ("What is X?", "X is a framework for building APIs."),
            ("What is Y?", "We discussed the weather today."),
        ])
        assert len(scores) == 2
        assert all(0.0 <= s <= 1.0 for s in scores)
