"""Tests for the heuristic fallback module."""

from cortex_abstention.heuristic import cosine_gap_score, text_overlap_score


class TestCosineGapScore:
    def test_large_gap_high_score(self):
        scores = [0.8, 0.5, 0.3, 0.2]
        assert cosine_gap_score(scores) > 0.5

    def test_small_gap_low_score(self):
        scores = [0.5, 0.48, 0.47, 0.46]
        assert cosine_gap_score(scores) < 0.2

    def test_single_result(self):
        assert cosine_gap_score([0.8]) == 0.0

    def test_empty(self):
        assert cosine_gap_score([]) == 0.0

    def test_bounded(self):
        scores = [1.0, 0.0]
        assert 0.0 <= cosine_gap_score(scores) <= 1.0


class TestTextOverlapScore:
    def test_full_overlap(self):
        score = text_overlap_score("TypeScript framework", "We use TypeScript framework here")
        assert score > 0.8

    def test_no_overlap(self):
        score = text_overlap_score("recipe chocolate cake", "API authentication middleware")
        assert score < 0.2

    def test_partial_overlap(self):
        score = text_overlap_score("TypeScript API design", "The new API was built with Python")
        assert 0.1 < score < 0.8

    def test_stopwords_ignored(self):
        score = text_overlap_score("the and is of", "something completely different")
        assert score == 0.5  # empty after stopword removal → default 0.5
