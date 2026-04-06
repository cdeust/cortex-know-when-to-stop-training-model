"""Heuristic fallback for retrieval abstention.

Used when the ONNX model is not available. Based on diagnostic data
from BEAM benchmark (Cortex project, April 2026):

- Raw cosine gap between rank 1 and rank 2: Cohen's d = 1.01
  (only signal with large effect size for abstention vs answerable)
- Text overlap: simple but effective for extreme cases
"""

from __future__ import annotations

import re


def cosine_gap_score(scores: list[float]) -> float:
    """Score based on gap between rank 1 and rank 2 raw cosine scores.

    Diagnostic data (BEAM, 5 conversations, 100 questions):
      Abstention avg gap: 0.106
      Answerable avg gap: 0.048
      Cohen's d: 1.01 (large effect)

    Higher gap = more discriminative retrieval = less likely abstention.
    Lower gap = flat distribution = more likely abstention.

    Args:
        scores: Raw cosine similarity scores, descending order.

    Returns:
        float in [0, 1]. High = confident retrieval, low = should abstain.
    """
    if len(scores) < 2:
        return 0.0
    gap = scores[0] - scores[1]
    # Normalize: gap typically ranges 0-0.3
    return min(gap / 0.3, 1.0)


def text_overlap_score(query: str, passage: str) -> float:
    """Simple token overlap between query and passage.

    Heuristic fallback when no model is available.
    Measures what fraction of query content words appear in the passage.

    Args:
        query: Search query.
        passage: Retrieved passage.

    Returns:
        float in [0, 1]. Fraction of query tokens found in passage.
    """
    _STOPWORDS = {
        "the", "a", "an", "is", "was", "are", "were", "be", "been",
        "being", "have", "has", "had", "do", "does", "did", "will",
        "would", "could", "should", "may", "might", "can", "shall",
        "to", "of", "in", "for", "on", "with", "at", "by", "from",
        "as", "into", "through", "during", "before", "after", "above",
        "below", "between", "out", "off", "over", "under", "again",
        "further", "then", "once", "here", "there", "when", "where",
        "why", "how", "all", "each", "every", "both", "few", "more",
        "most", "other", "some", "such", "no", "nor", "not", "only",
        "own", "same", "so", "than", "too", "very", "just", "because",
        "but", "and", "or", "if", "while", "about", "what", "which",
        "who", "whom", "this", "that", "these", "those", "i", "me",
        "my", "we", "our", "you", "your", "he", "him", "his", "she",
        "her", "it", "its", "they", "them", "their",
    }

    q_tokens = set(re.findall(r"\w+", query.lower())) - _STOPWORDS
    if not q_tokens:
        return 0.5

    p_lower = passage.lower()
    matches = sum(1 for t in q_tokens if t in p_lower)
    return matches / len(q_tokens)
