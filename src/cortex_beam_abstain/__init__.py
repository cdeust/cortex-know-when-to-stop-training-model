"""Cortex Abstention — retrieval confidence classifier.

Teaches RAG systems to say "I don't know" when retrieved passages
don't actually answer the query.
"""

from cortex_beam_abstain.classifier import AbstentionClassifier
from cortex_beam_abstain.heuristic import cosine_gap_score

__all__ = ["AbstentionClassifier", "cosine_gap_score"]
__version__ = "0.1.0"
