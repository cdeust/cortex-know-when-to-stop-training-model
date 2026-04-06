"""Retrieval abstention classifier.

Determines whether a retrieved passage actually answers a query.
Uses a fine-tuned DistilBERT model (ONNX) for fast inference.
Falls back to cosine gap heuristic if model is unavailable.

The model is trained on community-contributed (query, passage, label) pairs
and evaluated against BEAM abstention (Tavakoli et al., ICLR 2026).
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

_DEFAULT_MODEL_ID = "cdeust/cortex-abstention-v1"
_DEFAULT_THRESHOLD = 0.3
_MAX_LENGTH = 256


class AbstentionClassifier:
    """Binary classifier for retrieval relevance.

    Predicts whether a retrieved passage contains information that
    answers the given query. Optimized for retrieval abstention —
    detecting when to return empty results instead of irrelevant ones.

    Args:
        model_path: Path to ONNX model file. If None, auto-downloads
            from HuggingFace.
        threshold: Score below which to abstain. Default 0.3.
        use_heuristic: If True, uses cosine gap heuristic instead of model.
    """

    def __init__(
        self,
        model_path: str | Path | None = None,
        threshold: float = _DEFAULT_THRESHOLD,
        use_heuristic: bool = False,
    ) -> None:
        self.threshold = threshold
        self._session: Any = None
        self._tokenizer: Any = None
        self._use_heuristic = use_heuristic

        if use_heuristic:
            return

        resolved = self._resolve_model(model_path)
        if resolved:
            self._load_onnx(resolved)
        else:
            logger.warning(
                "Model not found, falling back to heuristic mode. "
                "Install a model: pip install cortex-abstention[torch] "
                "then train or download from HuggingFace."
            )
            self._use_heuristic = True

    def _resolve_model(self, model_path: str | Path | None) -> Path | None:
        """Find or download the ONNX model."""
        if model_path:
            p = Path(model_path)
            if p.exists():
                return p
            return None

        # Check local cache
        cache = Path.home() / ".cache" / "cortex-abstention"
        local = cache / "model.onnx"
        if local.exists():
            return local

        # Try auto-download
        try:
            from cortex_abstention.model_hub import download_model

            return download_model(_DEFAULT_MODEL_ID, cache_dir=cache)
        except Exception as e:
            logger.debug("Auto-download failed: %s", e)
            return None

    def _load_onnx(self, model_path: Path) -> None:
        """Load ONNX model and tokenizer."""
        try:
            import onnxruntime as ort
            from tokenizers import Tokenizer

            self._session = ort.InferenceSession(
                str(model_path),
                providers=["CPUExecutionProvider"],
            )

            # Look for tokenizer.json alongside model
            tok_path = model_path.parent / "tokenizer.json"
            if tok_path.exists():
                self._tokenizer = Tokenizer.from_file(str(tok_path))
            else:
                # Fall back to HuggingFace tokenizer
                from tokenizers import Tokenizer as HFTokenizer

                self._tokenizer = HFTokenizer.from_pretrained(
                    "distilbert-base-uncased"
                )

            logger.info("Loaded abstention model from %s", model_path)
        except ImportError as e:
            logger.warning("ONNX runtime not available: %s", e)
            self._use_heuristic = True
        except Exception as e:
            logger.warning("Failed to load model: %s", e)
            self._use_heuristic = True

    def predict(self, query: str, passage: str) -> float:
        """Predict relevance of a passage to a query.

        Returns:
            float in [0, 1]. High = relevant, low = irrelevant.
        """
        if self._use_heuristic:
            from cortex_abstention.heuristic import text_overlap_score

            return text_overlap_score(query, passage)

        return self._predict_onnx(query, passage)

    def _predict_onnx(self, query: str, passage: str) -> float:
        """Run ONNX inference on a (query, passage) pair."""
        if self._session is None or self._tokenizer is None:
            return 0.5

        # Tokenize as pair: [CLS] query [SEP] passage [SEP]
        encoded = self._tokenizer.encode(query, passage)
        ids = encoded.ids[:_MAX_LENGTH]
        mask = encoded.attention_mask[:_MAX_LENGTH]
        type_ids = encoded.type_ids[:_MAX_LENGTH]

        # Pad to max length
        pad_len = _MAX_LENGTH - len(ids)
        ids = ids + [0] * pad_len
        mask = mask + [0] * pad_len
        type_ids = type_ids + [0] * pad_len

        inputs = {
            "input_ids": np.array([ids], dtype=np.int64),
            "attention_mask": np.array([mask], dtype=np.int64),
            "token_type_ids": np.array([type_ids], dtype=np.int64),
        }

        outputs = self._session.run(None, inputs)
        logits = outputs[0][0]  # [irrelevant, relevant]

        # Softmax
        exp_logits = np.exp(logits - np.max(logits))
        probs = exp_logits / exp_logits.sum()

        return float(probs[1])  # P(relevant)

    def predict_batch(self, pairs: list[tuple[str, str]]) -> list[float]:
        """Batch prediction for efficiency.

        Args:
            pairs: List of (query, passage) tuples.

        Returns:
            List of relevance scores.
        """
        return [self.predict(q, p) for q, p in pairs]

    def should_abstain(
        self,
        query: str,
        passages: list[str],
        threshold: float | None = None,
    ) -> bool:
        """Determine if the system should abstain from answering.

        Returns True if ALL passages score below threshold.

        Args:
            query: The search query.
            passages: Retrieved passages to evaluate.
            threshold: Override default threshold.
        """
        if not passages:
            return True

        thresh = threshold if threshold is not None else self.threshold
        scores = [self.predict(query, p) for p in passages]
        return all(s < thresh for s in scores)
