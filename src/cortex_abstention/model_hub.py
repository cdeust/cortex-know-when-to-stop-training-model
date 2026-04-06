"""Model download and caching from HuggingFace Hub."""

from __future__ import annotations

import logging
import urllib.request
from pathlib import Path

logger = logging.getLogger(__name__)

_HF_BASE = "https://huggingface.co/{model_id}/resolve/main/{filename}"


def download_model(
    model_id: str = "cdeust/cortex-abstention-v1",
    cache_dir: Path | None = None,
) -> Path | None:
    """Download ONNX model and tokenizer from HuggingFace Hub.

    Args:
        model_id: HuggingFace model identifier.
        cache_dir: Local cache directory. Defaults to ~/.cache/cortex-abstention.

    Returns:
        Path to downloaded ONNX model file, or None on failure.
    """
    if cache_dir is None:
        cache_dir = Path.home() / ".cache" / "cortex-abstention"

    cache_dir.mkdir(parents=True, exist_ok=True)
    model_path = cache_dir / "model.onnx"
    tokenizer_path = cache_dir / "tokenizer.json"

    if model_path.exists():
        return model_path

    files = {
        "model.onnx": model_path,
        "tokenizer.json": tokenizer_path,
    }

    for filename, local_path in files.items():
        if local_path.exists():
            continue

        url = _HF_BASE.format(model_id=model_id, filename=filename)
        logger.info("Downloading %s from %s", filename, url)

        try:
            urllib.request.urlretrieve(url, str(local_path))
            logger.info("Saved %s (%d bytes)", local_path, local_path.stat().st_size)
        except Exception as e:
            logger.warning("Failed to download %s: %s", filename, e)
            # Clean up partial download
            if local_path.exists():
                local_path.unlink()
            return None

    return model_path
