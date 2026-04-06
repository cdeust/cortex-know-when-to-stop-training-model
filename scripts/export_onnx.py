#!/usr/bin/env python3
"""Export trained model to ONNX with optional INT8 quantization.

Usage:
    python scripts/export_onnx.py --checkpoint checkpoints/best --output models/abstention.onnx
    python scripts/export_onnx.py --checkpoint checkpoints/best --output models/abstention.onnx --quantize int8
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

try:
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
except ImportError:
    print("PyTorch/transformers required. Install: pip install cortex-beam-abstain[torch]")
    sys.exit(1)


class ExportWrapper(torch.nn.Module):
    """Wrap model so positional args match ONNX expectations.

    Forces eager attention to avoid SDPA tracer warnings about
    shape-dependent control flow being baked into the graph.
    """

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, input_ids, attention_mask):
        return self.model(input_ids=input_ids, attention_mask=attention_mask).logits


def export_onnx(checkpoint: Path, output: Path, max_length: int = 256) -> None:
    """Export PyTorch checkpoint to ONNX."""
    # Force eager attention — SDPA uses shape-dependent control flow
    # that ONNX trace cannot capture properly (TracerWarnings).
    model = AutoModelForSequenceClassification.from_pretrained(
        str(checkpoint),
        attn_implementation="eager",
    )
    tokenizer = AutoTokenizer.from_pretrained(str(checkpoint))
    model.eval()
    wrapped = ExportWrapper(model)
    wrapped.eval()

    # Dummy input for tracing
    dummy = tokenizer(
        "What color was the car?",
        "They discussed the API rate limiting strategy for the new endpoint.",
        padding="max_length",
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    )

    output.parent.mkdir(parents=True, exist_ok=True)

    # DistilBERT signature: input_ids, attention_mask only
    inputs = (dummy["input_ids"], dummy["attention_mask"])
    torch.onnx.export(
        wrapped,
        inputs,
        str(output),
        input_names=["input_ids", "attention_mask"],
        output_names=["logits"],
        dynamic_axes={
            "input_ids": {0: "batch"},
            "attention_mask": {0: "batch"},
            "logits": {0: "batch"},
        },
        opset_version=14,
        dynamo=False,
    )
    print(f"Exported ONNX model to {output} ({output.stat().st_size / 1024 / 1024:.1f} MB)")

    # Also save tokenizer alongside model
    tokenizer.save_pretrained(str(output.parent))
    print(f"Saved tokenizer to {output.parent}")

    # Verify ONNX matches PyTorch on multiple input shapes
    _verify_export(wrapped, tokenizer, output, max_length)


def _verify_export(model, tokenizer, onnx_path: Path, max_length: int) -> None:
    """Verify ONNX inference matches PyTorch on varied inputs."""
    import onnxruntime as ort
    import numpy as np

    session = ort.InferenceSession(str(onnx_path))

    test_cases = [
        ("Short query", "Short passage"),
        ("What is the user preferred database?", "I always use PostgreSQL for all my projects"),
        ("How should I format code?", "Always use 2-space indentation in TypeScript"),
        (
            "What library is used for authentication?",
            "We use OAuth2 with JWT tokens via passport-jwt for our REST API. "
            "The implementation handles token refresh and revocation.",
        ),
    ]

    max_diff = 0.0
    for q, p in test_cases:
        encoded = tokenizer(
            q, p,
            padding="max_length",
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )

        # PyTorch
        with torch.no_grad():
            torch_out = model(encoded["input_ids"], encoded["attention_mask"]).numpy()

        # ONNX
        onnx_out = session.run(
            None,
            {
                "input_ids": encoded["input_ids"].numpy().astype(np.int64),
                "attention_mask": encoded["attention_mask"].numpy().astype(np.int64),
            },
        )[0]

        diff = float(np.abs(torch_out - onnx_out).max())
        max_diff = max(max_diff, diff)

    if max_diff < 1e-3:
        print(f"Verification PASSED: max diff = {max_diff:.6f}")
    else:
        print(f"Verification FAILED: max diff = {max_diff:.6f} (>1e-3)")
        raise SystemExit(1)


def quantize_int8(model_path: Path) -> Path:
    """Apply INT8 dynamic quantization to ONNX model."""
    try:
        from onnxruntime.quantization import QuantType, quantize_dynamic
    except ImportError:
        print("onnxruntime-extensions required for quantization.")
        print("Install: pip install onnxruntime")
        sys.exit(1)

    output = model_path.with_suffix(".int8.onnx")
    quantize_dynamic(
        str(model_path),
        str(output),
        weight_type=QuantType.QInt8,
    )
    reduction = 1 - output.stat().st_size / model_path.stat().st_size
    print(f"Quantized to {output} ({output.stat().st_size / 1024 / 1024:.1f} MB, "
          f"{reduction:.0%} smaller)")
    return output


def main():
    parser = argparse.ArgumentParser(description="Export to ONNX")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("models/abstention.onnx"))
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--quantize", choices=["int8"], default=None)
    args = parser.parse_args()

    export_onnx(args.checkpoint, args.output, args.max_length)

    if args.quantize == "int8":
        quantize_int8(args.output)


if __name__ == "__main__":
    main()
