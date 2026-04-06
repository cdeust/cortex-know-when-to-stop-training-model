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
    print("PyTorch/transformers required. Install: pip install cortex-abstention[torch]")
    sys.exit(1)


def export_onnx(checkpoint: Path, output: Path, max_length: int = 256) -> None:
    """Export PyTorch checkpoint to ONNX."""
    model = AutoModelForSequenceClassification.from_pretrained(str(checkpoint))
    tokenizer = AutoTokenizer.from_pretrained(str(checkpoint))
    model.eval()

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

    torch.onnx.export(
        model,
        (dummy["input_ids"], dummy["attention_mask"], dummy["token_type_ids"]),
        str(output),
        input_names=["input_ids", "attention_mask", "token_type_ids"],
        output_names=["logits"],
        dynamic_axes={
            "input_ids": {0: "batch"},
            "attention_mask": {0: "batch"},
            "token_type_ids": {0: "batch"},
            "logits": {0: "batch"},
        },
        opset_version=14,
    )
    print(f"Exported ONNX model to {output} ({output.stat().st_size / 1024 / 1024:.1f} MB)")

    # Also save tokenizer alongside model
    tokenizer.save_pretrained(str(output.parent))
    print(f"Saved tokenizer to {output.parent}")


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
