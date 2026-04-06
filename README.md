# cortex-beam-abstain

### Every RAG system returns an answer even when it doesn't have one. This model teaches retrieval to say "I don't know."

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://python.org)

A lightweight, community-trainable model for retrieval abstention — built for the [BEAM benchmark](https://arxiv.org/abs/2510.27246) (ICLR 2026), usable in any RAG system.

---

## The Problem

Retrieval-Augmented Generation systems always return results. Ask about something that doesn't exist in the knowledge base, and the system confidently hands you the least-bad match. The reader LLM then hallucinates an answer based on irrelevant context.

On the [BEAM benchmark](https://arxiv.org/abs/2510.27246) — the hardest long-term memory benchmark — abstention is where every system fails. Diagnostic data on BEAM shows abstention queries get **higher** retrieval scores (avg 0.926) than many answerable queries. Score-based thresholds cannot discriminate.

Existing NLI models (MNLI/SNLI) check logical entailment, not retrieval relevance. They weren't designed for this task.

## The Solution

A binary classifier specifically trained to answer: **"Does this passage contain information that answers this query?"**

| Property | Value |
|---|---|
| Architecture | DistilBERT fine-tuned (66M params) |
| Input | `(query, passage)` pair |
| Output | confidence score `[0, 1]` |
| Model size | 64 MB (INT8 quantized ONNX) |
| Inference | ~32 ms per pair on CPU |
| Training | MLX (Apple Silicon) or PyTorch |

## Current Status

**v0.1 — first trained model, working but biased.**

| Metric | Score |
|---|---|
| Accuracy | 83.3% |
| Precision | 72.6% |
| Recall | 98.4% |
| F1 | 0.836 |

**Known limitation:** the model is biased toward "relevant" because the training data isn't balanced enough. Recall is excellent (catches almost all relevant passages) but precision needs work. v0.2 will rebalance with more hard negatives and retrain.

The model improves over time through community-contributed labeled data — see [Contributing](#contributing-data).

## Quick Start

### Install (when published to PyPI)

```bash
pip install cortex-beam-abstain
```

### Use

```python
from cortex_beam_abstain import AbstentionClassifier

clf = AbstentionClassifier()  # auto-downloads model from HuggingFace

# Single prediction
score = clf.predict(
    query="What recipe did they discuss?",
    passage="We talked about the new API design for authentication.",
)
# score < 0.3 → should abstain

# Batch prediction
scores = clf.predict_batch([
    ("What language does the user prefer?", "The user said they always use TypeScript"),
    ("What recipe did they discuss?", "We fixed the database migration issue"),
])

# Decision: should the system abstain entirely?
if clf.should_abstain("query", ["passage1", "passage2"], threshold=0.3):
    return []  # No relevant results — abstain
```

### Fallback mode

If no model is available, the classifier falls back to a token-overlap heuristic:

```python
clf = AbstentionClassifier(use_heuristic=True)
```

Or to the raw cosine gap heuristic from BEAM diagnostic data (Cohen's d = 1.01).

## Training Your Own

### Generate seed data from BEAM

```bash
python scripts/generate_seed_data.py --output data/seed/beam.jsonl --limit 20
```

### Train (PyTorch — currently the only working backend)

```bash
python scripts/train_torch.py \
    --data data/ \
    --output checkpoints/v1 \
    --epochs 3 \
    --lr 2e-5 \
    --batch-size 16 \
    --eval
```

### Export to ONNX (with INT8 quantization)

```bash
python scripts/export_onnx.py \
    --checkpoint checkpoints/v1 \
    --output models/abstention.onnx \
    --quantize int8
```

The export script verifies that the ONNX output matches PyTorch (max diff < 1e-3) before quantizing.

### MLX training (Apple Silicon)

MLX training script is scaffolded in `scripts/train_mlx.py` but the LoRA fine-tuning loop is still under development. Use PyTorch for now.

## Contributing Data

This model only gets better with more labeled data. Every contribution helps.

### Data format

JSONL files in `data/community/`:

```json
{"query": "What color was the car?", "passage": "They discussed the API rate limiting strategy.", "label": "irrelevant", "source": "user", "contributor": "your_github_handle"}
{"query": "What framework do they use?", "passage": "We decided to switch from React to Vue because.", "label": "relevant", "source": "user", "contributor": "your_github_handle"}
```

### How to contribute

1. Fork this repo
2. Add labeled JSONL files to `data/community/your_handle.jsonl`
3. Run `python scripts/validate_data.py` to check format
4. Open a PR — CI validates automatically

The most valuable contributions are **hard negatives** — passages that are topically similar to the query but don't actually answer it. These are exactly the cases where naive retrieval fails.

See [CONTRIBUTING.md](CONTRIBUTING.md) for labeling guidelines.

## Architecture

- **Base model**: DistilBERT (`distilbert-base-uncased`, 66M params)
- **Fine-tuning**: full fine-tune via HuggingFace Trainer (LoRA via MLX is WIP)
- **Classification**: binary (relevant / irrelevant)
- **Input format**: `[CLS] query [SEP] passage [SEP]`, max 256 tokens
- **Export**: ONNX opset 14, INT8 dynamic quantization
- **Verification**: ONNX outputs verified against PyTorch (max diff = 0.000001)
- **Fallback**: raw cosine gap heuristic (Cohen's d = 1.01 on BEAM diagnostic)

## Why "cortex-beam-abstain"?

- **cortex** — part of the [Cortex](https://github.com/cdeust/Cortex) memory system family
- **beam** — trained for and evaluated against [BEAM](https://github.com/mohammadtavakoli78/BEAM)
- **abstain** — what it teaches RAG to do when it doesn't actually know

The repo name (`cortex-know-when-to-stop-training-model`) is the long descriptive form. The Python package (`cortex_beam_abstain`) is the terse version for imports.

## Related Projects

- [Cortex](https://github.com/cdeust/Cortex) — Persistent memory for Claude Code (parent project)
- [BEAM Benchmark](https://github.com/mohammadtavakoli78/BEAM) — Long-term memory evaluation (ICLR 2026)

## License

MIT
