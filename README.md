# Cortex Abstention

### Every RAG system returns an answer even when it doesn't have one. This model teaches retrieval to say "I don't know."

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://python.org)

---

## The Problem

Retrieval-Augmented Generation systems always return results. Ask about something that doesn't exist in the knowledge base, and the system confidently hands you the least-bad match. The reader LLM then hallucinates an answer based on irrelevant context.

On the [BEAM benchmark](https://arxiv.org/abs/2510.27246) (ICLR 2026) — the hardest long-term memory benchmark — abstention is where every system fails:

| System | Abstention Score | Overall |
|---|---|---|
| LIGHT (Llama-4-Maverick) | 0.750 | 0.266 |
| Cortex (Claude Opus 4.6) | 0.450 | 0.404 |

Score-based thresholds don't work. Our diagnostic data shows abstention queries get **higher** retrieval scores (avg 0.926) than many answerable queries. The retrieval finds plausible but non-answering passages.

Existing NLI models (trained on MNLI/SNLI) check logical entailment, not retrieval relevance. They weren't designed for this.

## The Solution

A community-trained binary classifier specifically for retrieval abstention:

- **Input**: (query, retrieved passage) pair
- **Output**: confidence score [0, 1] — how likely the passage answers the query
- **Architecture**: Fine-tuned DistilBERT, <50MB, <10ms on CPU
- **Training**: MLX (Apple Silicon) with PyTorch fallback
- **Inference**: ONNX for cross-platform deployment

The model improves over time through community-contributed labeled data.

## Quick Start

```python
from cortex_beam_abstain import AbstentionClassifier

clf = AbstentionClassifier()  # auto-downloads model from HuggingFace

# Single prediction
score = clf.predict(
    query="What recipe did they discuss?",
    passage="We talked about the new API design for authentication..."
)
# score ≈ 0.05 → irrelevant, should abstain

# Batch prediction
scores = clf.predict_batch([
    ("What language do they prefer?", "The user said they always use TypeScript"),
    ("What recipe did they discuss?", "We fixed the database migration issue"),
])
# scores ≈ [0.92, 0.03]

# Decision
if clf.should_abstain("query", ["passage1", "passage2"], threshold=0.3):
    return []  # No relevant results
```

## Training

### With MLX (Apple Silicon)

```bash
pip install cortex-beam-abstain[mlx]
python scripts/train_mlx.py --data data/ --epochs 3 --lr 2e-5
```

### With PyTorch

```bash
pip install cortex-beam-abstain[torch]
python scripts/train_torch.py --data data/ --epochs 3 --lr 2e-5
```

### Export to ONNX

```bash
python scripts/export_onnx.py --checkpoint checkpoints/best --output models/abstention.onnx --quantize int8
```

## Contributing Data

This model gets better with more labeled data. Every contribution helps.

### Data Format

JSONL files in `data/community/`:

```json
{"query": "What color was the car?", "passage": "They discussed the API rate limiting strategy...", "label": "irrelevant", "source": "user", "contributor": "your_github_handle"}
{"query": "What framework do they use?", "passage": "We decided to switch from React to Vue because...", "label": "relevant", "source": "user", "contributor": "your_github_handle"}
```

### How to Contribute

1. Fork this repo
2. Add labeled JSONL files to `data/community/your_handle.jsonl`
3. Run `python scripts/validate_data.py` to check format
4. Open a PR — CI validates automatically

See [CONTRIBUTING.md](CONTRIBUTING.md) for labeling guidelines and quality standards.

## Evaluation

```bash
python evaluation/eval_beam.py --model models/abstention.onnx --split 100K
```

## Architecture

- **Base model**: DistilBERT (66M params, distilled from BERT-base)
- **Fine-tuning**: LoRA (rank 8) via MLX or full fine-tune via PyTorch
- **Classification**: Binary (relevant / irrelevant)
- **Input format**: `[CLS] query [SEP] passage [SEP]`
- **Max tokens**: 256
- **Export**: ONNX with INT8 quantization (~25MB)
- **Fallback**: Raw cosine gap heuristic when model unavailable (Cohen's d=1.01)

## Related Projects

- [Cortex](https://github.com/cdeust/Cortex) — Persistent memory for Claude Code (parent project)
- [BEAM Benchmark](https://github.com/mohammadtavakoli78/BEAM) — Long-term memory evaluation

## License

MIT
