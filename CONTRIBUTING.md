# Contributing to Cortex Abstention

Every labeled data point makes the model better. Here's how to help.

## Contributing Data

### Format

Create a JSONL file in `data/community/your_github_handle.jsonl`:

```json
{"query": "What color was the car?", "passage": "They discussed the new API design for the authentication module...", "label": "irrelevant", "source": "user", "contributor": "your_handle"}
{"query": "What framework do they use?", "passage": "We decided to switch from React to Vue because of the composition API...", "label": "relevant", "source": "user", "contributor": "your_handle"}
```

### Fields

| Field | Required | Values |
|---|---|---|
| `query` | Yes | The search query (min 5 chars) |
| `passage` | Yes | The retrieved passage (min 10 chars) |
| `label` | Yes | `relevant` or `irrelevant` |
| `source` | No | `user`, `beam`, `synthetic`, `locomo`, `longmemeval` |
| `contributor` | No | Your GitHub handle |

### Labeling Guidelines

**Label as `relevant` when:**
- The passage directly answers the query
- The passage contains the specific information the query asks about
- A human reading the passage could answer the query from it

**Label as `irrelevant` when:**
- The passage is about the same general topic but doesn't answer the specific question
- The passage discusses something else entirely
- The passage is from the same conversation but a different topic

**Hard cases (the most valuable labels):**
- Passage is topically similar but doesn't contain the answer
- Passage mentions the same entities but in a different context
- Passage partially answers but misses the key information

### Steps

1. Fork this repo
2. Create `data/community/your_handle.jsonl`
3. Run validation: `python scripts/validate_data.py`
4. Open a PR

CI will validate your data automatically.

## Contributing Code

### Setup

```bash
git clone https://github.com/cdeust/cortex-beam-abstain.git
cd cortex-beam-abstain
pip install -e ".[dev]"
```

### Tests

```bash
pytest tests/
```

### Style

```bash
ruff check .
ruff format .
```

## Ideas Welcome

Open an issue if you have ideas for:
- Better model architectures
- New training data sources
- Evaluation improvements
- Integration with other RAG systems
