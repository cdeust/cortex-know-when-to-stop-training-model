#!/usr/bin/env bash
# Source: pyproject.toml, tests and scripts/validate_data.py at ece014d1.
set -euo pipefail
cd "$(dirname "$0")/../.."
export PATH="$PWD/.venv/bin:$PATH"
case "${1:?phase required}" in
  setup)
    # This repository has no lockfile. Do not claim a frozen install.
    uv venv .venv
    uv pip install --python .venv/bin/python -e '.[dev]'
    ;;
  unit)
    python -m pytest -q tests/test_heuristic.py
    ;;
  regression)
    python -m pytest -q tests/test_classifier.py
    python scripts/validate_data.py
    ;;
  e2e)
    python .github/ci/api_probe.py
    ;;
  environment)
    python -c 'import onnxruntime as ort; from tokenizers import Tokenizer; import numpy; assert "CPUExecutionProvider" in ort.get_available_providers(); print("Native ONNX CPU provider, NumPy and tokenizers available")'
    ;;
  installation)
    python .github/ci/install_probe.py
    ;;
  release-version)
    python -c 'import tomllib; v=tomllib.load(open("pyproject.toml", "rb"))["project"]["version"]; print("version="+v); print("tag=v"+v)'
    ;;
  *) echo "Unknown phase: $1" >&2; exit 2 ;;
esac
