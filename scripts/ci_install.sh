#!/usr/bin/env bash
# The install cell of ci.json: a fresh virtual environment under CI_WORK on
# each OS, `pip install .` into it, then the package imports and the
# classifier answers. Fails on the first error.
set -euo pipefail
: "${CI_OS:?CI_OS is unset}" "${CI_WORK:?CI_WORK is unset}"
# shellcheck source-path=SCRIPTDIR
source "$(dirname "${BASH_SOURCE[0]}")/ci_python.sh"
PY="$(select_ci_python)"
"$PY" --version
"$PY" -m venv "$CI_WORK/venv"
if [ -x "$CI_WORK/venv/bin/python" ]; then VPY="$CI_WORK/venv/bin/python"; else VPY="$CI_WORK/venv/Scripts/python.exe"; fi
"$VPY" -m pip install --quiet .
"$VPY" -c '
from cortex_beam_abstain import AbstentionClassifier
score = AbstentionClassifier(use_heuristic=True).predict("What color is the car?", "The car was red.")
assert 0.0 <= score <= 1.0, score
print("installed: heuristic classifier answers", score)
'
