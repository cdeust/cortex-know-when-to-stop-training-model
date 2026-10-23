#!/usr/bin/env bash
# The environment cell of ci.json: on a bare runner (CI_OS linux, macos or
# windows), the package installs with the runner's own Python into a scratch
# target and imports. Fails on the first error.
set -euo pipefail
: "${CI_OS:?CI_OS is unset}" "${CI_WORK:?CI_WORK is unset}"
# shellcheck source-path=SCRIPTDIR
source "$(dirname "${BASH_SOURCE[0]}")/ci_python.sh"
PY="$(select_ci_python)"
"$PY" --version
"$PY" -m pip install --quiet --target "$CI_WORK/site" .
PYTHONPATH="$CI_WORK/site" "$PY" -c 'import cortex_beam_abstain as m; print("cortex_beam_abstain", m.__version__, "on", __import__("sys").platform)'
