#!/usr/bin/env bash
# Job 120237534290: Python 3.15 cannot install the required onnxruntime wheel.
# Source (cached Python 3.12): https://github.com/actions/runner-images/blob/macos-26-arm64/20261013.0424/images/macos/macos-26-arm64-Readme.md
# Source (cp312 macOS ARM64 wheel): https://pypi.org/project/onnxruntime/1.31.0/#files

is_python312() {
  "$1" -c 'import sys; raise SystemExit(sys.version_info[:2] != (3, 12))'
}

select_macos_python() {
  local architecture installation candidate
  case "${RUNNER_ARCH:-$(uname -m)}" in
    ARM64|arm64|aarch64) architecture=arm64 ;;
    X64|x64|x86_64) architecture=x64 ;;
    *) echo 'Unsupported macOS runner architecture' >&2; return 1 ;;
  esac
  # Source (layout/marker): https://github.com/actions/toolkit/blob/main/packages/tool-cache/src/tool-cache.ts
  if [ -n "${RUNNER_TOOL_CACHE:-}" ]; then
    for installation in "$RUNNER_TOOL_CACHE"/Python/3.12.*/"$architecture"; do
      candidate="$installation/bin/python3"
      if [ -f "$installation.complete" ] && [ -x "$candidate" ] && is_python312 "$candidate"; then
        printf '%s\n' "$candidate"
        return 0
      fi
    done
  fi
  candidate="$(command -v python3.12 || true)"
  if [ -n "$candidate" ] && is_python312 "$candidate"; then
    printf '%s\n' "$candidate"
    return 0
  fi
  echo 'Python 3.12 must already be installed on macOS (runner cache or PATH)' >&2
  return 1
}

select_ci_python() {
  if [ "${CI_OS:?CI_OS is unset}" = macos ]; then
    select_macos_python
  else
    command -v python3 || command -v python
  fi
}
