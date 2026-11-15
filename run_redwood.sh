#!/usr/bin/env bash
# Test su Redwood (formulazione B, x4).
cd "$(dirname "$0")"
PY="${PYTHON:-python3}"
command -v "$PY" >/dev/null 2>&1 || PY=python
"$PY" run_benchmark.py --dataset redwood "$@"
