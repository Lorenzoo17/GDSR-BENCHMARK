#!/usr/bin/env bash
# Test su TOFDSR (protocollo completo, FP32); stampa la tabella dei risultati.
# Argomenti extra passati a run_benchmark.py, es.:  bash run_tofdsr.sh --limit 50 --amp
cd "$(dirname "$0")"
PY="${PYTHON:-python3}"
command -v "$PY" >/dev/null 2>&1 || PY=python
"$PY" run_benchmark.py --dataset tofdsr "$@"
