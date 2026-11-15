#!/usr/bin/env bash
# Protocollo del paper su TOFDSR: prima FP32 (riferimento, TF32 spento), poi FP16.
cd "$(dirname "$0")"
PY="${PYTHON:-python3}"
command -v "$PY" >/dev/null 2>&1 || PY=python
"$PY" run_benchmark.py --dataset tofdsr --precision fp32 "$@"
"$PY" run_benchmark.py --dataset tofdsr --precision fp16 "$@"
