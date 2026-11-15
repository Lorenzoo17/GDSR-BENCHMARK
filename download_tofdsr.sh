#!/usr/bin/env bash
# Scarica il dataset TOFDSR, lo estrae in datasets/TOFDSR e aggiorna config.json.
# Opzioni utili: --solo-test   --da-archivio ~/Download/TOFDSR.zip   --dest /dati/TOFDSR
cd "$(dirname "$0")"
PY="${PYTHON:-python3}"
command -v "$PY" >/dev/null 2>&1 || PY=python
"$PY" download_tofdsr.py "$@"
