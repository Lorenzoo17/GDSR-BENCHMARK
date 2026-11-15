#!/usr/bin/env bash
# Scarica le reti del benchmark e i pesi. Uso:  bash setup.sh   (aggiungi --pip per installare i pacchetti)
cd "$(dirname "$0")"
PY="${PYTHON:-python3}"
command -v "$PY" >/dev/null 2>&1 || PY=python
"$PY" download_models.py "$@"
