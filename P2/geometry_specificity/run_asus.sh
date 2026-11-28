#!/usr/bin/env bash
set -euo pipefail
root=$(cd "$(dirname "$0")" && pwd)
assets=$(cd "$root/../.." && pwd)
pilot="${GEOMETRY_PILOT:-$assets/tta_pilot}"
dev="${GEOMETRY_DEV_PAIRS:-$assets/tta_p3_translation/pairs_frozen.csv}"
export PYTHONPATH="$assets/tta_pilot/.venv/lib/python3.10/site-packages${PYTHONPATH:+:$PYTHONPATH}"
export CUDA_VISIBLE_DEVICES="${GPU_UUID:?set GPU_UUID to the selected idle GPU}"
export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1
py="${GEOMETRY_PYTHON:?set GEOMETRY_PYTHON to the compatible Python 3.10 executable}"
out="${GEOMETRY_OUTPUT:?set GEOMETRY_OUTPUT to the writable GPU output directory}"
"$py" -Bu "$root/run.py" --pilot "$pilot" --old "$pilot/pairs.csv" --dev "$dev" --out "$out" --device cuda --batch 1 --workers 1
"$py" -B "$root/cases.py" --pilot "$pilot" --pairs "$dev" --root "$out"
"$py" -B "$root/verify.py" --root "$out"
"$py" -B "$root/report.py" --root "$out"
