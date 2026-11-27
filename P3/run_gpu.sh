#!/usr/bin/env bash
set -euo pipefail

zone=${1:?pass Google Cloud zone}
gpu_name=${2:-NVIDIA L4}
template=${3:-}
python=P1/baseline/.venv/bin/python
out=P3/runtime/gpu
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1

"$python" -B P3/freeze_holdout.py --out "$out/holdout"
for study in development-3 development-1.5 development-6 holdout-hps holdout-clip; do
    run="$out/$study"
    args=(--out "$run" --zone "$zone" --gpu-name "$gpu_name")
    if [[ -n "$template" ]]; then
        args+=(--instance-template "$template")
    fi
    pairs=P2/translation/pairs_frozen.csv
    case "$study" in
        development-3) args+=(--with-brightness) ;;
        development-1.5) args+=(--amplitude-pct 1.5) ;;
        development-6) args+=(--amplitude-pct 6) ;;
        holdout-hps)
            pairs="$out/holdout/pairs.csv"
            args+=(--with-brightness)
            ;;
        holdout-clip)
            pairs="$out/holdout/pairs.csv"
            args+=(--scorer clip)
            ;;
    esac
    "$python" -Bu P3/score_gpu.py --pairs "$pairs" "${args[@]}"
    "$python" -B P3/analyze.py --pairs "$pairs" --scores "$run/scores.csv" \
        --timings "$run/timings.csv" --environment "$run/environment.json" --out "$run"
done
