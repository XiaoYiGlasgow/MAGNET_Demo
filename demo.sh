#!/bin/bash
# Inference walk-through of one shipped sample (demo_infer.py show) + bird's-eye-view figure in figures/demo/.
#
# Usage:  bash demo.sh [SCENARIO | random] [GPU_ID | cpu] [SAMPLE]
#   SCENARIO  random (default) | high_lateral_acceleration | near_multiple_vehicles
#   GPU_ID    default cpu
#   SAMPLE    optional position 0-7 in the shipped batch; random if omitted
# Examples:  bash demo.sh
#            bash demo.sh near_multiple_vehicles 0 5
# Choose the interpreter with PYTHON=/path/to/python.
set -euo pipefail
SC=${1:-random}
DEVARG=${2:-cpu}
REPO=$(cd "$(dirname "$0")" && pwd)
PY=${PYTHON:-python}

case "$SC" in
  random) SC=$([ $((RANDOM % 2)) = 0 ] && echo high_lateral_acceleration || echo near_multiple_vehicles) ;;
  high_lateral_acceleration|near_multiple_vehicles) ;;
  *) echo "unknown scenario: $SC"; sed -n 4,7p "$0"; exit 1 ;;
esac
K=${3:-$((RANDOM % 8))}
case "$K" in [0-7]) ;; *) echo "SAMPLE must be 0-7 (position in the shipped batch of 8)"; exit 1 ;; esac

if ! "$PY" -c "import torch, torch_geometric" 2> /dev/null; then
  echo "torch / torch_geometric not found for '$PY': install requirements.txt first (README section 2)"
  exit 1
fi
if [ "$DEVARG" != "cpu" ] && ! "$PY" -c "import torch, sys; sys.exit(not torch.cuda.is_available())" 2> /dev/null; then
  echo "CUDA is not available for '$PY' (CPU build of torch?): use cpu, or install the CUDA build of torch"
  exit 1
fi
if [ "$DEVARG" = "cpu" ]; then DEV=cpu; else export CUDA_VISIBLE_DEVICES=$DEVARG; DEV=cuda:0; fi

cd "$REPO"
"$PY" demo_infer.py show --scenario "$SC" --sample "$K" --device "$DEV" --fig_dir figures/demo 2>&1 \
  | grep -v "Warning: nuPlan"
