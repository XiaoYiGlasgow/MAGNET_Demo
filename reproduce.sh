#!/bin/bash
# One command for the whole reproduction check (README section 3):
#   [1] every Table 2 cell against the training records shipped in checkpoints/magnet/
#   [2] the shipped evaluation batches through the original validate() with the 8 checkpoints, against the reference
#       values of the complete validation splits, and those reference values against the training records
#
# Usage:  bash reproduce.sh [GPU_ID | cpu]      (default: cpu; about 1 minute on CPU)
# Choose the interpreter with PYTHON=/path/to/python.
set -u
DEVARG=${1:-cpu}
REPO=$(cd "$(dirname "$0")" && pwd)
PY=${PYTHON:-python}
cd "$REPO"
if ! "$PY" -c "import torch, torch_geometric" 2> /dev/null; then
  echo "torch / torch_geometric not found for '$PY': install requirements.txt first (README section 2)"
  exit 1
fi
if [ "$DEVARG" != "cpu" ] && ! "$PY" -c "import torch, sys; sys.exit(not torch.cuda.is_available())" 2> /dev/null; then
  echo "CUDA is not available for '$PY' (CPU build of torch?): use cpu, or install the CUDA build of torch"
  exit 1
fi
if [ "$DEVARG" = "cpu" ]; then DEV=cpu; else export CUDA_VISIBLE_DEVICES=$DEVARG; DEV=cuda:0; fi
mkdir -p runs
LOG=runs/validate_debug_$(date +%Y%m%d_%H%M%S).log

echo "== [1] Table 2 against the training records (no computation)"
T2=$("$PY" check_table2.py)
echo "$T2"
N_MATCH=$(echo "$T2" | grep -c "MATCH$")

echo
echo "== [2] Shipped batches through the original validate() (device: $DEV)"
"$PY" demo_infer.py verify --device "$DEV" --log "$LOG" 2>&1 | grep -v "Warning: nuPlan"
V=${PIPESTATUS[0]}

echo
if [ "$N_MATCH" = "20" ] && [ "$V" = "0" ]; then
  echo "[reproduce] PASS: 20/20 Table 2 cells match the training records; 8/8 shipped batches reproduced; shipped reference values agree with the training records (8/8)"
else
  echo "[reproduce] FAIL: Table 2 cells matched $N_MATCH/20, verify exit code $V"
  exit 1
fi
