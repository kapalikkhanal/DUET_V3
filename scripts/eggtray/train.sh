#!/usr/bin/env bash
# Usage: bash scripts/eggtray/train.sh TASK RUN_NAME [NUM_ENVS] [ITERATIONS] [SEED]
# Foreground; trainer output goes through the compact logger (MuJoCo Warp
# prints one ccd_iterations warning per heightfield contact).
set -euo pipefail
ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)
PY=${PYTHON:-python}  # set PYTHON to your env's interpreter if needed
TASK=${1:?task required, e.g. Unitree-G1-23Dof-EggTray-Poultry}
NAME=${2:?run name required}
ENVS=${3:-1024}
ITERS=${4:-500}
SEED=${5:-1}
[[ "$NAME" =~ ^[a-zA-Z0-9_-]+$ ]] || { echo 'Use letters, digits, underscores or dashes in RUN_NAME.' >&2; exit 1; }
cd "$ROOT"
mkdir -p logs/eggtray
PYTHONPATH=. PYTHONUNBUFFERED=1 WANDB_CONSOLE=off CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0} \
  "$PY" scripts/eggtray/train.py "$TASK" \
  --env.scene.num-envs "$ENVS" --agent.max-iterations "$ITERS" --agent.seed "$SEED" \
  --agent.run-name "$NAME" \
  2>&1 | "$PY" -u scripts/tools/compact_log.py "logs/eggtray/${NAME}.log"
