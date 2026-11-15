#!/usr/bin/env bash
# Train the PoultrySurface task from scratch. Runs in the foreground; the full
# trainer output goes to logs/poultry_<RUN_NAME>.log and TensorBoard files to
# logs/rsl_rl/DUET_G1_23dof_poultry/<date>_<RUN_NAME>/.
#
#   bash scripts/poultry/train.sh RUN_NAME [MAX_ITERS] [NUM_ENVS] [SEED]
#
#   bash scripts/poultry/train.sh trial2k 2000          # 2k-iteration trial
#   bash scripts/poultry/train.sh full 50000 4096 1     # the real run
#
# Resume a stopped run (same RUN_NAME is fine, a new dated folder is made):
#   RESUME=<run folder name> CHECKPOINT=model_2000.pt \
#     bash scripts/poultry/train.sh full 30000
# MAX_ITERS is then the number of ADDITIONAL iterations.
#
# TASK=<task id> trains another task with the same launcher, e.g. the lean
# variant: TASK=Unitree-G1-23Dof-Duet-PoultrySurface-Lean (see add_lean_input.py).
#
# Set PYTHON to the interpreter that has mjlab 1.2.0 (default: python).
set -euo pipefail
ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)
PY=${PYTHON:-python}
NAME=${1:?run name required}
ITERS=${2:-50000}
ENVS=${3:-4096}
SEED=${4:-1}
[[ "$NAME" =~ ^[a-zA-Z0-9_-]+$ ]] || { echo 'RUN_NAME: letters, digits, _ and - only.' >&2; exit 1; }
[[ "$ITERS" =~ ^[1-9][0-9]*$ && "$ENVS" =~ ^[1-9][0-9]*$ && "$SEED" =~ ^[0-9]+$ ]] || exit 1

cd "$ROOT"
mkdir -p logs
RESUME_ARGS=()
if [[ -n "${RESUME:-}" ]]; then
  RESUME_ARGS=(--agent.resume True --agent.load-run "$RESUME"
               --agent.load-checkpoint "${CHECKPOINT:?set CHECKPOINT=model_N.pt with RESUME}")
fi

echo "PoultrySurface: $ITERS iterations, $ENVS robots, seed $SEED, run '$NAME'. Log: logs/poultry_${NAME}.log"
PYTHONPATH=. PYTHONUNBUFFERED=1 WANDB_MODE=disabled \
  "$PY" scripts/train.py "${TASK:-Unitree-G1-23Dof-Duet-PoultrySurface}" \
  --env.scene.num-envs "$ENVS" --agent.max-iterations "$ITERS" --agent.seed "$SEED" \
  --agent.logger tensorboard --agent.upload-model False --agent.run-name "$NAME" \
  "${RESUME_ARGS[@]}" > "logs/poultry_${NAME}.log" 2>&1
echo "PoultrySurface run '$NAME' finished."
