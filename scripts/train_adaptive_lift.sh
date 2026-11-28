#!/usr/bin/env bash
# Run from the environment used to train the fixed LiftFirst policy.
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.."
mode=${1:-train}
if [[ $# -gt 1 || ( "$mode" != train && "$mode" != check ) ]]; then
  echo "Usage: bash scripts/train_adaptive_lift.sh [check|train]" >&2
  exit 2
fi
py=${PYTHON:-python}
envs=${NUM_ENVS:-4096}
[[ "$envs" =~ ^[1-9][0-9]*$ ]] || { echo "NUM_ENVS must be a positive integer." >&2; exit 2; }
export PYTHONPATH=. PYTHONUNBUFFERED=1 MUJOCO_GL=egl
mkdir -p logs
logdir=$(mktemp -d logs/Adaptive_Lift_XXXXXX)
echo "Adaptive_Lift console logs: $logdir"
"$py" -m unittest discover -s tests/adaptive_lift -v
run_logged() {
  local logfile=$1
  shift
  "$@" 2>&1 | "$py" -u scripts/tools/compact_log.py "$logfile" || { tail -40 "$logfile" >&2; return 1; }
}
run_logged "$logdir/terrain.log" "$py" scripts/check_adaptive_lift.py
tail -3 "$logdir/terrain.log"
train=( "$py" scripts/train.py Adaptive_Lift --agent.resume False --agent.seed 42
        --agent.logger tensorboard --agent.upload-model False )
# Distinct names for each invocation. Both processes construct their own policy and optimizer.
tag=${logdir##*/}
run_logged "$logdir/smoke.log" "${train[@]}" --agent.run-name "${tag}_smoke" \
  --env.scene.num-envs 32 --agent.max-iterations 2
echo "Adaptive_Lift preflight passed."
if [[ "$mode" == train ]]; then
  echo "Launching $envs environments. Follow $logdir/train.log"
  run_logged "$logdir/train.log" "${train[@]}" --agent.run-name "$tag" --env.scene.num-envs "$envs"
fi
