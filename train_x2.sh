#!/bin/bash

TASK="X2-AMP-Flat"
NUM_ENVS="${NUM_ENVS:-8192}"
EXP_NAME="${EXP_NAME:-x2_walk}"

exec uv run python train.py "$TASK" \
  --env.scene.num-envs="$NUM_ENVS" \
  --agent.experiment-name="$EXP_NAME" \
  --agent.max-iterations=60000 \
  --video True \
  --video-interval 20000 \
  "$@"
