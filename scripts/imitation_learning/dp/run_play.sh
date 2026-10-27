#!/usr/bin/env bash
set -euo pipefail
# Supply a matching checkpoint, statistics file, task YAML and registered task ID.
# Historical banana-to-bowl checkpoints require their original bowl YAML;
# they must not be silently evaluated as the new banana-to-plate task.
: "${DP_CHECKPOINT:?Set DP_CHECKPOINT for the trained task}"
: "${DP_STAT:?Set DP_STAT for the trained task}"
: "${DP_TASK_CONFIG:?Set DP_TASK_CONFIG for the trained task}"
: "${DP_TASK_ID:?Set DP_TASK_ID for the trained task}"

# pick_banana_place_bowl_nogap
CUDA_VISIBLE_DEVICES=0 python scripts/imitation_learning/dp/play.py \
    --checkpoint "${DP_CHECKPOINT}" \
    --stat "${DP_STAT}" \
    --task_config "${DP_TASK_CONFIG}" \
    --task "${DP_TASK_ID}" \
    --horizon 500 --enable_cameras  --num_rollouts 30 --headless --action_horizon 4
