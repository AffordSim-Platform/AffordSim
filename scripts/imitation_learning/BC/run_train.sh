#!/bin/bash

# 设置遇到错误时自动退出
set -e

echo "开始运行 BC 训练脚本..."

# 运行训练命令
# pick_banana_and_place_on_plate_final
CUDA_VISIBLE_DEVICES=1 python scripts/imitation_learning/BC/train_bc.py \
    --dataset dataset/r2s2r_data/task_gen/pick_place/pick_banana_and_place_on_plate_final/merged_trajectories_robo.hdf5 \
    --stat dataset/r2s2r_data/task_gen/pick_place/pick_banana_and_place_on_plate_final/dataset_stat.json \
    --total_iters 200000 \
    --save_freq 1000 \
    --eval_task_config source/isaaclab_tasks/isaaclab_tasks/real2sim/tasks/pick/config/sceneconfig/task/pick/pick_banana_place_in_plate.yaml \
    --eval_task Pick_Place_banana-v0 \
    --eval_num_rollouts 5 \
    --eval_horizon 250 \
    --batch_size 48 \
    --action_chunk_size 8 --action_horizon 4 --no_remove_idle

# pour_red_cup_easy
# CUDA_VISIBLE_DEVICES=2 python scripts/imitation_learning/BC/train_bc.py \
#     --dataset dataset/r2s2r_data/task_gen/pour_with_affordance/pour_red_cup_easy/merged_trajectories_robo.hdf5 \
#     --stat dataset/r2s2r_data/task_gen/pour_with_affordance/pour_red_cup_easy/dataset_stat.json \
#     --total_iters 200000 \
#     --save_freq 1000 \
#     --eval_task_config source/isaaclab_tasks/isaaclab_tasks/real2sim/tasks/pick/config/sceneconfig/task/pour/pour_redcup.yaml \
#     --eval_task Pick_cupwithhandle_Pour-v0 \
#     --eval_num_rollouts 5 \
#     --eval_horizon 250 \
#     --batch_size 48 \
#     --action_chunk_size 8 --action_horizon 4 --no_remove_idle

# pour_pan
# CUDA_VISIBLE_DEVICES=3 python scripts/imitation_learning/BC/train_bc.py \
#     --dataset dataset/r2s2r_data/task_gen/pour_with_affordance/pour_affordence_pan/merged_trajectories_robo.hdf5 \
#     --stat dataset/r2s2r_data/task_gen/pour_with_affordance/pour_affordence_pan/dataset_stat.json \
#     --total_iters 200000 \
#     --save_freq 1000 \
#     --eval_task_config source/isaaclab_tasks/isaaclab_tasks/real2sim/tasks/pick/config/sceneconfig/task/pour/pour_pan.yaml \
#     --eval_task Pick_panwithhandle_Pour-v0 \
#     --eval_num_rollouts 5 \
#     --eval_horizon 250 \
#     --batch_size 48 \
#     --action_chunk_size 8 --action_horizon 4 --no_remove_idle

# pour_basket
# CUDA_VISIBLE_DEVICES=4 python scripts/imitation_learning/BC/train_bc.py \
#     --dataset dataset/r2s2r_data/task_gen/pour_with_affordance/pour_affordence_basket/merged_trajectories_robo.hdf5 \
#     --stat dataset/r2s2r_data/task_gen/pour_with_affordance/pour_affordence_basket/dataset_stat.json \
#     --total_iters 200000 \
#     --save_freq 1000 \
#     --eval_task_config source/isaaclab_tasks/isaaclab_tasks/real2sim/tasks/pick/config/sceneconfig/task/pour/pour_basket.yaml \
#     --eval_task Pick_basketwithhandle_Pour-v0 \
#     --eval_num_rollouts 5 \
#     --eval_horizon 250 \
#     --batch_size 48 \
#     --action_chunk_size 8 --action_horizon 4 --no_remove_idle

echo "训练运行结束！"
