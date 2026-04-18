#!/bin/bash

# 设置遇到错误时自动退出
set -e

echo "开始运行 Diffusion Policy 训练脚本..."

# 运行训练命令
# pick_banana_and_place_on_plate_final
python scripts/imitation_learning/diffusion_policy/train.py \
    --dataset /home/limingyang/project/Real2Sim2Real-Pipeline-affordsim_clean/dataset/r2s2r_data/task_gen/pick_place/pick_banana_and_place_on_plate_final/merged_trajectories_robo.hdf5 \
    --stat /home/limingyang/project/Real2Sim2Real-Pipeline-affordsim_clean/dataset/r2s2r_data/task_gen/pick_place/pick_banana_and_place_on_plate_final/dataset_stat.json \
    --epochs 1000 \
    --save_every 50 \
    --eval_task_config /home/limingyang/project/Real2Sim2Real-Pipeline-affordsim_clean/source/isaaclab_tasks/isaaclab_tasks/real2sim/tasks/pick/config/sceneconfig/task/pick/pick_banana_place_in_plate.yaml \
    --eval_task Pick_Place_banana-v0 \
    --eval_num_rollouts 5 \
    --eval_horizon 300

# pour_red_cup
# python scripts/imitation_learning/diffusion_policy/train.py \
#     --dataset /home/limingyang/project/Real2Sim2Real-Pipeline-affordsim_clean/dataset/r2s2r_data/task_gen/pour_with_affordance/pour_red_cup/merged_trajectories_robo.hdf5 \
#     --stat /home/limingyang/project/Real2Sim2Real-Pipeline-affordsim_clean/dataset/r2s2r_data/task_gen/pour_with_affordance/pour_red_cup/dataset_stat.json \
#     --epochs 1000 \
#     --save_every 50 \
#     --eval_task_config /home/limingyang/project/Real2Sim2Real-Pipeline-affordsim_clean/source/isaaclab_tasks/isaaclab_tasks/real2sim/tasks/pick/config/sceneconfig/task/pour/pour_affordence_redcup_new.yaml \
#     --eval_task Pick_cupwithhandle_Pour-v0 \
#     --eval_num_rollouts 5 \
#     --eval_horizon 300

echo "训练运行结束！"