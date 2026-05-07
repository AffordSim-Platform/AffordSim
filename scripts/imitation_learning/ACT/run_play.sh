#!/bin/bash

# 
set -e

echo " ACT ..."

# 
# pick_banana_and_place_on_plate_final

# CUDA_VISIBLE_DEVICES=3 python scripts/imitation_learning/ACT/play_act.py \
#     --checkpoint logs/act_chunk16_horizon8_h512_z32_merged_trajectories_robo_Pick_Place_banana-v0_1775922529/checkpoints/best_eval.pt \
#     --stat dataset/r2s2r_data/task_gen/pick_place/pick_banana_and_place_on_plate_final/dataset_stat.json \
#     --task Pick_Place_banana-v0 \
#     --task_config source/isaaclab_tasks/isaaclab_tasks/real2sim/tasks/pick/config/sceneconfig/task/pick/pick_banana_place_in_plate.yaml \
#     --horizon 250 --enable_cameras --save_img_interval 0 --num_rollouts 30 --headless

# pour_red_cup_easy
# CUDA_VISIBLE_DEVICES=4 python scripts/imitation_learning/ACT/play_act.py \
#     --checkpoint logs/act_chunk16_horizon8_h512_z32_merged_trajectories_robo_Pick_cupwithhandle_Pour-v0_1775925660/checkpoints/iter_50000.pt \
#     --stat dataset/r2s2r_data/task_gen/pour_with_affordance/pour_red_cup_easy/dataset_stat.json \
#     --task Pick_cupwithhandle_Pour-v0 \
#     --task_config source/isaaclab_tasks/isaaclab_tasks/real2sim/tasks/pick/config/sceneconfig/task/pour/pour_redcup.yaml \
#     --horizon 250 --enable_cameras --save_img_interval 0 --num_rollouts 30 --headless

# pour_pan
# CUDA_VISIBLE_DEVICES=5 python scripts/imitation_learning/ACT/play_act.py \
#     --checkpoint logs/act_chunk16_horizon8_h512_z32_merged_trajectories_robo_Pick_panwithhandle_Pour-v0_1775925674/checkpoints/iter_100000.pt \
#     --stat dataset/r2s2r_data/task_gen/pour_with_affordance/pour_affordence_pan/dataset_stat.json \
#     --task Pick_panwithhandle_Pour-v0 \
#     --task_config source/isaaclab_tasks/isaaclab_tasks/real2sim/tasks/pick/config/sceneconfig/task/pour/pour_pan.yaml \
#     --horizon 250 --enable_cameras --save_img_interval 0 --num_rollouts 30 --headless

# pour_basket
# CUDA_VISIBLE_DEVICES=6 python scripts/imitation_learning/ACT/play_act.py \
#     --checkpoint logs/act_chunk16_horizon8_h512_z32_merged_trajectories_robo_Pick_basketwithhandle_Pour-v0_1775925696/checkpoints/iter_50000.pt \
#     --stat dataset/r2s2r_data/task_gen/pour_with_affordance/pour_affordence_basket/dataset_stat.json \
#     --task Pick_basketwithhandle_Pour-v0 \
#     --task_config source/isaaclab_tasks/isaaclab_tasks/real2sim/tasks/pick/config/sceneconfig/task/pour/pour_basket.yaml \
#     --horizon 250 --enable_cameras --save_img_interval 0 --num_rollouts 30 --headless

# hang cup
# CUDA_VISIBLE_DEVICES=5 python scripts/imitation_learning/ACT/play_act.py \
#     --checkpoint logs/act_chunk16_horizon8_h512_z32_merged_trajectories_robo_Pick_Hang_WhiteCup_On_Rack-v0_1776080494/checkpoints/iter_50000.pt \
#     --stat dataset/r2s2r_data/task_gen/hang_cup/dataset_stat.json \
#     --task Pick_Hang_WhiteCup_On_Rack-v0 \
#     --task_config source/isaaclab_tasks/isaaclab_tasks/real2sim/tasks/config/task/hang/hang_cup.yaml \
#     --horizon 250 --enable_cameras --save_img_interval 0 --num_rollouts 30 --headless

# pull cabinet7
# CUDA_VISIBLE_DEVICES=0 python scripts/imitation_learning/ACT/play_act.py \
#     --checkpoint logs/act_chunk16_horizon8_h512_z32_merged_trajectories_robo_Pick_Pull_Cabinet7-v0_1776279687/checkpoints/iter_50000.pt \
#     --stat dataset/r2s2r_data/task_gen/pull/pull_cabinet7/dataset_stat.json \
#     --task Pick_Pull_Cabinet7-v0 \
#     --task_config source/isaaclab_tasks/isaaclab_tasks/real2sim/tasks/config/task/pull_push/pull_cabinet7.yaml \
#     --horizon 250 --enable_cameras --save_img_interval 0 --num_rollouts 30 --headless

# pick mug cup place coffeemachine
# CUDA_VISIBLE_DEVICES=4 python scripts/imitation_learning/ACT/play_act.py \
#     --checkpoint logs/act_chunk16_horizon8_h512_z32_merged_trajectories_robo_Pick_MugCup_Place_Coffeemachine-v0_1776279009/checkpoints/iter_28000.pt \
#     --stat dataset/r2s2r_data/task_gen/pick_place/pick_mugcup_place_coffeemachine/dataset_stat.json \
#     --task Pick_MugCup_Place_Coffeemachine-v0 \
#     --task_config source/isaaclab_tasks/isaaclab_tasks/real2sim/tasks/config/task/pick/pick_mugcup_place_coffeemachine.yaml \
#     --horizon 250 --enable_cameras --save_img_interval 0 --num_rollouts 30 --headless

# pick cup without handle pour
# CUDA_VISIBLE_DEVICES=0 python scripts/imitation_learning/ACT/play_act.py \
#     --checkpoint logs/act_chunk16_horizon8_h512_z32_merged_trajectories_robo_Pick_CupWithouthandle_Pour-v0_1776346142/checkpoints/iter_6000.pt \
#     --stat dataset/r2s2r_data/task_gen/pour_with_affordance/pour_withouthandlecup/dataset_stat.json \
#     --task Pick_CupWithouthandle_Pour-v0 \
#     --task_config source/isaaclab_tasks/isaaclab_tasks/real2sim/tasks/config/task/pour/pour_papercup.yaml \
#     --horizon 250 --enable_cameras --save_img_interval 0 --num_rollouts 30 --headless

# pick_pinkcup_place_woodshelf
# CUDA_VISIBLE_DEVICES=0 python scripts/imitation_learning/ACT/play_act.py \
#     --checkpoint logs/act_chunk16_horizon8_h512_z32_merged_trajectories_robo_Pick_PinkCup_Place_WoodShelf-v0_1776439611/checkpoints/iter_50000.pt \
#     --stat dataset/r2s2r_data/task_gen/pick_place/pick_pinkcup_place_woodshelf/dataset_stat.json \
#     --task Pick_PinkCup_Place_WoodShelf-v0 \
#     --task_config source/isaaclab_tasks/isaaclab_tasks/real2sim/tasks/config/task/pick/pick_cup_place_woodshelf.yaml \
#     --horizon 250 --enable_cameras --save_img_interval 0 --num_rollouts 30 --headless

# hang redmug
# CUDA_VISIBLE_DEVICES=3 python scripts/imitation_learning/ACT/play_act.py \
#     --checkpoint logs/act_chunk16_horizon8_h512_z32_merged_trajectories_robo_Pick_Hang_RedMug_On_Rack-v0_1776440528/checkpoints/iter_22000.pt \
#     --stat dataset/r2s2r_data/task_gen/hang_cup/redmug/dataset_stat.json \
#     --task Pick_Hang_RedMug_On_Rack-v0 \
#     --task_config source/isaaclab_tasks/isaaclab_tasks/real2sim/tasks/config/task/hang/hang_mugcup.yaml \
#     --horizon 250 --enable_cameras --save_img_interval 0 --num_rollouts 30 --headless

# stack cube
# CUDA_VISIBLE_DEVICES=4 python scripts/imitation_learning/ACT/play_act.py \
#     --checkpoint logs/act_chunk16_horizon8_h512_z32_merged_trajectories_robo_Pick_Stack_Cube-v0_1776440261/checkpoints/iter_30000.pt \
#     --stat dataset/r2s2r_data/task_gen/stack/stack_cube417/dataset_stat.json \
#     --task Pick_Stack_Cube-v0 \
#     --task_config source/isaaclab_tasks/isaaclab_tasks/real2sim/tasks/config/task/stack/stack_cube.yaml \
#     --horizon 250 --enable_cameras --save_img_interval 0 --num_rollouts 30 --headless

# pick_lid_place_pot
# CUDA_VISIBLE_DEVICES=0 python scripts/imitation_learning/ACT/play_act.py \
#     --checkpoint logs/act_chunk16_horizon8_h512_z32_merged_trajectories_robo_Pick_Lid_Place_Pot-v0_1776586841/checkpoints/iter_4000.pt \
#     --stat dataset/r2s2r_data/task_gen/pick_place/pick_lid_place_pot/dataset_stat.json \
#     --task_config source/isaaclab_tasks/isaaclab_tasks/real2sim/tasks/config/task/pick/pick_lid_place_pot.yaml \
#     --task Pick_Lid_Place_Pot-v0 \
#     --horizon 250 --enable_cameras --save_img_interval 0 --num_rollouts 30 --headless

# # push_coffeecan
# CUDA_VISIBLE_DEVICES=1 python scripts/imitation_learning/ACT/play_act.py \
#     --checkpoint logs/act_chunk16_horizon8_h512_z32_merged_trajectories_robo_Pick_Push_Coffeecan-v0_1776586960/checkpoints/iter_6000.pt \
#     --stat dataset/r2s2r_data/task_gen/push/push_coffeecan/dataset_stat.json \
#     --task_config source/isaaclab_tasks/isaaclab_tasks/real2sim/tasks/config/task/pull_push/push_coffeecan.yaml \
#     --task Pick_Push_Coffeecan-v0 \
#     --horizon 250 --enable_cameras --save_img_interval 0 --num_rollouts 30 --headless

# open_microwave
# CUDA_VISIBLE_DEVICES=2 python scripts/imitation_learning/ACT/play_act.py \
#     --checkpoint logs/act_chunk16_horizon8_h512_z32_merged_trajectories_robo_Pick_Open_Microwave-v0_1776587252/checkpoints/iter_26000.pt \
#     --stat dataset/r2s2r_data/task_gen/open/open_microwave/dataset_stat.json \
#     --task_config source/isaaclab_tasks/isaaclab_tasks/real2sim/tasks/config/task/open_close/open_microwave.yaml \
#     --task Pick_Open_Microwave-v0 \
#     --horizon 250 --enable_cameras --save_img_interval 0 --num_rollouts 30 --headless

# open_microwave
CUDA_VISIBLE_DEVICES=4 python scripts/imitation_learning/ACT/play_act.py \
    --checkpoint logs/act_chunk16_horizon8_h512_z32_merged_trajectories_robo_Pick_Pour_WhiteKettle4-v0_1776769102/checkpoints/iter_30000.pt \
    --stat dataset/r2s2r_data/task_gen/pour_with_affordance/pour_whitekettle4/dataset_stat.json \
    --task_config source/isaaclab_tasks/isaaclab_tasks/real2sim/tasks/config/task/pour/pour_whitekettle4.yaml \
    --task Pick_Pour_WhiteKettle4-v0 \
    --horizon 250 --enable_cameras --save_img_interval 0 --num_rollouts 30 --headless

echo ""