#!/bin/bash
# dp_rss . : isaacsim5 conda env (torch/torchvision/h5py/diffusers/einops).

set -e
echo " dp_rss (RSS-faithful Diffusion Policy) ..."


# CUDA_VISIBLE_DEVICES=4 python scripts/imitation_learning/dp_rss/train.py \
#     --dataset dataset/r2s2r_data/task_gen/pour_with_affordance/pour_whitekettle4/merged_trajectories_robo.hdf5 \
#     --stat dataset/r2s2r_data/task_gen/pour_with_affordance/pour_whitekettle4/dataset_stat.json \
#     --epochs 1000 --batch_size 64 --save_every 50 \
#     --pred_horizon 16 --obs_horizon 2 --action_horizon 8 \
#     --num_diffusion_steps 100 --beta_schedule squaredcos_cap_v2 \
#     --inference_scheduler ddim --num_inference_steps 16 \
#     --eval_task_config source/isaaclab_tasks/isaaclab_tasks/real2sim/tasks/config/task/pour/pour_whitekettle4.yaml \
#     --eval_task Pick_Pour_WhiteKettle4-v0 \
#     --eval_num_rollouts 5 --eval_horizon 300

# CUDA_VISIBLE_DEVICES=0 python scripts/imitation_learning/dp_rss/train.py \
#     --dataset dataset/r2s2r_data/task_gen/pick_place/pick_banana_place_bowl_nogap/merged_trajectories_robo.hdf5 \
#     --stat dataset/r2s2r_data/task_gen/pick_place/pick_banana_place_bowl_nogap/dataset_stat.json \
#     --epochs 1000 --batch_size 64 --save_every 50 \
#     --pred_horizon 16 --obs_horizon 2 --action_horizon 8 \
#     --num_diffusion_steps 100 --beta_schedule squaredcos_cap_v2 \
#     --inference_scheduler ddim --num_inference_steps 16 \
#     --eval_task_config source/isaaclab_tasks/isaaclab_tasks/real2sim/tasks/config/task/pick/pick_banana_place_bowl.yaml \
#     --eval_task Pick_Banana_Place_Bowl_Nogap_joint-v0 \
#     --eval_num_rollouts 5 --eval_horizon 300 \
#     --eval_joint_delta_mode cumulative --action_type joint_delta

# CUDA_VISIBLE_DEVICES=1 python scripts/imitation_learning/dp_rss/train.py \
#     --dataset dataset/r2s2r_data/task_gen/pick_place/pick_banana_place_bowl_nogap_noideal/merged_trajectories_joint_robo.hdf5 \
#     --stat dataset/r2s2r_data/task_gen/pick_place/pick_banana_place_bowl_nogap_noideal/dataset_stat.json \
#     --epochs 1000 --batch_size 64 --save_every 50 \
#     --pred_horizon 16 --obs_horizon 2 --action_horizon 8 \
#     --num_diffusion_steps 100 --beta_schedule squaredcos_cap_v2 \
#     --inference_scheduler ddim --num_inference_steps 16 \
#     --eval_task_config source/isaaclab_tasks/isaaclab_tasks/real2sim/tasks/config/task/pick/pick_banana_place_bowl.yaml \
#     --eval_task Pick_Banana_Place_Bowl_Nogap_joint-v0 \
#     --eval_num_rollouts 5 --eval_horizon 300 \
#     --eval_joint_delta_mode cumulative --action_type joint_delta

# CUDA_VISIBLE_DEVICES=0 python scripts/imitation_learning/dp_rss/train.py \
#     --dataset dataset/r2s2r_data/task_gen/pick_place/pick_banana_place_bowl_nogap_noideal/merged_trajectories_joint_robo.hdf5 \
#     --stat dataset/r2s2r_data/task_gen/pick_place/pick_banana_place_bowl_nogap_noideal/dataset_stat.json \
#     --epochs 1000 --batch_size 64 --save_every 50 \
#     --pred_horizon 16 --obs_horizon 2 --action_horizon 8 \
#     --num_diffusion_steps 100 --beta_schedule squaredcos_cap_v2 \
#     --inference_scheduler ddim --num_inference_steps 16 \
#     --eval_task_config source/isaaclab_tasks/isaaclab_tasks/real2sim/tasks/config/task/pick/pick_banana_place_bowl.yaml \
#     --eval_task Pick_Banana_Place_Bowl_Nogap_joint-v0 \
#     --eval_num_rollouts 5 --eval_horizon 300 \
#     --eval_joint_delta_mode cumulative --action_type joint_delta --no_image_aug

# CUDA_VISIBLE_DEVICES=2 python scripts/imitation_learning/dp_rss/train.py \
#     --dataset dataset/r2s2r_data/task_gen/pick_place/pick_banana_place_bowl_nogap_noideal/merged_trajectories_ee_robo.hdf5 \
#     --stat dataset/r2s2r_data/task_gen/pick_place/pick_banana_place_bowl_nogap_noideal/dataset_stat_ee.json \
#     --epochs 1000 --batch_size 64 --save_every 50 \
#     --pred_horizon 16 --obs_horizon 2 --action_horizon 8 \
#     --num_diffusion_steps 100 --beta_schedule squaredcos_cap_v2 \
#     --inference_scheduler ddim --num_inference_steps 16 \
#     --eval_task_config source/isaaclab_tasks/isaaclab_tasks/real2sim/tasks/config/task/pick/pick_banana_place_bowl.yaml \
#     --eval_task Pick_Banana_Place_Bowl_Nogap-v0 \
#     --eval_num_rollouts 5 --eval_horizon 300 \
#     --eval_joint_delta_mode cumulative --action_type ee_delta

# CUDA_VISIBLE_DEVICES=6 python scripts/imitation_learning/dp_rss/train.py \
#     --dataset dataset/r2s2r_data/task_gen/pour_with_affordance/pour_redcup422/merged_trajectories_robo_no_qvel.hdf5 \
#     --stat dataset/r2s2r_data/task_gen/pour_with_affordance/pour_redcup422/dataset_stat_no_qvel.json \
#     --epochs 1000 --batch_size 64 --save_every 50 \
#     --pred_horizon 16 --obs_horizon 2 --action_horizon 8 \
#     --num_diffusion_steps 100 --beta_schedule squaredcos_cap_v2 \
#     --inference_scheduler ddim --num_inference_steps 16 \
#     --eval_task_config source/isaaclab_tasks/isaaclab_tasks/real2sim/tasks/config/task/pour/pour_mugcup.yaml \
#     --eval_task Pick_cupwithhandle_Pour-v0 \
#     --eval_num_rollouts 5 --eval_horizon 400 \
#     --eval_joint_delta_mode cumulative --action_type ee_delta --no_qvel

CUDA_VISIBLE_DEVICES=7 python scripts/imitation_learning/dp_rss/train.py \
    --dataset dataset/r2s2r_data/task_gen/pour_with_affordance/pour_redcup422/merged_trajectories_robo.hdf5 \
    --stat dataset/r2s2r_data/task_gen/pour_with_affordance/pour_redcup422/dataset_stat.json \
    --epochs 1000 --batch_size 64 --save_every 50 \
    --pred_horizon 16 --obs_horizon 2 --action_horizon 8 \
    --num_diffusion_steps 100 --beta_schedule squaredcos_cap_v2 \
    --inference_scheduler ddim --num_inference_steps 16 \
    --eval_task_config source/isaaclab_tasks/isaaclab_tasks/real2sim/tasks/config/task/pour/pour_mugcup.yaml \
    --eval_task Pick_cupwithhandle_Pour-v0 \
    --eval_num_rollouts 5 --eval_horizon 400 \
    --eval_joint_delta_mode cumulative --action_type ee_delta

echo ""
