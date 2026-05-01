# pick_banana_and_place_on_plate_final
# CUDA_VISIBLE_DEVICES=0 python scripts/imitation_learning/ACT/train_act.py \
#     --dataset dataset/r2s2r_data/task_gen/pick_place/pick_banana_and_place_on_plate_final/merged_trajectories_robo.hdf5 \
#     --stat dataset/r2s2r_data/task_gen/pick_place/pick_banana_and_place_on_plate_final/dataset_stat.json \
#     --total_iters 200000 \
#     --batch_size 200 \
#     --save_freq 2000 \
#     --action_chunk_size 16 \
#     --action_horizon 8 \
#     --kl_weight 10.0 \
#     --eval_task_config source/isaaclab_tasks/isaaclab_tasks/real2sim/tasks/pick/config/sceneconfig/task/pick/pick_banana_place_in_plate.yaml \
#     --eval_task Pick_Place_banana-v0 --eval_num_rollouts 5

# pour_red_cup_easy
# CUDA_VISIBLE_DEVICES=5 python scripts/imitation_learning/ACT/train_act.py \
#     --dataset dataset/r2s2r_data/task_gen/pour_with_affordance/pour_red_cup_easy/merged_trajectories_robo.hdf5 \
#     --stat dataset/r2s2r_data/task_gen/pour_with_affordance/pour_red_cup_easy/dataset_stat.json \
#     --total_iters 200000 \
#     --batch_size 200 \
#     --save_freq 2000 \
#     --action_chunk_size 16 \
#     --action_horizon 8 \
#     --kl_weight 10.0 \
#     --eval_task_config source/isaaclab_tasks/isaaclab_tasks/real2sim/tasks/pick/config/sceneconfig/task/pour/pour_affordence_redcup_new.yaml \
#     --eval_task Pick_cupwithhandle_Pour-v0 --eval_num_rollouts 5

# pour_pan
# CUDA_VISIBLE_DEVICES=6 python scripts/imitation_learning/ACT/train_act.py \
#     --dataset dataset/r2s2r_data/task_gen/pour_with_affordance/pour_affordence_pan/merged_trajectories_robo.hdf5 \
#     --stat dataset/r2s2r_data/task_gen/pour_with_affordance/pour_affordence_pan/dataset_stat.json \
#     --total_iters 200000 \
#     --batch_size 200 \
#     --save_freq 2000 \
#     --action_chunk_size 16 \
#     --action_horizon 8 \
#     --kl_weight 10.0 \
#     --eval_task_config source/isaaclab_tasks/isaaclab_tasks/real2sim/tasks/pick/config/sceneconfig/task/pour/pour_affordence_pan.yaml \
#     --eval_task Pick_panwithhandle_Pour-v0 --eval_num_rollouts 5

# # pour_basket
# CUDA_VISIBLE_DEVICES=7 python scripts/imitation_learning/ACT/train_act.py \
#     --dataset dataset/r2s2r_data/task_gen/pour_with_affordance/pour_affordence_basket/merged_trajectories_robo.hdf5 \
#     --stat dataset/r2s2r_data/task_gen/pour_with_affordance/pour_affordence_basket/dataset_stat.json \
#     --total_iters 200000 \
#     --batch_size 200 \
#     --save_freq 2000 \
#     --action_chunk_size 16 \
#     --action_horizon 8 \
#     --kl_weight 10.0 \
#     --eval_task_config source/isaaclab_tasks/isaaclab_tasks/real2sim/tasks/pick/config/sceneconfig/task/pour/pour_affordence_basket.yaml \
#     --eval_task Pick_basketwithhandle_Pour-v0 --eval_num_rollouts 5

# hang cup
# CUDA_VISIBLE_DEVICES=6 python scripts/imitation_learning/ACT/train_act.py \
#     --dataset dataset/r2s2r_data/task_gen/hang_cup/merged_trajectories_robo.hdf5 \
#     --stat dataset/r2s2r_data/task_gen/hang_cup/dataset_stat.json \
#     --total_iters 200000 \
#     --batch_size 200 \
#     --save_freq 2000 \
#     --action_chunk_size 16 \
#     --action_horizon 8 \
#     --kl_weight 10.0 \
#     --eval_task_config source/isaaclab_tasks/isaaclab_tasks/real2sim/tasks/pick/config/sceneconfig/task/hang/hang_whitecup_on_rack.yaml \
#     --eval_task Pick_Hang_WhiteCup_On_Rack-v0 --eval_num_rollouts 5



# stack cube
# CUDA_VISIBLE_DEVICES=6 python scripts/imitation_learning/ACT/train_act.py \
#     --dataset dataset/r2s2r_data/task_gen/stack/stack_cube417/merged_trajectories_robo.hdf5 \
#     --stat dataset/r2s2r_data/task_gen/stack/stack_cube417/dataset_stat.json \
#     --total_iters 200000 \
#     --batch_size 200 \
#     --save_freq 2000 \
#     --action_chunk_size 16 \
#     --action_horizon 8 \
#     --kl_weight 10.0 \
#     --eval_task_config source/isaaclab_tasks/isaaclab_tasks/real2sim/tasks/config/task/stack/stack_cube.yaml \
#     --eval_task Pick_Stack_Cube-v0 \
#     --eval_num_rollouts 5 \
#     --eval_horizon 300

# pull cabinet
# CUDA_VISIBLE_DEVICES=2 python scripts/imitation_learning/ACT/train_act.py \
#     --dataset dataset/r2s2r_data/task_gen/pull/pull_cabinet7/merged_trajectories_robo.hdf5 \
#     --stat dataset/r2s2r_data/task_gen/pull/pull_cabinet7/dataset_stat.json \
#     --total_iters 200000 \
#     --batch_size 200 \
#     --save_freq 2000 \
#     --action_chunk_size 16 \
#     --action_horizon 8 \
#     --kl_weight 10.0 \
#     --eval_task_config source/isaaclab_tasks/isaaclab_tasks/real2sim/tasks/pick/config/sceneconfig/task/pull/pull_cabinet7.yaml \
#     --eval_task Pick_Pull_Cabinet7-v0 \
#     --eval_num_rollouts 5 \
#     --eval_horizon 300

# pick_mugcup_place_coffeemachine
# CUDA_VISIBLE_DEVICES=3 python scripts/imitation_learning/ACT/train_act.py \
#     --dataset dataset/r2s2r_data/task_gen/pick_place/pick_mugcup_place_coffeemachine/merged_trajectories_robo.hdf5 \
#     --stat dataset/r2s2r_data/task_gen/pick_place/pick_mugcup_place_coffeemachine/dataset_stat.json \
#     --total_iters 200000 \
#     --batch_size 200 \
#     --save_freq 2000 \
#     --action_chunk_size 16 \
#     --action_horizon 8 \
#     --kl_weight 10.0 \
#     --eval_task_config source/isaaclab_tasks/isaaclab_tasks/real2sim/tasks/config/task/pick/pick_mugcup_place_coffeemachine.yaml \
#     --eval_task Pick_MugCup_Place_Coffeemachine-v0 \
#     --eval_num_rollouts 5 \
#     --eval_horizon 300

# pick_cupwithouthandle and pour
# CUDA_VISIBLE_DEVICES=6 python scripts/imitation_learning/ACT/train_act.py \
#     --dataset dataset/r2s2r_data/task_gen/pour_with_affordance/pour_withouthandlecup/merged_trajectories_robo.hdf5 \
#     --stat dataset/r2s2r_data/task_gen/pour_with_affordance/pour_withouthandlecup/dataset_stat.json \
#     --total_iters 200000 \
#     --batch_size 200 \
#     --save_freq 2000 \
#     --action_chunk_size 16 \
#     --action_horizon 8 \
#     --kl_weight 10.0 \
#     --eval_task_config source/isaaclab_tasks/isaaclab_tasks/real2sim/tasks/config/task/pour/pour_papercup.yaml \
#     --eval_task Pick_CupWithouthandle_Pour-v0 \
#     --eval_num_rollouts 5 \
#     --eval_horizon 300

# pick_pinkcup_place_woodshelf
# CUDA_VISIBLE_DEVICES=3 python scripts/imitation_learning/ACT/train_act.py \
#     --dataset dataset/r2s2r_data/task_gen/pick_place/pick_pinkcup_place_woodshelf/merged_trajectories_robo.hdf5 \
#     --stat dataset/r2s2r_data/task_gen/pick_place/pick_pinkcup_place_woodshelf/dataset_stat.json \
#     --total_iters 200000 \
#     --batch_size 180 \
#     --save_freq 2000 \
#     --action_chunk_size 16 \
#     --action_horizon 8 \
#     --kl_weight 10.0 \
#     --eval_task_config source/isaaclab_tasks/isaaclab_tasks/real2sim/tasks/config/task/pick/pick_cup_place_woodshelf.yaml \
#     --eval_task Pick_PinkCup_Place_WoodShelf-v0 \
#     --eval_num_rollouts 5 \
#     --eval_horizon 300

# hang redmug
# CUDA_VISIBLE_DEVICES=7 python scripts/imitation_learning/ACT/train_act.py \
#     --dataset dataset/r2s2r_data/task_gen/hang_cup/redmug/merged_trajectories_robo.hdf5 \
#     --stat dataset/r2s2r_data/task_gen/hang_cup/redmug/dataset_stat.json \
#     --total_iters 200000 \
#     --batch_size 180 \
#     --save_freq 2000 \
#     --action_chunk_size 16 \
#     --action_horizon 8 \
#     --kl_weight 10.0 \
#     --eval_task_config source/isaaclab_tasks/isaaclab_tasks/real2sim/tasks/config/task/hang/hang_mugcup.yaml \
#     --eval_task Pick_Hang_RedMug_On_Rack-v0 \
#     --eval_num_rollouts 5 \
#     --eval_horizon 300

# pick_lid_place_pot
# CUDA_VISIBLE_DEVICES=0 python scripts/imitation_learning/ACT/train_act.py \
#     --dataset dataset/r2s2r_data/task_gen/pick_place/pick_lid_place_pot/merged_trajectories_robo.hdf5 \
#     --stat dataset/r2s2r_data/task_gen/pick_place/pick_lid_place_pot/dataset_stat.json \
#     --total_iters 200000 \
#     --batch_size 180 \
#     --save_freq 2000 \
#     --action_chunk_size 16 \
#     --action_horizon 8 \
#     --kl_weight 10.0 \
#     --eval_task_config source/isaaclab_tasks/isaaclab_tasks/real2sim/tasks/config/task/pick/pick_lid_place_pot.yaml \
#     --eval_task Pick_Lid_Place_Pot-v0 \
#     --eval_num_rollouts 5 \
#     --eval_horizon 300

# # push_coffeecan
# CUDA_VISIBLE_DEVICES=2 python scripts/imitation_learning/ACT/train_act.py \
#     --dataset dataset/r2s2r_data/task_gen/push/push_coffeecan/merged_trajectories_robo.hdf5 \
#     --stat dataset/r2s2r_data/task_gen/push/push_coffeecan/dataset_stat.json \
#     --total_iters 200000 \
#     --batch_size 180 \
#     --save_freq 2000 \
#     --action_chunk_size 16 \
#     --action_horizon 8 \
#     --kl_weight 10.0 \
#     --eval_task_config source/isaaclab_tasks/isaaclab_tasks/real2sim/tasks/config/task/pull_push/push_coffeecan.yaml \
#     --eval_task Pick_Push_Coffeecan-v0 \
#     --eval_num_rollouts 5 \
#     --eval_horizon 300

# open_microwave
# CUDA_VISIBLE_DEVICES=4 python scripts/imitation_learning/ACT/train_act.py \
#     --dataset dataset/r2s2r_data/task_gen/open/open_microwave/merged_trajectories_robo.hdf5 \
#     --stat dataset/r2s2r_data/task_gen/open/open_microwave/dataset_stat.json \
#     --total_iters 200000 \
#     --batch_size 180 \
#     --save_freq 2000 \
#     --action_chunk_size 16 \
#     --action_horizon 8 \
#     --kl_weight 10.0 \
#     --eval_task_config source/isaaclab_tasks/isaaclab_tasks/real2sim/tasks/config/task/open_close/open_microwave.yaml \
#     --eval_task Pick_Open_Microwave-v0 \
#     --eval_num_rollouts 5 \
#     --eval_horizon 300

# pour_whitekettle4
# CUDA_VISIBLE_DEVICES=4 python scripts/imitation_learning/ACT/train_act.py \
#     --dataset dataset/r2s2r_data/task_gen/pour_with_affordance/pour_whitekettle4/merged_trajectories_robo.hdf5 \
#     --stat dataset/r2s2r_data/task_gen/pour_with_affordance/pour_whitekettle4/dataset_stat.json \
#     --total_iters 200000 \
#     --batch_size 180 \
#     --save_freq 2000 \
#     --action_chunk_size 16 \
#     --action_horizon 8 \
#     --kl_weight 10.0 \
#     --eval_task_config source/isaaclab_tasks/isaaclab_tasks/real2sim/tasks/config/task/pour/pour_whitekettle4.yaml \
#     --eval_task Pick_Pour_WhiteKettle4-v0 \
#     --eval_num_rollouts 5 \
#     --eval_horizon 300
