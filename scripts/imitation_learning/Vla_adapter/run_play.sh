# python scripts/imitation_learning/Vla_adapter/play_vla_adapter.py \
#     --task Pick_cupwithhandle_Pour-v0 \
#     --server_host 10.184.17.177 \
#     --server_port 8001 \
#     --unnorm_key pour_redcup422_ee \
#     --prompt "pick up the red mug, place it over the yellow bowl and tilt it." \
#     --task_config source/isaaclab_tasks/isaaclab_tasks/real2sim/tasks/config/task/pour/pour_mugcup.yaml \
#     --num_rollouts 30 \
#     --horizon 400 \
#     --action_horizon 8 --headless --enable_cameras

python scripts/imitation_learning/Vla_adapter/play_vla_adapter.py \
    --task Pick_Place_banana-v0 \
    --server_host 10.184.17.177 \
    --server_port 8001 \
    --unnorm_key pick_banana_place_in_white_plate_ee \
    --prompt "Pick up the banana and put it on the white plate." \
    --task_config source/isaaclab_tasks/isaaclab_tasks/real2sim/tasks/config/task/pick/pick_banana_place_in_plate.yaml \
    --num_rollouts 30 \
    --horizon 400 \
    --action_horizon 8 --headless --enable_cameras