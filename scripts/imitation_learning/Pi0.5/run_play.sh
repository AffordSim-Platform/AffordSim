python scripts/imitation_learning/Pi0.5/play.py --task Pick_Place_banana-v0 \
    --server_host 10.184.17.177 --server_port 8000 \
    --prompt "pick banana and place it on the white plate" \
    --task_config source/isaaclab_tasks/isaaclab_tasks/real2sim/tasks/pick/config/sceneconfig/task/pick/pick_banana_place_in_plate.yaml \
    --horizon 250 --num_rollouts 30 --enable_cameras --headless
