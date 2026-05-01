# pick_banana_place_bowl_nogap
CUDA_VISIBLE_DEVICES=0 python scripts/imitation_learning/dp/play.py \
    --checkpoint logs/dp_rss/dp_rss_Pick_Banana_Place_Bowl_Nogap_joint-v0_20260424020126/models/model_epoch_400.pth \
    --stat dataset/r2s2r_data/task_gen/pick_place/pick_banana_place_bowl_nogap_noideal/dataset_stat.json \
    --task_config source/isaaclab_tasks/isaaclab_tasks/real2sim/tasks/config/task/pick/pick_banana_place_bowl.yaml \
    --task Pick_Banana_Place_Bowl_Nogap_joint-v0 \
    --horizon 500 --enable_cameras  --num_rollouts 30 --headless --action_horizon 4
