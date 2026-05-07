set -e

echo " BC ..."

# python scripts/imitation_learning/BC/play_bc.py \
#     --checkpoint logs/bc_rnn_seq10_chunk8_horizon4_merged_trajectories_robo.hdf5_Pick_Place_banana-v0_1775850763/checkpoints/iter_20000.pt \
#     --dataset_stat dataset/r2s2r_data/task_gen/pick_place/pick_banana_and_place_on_plate_final/dataset_stat.json \
#     --task Pick_Place_banana-v0 \
#     --task_config source/isaaclab_tasks/isaaclab_tasks/real2sim/tasks/pick/config/sceneconfig/task/pick/pick_banana_place_in_plate.yaml \
#     --horizon 250 --enable_cameras --save_img_interval 0 --num_rollouts 30 --headless

CUDA_VISIBLE_DEVICES=7 python scripts/imitation_learning/BC/play_bc.py \
    --checkpoint logs/bc_gnll_chunk8_horizon4_merged_trajectories_robo.hdf5_Pick_Place_banana-v0_1775922586/checkpoints/iter_60000.pt \
    --dataset_stat dataset/r2s2r_data/task_gen/pick_place/pick_banana_place_in_plate_fix/dataset_stat.json \
    --task Pick_Place_banana-v0 \
    --task_config source/isaaclab_tasks/isaaclab_tasks/real2sim/tasks/pick/config/sceneconfig/task/pick/pick_banana_fix.yaml \
    --horizon 250 --enable_cameras --save_img_interval 0 --num_rollouts 30 --action_horizon 4 --headless

echo ""
