python scripts/imitation_learning/diffusion_policy/play_real2sim.py \
    --checkpoint /path/to/model_best.pth \
    --stat /path/to/dataset_stat.json \
    --task_config /path/to/task_config.yaml \
    --task Pick_Place_banana-v0 \
    --port 8070 --horizon 300