```bash
python scripts/reinforcement_learning/rsl_rl/play.py \
    --task Isaac-Lift-Cube-Franka-ViT-v0 --num_envs 32 --enable_cameras
```

```bash
python scripts/reinforcement_learning/rsl_rl/train.py \
    --task Isaac-Lift-Cube-Franka-ViT-v0 --num_envs 512 --enable_cameras --headless
```

```bash
python scripts/reinforcement_learning/rsl_rl/train.py \
    --task FrankaFr3-DefaultGripper-IKRel-VisPick-v0 --num_envs 512 --enable_cameras --headless
```

```bash
python scripts/reinforcement_learning/rsl_rl/play.py \
    --task FrankaFr3-DefaultGripper-IKRel-VisPick-v0 --num_envs 1 --enable_cameras
```

```bash
python scripts/reinforcement_learning/rsl_rl/train.py \
    --task FrankaFr3-OurGripper-IKRel-PPOBC-VisPick-v0 --num_envs 512 --enable_cameras --headless
```

```bash
python scripts/reinforcement_learning/rsl_rl/play.py \
    --task FrankaFr3-OurGripper-IKRel-PPOBC-VisPick-v0 --num_envs 1 --enable_cameras
```

# Robomimic BC

```bash
python scripts/imitation_learning/robomimic/train.py \
    --task Pick_Place_banana-v0 --algo bc \
    --dataset /home/lmy/code/Real2Sim2Real-Pipeline-auto_task/dataset/r2s2r_data/task_gen/pick_place/pick_banana_and_place_on_plate/hdf5/merged_trajectories_robomimic.hdf5
```

```bash
python scripts/imitation_learning/robomimic/play.py \
    --task Pick_Place_banana-v0 \
    --checkpoint /home/lmy/code/Real2Sim2Real-Pipeline-auto_task/logs/robomimic/Pick_Place_banana-v0/bc/20260329003310/models/model_epoch_100.pth \
    --dataset_stat /home/lmy/code/Real2Sim2Real-Pipeline-auto_task/dataset/r2s2r_data/task_gen/pick_place/pick_banana_and_place_on_plate/hdf5/dataset_stat.json \
    --enable_cameras
```

# BC

```bash
python scripts/imitation_learning/BC/play_bc.py \
    --task Pick_Place_banana-v0 \
    --checkpoint /home/lmy/code/Real2Sim2Real-Pipeline-auto_task/runs/bc_hdf5merged_trajectories_1.hdf5_1775119766/checkpoints/iter_5000.pt \
    --dataset_stat /home/lmy/code/Real2Sim2Real-Pipeline-auto_task/dataset/r2s2r_data/task_gen/pick_place/pick_banana_and_place_on_plate_new/dataset_stat.json \
    --enable_cameras \
    --task_config /home/lmy/code/Real2Sim2Real-Pipeline-auto_task/source/isaaclab_tasks/isaaclab_tasks/real2sim/tasks/task_gen/config/sceneconfig/generated_task_config.yaml

python scripts/imitation_learning/BC/train_bc.py \
    --task Pick_Place_banana-v0 \
    --demo_path /home/lmy/code/Real2Sim2Real-Pipeline-auto_task/dataset/r2s2r_data/task_gen/pick_place/pick_banana_and_place_on_plate_new/hdf5merged_trajectories_1.hdf5
```

# DP

```bash
python scripts/imitation_learning/diffusion_policy/play.py \
    --checkpoint logs/diffusion_policy/diffusion_policy_20260330004109/models/model_epoch_500.pth \
    --stat dataset/r2s2r_data/task_gen/pick_place/pick_banana_and_place_on_plate/hdf5/dataset_stat.json \
    --task Pick_Place_banana-v0 \
    --task_config source/isaaclab_tasks/isaaclab_tasks/real2sim/tasks/task_gen/config/sceneconfig/generated_task_config.yaml \
    --horizon 300 --enable_cameras
```

```bash
python scripts/imitation_learning/diffusion_policy/train.py \
    --dataset /home/lmy/code/Real2Sim2Real-Pipeline-auto_task/dataset/r2s2r_data/task_gen/pick_place/pick_banana_and_place_on_plate_new/hdf5merged_trajectories.hdf5 \
    --stat /home/lmy/code/Real2Sim2Real-Pipeline-auto_task/dataset/r2s2r_data/task_gen/pick_place/pick_banana_and_place_on_plate_new/dataset_stat.json \
    --epochs 1000 --batch_size 64 --pred_horizon 16 --obs_horizon 2 
```

# RDT

```bash
python scripts/imitation_learning/RoboticsDiffusionTransformer/play_isaaclab.py \
    --task Pick_Place_banana-v0 \
    --checkpoint scripts/imitation_learning/RoboticsDiffusionTransformer/checkpoints/models--robotics-diffusion-transformer--rdt-170m/checkpoint-200000/pytorch_model/mp_rank_00_model_states.pt \
    --lang_embed_path scripts/imitation_learning/RoboticsDiffusionTransformer/data/datasets/rdt-ft-data/franka_fr3/pick_banana_and_place_in_plate/lang_embed_0.pt \
    --data_stat scripts/imitation_learning/RoboticsDiffusionTransformer/data/datasets/rdt-ft-data/franka_fr3/pick_banana_and_place_in_plate/data_stat.json \
    --task_config source/isaaclab_tasks/isaaclab_tasks/real2sim/tasks/task_gen/config/sceneconfig/generated_task_config.yaml \
    --horizon 400 --num_rollouts 5 --enable_cameras
```
