# Pick Banana Place Plate: RGB-only DP3 Runbook

日期：2026-05-28

这份 runbook 走真机迁移路径。训练和仿真验证都不要读仿真 depth / point cloud，也不要用背景替换图：

```text
AffordSim 纯仿真 RGB original_images/external_cam_side
  -> Depth-Anything 估计相对深度
  -> RGBD 反投影 xyzrgb point_cloud
  -> DP3 训练 / DP3 仿真验证
```

当前已经准备好的 clean 数据集：

```text
/home/sxy/code/260517_dp3/sdp_hlq/data/datasets/hdf5/pick_banana_place_plate_singleview_motionplan26_rgb_depthanything_pc_abs_clean.hdf5
```

当前训练：

```text
screen: 2562980.banana_dp3_depthanything_train
run: 260528_dp3_pick_banana_rgb_depthanything_vits_b4
log: /home/sxy/code/260517_dp3/sdp_hlq/logs/260528_dp3_pick_banana_rgb_depthanything_vits_b4_train.txt
output: /home/sxy/code/260517_dp3/sdp_hlq/data/outputs/dp3/pick_banana_place_plate_singleview_motionplan26_rgb_depthanything/260528_dp3_pick_banana_rgb_depthanything_vits_b4
latest ckpt: /home/sxy/code/260517_dp3/sdp_hlq/data/outputs/dp3/pick_banana_place_plate_singleview_motionplan26_rgb_depthanything/260528_dp3_pick_banana_rgb_depthanything_vits_b4/checkpoints/latest.ckpt
```

## 0. 2026-05-28 sim-depth pivot

当前这批 `20260527_165208/20260527_165216` raw episode 只能支持 RGB-only Depth-Anything 路线，不能直接重建仿真点云：

```text
episode_0001.npy keys:
obs_policy_actions, obs_policy_joint_pos, obs_policy_joint_vel,
obs_policy_arm_joint_pos, obs_policy_arm_joint_vel,
obs_policy_arm_ee_pose, obs_policy_gripper_pos, obs_policy_arm_ee_vel,
obs_policy_external_cam_side, obs_policy_wrist_cam,
obs_policy_lastactions, obs_policy_joint_actions

missing:
obs_policy_external_cam_side_distance_to_image_plane
depth_images/
point_cloud
```

已新增 sim-depth 数据链路：

```text
采集配置:
/home/sxy/code/isaacsim_env/AffordSim/source/isaaclab_tasks/isaaclab_tasks/real2sim/tasks/config/task/composite/pick_banana_place_plate_motionplan25_singleview_simdepth_260528.yaml

转换脚本:
/home/sxy/code/isaacsim_env/AffordSim/scripts/tools/convert_motionplan_sim_depth_to_dp3_hdf5.py
```

采集时必须同时满足：

```yaml
runtime:
  use_background_replace: false
  save_original_images: true
  save_depth_images: true
```

原因：只有 `use_background_replace: false` 时相机 data_types 才包含 `distance_to_image_plane`；只有 `save_depth_images: true` 时 `pick_r2s_env_cfg.py` 会把 distance observation 注册进 policy obs 并落盘到 `depth_images/*distance_to_image_plane/env0_ep*/step_*.png`。

推荐顺序：

```bash
# 1) 本地重采 25 条带 sim depth 的 motionplan demo
cd /home/sxy/code/isaacsim_env/AffordSim
source /home/sxy/miniconda3/etc/profile.d/conda.sh
conda activate env_isaaclab

python scripts/run_data_collect_improve.py \
  --skip_editor \
  --task_name pick_banana_place_plate \
  --task_config_path /home/sxy/code/isaacsim_env/AffordSim/source/isaaclab_tasks/isaaclab_tasks/real2sim/tasks/config/task/composite/pick_banana_place_plate_motionplan25_singleview_simdepth_260528.yaml \
  --episodes 25 \
  --num_envs 1 \
  --data_dir /home/sxy/code/isaacsim_env/AffordSim/data/r2s2r_data/pick_banana_place_plate_singleview_motionplan25_simdepth_260528/raw \
  --headless \
  --device cuda:0 \
  --enable_cameras \
  2>&1 | tee /home/sxy/code/isaacsim_env/AffordSim/data/r2s2r_data/pick_banana_place_plate_singleview_motionplan25_simdepth_260528_collect.log

# 2) 转成 DP3 HDF5，直接用仿真 depth 反投影 xyzrgb point_cloud
RAW_DIR=$(find /home/sxy/code/isaacsim_env/AffordSim/data/r2s2r_data/pick_banana_place_plate_singleview_motionplan25_simdepth_260528/raw -type f -name 'episode_*.npy' -printf '%h\n' | sort | tail -1)
OUT_HDF5=/home/sxy/code/260517_dp3/sdp_hlq/data/datasets/hdf5/pick_banana_place_plate_singleview_motionplan25_simdepth_pc_abs_clean.hdf5

/home/sxy/miniconda3/envs/env_isaaclab/bin/python \
  scripts/tools/convert_motionplan_sim_depth_to_dp3_hdf5.py \
  --input "$RAW_DIR" \
  --output "$OUT_HDF5" \
  --camera external_cam_side \
  --rgb-key obs_policy_external_cam_side \
  --depth-key obs_policy_external_cam_side_distance_to_image_plane \
  --eef-pose-key obs_policy_arm_ee_pose \
  --gripper-key obs_policy_gripper_pos \
  --num-points 1024 \
  --image-size 84 \
  --stride 2 \
  --min-depth 0.05 \
  --max-depth 2.0 \
  --z-min 0.02 \
  --pointcloud-frame world \
  --min-steps 10 \
  --skip-errors
```

训练选择：

```text
本地 RTX 4080 当前在跑 banana_dp3_depthanything_train，约 epoch 644/1000，占用约 8.3GB 显存。
gpu132 当前 GPU6 空闲 34MiB/24564MiB，建议新 sim-depth DP3 正式训练放到 gpu132:6。
```

## 1. 通用变量

换任务时优先只改这些变量：

```bash
TASK_ID=pick_banana_place_plate_singleview_motionplan50_260527
TASK_NAME=pick_banana_place_plate
TASK_CONFIG=/home/sxy/code/isaacsim_env/AffordSim/source/isaaclab_tasks/isaaclab_tasks/real2sim/tasks/config/task/composite/pick_banana_place_plate_motionplan50_singleview_260527.yaml
CAMERA=external_cam_side
N_DEMO=26
RUN_NAME=260528_dp3_pick_banana_rgb_depthanything_vits_b4

AFFORDSIM=/home/sxy/code/isaacsim_env/AffordSim
DP3_ROOT=/home/sxy/code/260517_dp3/sdp_hlq
DEPTH_ANYTHING_ROOT=/home/sxy/code/depthanything/Depth-Anything

RAW_BASE=$AFFORDSIM/data/r2s2r_data/$TASK_ID/raw
DATASET=$DP3_ROOT/data/datasets/hdf5/${TASK_ID}_rgb_depthanything_pc_abs_clean.hdf5
```

推荐相机和点云参数先保持一致：

```text
camera: external_cam_side
depth encoder: vits
point_cloud: [1024, 6], xyzrgb, rgb in [0, 1]
pointcloud_frame: world
min_depth: 0.25
max_depth: 1.4
stride: 2
```

如果以后换相机，必须同步改 conversion / eval 的 camera intrinsic、pose 或 camera cfg，否则点云坐标会错。

## 2. MotionPlan 采集

配置文件里建议保持：

```yaml
runtime:
  save_npy: true
  save_camera_images: true
  save_original_images: true
  save_composite_images: false
  save_depth_images: false
  use_background_replace: false
  enable_motion_planning: true
  sim_camera_resolution: [640, 480]
```

采集命令：

```bash
cd /home/sxy/code/isaacsim_env/AffordSim
source /home/sxy/miniconda3/etc/profile.d/conda.sh
conda activate env_isaaclab

python scripts/run_data_collect_improve.py \
  --skip_editor \
  --task_name "$TASK_NAME" \
  --task_config_path "$TASK_CONFIG" \
  --episodes 50 \
  --num_envs 1 \
  --data_dir "$RAW_BASE" \
  --headless \
  --device cuda:0 \
  --enable_cameras \
  2>&1 | tee "$AFFORDSIM/data/r2s2r_data/${TASK_ID}_collect.log"
```

找最新 raw episode 目录：

```bash
RAW_DIR=$(find "$RAW_BASE" -type f -name 'episode_*.npy' -printf '%h\n' | sort | tail -1)
echo "$RAW_DIR"
find "$RAW_DIR" -maxdepth 1 -name 'episode_*.npy' | wc -l
```

检查必须有纯仿真 RGB：

```bash
find "$RAW_DIR/original_images/$CAMERA" -maxdepth 2 -name 'step_*.png' | head
find "$RAW_DIR/original_images/$CAMERA" -mindepth 1 -maxdepth 1 -type d | wc -l
```

抽查每条 episode 的 RGB 帧数，帧数特别少的后面会被过滤：

```bash
for d in "$RAW_DIR/original_images/$CAMERA"/env0_ep*; do
  printf "%s " "$(basename "$d")"
  find "$d" -maxdepth 1 -name 'step_*.png' | wc -l
done | sort -V | head -20
```

## 3. Depth-Anything 环境

Depth-Anything 指南在：

```text
/home/sxy/code/depthanything/Depth-Anything/RGB_TO_RGBD_GUIDE.md
```

单张或文件夹独立估深可以这样跑，主要用于人工看深度图质量：

```bash
cd /home/sxy/code/depthanything/Depth-Anything
source /home/sxy/miniconda3/etc/profile.d/conda.sh
conda activate depth-anything

python run.py \
  --encoder vits \
  --img-path /path/to/images_or_one_image \
  --outdir /tmp/depthanything_preview \
  --pred-only \
  --grayscale
```

DP3 数据转换不需要先手动生成深度图，下面的转换脚本会在 Python 里直接加载 Depth-Anything，并且只用 RGB 生成点云。

## 4. RGB -> Depth-Anything -> DP3 HDF5

使用脚本：

```text
/home/sxy/code/isaacsim_env/AffordSim/scripts/tools/convert_motionplan_rgb_depthanything_to_dp3_hdf5.py
```

当前任务的转换命令：

```bash
cd /home/sxy/code/isaacsim_env/AffordSim
source /home/sxy/miniconda3/etc/profile.d/conda.sh
conda activate env_isaaclab

RAW_DIR=/home/sxy/code/isaacsim_env/AffordSim/data/r2s2r_data/pick_banana_place_plate_singleview_motionplan50_260527/raw/20260527_165208/20260527_165216
OUT_HDF5=/home/sxy/code/260517_dp3/sdp_hlq/data/datasets/hdf5/pick_banana_place_plate_singleview_motionplan26_rgb_depthanything_pc_abs_clean.hdf5
DEPTH_SAVE=/home/sxy/code/isaacsim_env/AffordSim/data/r2s2r_data/pick_banana_place_plate_singleview_motionplan50_260527/depthanything_vits_depth_32

/home/sxy/miniconda3/envs/env_isaaclab/bin/python \
  scripts/tools/convert_motionplan_rgb_depthanything_to_dp3_hdf5.py \
  --input "$RAW_DIR" \
  --output "$OUT_HDF5" \
  --camera external_cam_side \
  --rgb-key obs_policy_external_cam_side \
  --eef-pose-key obs_policy_arm_ee_pose \
  --gripper-key obs_policy_gripper_pos \
  --depth-anything-root /home/sxy/code/depthanything/Depth-Anything \
  --encoder vits \
  --device cuda \
  --batch-size 8 \
  --num-points 1024 \
  --image-size 84 \
  --stride 2 \
  --min-depth 0.25 \
  --max-depth 1.4 \
  --z-min 0.02 \
  --pointcloud-frame world \
  --min-steps 10 \
  --skip-errors \
  --save-depth-dir "$DEPTH_SAVE"
```

换任务模板：

```bash
RAW_DIR=$(find "$RAW_BASE" -type f -name 'episode_*.npy' -printf '%h\n' | sort | tail -1)
OUT_HDF5="$DP3_ROOT/data/datasets/hdf5/${TASK_ID}_rgb_depthanything_pc_abs_clean.hdf5"
DEPTH_SAVE="$AFFORDSIM/data/r2s2r_data/${TASK_ID}/depthanything_vits_depth"

cd "$AFFORDSIM"
/home/sxy/miniconda3/envs/env_isaaclab/bin/python \
  scripts/tools/convert_motionplan_rgb_depthanything_to_dp3_hdf5.py \
  --input "$RAW_DIR" \
  --output "$OUT_HDF5" \
  --camera "$CAMERA" \
  --rgb-key obs_policy_external_cam_side \
  --eef-pose-key obs_policy_arm_ee_pose \
  --gripper-key obs_policy_gripper_pos \
  --depth-anything-root "$DEPTH_ANYTHING_ROOT" \
  --encoder vits \
  --device cuda \
  --batch-size 8 \
  --num-points 1024 \
  --image-size 84 \
  --stride 2 \
  --min-depth 0.25 \
  --max-depth 1.4 \
  --z-min 0.02 \
  --pointcloud-frame world \
  --min-steps 10 \
  --skip-errors \
  --save-depth-dir "$DEPTH_SAVE"
```

如果已经生成了一个包含短坏轨迹的 HDF5，也可以再过滤：

```bash
/home/sxy/miniconda3/envs/env_isaaclab/bin/python \
  scripts/tools/filter_dp3_hdf5_by_length.py \
  --input /path/to/full.hdf5 \
  --output /path/to/clean.hdf5 \
  --min-steps 10
```

## 5. 检查 HDF5

```bash
/home/sxy/miniconda3/envs/maniskill260128/bin/python - <<'PY'
import h5py
p = "/home/sxy/code/260517_dp3/sdp_hlq/data/datasets/hdf5/pick_banana_place_plate_singleview_motionplan26_rgb_depthanything_pc_abs_clean.hdf5"
with h5py.File(p, "r") as f:
    demos = sorted(f["data"].keys(), key=lambda x: int(x.split("_")[1]))
    print("demos:", len(demos))
    print("total:", f["data"].attrs.get("total"))
    d = f["data"][demos[0]]
    print("actions:", d["actions"].shape, d["actions"].dtype)
    for k, v in d["obs"].items():
        print("obs/" + k, v.shape, v.dtype)
    print("uses_sim_depth:", f.attrs.get("uses_sim_depth"))
    print("uses_background_replacement:", f.attrs.get("uses_background_replacement"))
    print("depth_source:", f.attrs.get("depth_source"))
PY
```

当前 clean 数据集检查结果：

```text
demos: 26
samples: 2946
actions: [T, 7]
obs/point_cloud: [T, 1024, 6]
obs/robot0_eye_in_hand_image: [T, 84, 84, 3]
obs/robot0_eef_pos: [T, 3]
obs/robot0_eef_quat: [T, 4]
obs/robot0_gripper_qpos: [T, 2]
uses_sim_depth: false
uses_background_replacement: false
depth_source: Depth-Anything
```

## 6. DP3 Smoke Test

```bash
cd /home/sxy/code/260517_dp3/sdp_hlq
source /home/sxy/miniconda3/etc/profile.d/conda.sh
conda activate maniskill260128

TASK_NAME=pick_banana_place_plate_singleview_motionplan26_rgb_depthanything \
DATASET_PATH=/home/sxy/code/260517_dp3/sdp_hlq/data/datasets/hdf5/pick_banana_place_plate_singleview_motionplan26_rgb_depthanything_pc_abs_clean.hdf5 \
N_DEMO=26 \
GPU=0 \
BATCH_SIZE=4 \
NUM_WORKERS=1 \
NUM_EPOCHS=1 \
MAX_TRAIN_STEPS=1 \
CHECKPOINT_EVERY=1 \
RUN_NAME=smoke_dp3_pick_banana_rgb_depthanything \
bash scripts/train_hug_cup_realbot_dp3_local.sh
```

已通过的 smoke 输出：

```text
/tmp/dp3_smoke_pick_banana_depthanything_workers1/checkpoints/latest.ckpt
```

## 7. 正式训练 DP3

推荐 16GB 显存先用 `BATCH_SIZE=4`，更稳；显存足够再试 `BATCH_SIZE=8`。

```bash
cd /home/sxy/code/260517_dp3/sdp_hlq
source /home/sxy/miniconda3/etc/profile.d/conda.sh
conda activate maniskill260128

TASK_NAME=pick_banana_place_plate_singleview_motionplan26_rgb_depthanything \
DATASET_PATH=/home/sxy/code/260517_dp3/sdp_hlq/data/datasets/hdf5/pick_banana_place_plate_singleview_motionplan26_rgb_depthanything_pc_abs_clean.hdf5 \
N_DEMO=26 \
GPU=0 \
BATCH_SIZE=4 \
NUM_WORKERS=4 \
NUM_EPOCHS=1000 \
CHECKPOINT_EVERY=50 \
RUN_NAME=260528_dp3_pick_banana_rgb_depthanything_vits_b4 \
bash scripts/train_hug_cup_realbot_dp3_local.sh
```

后台 screen 版本：

```bash
screen -dmS banana_dp3_depthanything_train bash -lc '
cd /home/sxy/code/260517_dp3/sdp_hlq
source /home/sxy/miniconda3/etc/profile.d/conda.sh
conda activate maniskill260128
TASK_NAME=pick_banana_place_plate_singleview_motionplan26_rgb_depthanything \
DATASET_PATH=/home/sxy/code/260517_dp3/sdp_hlq/data/datasets/hdf5/pick_banana_place_plate_singleview_motionplan26_rgb_depthanything_pc_abs_clean.hdf5 \
N_DEMO=26 \
GPU=0 \
BATCH_SIZE=4 \
NUM_WORKERS=4 \
NUM_EPOCHS=1000 \
CHECKPOINT_EVERY=50 \
RUN_NAME=260528_dp3_pick_banana_rgb_depthanything_vits_b4 \
bash scripts/train_hug_cup_realbot_dp3_local.sh \
2>&1 | tee /home/sxy/code/260517_dp3/sdp_hlq/logs/260528_dp3_pick_banana_rgb_depthanything_vits_b4_train.txt
'
```

查看训练：

```bash
screen -ls
screen -r banana_dp3_depthanything_train
tail -f /home/sxy/code/260517_dp3/sdp_hlq/logs/260528_dp3_pick_banana_rgb_depthanything_vits_b4_train.txt
find /home/sxy/code/260517_dp3/sdp_hlq/data/outputs/dp3/pick_banana_place_plate_singleview_motionplan26_rgb_depthanything/260528_dp3_pick_banana_rgb_depthanything_vits_b4/checkpoints -maxdepth 1 -type f | sort
```

## 8. 仿真验证

验证也走 RGB-only 链路：

```text
sim RGB external_cam_side
  -> Depth-Anything
  -> xyzrgb point_cloud
  -> DP3 policy
  -> absolute EE action
  -> IKRel delta action
  -> AffordSim step
```

使用脚本：

```text
/home/sxy/code/isaacsim_env/AffordSim/scripts/imitation_learning/dp3/rgb_depth_pointcloud.py
/home/sxy/code/isaacsim_env/AffordSim/scripts/imitation_learning/dp3/eval_depthanything_sim.py
/home/sxy/code/isaacsim_env/AffordSim/scripts/imitation_learning/dp3/run_eval_pick_banana_depthanything.sh
```

后台验证仍然用 `--headless --enable_cameras`，wrapper 默认会加：

```text
--experience isaaclab.python.rendering.kit
```

原因：当前机器上默认 `isaaclab.python.headless.rendering.kit` 会在 RGB annotator 初始化时报 `Unable to write from unknown dtype, kind=f, size=0`。强制使用 `rendering.kit` 但保持 headless 可以后台跑，不弹 Isaac 界面，并且正常保存 JSON/MP4。

当前任务验证命令：

```bash
cd /home/sxy/code/isaacsim_env/AffordSim
source /home/sxy/miniconda3/etc/profile.d/conda.sh
conda activate env_isaaclab

CHECKPOINT=/home/sxy/code/260517_dp3/sdp_hlq/data/outputs/dp3/pick_banana_place_plate_singleview_motionplan26_rgb_depthanything/260528_dp3_pick_banana_rgb_depthanything_vits_b4/checkpoints/latest.ckpt \
TASK_CONFIG=/home/sxy/code/isaacsim_env/AffordSim/source/isaaclab_tasks/isaaclab_tasks/real2sim/tasks/config/task/composite/pick_banana_place_plate_motionplan50_singleview_260527.yaml \
NUM_ROLLOUTS=3 \
HORIZON=350 \
OUTPUT_JSON=/tmp/dp3_pick_banana_depthanything_sim_eval.json \
bash scripts/imitation_learning/dp3/run_eval_pick_banana_depthanything.sh
```

本次 smoke eval：

```text
checkpoint: /home/sxy/code/260517_dp3/sdp_hlq/data/outputs/dp3/pick_banana_place_plate_singleview_motionplan26_rgb_depthanything/260528_dp3_pick_banana_rgb_depthanything_vits_b4/checkpoints/latest.ckpt
result: /home/sxy/code/isaacsim_env/video/260528/pick_banana_place_plate_dp3_depthanything_latest_eval1.json
video: /home/sxy/code/isaacsim_env/video/260528/pick_banana_place_plate_dp3_depthanything_latest_eval1_640x480.mp4
video_shape: 640x480, 351 frames, 10 FPS
num_rollouts: 1
overall_success_rate: 0/1 = 0%
stage_success_rate: 00_pick 0/1, 01_place 0/1
失败现象: pick_lift_delta_z=0.0194m，小于当前 0.05m pick 阈值；place 未成功
```

查看结果：

```bash
cat /tmp/dp3_pick_banana_depthanything_sim_eval.json
find "$(dirname /tmp/dp3_pick_banana_depthanything_sim_eval.json)/videos" -maxdepth 1 -name '*.mp4' 2>/dev/null | sort
```

换任务验证模板：

```bash
CHECKPOINT=/path/to/latest.ckpt \
TASK_CONFIG=/path/to/task.yaml \
NUM_ROLLOUTS=10 \
HORIZON=350 \
OUTPUT_JSON=/tmp/${TASK_ID}_dp3_depthanything_eval.json \
bash scripts/imitation_learning/dp3/run_eval_pick_banana_depthanything.sh \
  --task FrankaFr3-DefaultGripper-IKRel-VisPick-v0
```

注意：DP3 训练 action 是绝对 EE pose，验证脚本会转成 IKRel delta action，所以默认用 `FrankaFr3-DefaultGripper-IKRel-VisPick-v0`。如果换机器人或 env action space，需要检查这个转换是否仍然匹配。

## 9. 后续换任务 Checklist

每个新任务按这个顺序跑：

1. 改 task yaml，保证 `save_original_images: true`、`save_depth_images: false`、`use_background_replace: false`。
2. 跑 motionplan 采集，确认 `original_images/external_cam_side/env0_ep*/step_*.png` 存在。
3. 跑 `convert_motionplan_rgb_depthanything_to_dp3_hdf5.py`，生成 clean DP3 HDF5。
4. 检查 HDF5 attrs 必须是 `uses_sim_depth: false` 和 `uses_background_replacement: false`。
5. 跑 DP3 smoke test。
6. 正式训练 DP3。
7. 用 `eval_depthanything_sim.py` 做仿真验证，仍然不读仿真 depth / point cloud。

最容易出错的地方：

```text
RGB 帧数太少: 该 demo 会被 --min-steps 过滤
相机换了但 intrinsic/pose 没换: 点云坐标会错
验证用了 Joint env: DP3 absolute EE action 语义对不上
Depth-Anything 深度范围不合适: 调 --min-depth / --max-depth
batch size 过大: DP3 训练 OOM，先用 BATCH_SIZE=4
```
