# 仿真实验记录

日期：2026-05-28

这个文件用于记录不同方法在各个仿真任务上的成功率，以及每个任务对应的配置、数据、模型和备注。成功率统一建议写成 `成功数/总数 = 百分比`，例如 `7/10 = 70%`。

## 成功率矩阵

| 方法 / 任务 | pick_banana_place_plate |
|---|---:|
| FDTR | TBD |
| DemoGen | TBD |
| 1demo | TBD |
| 10demo | TBD |
| 25demo | `0/1 = 0%` smoke eval |

当前 DP3 RGB-only 训练先记到 `25demo` 这一行。

## 任务索引

| 任务名 | 简述 | YAML 配置 | 数据位置 | 模型 / checkpoint | 验证命令 / 结果 | 备注 |
|---|---|---|---|---|---|---|
| pick_banana_place_plate | 抓取黄色香蕉并放到白色盘子里 | `/home/sxy/code/isaacsim_env/AffordSim/source/isaaclab_tasks/isaaclab_tasks/real2sim/tasks/config/task/composite/pick_banana_place_plate_motionplan50_singleview_260527.yaml` | `/home/sxy/code/260517_dp3/sdp_hlq/data/datasets/hdf5/pick_banana_place_plate_singleview_motionplan26_rgb_depthanything_pc_abs_clean.hdf5` | `/home/sxy/code/260517_dp3/sdp_hlq/data/outputs/dp3/pick_banana_place_plate_singleview_motionplan26_rgb_depthanything/260528_dp3_pick_banana_rgb_depthanything_vits_b4/checkpoints/latest.ckpt` | `/home/sxy/code/isaacsim_env/video/260528/pick_banana_place_plate_dp3_depthanything_latest_eval1.json`: `0/1 = 0%` | DP3 RGB-only，按 25demo 记录；输入链路为 RGB -> Depth-Anything -> point cloud；MP4: `/home/sxy/code/isaacsim_env/video/260528/pick_banana_place_plate_dp3_depthanything_latest_eval1_640x480.mp4`；单次 smoke eval，后续需多 rollout |

## 方法说明

| 方法 | 说明 | 默认记录内容 |
|---|---|---|
| FDTR | FDTR 方法的仿真成功率 | 成功率、失败类型、使用的 checkpoint/config |
| DemoGen | DemoGen 或 motionplan/demonstration generation baseline | 成功率、demo 数、任务配置 |
| 1demo | 只用 1 条 demo 训练或适配的方法 | 成功率、数据集路径、checkpoint |
| 10demo | 使用 10 条 demo 的方法 | 成功率、数据集路径、checkpoint |
| 25demo | 使用约 25 条 demo 的方法 | 成功率、数据集路径、checkpoint |

## 当前 DP3 25demo 配置

2026-05-28 sim-depth 复核：

```text
当前 RGB-only raw 不含 sim depth / point_cloud，不能直接转 sim point cloud：
/home/sxy/code/isaacsim_env/AffordSim/data/r2s2r_data/pick_banana_place_plate_singleview_motionplan50_260527/raw/20260527_165208/20260527_165216

已准备新的 sim-depth 采集配置和转换脚本：
/home/sxy/code/isaacsim_env/AffordSim/source/isaaclab_tasks/isaaclab_tasks/real2sim/tasks/config/task/composite/pick_banana_place_plate_motionplan25_singleview_simdepth_260528.yaml
/home/sxy/code/isaacsim_env/AffordSim/scripts/tools/convert_motionplan_sim_depth_to_dp3_hdf5.py

建议流程：
本地重采 25 条带 depth_images 的 motionplan demo -> 转 HDF5 ->
gpu132 GPU6 训练 DP3 run: 260528_dp3_pick_banana_simdepth_b4
```

训练数据：

```text
/home/sxy/code/260517_dp3/sdp_hlq/data/datasets/hdf5/pick_banana_place_plate_singleview_motionplan26_rgb_depthanything_pc_abs_clean.hdf5
```

训练输出：

```text
/home/sxy/code/260517_dp3/sdp_hlq/data/outputs/dp3/pick_banana_place_plate_singleview_motionplan26_rgb_depthanything/260528_dp3_pick_banana_rgb_depthanything_vits_b4
```

验证链路：

```text
sim RGB external_cam_side -> Depth-Anything -> xyzrgb point_cloud -> DP3 -> IKRel env
```

验证命令：

```bash
cd /home/sxy/code/isaacsim_env/AffordSim
source /home/sxy/miniconda3/etc/profile.d/conda.sh
conda activate env_isaaclab

CHECKPOINT=/home/sxy/code/260517_dp3/sdp_hlq/data/outputs/dp3/pick_banana_place_plate_singleview_motionplan26_rgb_depthanything/260528_dp3_pick_banana_rgb_depthanything_vits_b4/checkpoints/latest.ckpt \
TASK_CONFIG=/home/sxy/code/isaacsim_env/AffordSim/source/isaaclab_tasks/isaaclab_tasks/real2sim/tasks/config/task/composite/pick_banana_place_plate_motionplan50_singleview_260527.yaml \
NUM_ROLLOUTS=10 \
HORIZON=350 \
OUTPUT_JSON=/tmp/dp3_pick_banana_depthanything_sim_eval.json \
bash scripts/imitation_learning/dp3/run_eval_pick_banana_depthanything.sh
```

2026-05-28 smoke eval:

```text
checkpoint: /home/sxy/code/260517_dp3/sdp_hlq/data/outputs/dp3/pick_banana_place_plate_singleview_motionplan26_rgb_depthanything/260528_dp3_pick_banana_rgb_depthanything_vits_b4/checkpoints/latest.ckpt
result: /home/sxy/code/isaacsim_env/video/260528/pick_banana_place_plate_dp3_depthanything_latest_eval1.json
video: /home/sxy/code/isaacsim_env/video/260528/pick_banana_place_plate_dp3_depthanything_latest_eval1_640x480.mp4
video_shape: 640x480, 351 frames, 10 FPS
overall_success_rate: 0/1 = 0%
stage_success_rate: 00_pick 0/1, 01_place 0/1
notes: pick_lift_delta_z=0.0194m，低于 0.05m 成功阈值；place 未成功
```

## 新任务添加模板

在“成功率矩阵”里新增一列任务名，例如：

```text
pick_banana_place_plate | pick_carrot_place_plate | hang_mugcup
```

在“任务索引”里新增一行：

| 任务名 | 简述 | YAML 配置 | 数据位置 | 模型 / checkpoint | 验证命令 / 结果 | 备注 |
|---|---|---|---|---|---|---|
| new_task_name | TBD | `/path/to/task.yaml` | `/path/to/dataset.hdf5` | `/path/to/latest.ckpt` | `/path/to/eval.json` | TBD |

记录实验时建议同步填：

```text
任务随机化范围:
demo 数量:
训练方法:
训练 run name:
checkpoint:
rollout 数量:
成功率:
主要失败现象:
是否使用 RGB-only Depth-Anything 点云:
```
