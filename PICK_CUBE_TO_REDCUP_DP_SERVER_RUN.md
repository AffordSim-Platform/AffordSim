# Pick Cube To Redcup DP 服务器运行说明

> 更新日期：2026-04-13
> 适用目录：`/home/sxy/code/isaacsim_env/Real2Sim2Real-Pipeline_affordsim_clean`

## 1. 当前目标

这份文档只负责回答一件事：

如何按 `spain_task` 那套 `gpu132` 交接风格，在服务器上基于已经收好的 `pick_cube_to_redcup` 50 条轨迹运行 Diffusion Policy 训练，并知道日志和结果会落到哪里。

## 2. 服务器约定

默认远端机器：

- `gpu132`

推荐启动方式：

- `ssh gpu132`
- 用 `screen` 跑长训练
- 训练结束或中断后，把 `server / screen / 日志 / 输出目录 / checkpoint / 下一步计划` 回填到本项目 md

推荐远端项目目录：

```bash
/data1/user/suixinyang/code/isaacsim_env/Real2Sim2Real-Pipeline_affordsim_clean
```

如果 `gpu132` 上实际同步目录不是这个路径，先改这份文档，再启动正式实验。

## 3. 数据位置

正式训练轨迹目录：

```bash
/home/sxy/code/isaacsim_env/data/r2s2r_data/pick_cube_to_redcup_franka_fr3/20260410_152550/20260410_152605
```

当前磁盘状态：

- 任务：`pick_cube_to_redcup`
- 机器人：`franka_fr3`
- 原始 `episode_*.npy`：`50`

当前真实可用训练口径：

- 可用轨迹：`49`
- 坏轨迹：`episode_0047.npy`
- 错误：`UnpicklingError: pickle data was truncated`

## 4. 环境准备

本机转换继续使用 `env_isaaclab`。

`gpu132` 远端正式训练目前不走完整 IsaacLab 环境，而是走最小 DP 环境：

```bash
/data1/user/suixinyang/miniconda3/envs/dp_train/bin/python
```

远端项目目录：

```bash
cd /data1/user/suixinyang/code/isaacsim_env/Real2Sim2Real-Pipeline_affordsim_clean
```

如需重建该环境：

```bash
source /data1/user/suixinyang/miniconda3/etc/profile.d/conda.sh
conda create -y -n dp_train python=3.11 pip
conda run -n dp_train pip install torch==2.7.0 torchvision==0.22.0 --index-url https://download.pytorch.org/whl/cu128
conda run -n dp_train pip install h5py tensorboard numpy==1.26.0
```

## 5. 单实验运行

### 4.1 前视角单视角实验

脚本：

```bash
./scripts/imitation_learning/diffusion_policy/train_pick_cube_to_redcup_50_frontonly.sh
```

这条脚本会顺序做三件事：

1. `episode_XXXX.npy -> raw hdf5`
2. `raw hdf5 -> robomimic hdf5`
3. `front-only` DP 训练

### 4.2 单实验对应的文件

front-only 中间文件：

```bash
/home/sxy/code/isaacsim_env/data/dp_debug/pick_cube_to_redcup_50_frontonly_raw.hdf5
/home/sxy/code/isaacsim_env/data/dp_debug/pick_cube_to_redcup_50_frontonly_robomimic.hdf5
/home/sxy/code/isaacsim_env/data/dp_debug/pick_cube_to_redcup_50_frontonly_dataset_stat.json
```

front-only 输出目录：

```bash
/home/sxy/code/isaacsim_env/data/dp_debug/logs_pick_cube_to_redcup_50_frontonly
```

## 6. 批量实验运行

### 5.1 实验矩阵脚本

```bash
./scripts/imitation_learning/diffusion_policy/run_pick_cube_to_redcup_dp_matrix.sh
```

### 5.2 当前实验矩阵

```text
1. front_lr1e4_bs32_ph16_oh2
2. front_lr3e4_bs64_ph16_oh2
3. front_lr1e4_bs32_ph16_oh4
4. front_wrist_lr1e4_bs32_ph16_oh2
5. front_wrist_lr3e4_bs64_ph16_oh2
6. front_wrist_lr1e4_bs32_ph16_oh4
```

### 5.3 矩阵共用的转换文件

```bash
/home/sxy/code/isaacsim_env/data/dp_debug/pick_cube_to_redcup_49usable_raw.hdf5
/home/sxy/code/isaacsim_env/data/dp_debug/pick_cube_to_redcup_49usable_robomimic.hdf5
/home/sxy/code/isaacsim_env/data/dp_debug/pick_cube_to_redcup_49usable_dataset_stat.json
```

### 5.4 矩阵结果根目录

```bash
/home/sxy/code/isaacsim_env/data/dp_debug/dp_matrix_pick_cube_to_redcup_49usable
```

每个实验完成后，结果会落到类似：

```bash
/home/sxy/code/isaacsim_env/data/dp_debug/dp_matrix_pick_cube_to_redcup_49usable/<experiment_name>/diffusion_policy_<timestamp>/
```

## 7. DP 实际拿进去的数据

训练只使用下面这些：

- 图像：
  - `front`
  - 或 `front + wrist`
- 低维状态：
  - `qpos (7)`
  - `qvel (7)`
  - 拼起来是 `14` 维
- 动作标签：
  - `actions (7)`

训练阶段 sample 形状：

```text
obs_images: (T_o, V, 3, H, W)
obs_lowdim: (T_o, 14)
actions:    (T_a, 7)
```

其中：

- `V=1` 表示 `front`
- `V=2` 表示 `front,wrist`

## 8. 远端记录模板

正式在 `gpu132` 启动训练时，建议至少记录下面这些：

- server:
  - `gpu132`
- screen:
  - `<screen_name>`
- 日志:
  - `<remote_log_path>`
- 输出目录:
  - `<remote_output_dir>`
- 数据集:
  - `/home/sxy/code/isaacsim_env/data/r2s2r_data/pick_cube_to_redcup_franka_fr3/20260410_152550/20260410_152605`
- 实验配置:
  - `image_views=front` 或 `front,wrist`
  - `lr`
  - `batch_size`
  - `pred_horizon`
  - `obs_horizon`

## 9. 常用检查命令

### 9.1 看训练相关进程

```bash
ps -ef | rg 'diffusion_policy/train.py|run_pick_cube_to_redcup_dp_matrix.sh|train_pick_cube_to_redcup_50_frontonly.sh'
```

### 9.2 看转换产物

```bash
ls -lah /home/sxy/code/isaacsim_env/data/dp_debug
```

### 9.3 看实验目录

```bash
find /home/sxy/code/isaacsim_env/data/dp_debug/dp_matrix_pick_cube_to_redcup_49usable -maxdepth 2 -type d | sort
```

### 9.4 看 TensorBoard 事件和模型

```bash
find /home/sxy/code/isaacsim_env/data/dp_debug/dp_matrix_pick_cube_to_redcup_49usable -type f | rg 'config.json|events.out.tfevents|model_best.pth'
```

### 9.5 当前正式远端运行

截至 `2026-04-13 18:06 CST`：

- server:
  - `gpu132`
- 当前新矩阵输出目录:
  - `/data1/user/suixinyang/code/isaacsim_env/data/dp_debug/dp_matrix_pick_cube_to_redcup_49usable_v2_short_horizon`
- 当前 active screens:
  - `pickred_dp_v2_g1_20260413_1804`
  - `pickred_dp_v2_g7_20260413_1804`
  - `pickred_dp_v2_g0_20260413_1807`
- 当前对应日志:
  - `/data1/user/suixinyang/code/isaacsim_env/data/dp_debug/logs/pickred_dp_v2_g1_20260413_1804.log`
  - `/data1/user/suixinyang/code/isaacsim_env/data/dp_debug/logs/pickred_dp_v2_g7_20260413_1804.log`
  - `/data1/user/suixinyang/code/isaacsim_env/data/dp_debug/logs/pickred_dp_v2_g0_20260413_1807.log`
- 当前矩阵配置:
  - `front_lr1e4_bs64_ph8_oh2_ah4`
  - `front_lr3e4_bs64_ph8_oh2_ah4`
  - `front_lr1e4_bs64_ph12_oh2_ah4`
  - `front_wrist_lr1e4_bs64_ph8_oh2_ah4`
  - `front_wrist_lr3e4_bs64_ph8_oh2_ah4`
  - `front_wrist_lr1e4_bs64_ph12_oh2_ah4`
- GPU 绑定:
  - `GPU1`: `front_lr1e4_bs64_ph8_oh2_ah4` -> `front_wrist_lr1e4_bs64_ph8_oh2_ah4`
  - `GPU7`: `front_lr1e4_bs64_ph12_oh2_ah4` -> `front_wrist_lr1e4_bs64_ph12_oh2_ah4`
  - `GPU0`: `front_lr3e4_bs64_ph8_oh2_ah4` -> `front_wrist_lr3e4_bs64_ph8_oh2_ah4`
- 调整理由:
  - 上一轮 `pred_horizon=16, action_horizon=8` 更容易在 rollout 中累计误差
  - 当前改为短 horizon + 更频繁 replanning，优先验证能否换来更高成功率
- 当前进度摘要:
  - `front_lr1e4_bs64_ph8_oh2_ah4`: best valid loss 已到 `0.052989`
  - `front_lr1e4_bs64_ph12_oh2_ah4`: best valid loss 已到 `0.041235`
  - `front_lr3e4_bs64_ph8_oh2_ah4`: best valid loss 已到 `0.058432`
- 备注:
  - 原先计划放在 `GPU3` 的那一路因为显存不足 OOM，已迁到 `GPU0`
- 下一步:
  - 继续观察前 50-100 epoch 趋势
  - 等短 horizon 新矩阵出现明确最优线后，再做下一轮正式评估

### 9.7 历史评估记录（2026-04-12）

截至 `2026-04-13 20:14 CST`：

- 本机重评估 screen:
  - `pickred_reeval_old6_fix_20260413_1812`
- 本机 checkpoint 根目录:
  - `/home/sxy/code/isaacsim_env/data/dp_debug/eval_pick_cube_to_redcup_49usable_all`
- 本机重评估结果根目录:
  - `/home/sxy/code/isaacsim_env/data/dp_debug/re_eval_pick_cube_to_redcup_49usable_all`
- 当前状态:
  - 已按修正后的成功判据完成旧 6 个模型重评估
  - 当前没有活跃本机重评估进程
- 重评估结果:
  - `front_lr1e4_bs32_ph16_oh2`: `0/5`, success rate `0.00%`
  - `front_lr3e4_bs64_ph16_oh2`: `0/5`, success rate `0.00%`
  - `front_lr1e4_bs32_ph16_oh4`: `0/5`, success rate `0.00%`
  - `front_wrist_lr1e4_bs32_ph16_oh2`: `0/5`, success rate `0.00%`
  - `front_wrist_lr3e4_bs64_ph16_oh2`: `0/5`, success rate `0.00%`
  - `front_wrist_lr1e4_bs32_ph16_oh4`: `0/5`, success rate `0.00%`
- 视频导出方式:
  - 统一使用 `video_utils.py` 导出 H.264 mp4
  - 双视角默认导出 `front+wrist` 组合视频
- 视频落盘情况:
  - 单视角目录各有 `5` 个 `front.mp4`
  - 双视角目录各有 `5` 个 `front.mp4`、`5` 个 `wrist.mp4`、`5` 个 `combined.mp4`


### 9.6 当前本机状态

截至 `2026-04-12` 当前这次检查时：

- 本机评估环境：
  - `conda activate env_isaaclab`
- 评估脚本：
  - `scripts/imitation_learning/diffusion_policy/play.py`
- 当前新增能力：
  - `--debug_dir`
  - `--print_action_every`
  - `--result_json`
  - 单视角 checkpoint 推理支持 `image_views=front`
- 推荐评估命令模板：

```bash
source /home/sxy/miniconda3/etc/profile.d/conda.sh
conda activate env_isaaclab
python -u scripts/imitation_learning/diffusion_policy/play.py \
  --checkpoint /home/sxy/code/isaacsim_env/data/dp_debug/eval_pick_cube_to_redcup_49usable/front_best_model.pth \
  --stat /home/sxy/code/isaacsim_env/data/dp_debug/pick_cube_to_redcup_49usable_dataset_stat.json \
  --task FrankaFr3-OurGripper-Joint-VisPick-v0 \
  --task_config source/isaaclab_tasks/isaaclab_tasks/real2sim/tasks/pick/config/sceneconfig/task/pick/pick_cube_to_redcup.yaml \
  --num_rollouts 5 \
  --horizon 300 \
  --save_img_interval 0 \
  --print_action_every 20 \
  --result_json /home/sxy/code/isaacsim_env/data/dp_debug/eval_logs/pickred_front_eval_5.json \
  --debug_dir /home/sxy/code/isaacsim_env/data/dp_debug/play_debug_pickred/front \
  --headless --enable_cameras
```

- 当前首轮评估结果：
  - `pickred_front_eval_1.json` -> `0/1`
  - `pickred_frontwrist_eval_1.json` -> `0/1`
- 当前 `5 rollout` 评估结果：
  - `pickred_front_eval_5.json` -> `0/5`
  - `pickred_frontwrist_eval_5.json` -> `0/5`
- 当前视频目录：
  - 单视角：
    - `/home/sxy/code/isaacsim_env/data/dp_debug/eval_videos/pickred_front_5`
  - 双视角：
    - `/home/sxy/code/isaacsim_env/data/dp_debug/eval_videos/pickred_frontwrist_5`

- 本机没有活跃的正式 DP 训练
- 正式矩阵已切到 `gpu132`

## 10. 时间预估

粗略预估：

- 数据转换：`10-25` 分钟
- 单个实验：`1.5-4` 小时
- 六组矩阵全部完成：`9-24` 小时

通常：

- `front-only` 比 `front+wrist` 更快
- `batch_size=64` 比 `32` 更快，但更吃显存

## 11. 结果总结文档

训练结果和结论统一回填到：

```bash
/home/sxy/code/isaacsim_env/Real2Sim2Real-Pipeline_affordsim_clean/PICK_CUBE_TO_REDCUP_DP_SUMMARY.md
```

## 12. 结论

如果后面换人接手，这个项目当前最重要的入口只有两个：

1. 跑单实验：
   `train_pick_cube_to_redcup_50_frontonly.sh`
2. 跑矩阵实验：
   `run_pick_cube_to_redcup_dp_matrix.sh`

如果是远端正式实验，再额外加一条：

3. 优先使用 `gpu132 + screen`，并把运行状态写回这份文档和 `PICK_CUBE_TO_REDCUP_DP_SUMMARY.md`

其他信息统一以这份文档、`PICK_CUBE_TO_REDCUP_DP_SUMMARY.md` 和项目根目录 `AGENTS.md` 为准，不需要再从聊天记录里回捞命令。

### 9.7 当前远端评估环境排查结论

截至 `2026-04-13`：

- 远端项目目录已同步到：
  - `/data1/user/suixinyang/code/isaacsim_env/Real2Sim2Real-Pipeline_affordsim_clean`
- 远端模型目录已同步到：
  - `/data1/user/suixinyang/code/isaacsim_env/models`
- 远端评估 yaml：
  - `/data1/user/suixinyang/code/isaacsim_env/data/dp_debug/pick_cube_to_redcup_gpu132_eval.yaml`
- 远端补装过的环境：
  - `/data1/user/suixinyang/miniconda3/envs/env_isaaclab`
  - `/data1/user/suixinyang/miniconda3/envs/env_isaaclab_native`

实际结论：

- `env_isaaclab` 虽然已经带有 `isaacsim 5.1.0` 包，但启动 `isaaclab.app.AppLauncher` 时会报：
  - `GLIBC_2.33 not found`
- `env_isaaclab_native` 已创建，但在 `gpu132` 上执行
  - `pip install "isaacsim[all,extscache]==5.1.0" --extra-index-url https://pypi.nvidia.com`
  - 仍然失败
- 仓库官方文档已经写明：Isaac Sim pip 安装要求 `GLIBC 2.35+`，官方支持 `Ubuntu 22.04 / 24.04`
- `gpu132` 当前系统是 `Ubuntu 20.04.6 / GLIBC 2.31`，不满足条件

因此这台机器当前策略应固定为：

- DP 训练：可以继续在 `gpu132` 跑
- IsaacLab rollout 评估：不要继续在 `gpu132` 原生环境上投入时间
- 如果必须上服务器评估，优先找 `Ubuntu 22.04+` 机器，或使用已可运行的 Isaac Sim 二进制 / 容器环境

### 9.8 按官方 Isaac Lab pip 页面复核

截至 `2026-04-13`，已按官方页面 `v2.3.1` 的命令在 `gpu132` 上复核：

- `pip install "isaaclab[isaacsim,all]==2.3.1" --extra-index-url https://pypi.nvidia.com`
- `pip install "isaaclab[isaacsim,all]==2.3.0" --extra-index-url https://pypi.nvidia.com`

两次都失败，错误一致：

- `Could not find a version that satisfies the requirement isaaclab==2.3.1`
- `Could not find a version that satisfies the requirement isaaclab==2.3.0`

因此对 `gpu132` 来说：

- 当前不能把 `IsaacLab pip` 页面当成可落地的远端评估安装方案
- 即使 `isaaclab` 包问题排除，`Ubuntu 20.04 / GLIBC 2.31` 依然不满足 `Isaac Sim 5.1` pip 运行条件
