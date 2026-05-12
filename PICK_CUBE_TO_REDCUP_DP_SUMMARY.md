# Pick Cube To Redcup DP 实验总结

> 更新日期：2026-04-13
> 范围：`/home/sxy/code/isaacsim_env/Real2Sim2Real-Pipeline_affordsim_clean`

## 1. 当前主线

目前这条线的目标很明确：

1. 为 `pick_cube_to_redcup` 收集可用于 Diffusion Policy 的轨迹数据
2. 确保数据中包含正面视角图像、机器人自身状态和动作
3. 将原始 `episode_XXXX.npy` 转成 DP 可训练的 HDF5
4. 在服务器上比较单视角和双视角，以及不同超参数配置

这条线已经不再停留在“能采到数据”，而是进入了“批量训练与比较”的阶段。

目前文档风格已经对齐到 `spain_task` 的交接方式：

- 明确默认远端机器是 `gpu132`
- 长训练优先 remote `screen`
- 需要固定写回 `server / screen / 日志 / 输出目录 / checkpoint / 下一步计划`


## 1.1 当前进行中的重训与已完成重评估（2026-04-13 20:14 CST）

当前已经进入下一轮正式动作：

1. 在 `gpu132` 上重新启动一轮更偏向可执行成功率的 DP 重训
2. 在本机 IsaacLab 环境里按修正后的成功判据，重评估旧 6 个 checkpoint 并保存 H.264 视频

本轮重训不再重复旧矩阵，而是基于当前定位结果做了两点收缩：

- 把 `pred_horizon` 从 `16` 缩到 `8` 或 `12`
- 把 `action_horizon` 从 `8` 缩到 `4`

这样做的原因很直接：

- `demo_0` 的 open-loop action replay 已经能成功
- 环境初始位姿和任务 yaml 一致
- 说明数据语义和环境执行基本对齐
- 当前更像是长 horizon rollout 下误差累计，而不是“环境错了”或“数据本身错了”

当前远端新矩阵根目录：

- `/data1/user/suixinyang/code/isaacsim_env/data/dp_debug/dp_matrix_pick_cube_to_redcup_49usable_v2_short_horizon`

当前远端 active screens：

- `gpu132 / GPU1 / screen=pickred_dp_v2_g1_20260413_1804`
  - log: `/data1/user/suixinyang/code/isaacsim_env/data/dp_debug/logs/pickred_dp_v2_g1_20260413_1804.log`
  - exps: `front_lr1e4_bs64_ph8_oh2_ah4`, `front_wrist_lr1e4_bs64_ph8_oh2_ah4`
- `gpu132 / GPU7 / screen=pickred_dp_v2_g7_20260413_1804`
  - log: `/data1/user/suixinyang/code/isaacsim_env/data/dp_debug/logs/pickred_dp_v2_g7_20260413_1804.log`
  - exps: `front_lr1e4_bs64_ph12_oh2_ah4`, `front_wrist_lr1e4_bs64_ph12_oh2_ah4`
- `gpu132 / GPU0 / screen=pickred_dp_v2_g0_20260413_1807`
  - log: `/data1/user/suixinyang/code/isaacsim_env/data/dp_debug/logs/pickred_dp_v2_g0_20260413_1807.log`
  - exps: `front_lr3e4_bs64_ph8_oh2_ah4`, `front_wrist_lr3e4_bs64_ph8_oh2_ah4`
  - 说明：原先放在 `GPU3` 的这一路因为显存不足 OOM，已迁到 `GPU0`

当前新矩阵早期状态：

- `front_lr1e4_bs64_ph8_oh2_ah4`: 到 `Epoch 18`，当前 best valid loss `0.052989`
- `front_lr1e4_bs64_ph12_oh2_ah4`: 到 `Epoch 17`，当前 best valid loss `0.041235`
- `front_lr3e4_bs64_ph8_oh2_ah4`: 到 `Epoch 9`，当前 best valid loss `0.058432`

这些早期 valid loss 已经明显低于上一轮同阶段表现，说明“短 horizon + 更频繁 replanning”的方向值得继续跑完。

当前本机旧模型重评估：

- screen: `pickred_reeval_old6_fix_20260413_1812`
- 本机 checkpoint 根目录：
  - `/home/sxy/code/isaacsim_env/data/dp_debug/eval_pick_cube_to_redcup_49usable_all`
- 结果根目录：
  - `/home/sxy/code/isaacsim_env/data/dp_debug/re_eval_pick_cube_to_redcup_49usable_all`
- 当前状态：
  - 旧 6 个 checkpoint 已全部重评估完成
  - `logs/*.json`、`logs/*.log`、`media/*` 都已落盘
- 重评估结论：
  - `front_lr1e4_bs32_ph16_oh2`: `0/5`, success rate `0.00%`
  - `front_lr3e4_bs64_ph16_oh2`: `0/5`, success rate `0.00%`
  - `front_lr1e4_bs32_ph16_oh4`: `0/5`, success rate `0.00%`
  - `front_wrist_lr1e4_bs32_ph16_oh2`: `0/5`, success rate `0.00%`
  - `front_wrist_lr3e4_bs64_ph16_oh2`: `0/5`, success rate `0.00%`
  - `front_wrist_lr1e4_bs32_ph16_oh4`: `0/5`, success rate `0.00%`
- 视频导出：
  - 单视角每个 checkpoint 导出 `5` 个 `front.mp4`
  - 双视角每个 checkpoint 导出 `5` 个 `front.mp4`、`5` 个 `wrist.mp4`、`5` 个 `combined.mp4`
- 说明：这轮重评估已经切到显式 `conda activate env_isaaclab` + 修正后的几何成功判据，不再使用之前那条错误 success term 链路

下一步计划：

1. 继续观察新矩阵前 `50-100 epoch` 的中期趋势，优先看单视角 `ph8/ph12`
2. 结合旧 6 模型 `0/5` 的重评估结果，优先等待短 horizon 新矩阵收敛后再做正式对比
3. 再决定是否把新矩阵继续扩到双视角更多组合，还是只保留最强单视角线继续加训

## 2. 当前数据集状态

### 2.1 正式训练数据

当前用于训练的正式数据目录是：

- `/home/sxy/code/isaacsim_env/data/r2s2r_data/pick_cube_to_redcup_franka_fr3/20260410_152550/20260410_152605`

当前原始目录规模：

- 原始 `episode_*.npy`：`50`
- 任务：`pick_cube_to_redcup`
- 机器人：`franka_fr3`

### 2.2 当前真实可用训练规模

虽然主目录里有 `50` 个 `episode_*.npy`，但 `episode_0047.npy` 当前已损坏，读取时报：

- `UnpicklingError: pickle data was truncated`

因此截至 `2026-04-11` 这次训练启动时，真实可用的 DP 训练数据是：

- 可用轨迹：`49`
- 跳过坏轨迹：`episode_0047.npy`
- 保留并补入：`episode_0048.npy`、`episode_0049.npy`

### 2.3 数据是怎么凑到 50 条的

正式主目录最初收到了 `48` 条轨迹，随后又补采了 `2` 条，并合并回主目录：

- 补采目录：
  `/home/sxy/code/isaacsim_env/data/r2s2r_data/pick_cube_to_redcup_franka_fr3/20260410_175520/20260410_175535`

最终主目录中名义上已经补齐：

- `episode_0048.npy`
- `episode_0049.npy`
- 前视角目录补到 `env0_ep50`

但实际训练时不能直接按“50 条全部可用”处理，因为 `episode_0047.npy` 已损坏。

## 3. DP 实际吃进去的数据

### 3.1 原始采集文件包含的字段

每个 `episode_XXXX.npy` 中当前保存的关键字段有：

- `obs_policy_actions`: `(T, 7)`
- `obs_policy_arm_ee_pose`: `(T, 7)`
- `obs_policy_q_pos`: `(T, 7)`
- `obs_policy_q_vel`: `(T, 7)`
- `obs_policy_external_cam_side`: `(T, H*W*3)`
- `obs_policy_external_cam_front`: `(T, H*W*3)`
- `obs_policy_wrist_cam`: `(T, H*W*3)`
- `obs_policy_lastactions`: `(T, 7)`
- `obs_policy_joint_actions`: `(T, 8)`

### 3.2 训练真正使用的字段

Diffusion Policy 训练目前只使用下面这些：

- 图像：
  - `front`
  - 或 `front + wrist`
- 低维状态：
  - `qpos (7)`
  - `qvel (7)`
  - 拼接后变成 `14` 维
- 监督目标：
  - `actions (7)`

当前没有送进主模型训练的字段：

- `side`
- `arm_ee_pose`
- `lastactions`
- `joint_actions`

### 3.3 转换后 HDF5 结构

原始 `npy` 会先转成 raw HDF5：

- `traj_X/action`
- `traj_X/observations/qpos`
- `traj_X/observations/qvel`
- `traj_X/observations/images/front`
- `traj_X/observations/images/wrist`

再转成 robomimic / DP 训练格式：

- `data/demo_X/actions`
- `data/demo_X/obs/qpos`
- `data/demo_X/obs/qvel`
- `data/demo_X/obs/front`
- `data/demo_X/obs/wrist`

训练时，`DiffusionPolicyDataset` 返回的单个样本是：

- `obs_images`: `(T_o, V, 3, H, W)`
- `obs_lowdim`: `(T_o, 14)`
- `actions`: `(T_a, 7)`

其中：

- `V = 1` 对应 `front`
- `V = 2` 对应 `front,wrist`
- `T_o = obs_horizon`
- `T_a = pred_horizon`

## 4. 已做的代码改动

### 4.1 采集侧

为了让轨迹里真正包含正面视角，相机配置已经补到任务配置中：

- `source/isaaclab_tasks/isaaclab_tasks/real2sim/tasks/pick/config/sceneconfig/franka_fr3/ourgripper/joint_vis_env_cfg.py`
- `source/isaaclab_tasks/isaaclab_tasks/real2sim/tasks/pick/config/sceneconfig/franka_fr3/ourgripper/ik_rel_ee_vis_env_cfg.py`
- `source/isaaclab_tasks/isaaclab_tasks/real2sim/tasks/pick/config/sceneconfig/franka_fr3/ourgripper/ik_rel_vis_env_cfg.py`

### 4.2 数据转换侧

新增或修改的关键脚本：

- `scripts/imitation_learning/diffusion_policy/convert_npy_dir_to_hdf5.py`
- `scripts/imitation_learning/robomimic/convert_hdf5_for_robomimic.py`

其中 `convert_hdf5_for_robomimic.py` 额外修了小数据集 train/valid split 的边界问题。

### 4.3 训练侧

为了支持“单视角”和“双视角”两种实验模式，已经改了：

- `scripts/imitation_learning/diffusion_policy/dataset.py`
- `scripts/imitation_learning/diffusion_policy/network.py`
- `scripts/imitation_learning/diffusion_policy/train.py`

新增能力：

- `--image_views front`
- `--image_views front,wrist`

## 5. 实验矩阵

目前正式远端矩阵改成了 6 组：

1. `front_lr1e4_bs32_ph16_oh2`
2. `front_lr3e4_bs64_ph16_oh2`
3. `front_lr1e4_bs32_ph16_oh4`
4. `front_wrist_lr1e4_bs32_ph16_oh2`
5. `front_wrist_lr3e4_bs64_ph16_oh2`
6. `front_wrist_lr1e4_bs32_ph16_oh4`

矩阵脚本：

- `scripts/imitation_learning/diffusion_policy/run_pick_cube_to_redcup_dp_matrix.sh`

这 6 组主要覆盖三个核心维度：

- 视角：
  - 前视角单视角
  - 前视角 + 腕部双视角
- 超参数：
  - `lr=1e-4, batch_size=32`
  - `lr=3e-4, batch_size=64`
- 观测历史：
  - `obs_horizon=2`
  - `obs_horizon=4`

## 6. 服务器口径

参考 `spain_task` 的运行习惯，当前这条线默认按下面口径交接：

- Preferred server: `gpu132`
- Long runs: remote `screen`
- 远端项目目录建议写成：
  - `/data1/user/suixinyang/code/isaacsim_env/Real2Sim2Real-Pipeline_affordsim_clean`
- 每次正式训练都应该补齐：
  - server
  - screen
  - remote log
  - remote output dir
  - best checkpoint
  - 当前结论
  - 下一步计划

### 6.1 当前远端评估环境结论

截至 `2026-04-13`，`gpu132` 的远端 IsaacLab rollout 评估环境已经做过一次完整排查，结论如下：

- 远端项目代码已同步到：
  - `/data1/user/suixinyang/code/isaacsim_env/Real2Sim2Real-Pipeline_affordsim_clean`
- 远端模型目录已同步到：
  - `/data1/user/suixinyang/code/isaacsim_env/models`
- 远端专用评估 yaml 已生成：
  - `/data1/user/suixinyang/code/isaacsim_env/data/dp_debug/pick_cube_to_redcup_gpu132_eval.yaml`

但是 `gpu132` 本机系统是：

- `Ubuntu 20.04.6 LTS`
- `GLIBC 2.31`

而当前仓库对应的 `Isaac Sim 5.1` pip 安装要求是：

- `GLIBC 2.35+`
- 官方支持 `Ubuntu 22.04 / 24.04`

因此这台机器的远端 rollout 评估目前有两个明确阻塞：

- 直接拷贝本机 `env_isaaclab` 到远端后，启动 `isaaclab.app.AppLauncher` 会报：
  - `GLIBC_2.33 not found`
- 在远端重新创建 `env_isaaclab_native` 并安装 `isaacsim[all,extscache]==5.1.0` 也失败
  - 仓库文档和实际安装结果都说明：`Ubuntu 20.04 / GLIBC 2.31` 不满足 pip 安装条件

当前结论：

- `gpu132` 适合继续做 DP 训练
- `gpu132` 不适合继续做 IsaacLab rollout 评估环境搭建
- 如果后续必须把评估也迁到服务器，应优先换到 `Ubuntu 22.04+` 的机器，或使用现成可运行的 Isaac Sim 二进制 / 容器环境

## 7. 结果目录约定

当前正式共享转换产物分成两套：

- 旧文件：
  - `/home/sxy/code/isaacsim_env/data/dp_debug/pick_cube_to_redcup_50_raw.hdf5`
  - 这份只对应旧的 `47` 条 raw 状态，不再作为正式训练入口
- 当前正式训练文件：
  - `/home/sxy/code/isaacsim_env/data/dp_debug/pick_cube_to_redcup_49usable_raw.hdf5`
  - `/home/sxy/code/isaacsim_env/data/dp_debug/pick_cube_to_redcup_49usable_robomimic.hdf5`
  - `/home/sxy/code/isaacsim_env/data/dp_debug/pick_cube_to_redcup_49usable_dataset_stat.json`

当前正式实验输出根目录：

- `/home/sxy/code/isaacsim_env/data/dp_debug/dp_matrix_pick_cube_to_redcup_49usable`

单个实验的典型输出目录结构：

- `.../<experiment_name>/diffusion_policy_<timestamp>/config.json`
- `.../<experiment_name>/diffusion_policy_<timestamp>/events.out.tfevents.*`
- `.../<experiment_name>/diffusion_policy_<timestamp>/models/model_best.pth`

## 8. 当前运行状态

截至 `2026-04-12` 当前这次检查时：

- 远端 `gpu132` 上 6 组训练已全部结束
- 远端 output root:
  - `/data1/user/suixinyang/code/isaacsim_env/data/dp_debug/dp_matrix_pick_cube_to_redcup_49usable`
- 当前训练结论：
  - 单视角可训出来
  - 当前这批 best valid loss 仍是单视角更好
  - 最好配置是 `front_lr3e4_bs64_ph16_oh2`

### 8.1 当前 6 组 best valid loss

| 实验 | best valid loss |
|---|---:|
| `front_lr1e4_bs32_ph16_oh2` | `0.010161` |
| `front_lr3e4_bs64_ph16_oh2` | `0.007817` |
| `front_lr1e4_bs32_ph16_oh4` | `0.009689` |
| `front_wrist_lr1e4_bs32_ph16_oh2` | `0.011216` |
| `front_wrist_lr3e4_bs64_ph16_oh2` | `0.011004` |
| `front_wrist_lr1e4_bs32_ph16_oh4` | `0.014250` |

### 8.2 当前成功率评估状态

目前成功率评估不在 `gpu132` 上做，原因是远端只同步了最小 DP 训练子集，没有完整 `IsaacLab/source` 运行依赖。

因此当前评估切回本机短跑验证，使用：

- checkpoint:
  - `/home/sxy/code/isaacsim_env/data/dp_debug/eval_pick_cube_to_redcup_49usable/front_best_model.pth`
  - `/home/sxy/code/isaacsim_env/data/dp_debug/eval_pick_cube_to_redcup_49usable/front_wrist_best_model.pth`
- 统计文件:
  - `/home/sxy/code/isaacsim_env/data/dp_debug/pick_cube_to_redcup_49usable_dataset_stat.json`
- 评估日志目录:
  - `/home/sxy/code/isaacsim_env/data/dp_debug/eval_logs`

首轮 `1 rollout` 冒烟结果：

| checkpoint | 视角 | rollouts | 成功率 |
|---|---|---:|---:|
| `front_best_model.pth` | `front` | `1` | `0/1 = 0.00%` |
| `front_wrist_best_model.pth` | `front,wrist` | `1` | `0/1 = 0.00%` |

对应结果文件：

- `/home/sxy/code/isaacsim_env/data/dp_debug/eval_logs/pickred_front_eval_1.json`
- `/home/sxy/code/isaacsim_env/data/dp_debug/eval_logs/pickred_frontwrist_eval_1.json`

当前判断：

- 训练 loss 已明显收敛，但首轮环境 rollout 还没有成功
- 这说明“拟合 loss 变好”还没有转成“任务成功率可用”
- 下一步应该扩大评估样本，并排查动作尺度、成功判据、episode horizon、以及训练/推理归一化是否完全一致

### 8.3 当前 `5 rollout` 正式评估

截至 `2026-04-12`，已对单视角 best 和双视角 best 各跑 `5` 次 rollout：

| checkpoint | 视角 | rollouts | 成功率 |
|---|---|---:|---:|
| `front_best_model.pth` | `front` | `5` | `0/5 = 0.00%` |
| `front_wrist_best_model.pth` | `front,wrist` | `5` | `0/5 = 0.00%` |

结果 json：

- `/home/sxy/code/isaacsim_env/data/dp_debug/eval_logs/pickred_front_eval_5.json`
- `/home/sxy/code/isaacsim_env/data/dp_debug/eval_logs/pickred_frontwrist_eval_5.json`

结果日志：

- `/home/sxy/code/isaacsim_env/data/dp_debug/eval_logs/pickred_front_eval_5.log`
- `/home/sxy/code/isaacsim_env/data/dp_debug/eval_logs/pickred_frontwrist_eval_5.log`

逐 trial 视频：

- 单视角 front 视频目录：
  - `/home/sxy/code/isaacsim_env/data/dp_debug/eval_videos/pickred_front_5`
- 双视角 front/wrist 视频目录：
  - `/home/sxy/code/isaacsim_env/data/dp_debug/eval_videos/pickred_frontwrist_5`

当前任务结论已经比较明确：

- 这批 DP 模型“训练收敛”不等于“任务成功”
- 当前单视角和双视角都没有跑出成功样本
- 单视角动作幅度更大，表现更像“有意图但没完成”
- 双视角动作整体更保守，更像停留在不稳定接近阶段
  - `front_lr3e4_bs64_ph16_oh2`
  - `front_lr1e4_bs32_ph16_oh4`
- 当前并行运行中的实验：
  - GPU 1:
    - screen: `pickred_dp49_m6_g1_20260411_0947`
    - exp: `front_wrist_lr1e4_bs32_ph16_oh2`
    - log: `/data1/user/suixinyang/code/isaacsim_env/data/dp_debug/logs/pickred_dp49_m6_g1_20260411_0947.log`
  - GPU 3:
    - screen: `pickred_dp49_g3_frontwrist_bs64_20260411_1547`
    - exp: `front_wrist_lr3e4_bs64_ph16_oh2`
    - log: `/data1/user/suixinyang/code/isaacsim_env/data/dp_debug/logs/pickred_dp49_g3_frontwrist_bs64_20260411_1547.log`
  - GPU 7:
    - screen: `pickred_dp49_g7_frontwrist_oh4_20260411_1547`
    - exp: `front_wrist_lr1e4_bs32_ph16_oh4`
    - log: `/data1/user/suixinyang/code/isaacsim_env/data/dp_debug/logs/pickred_dp49_g7_frontwrist_oh4_20260411_1547.log`
- 当前已确认的单前视角早期结果：
  - `front_lr1e4_bs32_ph16_oh2` 在早期已到 `valid loss 0.058768`
- 当前 best checkpoint 示例：
  - `/data1/user/suixinyang/code/isaacsim_env/data/dp_debug/dp_matrix_pick_cube_to_redcup_49usable/front_lr1e4_bs32_ph16_oh2/diffusion_policy_20260411094611/models/model_best.pth`

## 9. 当前未完成项

目前已经完成：

- 50 条轨迹收集
- 数据格式转换脚本
- 单视角 / 双视角训练代码改造
- 批量实验脚本
- 服务器运行说明文档

目前还需要继续补充：

1. 跑完 6 组矩阵并记录每组 best valid loss
2. 统计每组训练时间
3. 对比单前视角和前视角+腕部双视角
4. 对比 `obs_horizon=2` 和 `obs_horizon=4`
5. 判断单前视角是否已经足够把任务训出来
6. 继续补齐 checkpoint 路径整理

## 10. 接下来建议

如果继续推进，建议按这个顺序：

1. 继续盯 3 个 active screen 是否都稳定收敛
2. 确认剩余 3 组都能落盘 `config.json`、`events`、`model_best.pth`
3. 3 个双视角组跑完后，把单视角和双视角结果回填到本文件
4. 如果单前视角明显不够，再决定是否继续补：
   - 只用 wrist
   - 更长 `pred_horizon`
   - 更小 / 更大 batch

## 11. 结论

这条线当前已经具备了：

- 明确的原始 `50` 条目录状态
- 明确的 `49` 条可用训练口径
- 明确的数据格式
- 单 / 双视角可切换训练能力
- 带远端变量覆盖的可复跑矩阵脚本
- 可交接的文档入口

后续最重要的是把这 6 组矩阵稳定跑完，并直接回答：

- 单前视角能不能把 `pick_cube_to_redcup` 训起来
- 双视角相对单视角到底提升多少

### 6.2 按 Isaac Lab pip 页面复核的结果

`2026-04-13` 又按官方页面 `v2.3.1` 的 pip 方案在 `gpu132` 上做了复核。

尝试命令：

- `pip install "isaaclab[isaacsim,all]==2.3.1" --extra-index-url https://pypi.nvidia.com`
- `pip install "isaaclab[isaacsim,all]==2.3.0" --extra-index-url https://pypi.nvidia.com`

实际结果：

- 两条命令都直接报：
  - `Could not find a version that satisfies the requirement isaaclab==2.3.1`
  - `Could not find a version that satisfies the requirement isaaclab==2.3.0`

说明：

- 用户提供的 `v2.3.1` 页面里虽然写了 `isaaclab[isaacsim,all]==2.3.1`，但同页也写了当前 pip 包只覆盖 major release
- 在 `gpu132` 当前环境下，这条 pip 页面路线没有成功拿到可安装的 `isaaclab` 包
- 即便后续 `isaaclab` 包问题解决，`gpu132` 仍然还有 `Ubuntu 20.04 / GLIBC 2.31` 与 `Isaac Sim 5.1` 不兼容的问题
