# HDF5 轨迹格式

通用采集入口通过 `--save_trajectories` 启用数据保存，直接写入独立的 `episode_XXXX.hdf5`；使用旧配置中的 `runtime.save_npy: true` 也会写入 HDF5。正式 HDF5 采集不会额外保存逐帧图像或 NPY 轨迹；只采 GIF 的预览模式不受此设置影响。训练使用的字段与现有 HDF5 读取接口一致：

| 路径 | 内容 |
| --- | --- |
| `action` | 原仓库定义的 7 维动作：相邻记录帧的末端位置和姿态变化，以及源帧的夹爪命令。 |
| `observations/qpos`、`observations/qvel` | 机械臂实际关节位置和速度，按 `recordings/trajectory_fields/arm_joint_names` 的顺序。UR 为 6 轴，其他构型按实际关节数保存，不补零。 |
| `observations/images/front`、`observations/images/wrist` | 与 `action` 源帧对应的 RGB 图像，沿用原转换工具的中心裁剪和 256×256 缩放。 |
| `recordings/source_frame_index`、`recordings/next_frame_index` | 每个动作对应的源帧和下一帧编号。 |
| `recordings/trajectory_fields/obs_policy_joint_actions` | 源帧对应的原始关节目标命令。 |
| `recordings/trajectory_fields/observed_next_joint_command` | 到达下一记录帧时实际下发的关节目标命令，作为独立字段保留；不会替换训练主动作。 |
| `recordings/trajectory_fields/arm_joint_names` | 与实际关节数据对应的关节名称和顺序。 |

夹爪命令的数值、符号和末端位姿变化的计算沿用原采集代码。HDF5 采集保留静止控制步，包括倒水最大角度处的保持步。通过旧的帧分离采集转换出的 HDF5 还会保留 `recordings/cameras/.../encoded_bytes` 和原始 NPY 内容；通用入口直接写入训练数据，不生成 NPY 副本。10 秒 GIF/MP4 只是加速预览，不改变动作或仿真控制时间。

将同一构型的合格轨迹合并为 ACT/BC 读取的 `traj_0`、`traj_1` 等结构，例如：

```bash
python tools/merge_accepted_hdf5.py --episode-dir /path/to/episodes \
  --robot ur5e --output /path/to/ur5e_trajectories.hdf5
```

已有的专项采集清单也可以用于合并：

```bash
python tools/merge_accepted_hdf5.py \
  --manifest outputs/collection_50_hdf5_20261029/accepted_manifest.json \
  --robot ur5e --output /path/to/ur5e_trajectories.hdf5
```

可重复传入 `--task TASK_ID` 选择任务。工具会检查关节顺序、维度和动作约定，不会覆盖已有输出或将不同构型静默拼接。ACT、BC 的训练入口，以及 `tools/convert_hdf5_for_robomimic.py`，需要显式使用 `--no_remove_idle` 才能避免在读取/转换时再次丢弃静止步。

DP 的 HDF5 数据读取和模型输入维度按实际关节数推断。非 Franka 构型的在线 rollout 接口，以及 VLA-Adapter 和 Pi0.5 的跨构型输入协议，仍需各自验证；数据格式兼容不等于策略已完成训练或验证。
