# =============================================================================
# 各原语成功判定阈值常量
# 仅 yaml 中开启 success_eval 时, 这些常量才会被使用
# =============================================================================

# --- place 原语 ---
# place 成功判定距离阈值 (米):
# - 非 affordance 模式: 物体最终位置与目标 place 位置之间的最大 XY 距离
# - affordance 模式: 被放置物体 place affordance 与目标 place affordance 的最大距离
PLACE_SUCCESS_XY_THRESHOLD = 0.03
# place 时要求 pick 时的 z >= place 时的 z + Z_MARGIN, 防止穿透
PLACE_SUCCESS_Z_MARGIN = 0.0
# place 成功还要求夹爪已松开 (单侧 finger 关节位置 >= 该阈值, 米).
# Franka FR3 / Panda 完全张开 ≈ 0.04, 所以 0.025 留一些余量;
# 调小阈值更宽松 (允许半开), 调大更严格.
PLACE_SUCCESS_GRIPPER_OPEN_THRESHOLD = 0.018

# --- 通用 (place / pour / push / pull / hang) ---
# 物体落入 yaml 中 position_range.x / position_range.y 范围时允许的边界容差 (米)
POSITION_RANGE_TOLERANCE = 0.01

# --- pour 原语 ---
# 容器局部 z 轴在世界 z 方向上的分量阈值, 小于等于该值视为成功倾倒
# 0.0 → 容器轴需水平或更低; 取负值 → 要求倒得更彻底
POUR_SUCCESS_TILT_THRESHOLD = 0.0 # 0.707
# 倾倒时被倒物体与目标容器的最大 XY 距离 (米)
POUR_SUCCESS_XY_THRESHOLD = 0.1

# ---- pour 过程判定 (tilt 插值期间的逐步采样检查) ----
# 被倾倒物体的 pour tip 在世界 XY 与容器 pour center 的最大越界距离 (米)
# 实际容忍半径 = tilt_horizontal_offset + POUR_PROCESS_XY_TOLERANCE
POUR_PROCESS_XY_TOLERANCE = 0.1
# 单步 z_axis_world_z 允许的反向上升量 (容忍物理抖动 / IK 漂移)
# z_axis_world_z 物理含义: 杯子局部 +z 在世界 z 方向的分量
# 1.0 → 完全竖直; 0.0 → 水平; <0 → 倒扣; 理想过程是逐步减小
# 注: 总倾倒角度由终点检查 POUR_SUCCESS_TILT_THRESHOLD 保证, 此处只负责过程连续性
POUR_PROCESS_TILT_MONOTONIC_SLACK = 0.01

# --- push 原语 ---
# 物体最终位置与目标 push 位置之间允许的最大 XY 距离 (米)
PUSH_SUCCESS_XY_THRESHOLD = 0.05
# 物体本地 z 轴在世界 z 方向上的最小分量; 小于该值视为翻倒
# 1.0 → 完全直立; 0.8 ≈ 允许 ~37° 以内倾斜; 0.0 → 允许倒伏
PUSH_SUCCESS_TILT_THRESHOLD = 0.8

# --- pull 原语 ---
# pull 拉动时, articulation 上对应 prismatic 关节最小位移 (米)
PULL_SUCCESS_MIN_DISPLACEMENT = 0.08
# pull 评估时使用的关节名 (cabinet1 抽屉的 prismatic 关节)
PULL_SUCCESS_JOINT_NAME = "joint_2"

# --- open 原语 ---
# open 时 articulation 上 revolute 关节最小开启角度 (弧度), 默认 ≈57°
OPEN_SUCCESS_MIN_ANGLE_RAD = 1.0
# open 评估时使用的关节名 (microwave 门的 revolute 关节)
OPEN_SUCCESS_JOINT_NAME = "microjoint"

# --- close 原语 ---
# close 时 articulation 上 revolute 关节允许的最大闭合残差角度 (弧度), 默认 ≈14°
CLOSE_SUCCESS_MAX_ANGLE_RAD = 0.25
# close 评估时使用的关节名 (microwave 门的 revolute 关节)
CLOSE_SUCCESS_JOINT_NAME = "microjoint"

# --- hang 原语 ---
# 物体最终位置与（挂载点和物挂架中心点连线的中点）之间允许的最大 XY 距离 (米)
HANG_SUCCESS_XY_THRESHOLD = 0.05
# 物体在 env 局部坐标下要求达到的最低 Z 高度 (米), 防止物体掉到地面
HANG_SUCCESS_MIN_Z = 0.08

# --- hold 原语 ---
# 物体在 env 局部坐标下要求达到的最低 Z 高度 (米), 用于判定是否抬起
HOLD_SUCCESS_MIN_Z = 0.10

# --- pick 原语 ---
# 物体相对初始 z 至少抬升多少才算抓取成功 (米)
PICK_SUCCESS_LIFT_THRESHOLD = 0.02

# --- IK solution quality ---
# Max allowed joint-space jump (radians) between current joint position and IK solution per step.
# Normal fine-interpolation steps produce ~0.1-0.2 rad change; anything above this threshold
# indicates CuRobo returned a solution far from the current configuration.
IK_MAX_JOINT_JUMP = 0.5
