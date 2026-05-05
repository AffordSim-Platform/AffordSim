from pathlib import Path

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg

from isaaclab_tasks.real2sim import REAL2SIM_ROOT_DIR

ASSET_PATH = Path(REAL2SIM_ROOT_DIR).parents[3] / "models"


XARM7_CFG = ArticulationCfg(
    prim_path="{ENV_REGEX_NS}/xarm7",
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ASSET_PATH}/xarm7.usd",
        activate_contact_sensors=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=True,
            max_depenetration_velocity=5.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False,
            solver_position_iteration_count=64,
            solver_velocity_iteration_count=0,
        ),
        semantic_tags=[("class", "robot")],
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0, 0, 0),
        rot=(1, 0, 0, 0),
        joint_pos={
            "joint1": 0.0,
            "joint2": -0.829,   # -47.5 deg
            "joint3": 0.0,
            "joint4": 0.0,
            "joint5": 0.0,
            "joint6": 0.0,
            "joint7": 0.168,    # 9.6 deg
            "drive_joint": 0.0,
        },
    ),
    actuators={
        "xarm7_shoulder": ImplicitActuatorCfg(
            joint_names_expr=["joint1", "joint2", "joint3", "joint4"],
            effort_limit_sim=80.0,
            velocity_limit_sim=3.14,
            stiffness=400.0,
            damping=80.0,
        ),
        "xarm7_wrist": ImplicitActuatorCfg(
            joint_names_expr=["joint5", "joint6", "joint7"],
            effort_limit_sim=28.0,
            velocity_limit_sim=3.14,
            stiffness=400.0,
            damping=80.0,
        ),
        "xarm_gripper": ImplicitActuatorCfg(
            joint_names_expr=["drive_joint"],
            effort_limit_sim=16.5,
            velocity_limit_sim=130.0,
            stiffness=10.0,
            damping=0.1,
        ),
    },
    soft_joint_pos_limit_factor=0.9,
)
