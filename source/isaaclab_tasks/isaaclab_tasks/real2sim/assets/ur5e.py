import numpy as np
from pathlib import Path
import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab_tasks.real2sim import REAL2SIM_ROOT_DIR

ASSET_PATH = Path(REAL2SIM_ROOT_DIR).parents[3] / "models"

UR5E_ROBOTIQ85_CFG = ArticulationCfg(
    prim_path="{ENV_REGEX_NS}/ur5e",
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ASSET_PATH}/ur5e_robotiq85_new.usd",
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
        semantic_tags=[('class', 'robot')],
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0, 0, 0),
        rot=(1, 0, 0, 0),
        joint_pos={
            "shoulder_pan_joint":  0.0,
            "shoulder_lift_joint": -1.5708,   # -π/2
            "elbow_joint":          1.5708,   #  π/2
            "wrist_1_joint":       -1.5708,   # -π/2
            "wrist_2_joint":       -1.5708,   # -π/2
            "wrist_3_joint":        0.0,
            "finger_joint":         0.0,      # open
        },
    ),
    actuators={
        "ur5e_shoulder": ImplicitActuatorCfg(
            joint_names_expr=["shoulder_pan_joint", "shoulder_lift_joint", "elbow_joint"],
            effort_limit_sim=150.0,
            velocity_limit_sim=3.14,
            stiffness=400.0,
            damping=80.0,
        ),
        "ur5e_wrist": ImplicitActuatorCfg(
            joint_names_expr=["wrist_1_joint", "wrist_2_joint", "wrist_3_joint"],
            effort_limit_sim=28.0,
            velocity_limit_sim=3.14,
            stiffness=400.0,
            damping=80.0,
        ),
        "robotiq85_hand": ImplicitActuatorCfg(
            joint_names_expr=["finger_joint"],
            effort_limit_sim=16.5,
            velocity_limit_sim=130,
            stiffness=10,
            damping=0.1,
        ),
    },
    soft_joint_pos_limit_factor=0.9,
)
