from pathlib import Path

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg

from isaaclab_tasks.real2sim import REAL2SIM_ROOT_DIR

ASSET_PATH = Path(REAL2SIM_ROOT_DIR).parents[3] / "models"


KINOVA_ROBOTIQ85_CFG = ArticulationCfg(
    prim_path="{ENV_REGEX_NS}/kinova",
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ASSET_PATH}/kinova_robotiq85_new.usd",
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
            "joint_1": 0.0,
            "joint_2": 0.65,
            "joint_3": 0.0,
            "joint_4": 1.89,
            "joint_5": 0.0,
            "joint_6": 0.6,
            "joint_7": -1.57,
            "finger_joint": 0.0,
        },
    ),
    actuators={
        "kinova_arm": ImplicitActuatorCfg(
            joint_names_expr=[
                "joint_1",
                "joint_2",
                "joint_3",
                "joint_4",
                "joint_5",
                "joint_6",
                "joint_7",
            ],
            effort_limit_sim=80.0,
            velocity_limit_sim=2.5,
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
