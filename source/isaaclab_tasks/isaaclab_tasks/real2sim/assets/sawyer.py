from pathlib import Path

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg

from isaaclab_tasks.real2sim import REAL2SIM_ROOT_DIR

ASSET_PATH = Path(REAL2SIM_ROOT_DIR).parents[3] / "models"


SAWYER_ROBOTIQ85_CFG = ArticulationCfg(
    prim_path="{ENV_REGEX_NS}/sawyer",
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ASSET_PATH}/sawyer_new.usd",
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
            "right_j0": 0.0,
            "right_j1": -1.2,
            "right_j2": 0.0,
            "right_j3": 1.0471975511965976,
            "right_j4": 0.0,
            "right_j5": 1.3089969389957472,
            "right_j6": 0.0,
            "finger_joint": 0.0,
        },
    ),
    actuators={
        "sawyer_arm": ImplicitActuatorCfg(
            joint_names_expr=[
                "right_j0",
                "right_j1",
                "right_j2",
                "right_j3",
                "right_j4",
                "right_j5",
                "right_j6",
            ],
            effort_limit_sim=150.0,
            velocity_limit_sim=3.1415,
            stiffness=500.0,
            damping=50.0,
        ),
        "robotiq85_hand": ImplicitActuatorCfg(
            joint_names_expr=["finger_joint"],
            effort_limit_sim=16.5,
            velocity_limit_sim=130.0,
            stiffness=10.0,
            damping=0.1,
        ),
    },
    soft_joint_pos_limit_factor=0.9,
)
