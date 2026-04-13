import os
import numpy as np
from pathlib import Path
import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab_tasks.real2sim import REAL2SIM_ROOT_DIR

# Model files are located in the models folder at the project root
ASSET_PATH = Path(REAL2SIM_ROOT_DIR).parents[3] / "models"

FRANKA_FR3_CFG = ArticulationCfg(
    prim_path="{ENV_REGEX_NS}/fr3",
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ASSET_PATH}/franka_fr3_limit.usd",    
        activate_contact_sensors=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=True,
            max_depenetration_velocity=5.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False, 
            solver_position_iteration_count=64, 
            solver_velocity_iteration_count=0
        ),
        # Removed semantic_tags - now added precisely via events
        semantic_tags=[('class', 'robot')],
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0,0,0), 
        rot=(1,0,0,0), 
        joint_pos={
            "fr3_joint1": 0.0,
            "fr3_joint2": -1 / 5 * np.pi,
            "fr3_joint3": 0.0,
            "fr3_joint4": -4 / 5 * np.pi,
            "fr3_joint5": 0.0,
            "fr3_joint6": 3 / 5 * np.pi, #3 / 5 * np.pi
            "fr3_joint7": 1/4 * np.pi,
            "fr3_finger_joint.*": False,
        },
    ),
    actuators={
        "fr3_shoulder": ImplicitActuatorCfg(
            joint_names_expr=["fr3_joint[1-4]"],
            effort_limit_sim=87.0,
            velocity_limit_sim=2.175,
            stiffness=400.0,
            damping=80.0,
            # stiffness=80.0,
            # damping=4.0,
        ),
        "fr3_forearm": ImplicitActuatorCfg(
            joint_names_expr=["fr3_joint[5-7]"],
            effort_limit_sim=12.0,
            velocity_limit_sim=2.61,
            stiffness=400.0,
            damping=80.0,
            # stiffness=80.0,
            # damping=4.0,
        ),
        "fr3_hand": ImplicitActuatorCfg(
            joint_names_expr=["fr3_finger_joint.*"],
            effort_limit_sim=500.0,
            velocity_limit_sim=0.2,
            stiffness=5e3,
            damping=2e2,
        ),
    },
    soft_joint_pos_limit_factor = 0.9  # Used to compute soft limits; 0.9 is generally sufficient
)

FRANKA_FR3_OURGRIPPER_CFG = ArticulationCfg(
    prim_path="{ENV_REGEX_NS}/fr3",
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ASSET_PATH}/franka_fr3_ourgripper.usd",
        activate_contact_sensors=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=True,
            max_depenetration_velocity=5.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False, 
            solver_position_iteration_count=64, 
            solver_velocity_iteration_count=0
        ),
        semantic_tags=[('class', 'robot')],
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0,0,0), 
        rot=(1,0,0,0), 
        joint_pos={
            "fr3_joint1": 0.0,
            "fr3_joint2": -1 / 5 * np.pi,
            "fr3_joint3": 0.0,
            "fr3_joint4": -4 / 5 * np.pi,
            "fr3_joint5": 0.0,
            "fr3_joint6": 3 / 5 * np.pi,
            "fr3_joint7": 1/4 * np.pi,
            "fr3_finger_joint.*": False,
        },
    ),
    actuators={
        "fr3_shoulder": ImplicitActuatorCfg(
            joint_names_expr=["fr3_joint[1-4]"],
            effort_limit_sim=87.0,
            velocity_limit_sim=2.175,
            stiffness=400.0,
            damping=80.0,
            # stiffness=80.0,
            # damping=4.0,
        ),
        "fr3_forearm": ImplicitActuatorCfg(
            joint_names_expr=["fr3_joint[5-7]"],
            effort_limit_sim=12.0,
            velocity_limit_sim=2.61,
            stiffness=400.0,
            damping=80.0,
            # stiffness=80.0,
            # damping=4.0,
        ),
        "fr3_hand": ImplicitActuatorCfg(
            joint_names_expr=["fr3_finger_joint.*"],
            effort_limit_sim=500.0,
            velocity_limit_sim=0.2,
            stiffness=5e3,
            damping=2e2,
        ),
    },
    soft_joint_pos_limit_factor = 0.9
)

FRANKA_PANDA_OURGRIPPER_CFG = ArticulationCfg(
    prim_path="{ENV_REGEX_NS}/panda",
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ASSET_PATH}/franka.usd",  
        activate_contact_sensors=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=True,
            max_depenetration_velocity=5.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False, 
            solver_position_iteration_count=64, 
            solver_velocity_iteration_count=0
        ),
        
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0,0,0), 
        rot=(1,0,0,0), 
        joint_pos={
            "panda_joint1": 0.0,
            "panda_joint2": -1 / 5 * np.pi,
            "panda_joint3": 0.0,
            "panda_joint4": -4 / 5 * np.pi,
            "panda_joint5": 0.0,
            "panda_joint6": 3 / 5 * np.pi,
            "panda_joint7": 0,
            "panda_finger_joint.*": False,
        },
    ),
    actuators={
        "panda_shoulder": ImplicitActuatorCfg(
            joint_names_expr=["panda_joint[1-4]"],
            effort_limit_sim=87.0,
            velocity_limit_sim=2.175,
            stiffness=400.0,
            damping=80.0,
            # stiffness=80.0,
            # damping=4.0,
        ),
        "panda_forearm": ImplicitActuatorCfg(
            joint_names_expr=["panda_joint[5-7]"],
            effort_limit_sim=12.0,
            velocity_limit_sim=2.61,
            stiffness=400.0,
            damping=80.0,
            # stiffness=80.0,
            # damping=4.0,
        ),
        "panda_hand": ImplicitActuatorCfg(
            joint_names_expr=["panda_finger_joint.*"],
            effort_limit_sim=500.0,
            velocity_limit_sim=0.2,
            stiffness=5e3,
            damping=2e2,
        ),
    },
    soft_joint_pos_limit_factor = 0.9
)