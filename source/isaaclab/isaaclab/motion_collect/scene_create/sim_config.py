import os
import glob
from pathlib import Path

import numpy as np
from isaaclab.assets import AssetBaseCfg, RigidObjectCfg
from isaaclab.assets.articulation import ArticulationCfg
from isaaclab.sim.spawners.from_files.from_files_cfg import UsdFileCfg
from isaaclab.sim.schemas.schemas_cfg import RigidBodyPropertiesCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.utils import configclass
import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.sensors import CameraCfg, TiledCameraCfg


# Model files are located in the models directory at the project root
ASSET_PATH = "/home/hk/xuhaofan/Real2Sim2Real-Pipeline-auto_task_aff/models"
ALL_OBJECT_DIR = os.path.join(ASSET_PATH, "all_object")
@configclass
class SceneCfg(InteractiveSceneCfg):
    # lights
    dome_light = AssetBaseCfg(
        prim_path="/World/dome_light",
        spawn=sim_utils.DomeLightCfg(intensity=2000.0),
    )
    ground = AssetBaseCfg(
        prim_path="/World/defaultGroundPlane", 
        spawn=sim_utils.GroundPlaneCfg(),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, -0.8)),
    )
    table = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/table",
        spawn=sim_utils.UsdFileCfg(
            usd_path=f"{ASSET_PATH}/table.usd",
        ),
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(0, 0, 0),
            rot=(1.0, 0.0, 0.0, 0.0),
        ),
    )  

    robot = ArticulationCfg(
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
                effort_limit_sim=200.0,
                velocity_limit_sim=0.2,
                stiffness=2e3,
                damping=1e2,
            ),
        },
        soft_joint_pos_limit_factor = 0.9
    )

    fr3_base = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/fr3_base",
        spawn=sim_utils.UsdFileCfg(
            usd_path=f"{ASSET_PATH}/FR3_base.usd",
        ),
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(0, 0, 0),
            rot=(1.0, 0.0, 0.0, 0.0),
        ),
    )  

    pass  # objects are added dynamically below


# Dynamically scan all USD files and add object configs to SceneCfg
_usd_files = sorted(glob.glob(os.path.join(ALL_OBJECT_DIR, "*.usd")))
SIM_OBJECT_IDS = []
for _i, _usd in enumerate(_usd_files, start=1):
    _attr_name = f"object_{_i}"
    _cfg = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/" + _attr_name,
        init_state=RigidObjectCfg.InitialStateCfg(pos=[10.0 + _i, 0.0, 0.25], rot=[1, 0, 0, 0]),
        spawn=UsdFileCfg(usd_path=_usd, scale=(1, 1, 1)),
    )
    setattr(SceneCfg, _attr_name, _cfg)
    SIM_OBJECT_IDS.append(Path(_usd).stem)

SIM_NUM_OBJECTS = len(SIM_OBJECT_IDS)
print(f"[sim_config] Found {SIM_NUM_OBJECTS} USD files: {SIM_OBJECT_IDS}")
