from pathlib import Path

import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg, RigidObjectCfg
from isaaclab.sensors import CameraCfg, FrameTransformerCfg, TiledCameraCfg
from isaaclab.sensors.frame_transformer.frame_transformer_cfg import OffsetCfg
from isaaclab.utils import configclass

from isaaclab_tasks.real2sim.assets.franka import FRANKA_FR3_OURGRIPPER_CFG
from isaaclab_tasks.real2sim.tasks.pick import mdp
from isaaclab_tasks.real2sim.tasks.pick.pick_r2s_env_cfg import (
    EnvCfg,
    _get_scene_usd_path,
    _get_sim_camera_resolution,
    _get_use_background_replace,
)
from isaaclab_tasks.real2sim.utils.camera_cfg import fr3_default_side_cam, fr3_default_wrist_cam

_current = Path(__file__).resolve()
PROJECT_ROOT = _current
while PROJECT_ROOT != PROJECT_ROOT.parent:
    if (PROJECT_ROOT / "models").is_dir():
        break
    PROJECT_ROOT = PROJECT_ROOT.parent


@configclass
class FrankaFr3DefaultGripperIKRelVisPickEnvCfg(EnvCfg):
    def __post_init__(self):
        super().__post_init__()

        self.scene.robot = FRANKA_FR3_OURGRIPPER_CFG.replace(prim_path="{ENV_REGEX_NS}/fr3")

        self.actions.body = mdp.DifferentialInverseKinematicsActionCfg(
            delta_mode="origin",
            asset_name="robot",
            joint_names=["fr3_joint.*"],
            body_name="fr3_hand",
            controller=mdp.DifferentialIKControllerCfg(
                command_type="pose",
                use_relative_mode=True,
                ik_method="dls",
                ik_params={"lambda_val": 0.1},
            ),
        )
        self.actions.gripper_action = mdp.BinaryJointPositionActionCfg(
            asset_name="robot",
            joint_names=["fr3_finger.*"],
            open_command_expr={"fr3_finger_.*": 0.04},
            close_command_expr={"fr3_finger_.*": 0.0},
        )

        self.scene.basic_block_red = RigidObjectCfg(
            prim_path="{ENV_REGEX_NS}/basic_block_red",
            spawn=sim_utils.UsdFileCfg(
                usd_path=str(PROJECT_ROOT / "models/rigid_assets/basic_block.usd"),
                semantic_tags=[("class", "cube")],
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.0, 0.0)),
            ),
            init_state=RigidObjectCfg.InitialStateCfg(
                pos=(0.47, 0.068, 0.0483),
                rot=(1.0, 0.0, 0.0, 0.0),
            ),
        )
        self.scene.table = AssetBaseCfg(
            prim_path="{ENV_REGEX_NS}/table",
            spawn=sim_utils.UsdFileCfg(
                usd_path=_get_scene_usd_path(),
            ),
            init_state=RigidObjectCfg.InitialStateCfg(
                pos=(0, 0, 0),
                rot=(1.0, 0.0, 0.0, 0.0),
            ),
        )

        _cam_w, _cam_h = _get_sim_camera_resolution()
        if _get_use_background_replace():
            camera_types = ["rgb", "semantic_segmentation"]
        else:
            camera_types = ["rgb", "distance_to_image_plane"]

        self.scene.external_cam_side = TiledCameraCfg(
            prim_path="{ENV_REGEX_NS}/external_cam_side",
            height=_cam_h,
            width=_cam_w,
            data_types=camera_types,
            spawn=sim_utils.PinholeCameraCfg.from_intrinsic_matrix(
                intrinsic_matrix=fr3_default_side_cam["intrinsic_matrix"],
                height=_cam_h,
                width=_cam_w,
            ),
            offset=CameraCfg.OffsetCfg(
                pos=fr3_default_side_cam["pos"],
                rot=fr3_default_side_cam["rot"],
                convention="ros",
            ),
        )
        self.scene.wrist_cam = TiledCameraCfg(
            prim_path="{ENV_REGEX_NS}/fr3/urdf_01/fr3_hand/wrist_cam",
            height=_cam_h,
            width=_cam_w,
            data_types=camera_types,
            spawn=sim_utils.PinholeCameraCfg.from_intrinsic_matrix(
                intrinsic_matrix=fr3_default_wrist_cam["intrinsic_matrix"],
                height=_cam_h,
                width=_cam_w,
            ),
            offset=CameraCfg.OffsetCfg(
                pos=fr3_default_wrist_cam["pos"],
                rot=fr3_default_wrist_cam["rot"],
                convention="ros",
            ),
        )

        self.scene.ee_frame = FrameTransformerCfg(
            prim_path="{ENV_REGEX_NS}/fr3/urdf_01/fr3_link0",
            debug_vis=False,
            target_frames=[
                FrameTransformerCfg.FrameCfg(
                    prim_path="{ENV_REGEX_NS}/fr3/urdf_01/fr3_hand",
                    name="end_effector",
                    offset=OffsetCfg(pos=[0.0, 0.0, 0.1034]),
                ),
            ],
        )

        self.commands.object_pose.body_name = "fr3_hand"
        self.decimation = 10
        self.sim.dt = 1 / (60 * 2)
        self.sim.render_interval = 10
        self.set_scene("2")
