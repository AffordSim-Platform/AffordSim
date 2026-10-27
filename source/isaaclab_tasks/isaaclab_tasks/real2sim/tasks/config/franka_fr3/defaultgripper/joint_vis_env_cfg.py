import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg, RigidObjectCfg
from isaaclab.sensors import CameraCfg, FrameTransformerCfg, TiledCameraCfg
from isaaclab.sensors.frame_transformer.frame_transformer_cfg import OffsetCfg
from isaaclab.utils import configclass

from isaaclab_tasks.real2sim.assets.franka import FRANKA_FR3_CFG
from isaaclab_tasks.real2sim.tasks import mdp
from isaaclab_tasks.real2sim.tasks.config.franka_fr3.ourgripper.joint_vis_env_cfg import (
    FrankaFr3OurGripperJointVisPickEnvCfg,
)
from isaaclab_tasks.real2sim.tasks.pick_r2s_env_cfg import (
    _get_scene_usd_path,
    _get_sim_camera_resolution,
    _get_use_background_replace,
)
from isaaclab_tasks.real2sim.utils.camera_cfg import fr3_default_side_cam, fr3_default_wrist_cam

@configclass
class FrankaFr3DefaultGripperJointVisPickEnvCfg(FrankaFr3OurGripperJointVisPickEnvCfg):
    """FR3 with its stock fingers and calibrated 0.1034 m tool centre.

    Share the collection observations and required top cameras with the FR3
    collection configuration, then replace the robot, side/wrist cameras and
    tool frame with the stock-gripper settings.
    """
    def __post_init__(self):
        super().__post_init__()

        self.scene.robot = FRANKA_FR3_CFG.replace(prim_path="{ENV_REGEX_NS}/fr3")

        self.actions.body = mdp.JointPositionActionCfg(
            asset_name="robot",
            joint_names=["fr3_joint.*"],
            preserve_order=True,
            use_default_offset=False,
        )
        self.actions.gripper_action = mdp.BinaryJointPositionActionCfg(
            asset_name="robot",
            joint_names=["fr3_finger.*"],
            open_command_expr={"fr3_finger_.*": 0.04},
            close_command_expr={"fr3_finger_.*": 0.0},
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
            prim_path="{ENV_REGEX_NS}/fr3/fr3_hand/wrist_cam",
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
            prim_path="{ENV_REGEX_NS}/fr3/fr3_link0",
            debug_vis=False,
            target_frames=[
                FrameTransformerCfg.FrameCfg(
                    prim_path="{ENV_REGEX_NS}/fr3/fr3_hand",
                    name="end_effector",
                    offset=OffsetCfg(pos=[0.0, 0.0, 0.1034]),
                ),
            ],
        )

        self.commands.object_pose.body_name = "fr3_hand"
        self.decimation = 10
        self.sim.dt = 1 / (60 * 2)
        self.sim.render_interval = 10
