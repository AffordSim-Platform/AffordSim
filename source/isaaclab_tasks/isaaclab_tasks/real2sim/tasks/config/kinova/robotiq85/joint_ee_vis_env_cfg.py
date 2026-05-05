from pathlib import Path

import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import CameraCfg, FrameTransformerCfg, TiledCameraCfg
from isaaclab.sensors.frame_transformer.frame_transformer_cfg import OffsetCfg
from isaaclab.utils import configclass

from isaaclab_tasks.real2sim.assets.kinova import KINOVA_ROBOTIQ85_CFG
from isaaclab_tasks.real2sim.tasks import mdp
from isaaclab_tasks.real2sim.tasks.pick_r2s_env_cfg import (
    EnvCfg,
    _get_scene_usd_path,
    _get_sim_camera_resolution,
    _get_use_background_replace,
)
from isaaclab_tasks.real2sim.utils.camera_cfg import realsense_435_side_cam, realsense_435_wrist_cam, top_camera

_current = Path(__file__).resolve()
PROJECT_ROOT = _current
while PROJECT_ROOT != PROJECT_ROOT.parent:
    if (PROJECT_ROOT / "models").is_dir():
        break
    PROJECT_ROOT = PROJECT_ROOT.parent

_KINOVA_ARM_JOINTS = [
    "joint_1",
    "joint_2",
    "joint_3",
    "joint_4",
    "joint_5",
    "joint_6",
    "joint_7",
]


@configclass
class KinovaRobotiq85JointEEVisPickEnvCfg(EnvCfg):

    @configclass
    class ObservationCfg:
        @configclass
        class PolicyCfg(ObsGroup):
            arm_ee_pose = ObsTerm(
                func=mdp.arm_ee_pose,
                params={"body_name": "end_effector_link"},
            )
            gripper_pos = ObsTerm(
                func=mdp.gripper_pos,
                params={"joint_names_pattern": ["finger_joint"]},
            )
            arm_ee_vel = ObsTerm(
                func=mdp.arm_ee_vel,
                params={"body_name": "end_effector_link"},
            )

            semantic_segmentation_side = ObsTerm(
                func=mdp.image,
                params={
                    "sensor_cfg": SceneEntityCfg("external_cam_side"),
                    "data_type": "semantic_segmentation",
                    "normalize": False,
                },
            )
            semantic_segmentation_wrist = ObsTerm(
                func=mdp.image,
                params={
                    "sensor_cfg": SceneEntityCfg("wrist_cam"),
                    "data_type": "semantic_segmentation",
                    "normalize": False,
                },
            )

            external_cam_side = ObsTerm(
                func=mdp.image,
                params={
                    "sensor_cfg": SceneEntityCfg("external_cam_side"),
                    "data_type": "rgb",
                    "normalize": True,
                },
            )
            wrist_cam = ObsTerm(
                func=mdp.image,
                params={
                    "sensor_cfg": SceneEntityCfg("wrist_cam"),
                    "data_type": "rgb",
                    "normalize": True,
                },
            )

            def __post_init__(self) -> None:
                self.enable_corruption = True
                self.concatenate_terms = False
                if not _get_use_background_replace():
                    self.semantic_segmentation_side = None
                    self.semantic_segmentation_wrist = None

        policy: PolicyCfg = PolicyCfg()

    observations: ObservationCfg = ObservationCfg()

    def __post_init__(self):
        super().__post_init__()

        self.scene.robot = KINOVA_ROBOTIQ85_CFG.replace(prim_path="{ENV_REGEX_NS}/kinova")

        self.actions.body = mdp.JointPositionActionCfg(
            asset_name="robot",
            joint_names=_KINOVA_ARM_JOINTS,
            preserve_order=True,
            use_default_offset=False,
        )

        self.actions.gripper_action = mdp.BinaryJointPositionActionCfg(
            asset_name="robot",
            joint_names=["finger_joint"],
            open_command_expr={"finger_joint": 0.0},
            close_command_expr={"finger_joint": 0.8},
        )

        # TCP is approximated as 0.137 m ahead of end_effector_link.
        self.scene.ee_frame = FrameTransformerCfg(
            prim_path="{ENV_REGEX_NS}/kinova/gen3n7_instanceable/base_link",
            debug_vis=False,
            target_frames=[
                FrameTransformerCfg.FrameCfg(
                    prim_path="{ENV_REGEX_NS}/kinova/gen3n7_instanceable/end_effector_link",
                    name="end_effector",
                    offset=OffsetCfg(
                        pos=[0.0, 0.0, 0.137],
                    ),
                ),
            ],
        )

        _cam_w, _cam_h = _get_sim_camera_resolution()
        if _get_use_background_replace():
            self.scene.external_cam_side = TiledCameraCfg(
                prim_path="{ENV_REGEX_NS}/external_cam_side",
                height=_cam_h,
                width=_cam_w,
                data_types=["rgb", "semantic_segmentation"],
                spawn=sim_utils.PinholeCameraCfg.from_intrinsic_matrix(
                    intrinsic_matrix=realsense_435_side_cam["intrinsic_matrix"],
                    height=_cam_h,
                    width=_cam_w,
                ),
                offset=CameraCfg.OffsetCfg(
                    pos=realsense_435_side_cam["pos"],
                    rot=realsense_435_side_cam["rot"],
                    convention="ros",
                ),
            )

            self.scene.wrist_cam = TiledCameraCfg(
                prim_path="{ENV_REGEX_NS}/kinova/gen3n7_instanceable/end_effector_link/wrist_cam",
                height=_cam_h,
                width=_cam_w,
                data_types=["rgb", "semantic_segmentation"],
                spawn=sim_utils.PinholeCameraCfg.from_intrinsic_matrix(
                    intrinsic_matrix=realsense_435_wrist_cam["intrinsic_matrix"],
                    height=_cam_h,
                    width=_cam_w,
                ),
                offset=CameraCfg.OffsetCfg(
                    pos=realsense_435_wrist_cam["pos"],
                    rot=realsense_435_wrist_cam["rot"],
                    convention="ros",
                ),
            )
        else:
            self.scene.external_cam_side = TiledCameraCfg(
                prim_path="{ENV_REGEX_NS}/external_cam_side",
                height=_cam_h,
                width=_cam_w,
                data_types=["rgb", "distance_to_image_plane"],
                spawn=sim_utils.PinholeCameraCfg.from_intrinsic_matrix(
                    intrinsic_matrix=realsense_435_side_cam["intrinsic_matrix"],
                    height=_cam_h,
                    width=_cam_w,
                ),
                offset=CameraCfg.OffsetCfg(
                    pos=realsense_435_side_cam["pos"],
                    rot=realsense_435_side_cam["rot"],
                    convention="ros",
                ),
            )

            self.scene.wrist_cam = TiledCameraCfg(
                prim_path="{ENV_REGEX_NS}/kinova/gen3n7_instanceable/end_effector_link/wrist_cam",
                height=_cam_h,
                width=_cam_w,
                data_types=["rgb", "distance_to_image_plane"],
                spawn=sim_utils.PinholeCameraCfg.from_intrinsic_matrix(
                    intrinsic_matrix=realsense_435_wrist_cam["intrinsic_matrix"],
                    height=_cam_h,
                    width=_cam_w,
                ),
                offset=CameraCfg.OffsetCfg(
                    pos=realsense_435_wrist_cam["pos"],
                    rot=realsense_435_wrist_cam["rot"],
                    convention="ros",
                ),
            )

        self.scene.table = AssetBaseCfg(
            prim_path="{ENV_REGEX_NS}/table",
            spawn=sim_utils.UsdFileCfg(
                usd_path=_get_scene_usd_path(),
            ),
            init_state=AssetBaseCfg.InitialStateCfg(
                pos=(0, 0, 0),
                rot=(1.0, 0.0, 0.0, 0.0),
            ),
        )

        self.scene.top_rgb_camera = TiledCameraCfg(
            prim_path="{ENV_REGEX_NS}/front_rgb_camera",
            height=top_camera["image_height"],
            width=top_camera["image_width"],
            data_types=["rgb"],
            spawn=sim_utils.PinholeCameraCfg.from_intrinsic_matrix(
                intrinsic_matrix=top_camera["intrinsic_matrix"],
                height=top_camera["image_height"],
                width=top_camera["image_width"],
            ),
            offset=CameraCfg.OffsetCfg(
                pos=top_camera["pos"],
                rot=top_camera["rot"],
                convention="openl",
            ),
        )
        self.scene.top_depth_camera = TiledCameraCfg(
            prim_path="{ENV_REGEX_NS}/front_depth_camera",
            height=top_camera["image_height"],
            width=top_camera["image_width"],
            data_types=["depth"],
            spawn=sim_utils.PinholeCameraCfg.from_intrinsic_matrix(
                intrinsic_matrix=top_camera["intrinsic_matrix"],
                height=top_camera["image_height"],
                width=top_camera["image_width"],
            ),
            offset=CameraCfg.OffsetCfg(
                pos=top_camera["pos"],
                rot=top_camera["rot"],
                convention="openl",
            ),
        )

        self.commands.object_pose.body_name = "end_effector_link"

        self.ee_pose_obs = True
        self.decimation = 10
        self.sim.dt = 1 / (60 * 2)
        self.sim.render_interval = 10
        self.object_initialization_mode = "random"
