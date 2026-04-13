from pathlib import Path

import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg, RigidObjectCfg
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import CameraCfg, FrameTransformerCfg, TiledCameraCfg
from isaaclab.sensors.frame_transformer.frame_transformer_cfg import OffsetCfg
from isaaclab.utils import configclass

from isaaclab_tasks.real2sim.assets.franka import FRANKA_FR3_OURGRIPPER_CFG, FRANKA_FR3_CFG
from isaaclab_tasks.real2sim.tasks import mdp
from isaaclab_tasks.real2sim.tasks.pick_r2s_env_cfg import (
    EnvCfg,
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


@configclass
class FrankaFr3OurGripperJointVisPickEnvCfg(EnvCfg):
    @configclass
    class ObservationCfg:
        @configclass
        class PolicyCfg(ObsGroup):
            arm_ee_pose = ObsTerm(func=mdp.arm_ee_pose)
            # gripper_pos = ObsTerm(func=mdp.gripper_pos)
            # arm_ee_vel = ObsTerm(func=mdp.arm_ee_vel)
            q_pos = ObsTerm(func=mdp.arm_joint_pos)  # 7
            q_vel = ObsTerm(func=mdp.arm_joint_vel)  # 7

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
                self.enable_corruption = False
                self.concatenate_terms = False
                if not _get_use_background_replace():
                    self.semantic_segmentation_side = None
                    self.semantic_segmentation_wrist = None

        policy: PolicyCfg = PolicyCfg()

    observations: ObservationCfg = ObservationCfg()

    def __post_init__(self):
        super().__post_init__()

        self.scene.robot = FRANKA_FR3_OURGRIPPER_CFG.replace(prim_path="{ENV_REGEX_NS}/fr3")

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
                usd_path=str(PROJECT_ROOT / "models/table.usd"),
            ),
            init_state=RigidObjectCfg.InitialStateCfg(
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
                prim_path="{ENV_REGEX_NS}/fr3/urdf_01/fr3_hand/wrist_cam",
                # prim_path="{ENV_REGEX_NS}/fr3/fr3_hand/wrist_cam",
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
                prim_path="{ENV_REGEX_NS}/fr3/urdf_01/fr3_hand/wrist_cam",
                # prim_path="{ENV_REGEX_NS}/fr3/fr3_hand/wrist_cam",
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

        self.scene.ee_frame = FrameTransformerCfg(
            prim_path="{ENV_REGEX_NS}/fr3/urdf_01/fr3_link0",
            # prim_path="{ENV_REGEX_NS}/fr3/fr3_link0",
            debug_vis=False,
            target_frames=[
                FrameTransformerCfg.FrameCfg(
                    prim_path="{ENV_REGEX_NS}/fr3/urdf_01/fr3_hand",
                    # prim_path="{ENV_REGEX_NS}/fr3/fr3_hand",
                    name="end_effector",
                    offset=OffsetCfg(pos=[0.0, 0.0, 0.182]),
                ),
            ],
        )

        self.commands.object_pose.body_name = "fr3_hand"
        self.episode_length_s = 50
        self.decimation = 20
        self.sim.dt = 1 / (60 * 2)
        self.sim.render_interval = 10
