import isaaclab.sim as sim_utils
from isaaclab_tasks.real2sim.tasks.pick.pick_r2s_env_cfg import EnvCfg, _get_sim_camera_resolution, _get_use_background_replace
from isaaclab_tasks.real2sim.tasks.pick import mdp
from isaaclab.sensors import FrameTransformerCfg, CameraCfg, TiledCameraCfg
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.sensors.frame_transformer.frame_transformer_cfg import OffsetCfg
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import SceneEntityCfg
from isaaclab.assets import AssetBaseCfg, RigidObjectCfg
from isaaclab.utils import configclass
from isaaclab_tasks.real2sim.assets.ur5e import UR5E_ROBOTIQ85_CFG
from isaaclab_tasks.real2sim.utils.camera_cfg import top_camera, ur5e_side_cam, ur5e_wrist_cam

# UR5e arm joint names
_UR5E_ARM_JOINTS = [
    "shoulder_pan_joint", "shoulder_lift_joint", "elbow_joint",
    "wrist_1_joint", "wrist_2_joint", "wrist_3_joint",
]

@configclass
class UR5eRobotiq85IKRelEEVisPickEnvCfg(EnvCfg):

    @configclass
    class ObservationCfg:
        @configclass
        class PolicyCfg(ObsGroup):
            """Observations for policy - UR5e with Robotiq-85"""
            arm_ee_pose = ObsTerm(
                func=mdp.arm_ee_pose,
                params={"body_name": "wrist_3_link"},
            )  # 7
            gripper_pos = ObsTerm(
                func=mdp.gripper_pos,
                params={"joint_names_pattern": ["finger_joint"]},
            )  # 1
            arm_ee_vel = ObsTerm(
                func=mdp.arm_ee_vel,
                params={"body_name": "wrist_3_link"},
            )  # 6

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

        # --- Robot ---
        self.scene.robot = UR5E_ROBOTIQ85_CFG.replace(prim_path="{ENV_REGEX_NS}/ur5e")

        # --- Arm joint position action (for data collection via CuRobo) ---
        self.actions.body = mdp.JointPositionActionCfg(
            asset_name="robot",
            joint_names=_UR5E_ARM_JOINTS,
            preserve_order=True,
            use_default_offset=False,
        )

        # --- Gripper action (Robotiq 2F-85) ---
        # finger_joint range: [0.0 (open), 0.8 (closed)] rad
        self.actions.gripper_action = mdp.BinaryJointPositionActionCfg(
            asset_name="robot",
            joint_names=["finger_joint"],
            open_command_expr={"finger_joint": 0.0},
            close_command_expr={"finger_joint": 0.8},
        )

        # --- EE frame transformer ---
        # USD root prim is also named 'ur5e', so actual path is ur5e/ur5e/...
        # Robotiq 2F-85 TCP is approximately 0.17 m from wrist_3_link
        self.scene.ee_frame = FrameTransformerCfg(
            prim_path="{ENV_REGEX_NS}/ur5e/ur5e/base_link",
            debug_vis=False,
            target_frames=[
                FrameTransformerCfg.FrameCfg(
                    prim_path="{ENV_REGEX_NS}/ur5e/ur5e/wrist_3_link",
                    name="end_effector",
                    offset=OffsetCfg(
                        pos=[0.0, 0.0, 0.17],   # gripper TCP offset from wrist_3_link
                    ),
                ),
            ],
        )

        # --- Cameras ---
        _cam_w, _cam_h = _get_sim_camera_resolution()
        if _get_use_background_replace():
            self.scene.external_cam_side = TiledCameraCfg(
                prim_path="{ENV_REGEX_NS}/external_cam_side",
                height=_cam_h,
                width=_cam_w,
                data_types=["rgb", "semantic_segmentation"],
                spawn=sim_utils.PinholeCameraCfg.from_intrinsic_matrix(
                    intrinsic_matrix=ur5e_side_cam["intrinsic_matrix"],
                    height=_cam_h,
                    width=_cam_w,
                ),
                offset=CameraCfg.OffsetCfg(
                    pos=ur5e_side_cam["pos"],
                    rot=ur5e_side_cam["rot"], convention="ros"
                ),
            )

            self.scene.wrist_cam = TiledCameraCfg(
                prim_path="{ENV_REGEX_NS}/ur5e/ur5e/wrist_3_link/wrist_cam",
                height=_cam_h,
                width=_cam_w,
                data_types=["rgb", "semantic_segmentation"],
                spawn=sim_utils.PinholeCameraCfg.from_intrinsic_matrix(
                    intrinsic_matrix=ur5e_wrist_cam["intrinsic_matrix"],
                    height=_cam_h,
                    width=_cam_w,
                ),
                offset=CameraCfg.OffsetCfg(
                    pos=ur5e_wrist_cam["pos"],
                    rot=ur5e_wrist_cam["rot"], convention="ros"
                ),
            )
        else:
            self.scene.external_cam_side = TiledCameraCfg(
                prim_path="{ENV_REGEX_NS}/external_cam_side",
                height=_cam_h,
                width=_cam_w,
                data_types=["rgb", "distance_to_image_plane"],
                spawn=sim_utils.PinholeCameraCfg.from_intrinsic_matrix(
                    intrinsic_matrix=ur5e_side_cam["intrinsic_matrix"],
                    height=_cam_h,
                    width=_cam_w,
                ),
                offset=CameraCfg.OffsetCfg(
                    pos=ur5e_side_cam["pos"],
                    rot=ur5e_side_cam["rot"], convention="ros"
                ),
            )

            self.scene.wrist_cam = TiledCameraCfg(
                prim_path="{ENV_REGEX_NS}/ur5e/ur5e/wrist_3_link/wrist_cam",
                height=_cam_h,
                width=_cam_w,
                data_types=["rgb", "distance_to_image_plane"],
                spawn=sim_utils.PinholeCameraCfg.from_intrinsic_matrix(
                    intrinsic_matrix=ur5e_wrist_cam["intrinsic_matrix"],
                    height=_cam_h,
                    width=_cam_w,
                ),
                offset=CameraCfg.OffsetCfg(
                    pos=ur5e_wrist_cam["pos"],
                    rot=ur5e_wrist_cam["rot"], convention="ros"
                ),
            )

        # --- Table (legacy UR5e cabinet) ---
        # self.scene.table = AssetBaseCfg(
        #     prim_path="{ENV_REGEX_NS}/table",
        #     spawn=sim_utils.UsdFileCfg(
        #         usd_path="/home/hk/xuhaofan/Real2Sim2Real-Pipeline-auto_task_aff/models/cabinet.usd",
        #     ),
        #     init_state=AssetBaseCfg.InitialStateCfg(
        #         pos=(0, 0, 0),
        #         rot=(1.0, 0.0, 0.0, 0.0),
        #     ),
        # )

        # --- Table (aligned with Franka configs) ---
        self.scene.table = AssetBaseCfg(
            prim_path="{ENV_REGEX_NS}/table",
            spawn=sim_utils.UsdFileCfg(
                usd_path="/home/hk/xuhaofan/Real2Sim2Real-Pipeline-auto_task_aff/models/table.usd",
            ),
            init_state=AssetBaseCfg.InitialStateCfg(
                pos=(0, 0, 0),
                rot=(1.0, 0.0, 0.0, 0.0),
            ),
        )

        # --- top_rgb_camera / top_depth_camera (required by SceneCfg base) ---
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
                convention="openl"
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
                convention="openl"
            ),
        )

        # --- Commands: use wrist_3_link as reference body ---
        self.commands.object_pose.body_name = "wrist_3_link"

        self.ee_pose_obs = True
        self.decimation = 10
        self.sim.dt = 1 / (60 * 2)
        self.sim.render_interval = 10
        self.object_initialization_mode = "random"
