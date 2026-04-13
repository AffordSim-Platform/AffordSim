import isaaclab.sim as sim_utils
from isaaclab_tasks.real2sim.tasks.pick.pick_r2s_env_cfg import EnvCfg
from isaaclab_tasks.real2sim.tasks.pick import mdp
from isaaclab.sensors import FrameTransformerCfg, CameraCfg, TiledCameraCfg
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass
from isaaclab_tasks.real2sim.assets.franka import *
from isaaclab_tasks.real2sim.utils.camera_cfg import panda_front_cam, panda_wrist_cam, top_camera

ROBOT_CONFIGS = {
        "franka_fr3_default": FRANKA_FR3_CFG,
        "franka_fr3_ourgripper": FRANKA_FR3_OURGRIPPER_CFG,
        "franka_panda_ourgripper": FRANKA_PANDA_OURGRIPPER_CFG,}

@configclass
class FrankaPandaDefaultGripperIKRelVisPickEnvCfg(EnvCfg):
    def __post_init__(self):

        self.scene.robot = FRANKA_FR3_CFG

        self.actions.body = mdp.DifferentialInverseKinematicsActionCfg(
                delta_mode="previous",
                asset_name="robot",
                joint_names=["panda_joint.*"],
                body_name="panda_hand", 
                controller=mdp.DifferentialIKControllerCfg(
                    command_type="pose", 
                    use_relative_mode=True,
                    ik_method="dls",
                    ik_params={
                        "lambda_val": 0.1,
                    },
                ),
            )
        
        self.actions.gripper_action = mdp.BinaryJointPositionActionCfg(
            asset_name="robot",
            joint_names=["panda_finger.*"],
            open_command_expr={"panda_finger_.*": 0.04},
            close_command_expr={"panda_finger_.*": 0.0},
        )
        
        self.scene.external_cam_front = TiledCameraCfg(
        prim_path="{ENV_REGEX_NS}/external_cam_front",
        height=panda_front_cam["image_height"],
        width=panda_front_cam["image_width"],
        data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg.from_intrinsic_matrix(
            intrinsic_matrix=panda_front_cam["intrinsic_matrix"],
            height=panda_front_cam["image_height"],
            width=panda_front_cam["image_width"],
            ),
        offset=CameraCfg.OffsetCfg(
            pos=panda_front_cam["pos"],
            rot=panda_front_cam["rot"], convention="ros"
            ),
        )

        self.scene.wrist_cam = TiledCameraCfg(
            prim_path="{ENV_REGEX_NS}/fr3/urdf_01/fr3_hand/wrist_cam",
            height=panda_wrist_cam["image_height"],
            width=panda_wrist_cam["image_width"],
            data_types=["rgb"],
            spawn=sim_utils.PinholeCameraCfg.from_intrinsic_matrix(
                intrinsic_matrix=panda_wrist_cam["intrinsic_matrix"],
                height=panda_wrist_cam["image_height"],
                width=panda_wrist_cam["image_width"],
            ),
            offset=CameraCfg.OffsetCfg(
                pos=panda_wrist_cam["pos"],
                rot=panda_wrist_cam["rot"], convention="ros"
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
            offset=CameraCfg.OffsetCfg(pos=top_camera["pos"], rot=top_camera["rot"], convention="openl"),
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
            offset=CameraCfg.OffsetCfg(pos=top_camera["pos"], rot=top_camera["rot"], convention="openl"),
        )

        self.observations.policy.external_cam_front = ObsTerm(
                    func=mdp.image,
                    params={
                        "sensor_cfg": SceneEntityCfg("external_cam_front"),
                        "data_type": "rgb",
                        "normalize": False,
                        }
                    )
        
        self.observations.policy.wrist_cam = ObsTerm(
                    func=mdp.image,
                    params={
                        "sensor_cfg": SceneEntityCfg("wrist_cam"),
                        "data_type": "rgb",
                        "normalize": False,
                        }
                    )
        
        # Set command body_name (required for UniformPoseCommandCfg)
        self.commands.object_pose.body_name = "panda_hand"
        
        self.decimation = 50
        self.sim.dt = 1 / (60*2)
        self.sim.render_interval = 4
        
