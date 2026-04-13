import gymnasium as gym
from isaaclab_tasks.real2sim.tasks import agents


# gym.register(
#     id="FrankaFr3-OurGripper-IKRel-VisPick-v0",
#     entry_point="isaaclab.envs:ManagerBasedRLEnv",
#     disable_env_checker=True,
#     kwargs={
#         "env_cfg_entry_point": f"{__name__}.ik_rel_vis_env_cfg:FrankaFr3OurGripperIKRelVisPickEnvCfg",
#         "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:BasePPORunnerCfg",
#     },
# )

# gym.register(
#     id="FrankaFr3-OurGripper-Joint-VisPick-v0",
#     entry_point="isaaclab.envs:ManagerBasedRLEnv",
#     disable_env_checker=True,
#     kwargs={
#         "env_cfg_entry_point": f"{__name__}.joint_vis_env_cfg:FrankaFr3OurGripperJointVisPickEnvCfg",
#         "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:BasePPORunnerCfg",
#     },
# )

gym.register(
    id="FrankaFr3-OurGripper-IKRel-VisPick-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.ik_rel_vis_env_cfg:FrankaFr3OurGripperIKRelVisPickEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:LiftCubeCNNPPORunnerCfg",
    },
)

gym.register(
    id="FrankaFr3-OurGripper-Joint-VisPick-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.joint_vis_env_cfg:FrankaFr3OurGripperJointVisPickEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:LiftCubeCNNPPORunnerCfg",
    },
)

gym.register(
    id="FrankaFr3-OurGripper-IKRel-PPOBC-VisPick-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.ik_rel_vis_env_cfg:FrankaFr3OurGripperIKRelVisPickEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:LiftCubeCNNPPOBCRunnerCfg",
    },
)

gym.register(
    id="FrankaFr3-OurGripper-IKRel-EE-PPOBC-VisPick-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.ik_rel_ee_vis_env_cfg:FrankaFr3OurGripperIKRelEEVisPickEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:LiftCubeCNNPPOBCRunnerCfg",
    },
)