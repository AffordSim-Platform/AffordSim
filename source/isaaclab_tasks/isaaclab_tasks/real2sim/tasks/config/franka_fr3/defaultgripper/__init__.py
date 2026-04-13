import gymnasium as gym
from isaaclab_tasks.real2sim.tasks import agents


gym.register(
    id="FrankaFr3-DefaultGripper-IKRel-VisPick-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.ik_rel_vis_env_cfg:FrankaFr3DefaultGripperIKRelVisPickEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:LiftCubeCNNPPORunnerCfg",
    },
)

gym.register(
    id="FrankaFr3-DefaultGripper-Joint-VisPick-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.joint_vis_env_cfg:FrankaFr3DefaultGripperJointVisPickEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:LiftCubeCNNPPORunnerCfg",
    },
)

gym.register(
    id="FrankaFr3-DefaultGripper-IKRel-PPOBC-VisPick-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.ik_rel_vis_env_cfg:FrankaFr3DefaultGripperIKRelVisPickEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:LiftCubeCNNPPOBCRunnerCfg",
    },
)