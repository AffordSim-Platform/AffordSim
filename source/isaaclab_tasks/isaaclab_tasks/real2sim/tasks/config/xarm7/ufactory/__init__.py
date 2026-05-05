import gymnasium as gym

from isaaclab_tasks.real2sim.tasks import agents


gym.register(
    id="XArm7-UFactory-Joint-EE-VisPick-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.joint_ee_vis_env_cfg:XArm7UFactoryJointEEVisPickEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:LiftCubeCNNPPOBCRunnerCfg",
    },
)
