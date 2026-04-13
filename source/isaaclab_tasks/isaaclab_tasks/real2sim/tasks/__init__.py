import gymnasium as gym
from isaaclab_tasks.real2sim.tasks import agents

from isaaclab_tasks.real2sim.tasks.config.franka_fr3 import defaultgripper, ourgripper
from isaaclab_tasks.real2sim.tasks.config.franka_panda import defaultgripper as panda_defaultgripper

gym.register(
    id="Pick_Place_banana-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{ourgripper.__name__}.ik_rel_vis_env_cfg:FrankaFr3OurGripperIKRelVisPickEnvCfg",
        "robomimic_bc_cfg_entry_point": f"{agents.__name__}:robomimic/bc.json",
        "robomimic_bc_rnn_cfg_entry_point": f"{agents.__name__}:robomimic/bc_rnn.json",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:BasePPORunnerCfg",
    },
)

gym.register(
    id="Pick_cupwithhandle_Pour-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{ourgripper.__name__}.ik_rel_vis_env_cfg:FrankaFr3OurGripperIKRelVisPickEnvCfg",
        "robomimic_bc_cfg_entry_point": f"{agents.__name__}:robomimic/bc.json",
        "robomimic_bc_rnn_cfg_entry_point": f"{agents.__name__}:robomimic/bc_rnn.json",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:BasePPORunnerCfg",
    },
)


gym.register(
    id="Pick_basketwithhandle_Pour-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{ourgripper.__name__}.ik_rel_vis_env_cfg:FrankaFr3OurGripperIKRelVisPickEnvCfg",
        "robomimic_bc_cfg_entry_point": f"{agents.__name__}:robomimic/bc.json",
        "robomimic_bc_rnn_cfg_entry_point": f"{agents.__name__}:robomimic/bc_rnn.json",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:BasePPORunnerCfg",
    },
)


