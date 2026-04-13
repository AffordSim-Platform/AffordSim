# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.utils import configclass

from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlPpoActorCriticCfg, RslRlPpoAlgorithmCfg
from isaaclab_rl.rsl_rl.ViT_BC.cfg.cfg import ViTRslRlPpoActorCriticCfg, CNNRslRlPpoActorCriticCfg 

@configclass
class LiftCubePPORunnerCfg(RslRlOnPolicyRunnerCfg):
    num_steps_per_env = 24
    max_iterations = 1500
    save_interval = 50
    experiment_name = "franka_lift"
    policy = RslRlPpoActorCriticCfg(
        init_noise_std=1.0,
        actor_obs_normalization=False,
        critic_obs_normalization=False,
        actor_hidden_dims=[256, 128, 64],
        critic_hidden_dims=[256, 128, 64],
        activation="elu",
    )
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.006,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-4,
        schedule="adaptive",
        gamma=0.98,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )

@configclass
class LiftCubeViTPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    class_name: str = "OnPolicyRunnerViTBC"
    num_steps_per_env = 24
    max_iterations = 50000
    save_interval = 100
    experiment_name = ""  # same as task name
    policy = ViTRslRlPpoActorCriticCfg(
        init_noise_std=1.0,
        actor_hidden_dims=[512, 256, 128],
        critic_hidden_dims=[512, 256, 128],
        activation="elu",
    )
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.01,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-3,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )

@configclass
class LiftCubeCNNPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    class_name: str = "OnPolicyRunnerViTBC"
    num_steps_per_env = 24
    max_iterations = 50000
    save_interval = 100
    experiment_name = ""  # same as task name
    policy = CNNRslRlPpoActorCriticCfg(
        init_noise_std=1.0,
        actor_hidden_dims=[1024,512,256],
        critic_hidden_dims=[1024,512,256],
        activation="elu",
        noise_std_type="log",  # Changed from scalar to log for better numerical stability
        state_dependent_std=False,
    )
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.01,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1e-3,  # Reduced from 1e-3 for better stability with vision
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )