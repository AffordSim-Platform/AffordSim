# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlPpoActorCriticCfg, RslRlPpoAlgorithmCfg, RslRlBCOnPolicyRunnerCfg
from isaaclab_rl.rsl_rl.ViT_BC.cfg.cfg import ViTRslRlPpoActorCriticCfg, CNNRslRlPpoActorCriticCfg 

@configclass
class BasePPORunnerCfg(RslRlOnPolicyRunnerCfg):
    num_steps_per_env = 24
    max_iterations = 50000
    save_interval = 500
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

# @configclass
# class BasePPORunnerCfg(RslRlOnPolicyRunnerCfg):
#     num_steps_per_env = 24
#     max_iterations = 50000
#     save_interval = 500
#     experiment_name = ""  # same as task name
#     policy = RslRlPpoActorCriticCfg(
#         init_noise_std=1.0,
#         actor_hidden_dims=[256, 128, 64],
#         critic_hidden_dims=[256, 128, 64],
#         activation="elu",
#     )
#     algorithm = RslRlPpoAlgorithmCfg(
#         value_loss_coef=1.0,
#         use_clipped_value_loss=True,
#         clip_param=0.2,
#         entropy_coef=0.01,
#         num_learning_epochs=5,
#         num_mini_batches=4,
#         learning_rate=1.0e-3,
#         schedule="adaptive",
#         gamma=0.99,
#         lam=0.95,
#         desired_kl=0.01,
#         max_grad_norm=1.0,
#     )

@configclass
class LiftCubeCNNPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    class_name: str = "OnPolicyRunnerViT"
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

@configclass
class LiftCubeCNNPPOBCRunnerCfg(RslRlBCOnPolicyRunnerCfg):
    class_name: str = "OnPolicyRunnerViTBC"
    num_steps_per_env = 24
    max_iterations = 50000
    save_interval = 100
    experiment_name = ""  # same as task name
    # bc_data_path = "/data/limingyang_data/r2s2r_data/pick up cube/20251217_020343"
    bc_data_path = "/data/limingyang_data/r2s2r_data/pick up cube/20251223_221241" #20251225_233403 20251223_221241
    policy = CNNRslRlPpoActorCriticCfg(
        init_noise_std=1.0, #0.6
        actor_hidden_dims=[1024,512,256],
        critic_hidden_dims=[1024,512,256],
        activation="elu",
        noise_std_type="scalar",
        state_dependent_std=False,
    )
    algorithm = RslRlPpoAlgorithmCfg(
        class_name="PPOBC",
        value_loss_coef=1.0,
        # bc_coef=1.0,
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
        # bc_batch_size=32,
    )