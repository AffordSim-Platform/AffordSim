# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

# This file defines the configuration classes for the Concurrent Teacher-Student (CTS)
# reinforcement learning algorithm, designed for use with Isaac Lab.

from __future__ import annotations

from dataclasses import MISSING
from isaaclab.utils import configclass

# Import RSL-RL base configurations so we can inherit or reference them
from isaaclab_rl.rsl_rl.rl_cfg import RslRlPpoAlgorithmCfg, RslRlOnPolicyRunnerCfg

#########################
# Policy configurations #
#########################

@configclass
class ViTRslRlPpoActorCriticCfg:
    """Configuration for the PPO actor-critic networks with IBRL ViT architecture."""

    class_name: str = "ActorCriticViT" 
    actor_hidden_dims: list[int] = [512, 256, 128]
    critic_hidden_dims: list[int] = [512, 256, 128]
    activation: str = "elu" 
    init_noise_std: float = 1.0
    noise_std_type: str = "scalar"
    state_dependent_std: bool = False

@configclass
class CNNRslRlPpoActorCriticCfg:
    """Configuration for the PPO actor-critic networks with IBRL CNN architecture."""

    class_name: str = "ActorCriticCNN" 
    actor_hidden_dims: list[int] = [512, 256, 128]
    critic_hidden_dims: list[int] = [512, 256, 128]
    activation: str = "elu" 
    init_noise_std: float = 1.0
    noise_std_type: str = "scalar"
    state_dependent_std: bool = False
