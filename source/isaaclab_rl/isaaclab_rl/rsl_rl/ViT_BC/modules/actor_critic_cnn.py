# Copyright (c) 2021-2025, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
import torch.nn as nn
from tensordict import TensorDict
from torch.distributions import Normal
from typing import Any, NoReturn

from rsl_rl.networks import MLP, EmpiricalNormalization
from PIL import Image

USE_BACKGROUND_REPLACE = True

class NatureCNN(nn.Module):
    def __init__(self, sample_obs):
        super().__init__()

        extractors = {}

        self.out_features = 0
        self.img_feature_size = 256
        self.state_feature_size = 64
        self.img_n = 2
        self.img_c = 3
        self.img_h = 128
        self.img_w = 128
        self.img_flattened_size = self.img_c * self.img_h * self.img_w
        dict_obs = {}
        dict_obs["external_cam_side"] = sample_obs[:, -self.img_flattened_size:]
        dict_obs["wrist_cam"] = sample_obs[:, -2 * self.img_flattened_size : -self.img_flattened_size]
        dict_obs["state"] = sample_obs[:, : -2 * self.img_flattened_size]
        # dict_obs["state"] = sample_obs[:, : -2 * (self.img_flattened_size + self.img_segmentation_size)] if USE_BACKGROUND_REPLACE else sample_obs[:, : -(2 * self.img_flattened_size)]
        flattened_size = dict_obs["external_cam_side"].shape[-1]
        self.rgb_shape = (self.img_c, self.img_h, self.img_w)
        print(f"Inferred RGB image shape: {self.rgb_shape} from flattened size {flattened_size}")

        # here we use a NatureCNN architecture to process images, but any architecture is permissble here
        cnn = nn.Sequential(
            nn.Conv2d(
                in_channels=self.img_c,
                out_channels=32,
                kernel_size=8,
                stride=4,
                padding=0,
            ),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2, padding=0),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=0),
            nn.ReLU(),
            nn.Flatten(),
        )
        # to easily figure out the dimensions after flattening, we pass a test tensor
        with torch.no_grad():
            # Reshape flattened image back to (B, C, H, W) for dimension inference
            test_img = dict_obs["external_cam_side"].float().reshape(-1, *self.rgb_shape).cpu()
            n_flatten = cnn(test_img).shape[1]
            fc = nn.Sequential(nn.Linear(n_flatten, self.img_feature_size), nn.ReLU())
        extractors["camera"] = nn.Sequential(cnn, fc)
        self.out_features += self.img_feature_size * self.img_n
        state_dim = dict_obs["state"].shape[-1]
        print(f"CNN Encoder: State dimension {state_dim}  to  compressed to {self.state_feature_size}")
        
        extractors["state"] = nn.Linear(state_dim, self.state_feature_size)
        self.out_features += self.state_feature_size

        self.extractors = nn.ModuleDict(extractors)
        self.counter = 0

    def forward(self, observations) -> torch.Tensor:
        batch_size = observations.shape[0]
        dict_obs = {}
        dict_obs["wrist_cam"] = observations[:, -self.img_flattened_size:]
        dict_obs["external_cam_side"] = observations[:, -2 * self.img_flattened_size : -self.img_flattened_size]
        dict_obs["state"] =  observations[:, : -2 * self.img_flattened_size]
        # dict_obs["state"] =  observations[:, : -2 * (self.img_flattened_size + self.img_segmentation_size)] if USE_BACKGROUND_REPLACE else observations[:, : -(2 * self.img_flattened_size)]        
        # print(f"NatureCNN forward: observations shape {observations.shape}, wrist_cam shape {dict_obs['wrist_cam'].shape}, external_cam_front shape {dict_obs['external_cam_front'].shape}, state shape {dict_obs['state'].shape}")
        # if self.counter % 100 == 0:
        #     Image.fromarray((dict_obs["wrist_cam"][0] * 255).reshape(*self.rgb_shape).cpu().numpy().transpose(1, 2, 0).astype('uint8')).save(f'code/IsaacLab/logs/debug/wrist/debug_obs_{self.counter}.png')
        #     Image.fromarray((dict_obs["external_cam_side"][0] * 255).reshape(*self.rgb_shape).cpu().numpy().transpose(1, 2, 0).astype('uint8')).save(f'code/IsaacLab/logs/debug/font/debug_obs_{self.counter}.png')
        self.counter += 1
        
        state_lat = self.extractors["state"](dict_obs["state"])
        external_cam_lat = self.extractors["camera"](dict_obs["external_cam_side"].reshape(batch_size, *self.rgb_shape))
        wrist_cam_lat = self.extractors["camera"](dict_obs["wrist_cam"].reshape(batch_size, *self.rgb_shape))
        encoded_tensor_list = [state_lat, external_cam_lat, wrist_cam_lat]
        return torch.cat(encoded_tensor_list, dim=1)

class ActorCriticCNN(nn.Module):
    is_recurrent: bool = False

    def __init__(
        self,
        obs: TensorDict,
        obs_groups: dict[str, list[str]],
        num_actions: int,
        actor_obs_normalization: bool = False,
        critic_obs_normalization: bool = False,
        actor_hidden_dims: tuple[int] | list[int] = [256, 256, 256],
        critic_hidden_dims: tuple[int] | list[int] = [256, 256, 256],
        activation: str = "elu",
        init_noise_std: float = 1.0,
        noise_std_type: str = "scalar",
        state_dependent_std: bool = False,
        **kwargs: dict[str, Any],
    ) -> None:
        if kwargs:
            print(
                "ActorCritic.__init__ got unexpected arguments, which will be ignored: " + str([key for key in kwargs])
            )
        super().__init__()

        # Get the observation dimensions
        self.obs_groups = obs_groups
        
        # Initialize CNN feature extractor if rgb/state observations exist
        self.cnn_encoder = NatureCNN(obs['policy'][:, :98322]) # TODO
        cnn_feature_dim = self.cnn_encoder.out_features
        print(f"CNN Encoder initialized with output features: {cnn_feature_dim}")
        
        # Calculate dimensions for actor and critic
        num_actor_obs = cnn_feature_dim
        num_critic_obs = cnn_feature_dim

        self.state_dependent_std = state_dependent_std

        # Actor
        if self.state_dependent_std:
            self.actor = MLP(num_actor_obs, [2, num_actions], actor_hidden_dims, activation)
        else:
            self.actor = MLP(num_actor_obs, num_actions, actor_hidden_dims, activation)
        # Add tanh activation to the last layer of the actor
        self.actor.append(nn.Tanh())
        print(f"Actor MLP: {self.actor}")

        # Actor observation normalization
        self.actor_obs_normalization = actor_obs_normalization
        if actor_obs_normalization:
            self.actor_obs_normalizer = EmpiricalNormalization(num_actor_obs)
        else:
            self.actor_obs_normalizer = torch.nn.Identity()

        # Critic
        self.critic = MLP(num_critic_obs, 1, critic_hidden_dims, activation)
        print(f"Critic MLP: {self.critic}")

        # Critic observation normalization
        self.critic_obs_normalization = critic_obs_normalization
        if critic_obs_normalization:
            self.critic_obs_normalizer = EmpiricalNormalization(num_critic_obs)
        else:
            self.critic_obs_normalizer = torch.nn.Identity()

        # Action noise
        self.noise_std_type = noise_std_type
        if self.state_dependent_std:
            torch.nn.init.zeros_(self.actor[-2].weight[num_actions:])
            if self.noise_std_type == "scalar":
                torch.nn.init.constant_(self.actor[-2].bias[num_actions:], init_noise_std)
            elif self.noise_std_type == "log":
                torch.nn.init.constant_(
                    self.actor[-2].bias[num_actions:], torch.log(torch.tensor(init_noise_std + 1e-7))
                )
            else:
                raise ValueError(f"Unknown standard deviation type: {self.noise_std_type}. Should be 'scalar' or 'log'")
        else:
            if self.noise_std_type == "scalar":
                self.std = nn.Parameter(init_noise_std * torch.ones(num_actions))
            elif self.noise_std_type == "log":
                self.log_std = nn.Parameter(torch.log(init_noise_std * torch.ones(num_actions)))
            else:
                raise ValueError(f"Unknown standard deviation type: {self.noise_std_type}. Should be 'scalar' or 'log'")

        # Action distribution
        # Note: Populated in update_distribution
        self.distribution = None

        # Disable args validation for speedup
        Normal.set_default_validate_args(False)

    def reset(self, dones: torch.Tensor | None = None) -> None:
        pass

    def forward(self) -> NoReturn:
        raise NotImplementedError

    @property
    def action_mean(self) -> torch.Tensor:
        return self.distribution.mean

    @property
    def action_std(self) -> torch.Tensor:
        return self.distribution.stddev

    @property
    def entropy(self) -> torch.Tensor:
        return self.distribution.entropy().sum(dim=-1)

    def _update_distribution(self, obs: TensorDict) -> None:
        if self.state_dependent_std:
            # Compute mean and standard deviation
            mean_and_std = self.actor(obs)
            if self.noise_std_type == "scalar":
                mean, std = torch.unbind(mean_and_std, dim=-2)
            elif self.noise_std_type == "log":
                mean, log_std = torch.unbind(mean_and_std, dim=-2)
                std = torch.exp(log_std)
            else:
                raise ValueError(f"Unknown standard deviation type: {self.noise_std_type}. Should be 'scalar' or 'log'")
        else:
            # Compute mean
            mean = self.actor(obs)
            # Compute standard deviation
            if self.noise_std_type == "scalar":
                std = self.std.expand_as(mean)
            elif self.noise_std_type == "log":
                std = torch.exp(self.log_std).expand_as(mean)
            else:
                raise ValueError(f"Unknown standard deviation type: {self.noise_std_type}. Should be 'scalar' or 'log'")
        # Create distribution
        self.distribution = Normal(mean, std)

    def act(self, obs: TensorDict, **kwargs: dict[str, Any]) -> torch.Tensor:
        obs = self.get_actor_obs(obs)
        obs = self.actor_obs_normalizer(obs)
        self._update_distribution(obs)
        return self.distribution.sample()

    def act_inference(self, obs: TensorDict) -> torch.Tensor:
        obs = self.get_actor_obs(obs)
        obs = self.actor_obs_normalizer(obs)
        if self.state_dependent_std:
            return self.actor(obs)[..., 0, :]
        else:
            return self.actor(obs)

    def evaluate(self, obs: TensorDict, **kwargs: dict[str, Any]) -> torch.Tensor:
        obs = self.get_critic_obs(obs)
        obs = self.critic_obs_normalizer(obs)
        return self.critic(obs)

    def get_actor_obs(self, obs: TensorDict) -> torch.Tensor:
        return self.cnn_encoder(obs["policy"])

    def get_critic_obs(self, obs: TensorDict) -> torch.Tensor:
        return self.cnn_encoder(obs["policy"])

    def update_distribution(self, obs: torch.Tensor | TensorDict) -> None:
        """Public method to update distribution for BC training.
        
        Args:
            obs: Can be either:
                - TensorDict with 'policy' key (for normal RL)
                - Flattened observation tensor (for BC training)
        """
        # If obs is a TensorDict, extract and encode it
        if isinstance(obs, TensorDict):
            obs = self.get_actor_obs(obs)
        # If obs is already a flattened tensor, we need to reconstruct it as TensorDict for CNN encoder
        # For BC, we assume the flattened obs needs to be reshaped/processed
        # Since BC buffer returns flattened observations, we bypass CNN encoder if obs is already encoded
        else:
            # BC observations are already flattened, but actor expects CNN-encoded features
            # We need to pass through CNN encoder first
            # Assuming obs is (batch, flattened_size), we need to create a TensorDict
            # This is a workaround - ideally BC should pass TensorDict or pre-encoded features
            pass  # For now, assume obs is already in the right format for actor
        
        obs = self.actor_obs_normalizer(obs)
        
        if self.state_dependent_std:
            # Compute mean and standard deviation
            mean_and_std = self.actor(obs)
            if self.noise_std_type == "scalar":
                mean, std = torch.unbind(mean_and_std, dim=-2)
            elif self.noise_std_type == "log":
                mean, log_std = torch.unbind(mean_and_std, dim=-2)
                std = torch.exp(log_std)
            else:
                raise ValueError(f"Unknown standard deviation type: {self.noise_std_type}")
        else:
            # Compute mean
            mean = self.actor(obs)
            # Compute standard deviation
            if self.noise_std_type == "scalar":
                std = self.std.expand_as(mean)
            elif self.noise_std_type == "log":
                std = torch.exp(self.log_std).expand_as(mean)
            else:
                raise ValueError(f"Unknown standard deviation type: {self.noise_std_type}")
        
        # Create distribution
        self.distribution = Normal(mean, std)

    def get_actions_log_prob(self, actions: torch.Tensor) -> torch.Tensor:
        return self.distribution.log_prob(actions).sum(dim=-1)

    def update_normalization(self, obs: TensorDict) -> None:
        if self.actor_obs_normalization:
            actor_obs = self.get_actor_obs(obs)
            self.actor_obs_normalizer.update(actor_obs)
        if self.critic_obs_normalization:
            critic_obs = self.get_critic_obs(obs)
            self.critic_obs_normalizer.update(critic_obs)

    def load_state_dict(self, state_dict: dict, strict: bool = True) -> bool:
        """Load the parameters of the actor-critic model.

        Args:
            state_dict: State dictionary of the model.
            strict: Whether to strictly enforce that the keys in `state_dict` match the keys returned by this module's
                :meth:`state_dict` function.

        Returns:
            Whether this training resumes a previous training. This flag is used by the :func:`load` function of
                :class:`OnPolicyRunner` to determine how to load further parameters (relevant for, e.g., distillation).
        """
        super().load_state_dict(state_dict, strict=strict)
        return True
