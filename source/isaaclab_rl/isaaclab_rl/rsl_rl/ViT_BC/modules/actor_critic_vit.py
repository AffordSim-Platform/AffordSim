# Copyright (c) 2021-2025, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
import torch.nn as nn
from tensordict import TensorDict
from torch.distributions import Normal
import torch.nn.functional as F # Needed for interpolation
from typing import Any

# ---------------------------------------------------------------------------- #
#                           IBRL Network Components                            #
# ---------------------------------------------------------------------------- #

class PatchEmbed2(nn.Module):
    """
    Specific Patch Embedding from the IBRL paper: two convolutional layers.
    """
    def __init__(self, embed_dim=128, use_norm=False):
        super().__init__()
        layers = [
            nn.Conv2d(3, embed_dim, kernel_size=8, stride=4),
            nn.GroupNorm(embed_dim, embed_dim) if use_norm else nn.Identity(),
            nn.ReLU(),
            nn.Conv2d(embed_dim, embed_dim, kernel_size=3, stride=2),
        ]
        self.embed = nn.Sequential(*layers)
        self.patch_dim = embed_dim

    def forward(self, x: torch.Tensor):
        # x: [B, 3, H, W]
        y = self.embed(x)
        # y: [B, C, H', W']  to  [B, H'*W', C]
        y = y.flatten(2).transpose(1, 2)
        return y


class TransformerLayer(nn.Module):
    """
    Standard Transformer Encoder layer.
    """
    def __init__(self, embed_dim, num_head, dropout=0.0):
        super().__init__()
        self.layer_norm1 = nn.LayerNorm(embed_dim)
        self.mha = nn.MultiheadAttention(embed_dim, num_head, dropout=dropout, batch_first=True)
        self.layer_norm2 = nn.LayerNorm(embed_dim)
        self.linear1 = nn.Linear(embed_dim, 4 * embed_dim)
        self.linear2 = nn.Linear(4 * embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.GELU()

    def forward(self, x):
        # x: [Batch, Seq, Dim]
        attn_out, _ = self.mha(self.layer_norm1(x), self.layer_norm1(x), self.layer_norm1(x))
        x = x + self.dropout(attn_out)
        x = x + self.dropout(self._ff_block(self.layer_norm2(x)))
        return x

    def _ff_block(self, x):
        x = self.linear2(self.activation(self.linear1(x)))
        return x


class MinVit(nn.Module):
    """
    Shallow ViT Student Backbone from IBRL.
    """
    def __init__(self, img_size=500, embed_dim=128, num_head=4, depth=1):
        super().__init__()
        self.patch_embed = PatchEmbed2(embed_dim, use_norm=False)
        
        # Compute number of patches
        # L1: (500-8)/4 + 1 = 124
        # L2: (124-3)/2 + 1 = 61
        # Num Patches = 61 * 61 = 3721
        with torch.no_grad():
            dummy = torch.zeros(1, 3, img_size, img_size)
            dummy_out = self.patch_embed(dummy)
            self.num_patches = dummy_out.shape[1]

        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches, embed_dim))
        
        layers = [TransformerLayer(embed_dim, num_head) for _ in range(depth)]
        self.net = nn.Sequential(*layers)
        self.norm = nn.LayerNorm(embed_dim)

        # Initialize weights
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    def forward(self, x):
        # x: [B, 3, H, W]
        x = self.patch_embed(x)
        # Add positional encoding
        if x.shape[1] == self.num_patches:
             x = x + self.pos_embed
        else:
            # Simple fallback if input size changed (unlikely)
            x = x + self.pos_embed[:, :x.shape[1], :]
            
        x = self.net(x)
        return self.norm(x)


class SpatialEmb(nn.Module):
    """
    Core IBRL component: spatial aggregation layer.
    Fuses and aggregates ViT patch features with proprioception.
    """
    def __init__(self, num_patch, patch_dim, prop_dim, proj_dim, dropout=0.0):
        super().__init__()
        self.prop_dim = prop_dim
        
        # Input dimension = patch dimension + proprioception dimension
        proj_in_dim = patch_dim + prop_dim
        
        # Projection layer
        self.input_proj = nn.Sequential(
            nn.Linear(proj_in_dim, proj_dim),
            nn.LayerNorm(proj_dim),
            nn.ReLU(inplace=True),
        )
        
        # Learnable spatial weights [1, Num_Patch, Proj_Dim]
        self.weight = nn.Parameter(torch.zeros(1, num_patch, proj_dim))
        nn.init.normal_(self.weight)
        
        self.dropout = nn.Dropout(dropout)

    def forward(self, feat: torch.Tensor, prop: torch.Tensor):
        # feat: [B, Num_Patch, Patch_Dim]
        # prop: [B, Prop_Dim]
        
        if self.prop_dim > 0:
            # Broadcast prop and concatenate to each patch
            repeated_prop = prop.unsqueeze(1).repeat(1, feat.size(1), 1)
            feat = torch.cat((feat, repeated_prop), dim=-1)

        # Projection
        y = self.input_proj(feat)
        
        # Weighted sum (Spatial Aggregation)
        # weight * y  to  sum over patch dim
        z = (self.weight * y).sum(dim=1)
        z = self.dropout(z)
        return z


# ---------------------------------------------------------------------------- #
#                                 Main Class                                   #
# ---------------------------------------------------------------------------- #

class ActorCriticViT(nn.Module):
    is_recurrent: bool = False

    def __init__(
        self,
        obs: TensorDict,
        obs_groups: dict[str, list[str]],
        num_actions: int,
        actor_hidden_dims: tuple[int] | list[int] = [512, 256, 128],
        critic_hidden_dims: tuple[int] | list[int] = [512, 256, 128],
        activation: str = "elu",
        init_noise_std: float = 1.0,
        noise_std_type: str = "scalar",
        **kwargs: dict[str, Any],
    ) -> None:
        if kwargs:
            print("ActorCriticViT.__init__ got unexpected arguments: " + str([key for key in kwargs]))
        super().__init__()
        self.obs_groups = obs_groups
        num_actor_obs = 0
        for obs_group in obs_groups["policy"]:
            assert len(obs[obs_group].shape) == 2, "The ActorCritic module only supports 1D observations."
            num_actor_obs += obs[obs_group].shape[-1]
        num_critic_obs = 0
        for obs_group in obs_groups["critic"]:
            assert len(obs[obs_group].shape) == 2, "The ActorCritic module only supports 1D observations."
            num_critic_obs += obs[obs_group].shape[-1]
        # 1. Compute dimensions
        # Assume all observations are flattened into one large vector
        # Image part: 2 images * 3 channels * 500 * 500 = 1,500,000
        self.img_c = 3
        self.img_h = 128
        self.img_w = 128

        self.num_imgs = 2
        self.img_flat_dim = self.num_imgs * self.img_c * self.img_h * self.img_w

        self.img_h_enc = 96 
        self.img_w_enc = 96

        self.prop_dim = num_actor_obs - self.img_flat_dim
        assert self.prop_dim >= 0, f"Error: num_actor_obs ({num_actor_obs}) < required image dim ({self.img_flat_dim})"
        
        self.critic_prop_dim = num_critic_obs - self.img_flat_dim
        assert self.critic_prop_dim >= 0, "Critic obs dim too small for images"
        print(f"IBRL Arch: Input {self.img_h}x{self.img_w}  to  Resized to {self.img_h_enc}x{self.img_w_enc}")

        # 2. Initialize visual backbone (MinVit)
        # IBRL Student is typically very shallow (depth=1)
        self.vit_embed_dim = 128
        self.backbone = MinVit(
            img_size=self.img_h_enc, 
            embed_dim=self.vit_embed_dim, 
            depth=1, 
            num_head=4
        )
        
        # Since we have 2 images, features will be a concatenation of two sets of patches
        # Number of patches after ViT processes a single image
        self.patches_per_img = self.backbone.num_patches
        # Total number of patches (for SpatialEmb)
        self.total_patches = self.patches_per_img * self.num_imgs 
        
        # 3. Initialize Spatial Embeddings (Actor and Critic each have their own aggregation layer)
        # Projection dimension is typically large (e.g. 1024)
        self.spatial_proj_dim = 1024
        
        self.actor_spatial = SpatialEmb(
            num_patch=self.total_patches,
            patch_dim=self.vit_embed_dim,
            prop_dim=self.prop_dim,
            proj_dim=self.spatial_proj_dim,
            dropout=0.0
        )
        
        self.critic_spatial = SpatialEmb(
            num_patch=self.total_patches,
            patch_dim=self.vit_embed_dim,
            prop_dim=self.prop_dim, # PPO Critic is state-based only, so input is also prop
            proj_dim=self.spatial_proj_dim,
            dropout=0.0
        )

        # 4. Initialize MLP Heads
        activation_fn = get_activation(activation)
        
        # Actor MLP
        # : SpatialEmb (1024) + Skip Connection Prop (prop_dim)
        actor_in_dim = self.spatial_proj_dim + self.prop_dim
        
        actor_layers = []
        dims = [actor_in_dim] + list(actor_hidden_dims)
        for i in range(len(dims) - 1):
            actor_layers.append(nn.Linear(dims[i], dims[i+1]))
            actor_layers.append(nn.LayerNorm(dims[i+1])) # IBRL Actor uses LayerNorm
            actor_layers.append(activation_fn)
        
        actor_out_dim = num_actions
        actor_layers.append(nn.Linear(dims[-1], actor_out_dim))
        # IBRL typically adds Tanh at the output layer, but rsl_rl PPO usually outputs raw logits/means
        # No Tanh here to maintain rsl_rl PPO distribution logic
        self.actor_mlp = nn.Sequential(*actor_layers)

        # Critic MLP
        # : SpatialEmb (1024) + Skip Connection Prop (prop_dim)
        critic_in_dim = self.spatial_proj_dim + self.prop_dim
        
        critic_layers = []
        dims = [critic_in_dim] + list(critic_hidden_dims)
        for i in range(len(dims) - 1):
            critic_layers.append(nn.Linear(dims[i], dims[i+1]))
            critic_layers.append(nn.LayerNorm(dims[i+1]))
            critic_layers.append(activation_fn)
        
        critic_layers.append(nn.Linear(dims[-1], 1))
        self.critic_mlp = nn.Sequential(*critic_layers)

        # 5. Action Noise (Standard rsl_rl implementation)
        self.noise_std_type = noise_std_type
        if self.noise_std_type == "scalar":
            self.std = nn.Parameter(init_noise_std * torch.ones(num_actions))
        elif self.noise_std_type == "log":
            self.log_std = nn.Parameter(torch.log(torch.tensor(init_noise_std) * torch.ones(num_actions)))
        
        self.distribution = None
        Normal.set_default_validate_args(False)

        # Print info
        print(f"Backbone: MinVit (Depth=1, Embed={self.vit_embed_dim}, Total Patches={self.total_patches})")
        print(f"Actor Head: SpatialEmb  to  MLP({actor_hidden_dims})")
        print(f"Critic Head: SpatialEmb  to  MLP({critic_hidden_dims})")

    def _process_obs(self, obs_tensor: torch.Tensor, prop_dim: int):
        """
        Process flattened observation vector: split  to  reshape  to  backbone.
        """
        # 1. Split proprioception
        prop = obs_tensor[:, :prop_dim]
        
        imgs_flat = obs_tensor[:, prop_dim:]
        imgs = imgs_flat.view(-1, self.num_imgs, self.img_c, self.img_h, self.img_w)
        b, n, c, h, w = imgs.shape
        imgs_reshaped = imgs.reshape(b * n, c, h, w)

        imgs_resized = F.interpolate(
            imgs_reshaped, 
            size=(self.img_h_enc, self.img_w_enc), 
            mode='bilinear', 
            align_corners=False
        )
        
        features_stacked = self.backbone(imgs_resized)
        # Restore dimensions: [B, Total_Patches, Embed_Dim]
        features = features_stacked.view(b, n * self.patches_per_img, self.vit_embed_dim)
        
        return prop, features
    
    def reset(self, dones=None):
        pass

    def forward(self):
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

    def update_distribution(self, observations):
        prop, features = self._process_obs(observations, self.prop_dim)
        spatial_out = self.actor_spatial(features, prop)
        mlp_in = torch.cat([spatial_out, prop], dim=-1)
        mean = self.actor_mlp(mlp_in)
        if self.noise_std_type == "scalar":
            std = self.std.expand_as(mean)
        elif self.noise_std_type == "log":
            std = torch.exp(self.log_std).expand_as(mean)
        
        self.distribution = Normal(mean, std)

    def act(self, obs: TensorDict, **kwargs: dict[str, Any]) -> torch.Tensor:
        obs = self.get_actor_obs(obs)
        self.update_distribution(obs)
        return self.distribution.sample()

    def get_actions_log_prob(self, actions):
        return self.distribution.log_prob(actions).sum(dim=-1)

    def act_inference(self, obs: TensorDict) -> torch.Tensor:
        obs = self.get_actor_obs(obs)
        prop, features = self._process_obs(obs, self.prop_dim)
        spatial_out = self.actor_spatial(features, prop)
        mlp_in = torch.cat([spatial_out, prop], dim=-1)
        mean = self.actor_mlp(mlp_in)
        return mean

    def evaluate(self, obs: TensorDict, **kwargs: dict[str, Any]) -> torch.Tensor:
        obs = self.get_critic_obs(obs)
        # Critic processing pipeline
        prop, features = self._process_obs(obs, self.critic_prop_dim)
        
        # Use Critic's independent SpatialEmb
        spatial_out = self.critic_spatial(features, prop)
        
        mlp_in = torch.cat([spatial_out, prop], dim=-1)
        value = self.critic_mlp(mlp_in)
        return value
    
    def get_actor_obs(self, obs: TensorDict) -> torch.Tensor:
        obs_list = [obs[obs_group] for obs_group in self.obs_groups["policy"]]
        return torch.cat(obs_list, dim=-1)

    def get_critic_obs(self, obs: TensorDict) -> torch.Tensor:
        obs_list = [obs[obs_group] for obs_group in self.obs_groups["critic"]]
        return torch.cat(obs_list, dim=-1)
    
    def update_normalization(self, obs: TensorDict) -> None:
        pass
        
    def load_state_dict(self, state_dict, strict=True):
        super().load_state_dict(state_dict, strict=strict)
        return True

# Helper
def get_activation(act_name):
    if act_name == "elu":
        return nn.ELU()
    elif act_name == "selu":
        return nn.SELU()
    elif act_name == "relu":
        return nn.ReLU()
    elif act_name == "crelu":
        return nn.ReLU()
    elif act_name == "lrelu":
        return nn.LeakyReLU()
    elif act_name == "tanh":
        return nn.Tanh()
    elif act_name == "sigmoid":
        return nn.Sigmoid()
    else:
        print("invalid activation function!")
        return None