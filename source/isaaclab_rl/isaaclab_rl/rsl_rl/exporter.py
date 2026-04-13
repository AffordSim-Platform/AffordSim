# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import copy
import os
import torch


def export_policy_as_jit(policy: object, normalizer: object | None, path: str, filename="policy.pt"):
    """Export policy into a Torch JIT file.

    Args:
        policy: The policy torch module.
        normalizer: The empirical normalizer module. If None, Identity is used.
        path: The path to the saving directory.
        filename: The name of exported JIT file. Defaults to "policy.pt".
    """
    policy_exporter = _TorchPolicyExporter(policy, normalizer)
    policy_exporter.export(path, filename)


def export_policy_as_onnx(
    policy: object, path: str, normalizer: object | None = None, filename="policy.onnx", verbose=False
):
    """Export policy into a Torch ONNX file.

    Args:
        policy: The policy torch module.
        normalizer: The empirical normalizer module. If None, Identity is used.
        path: The path to the saving directory.
        filename: The name of exported ONNX file. Defaults to "policy.onnx".
        verbose: Whether to print the model summary. Defaults to False.
    """
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)
    policy_exporter = _OnnxPolicyExporter(policy, normalizer, verbose)
    policy_exporter.export(path, filename)


"""
Helper Classes - Private.
"""

class _CNNInferenceWrapper(torch.nn.Module):
    """Inference wrapper for ActorCriticCNN that removes training-only components and keeps only inference."""
    def __init__(self, cnn_policy):
        super().__init__()
        # Deep copy components needed for inference
        self.cnn_encoder = copy.deepcopy(cnn_policy.cnn_encoder)
        self.actor = copy.deepcopy(cnn_policy.actor)
        # Record input dimension (flattened obs)
        self.input_dim = None
        try:
            # Infer input dimension directly from cnn_encoder parameters
            self.img_n = self.cnn_encoder.img_n
            self.img_c = self.cnn_encoder.img_c
            self.img_h = self.cnn_encoder.img_h
            self.img_w = self.cnn_encoder.img_w
            self.img_flattened_size = self.img_c * self.img_h * self.img_w
            self.img_segmentation_size = self.img_h * self.img_w * 4
            self.state_feature_size = self.cnn_encoder.state_feature_size
            # Compute total input length
            # 2 images + 2 segmentations + state
            self.input_dim = (
                self.img_n * self.img_flattened_size +
                self.img_n * self.img_segmentation_size +
                self.cnn_encoder.extractors["state"].in_features
            )
        except Exception:
            self.input_dim = None

    def forward(self, obs_tensor: torch.Tensor) -> torch.Tensor:
        # obs_tensor: (batch, flattened_obs)
        features = self.cnn_encoder(obs_tensor)
        return self.actor(features)
    

class _ViTInferenceWrapper(torch.nn.Module):
    """Inference wrapper for ActorCriticViT that removes training-only components."""
    
    def __init__(self, vit_policy):
        super().__init__()
        # Copy all the components needed for inference
        self.img_c = vit_policy.img_c
        self.img_h = vit_policy.img_h
        self.img_w = vit_policy.img_w
        self.num_imgs = vit_policy.num_imgs
        self.img_h_enc = vit_policy.img_h_enc
        self.img_w_enc = vit_policy.img_w_enc
        self.prop_dim = vit_policy.prop_dim
        self.vit_embed_dim = vit_policy.vit_embed_dim
        self.patches_per_img = vit_policy.patches_per_img
        
        # Copy the network components
        self.backbone = copy.deepcopy(vit_policy.backbone)
        self.actor_spatial = copy.deepcopy(vit_policy.actor_spatial)
        self.actor_mlp = copy.deepcopy(vit_policy.actor_mlp)
        
    def forward(self, obs_tensor: torch.Tensor) -> torch.Tensor:
        # Process observation
        prop = obs_tensor[:, :self.prop_dim]
        imgs_flat = obs_tensor[:, self.prop_dim:]
        imgs = imgs_flat.view(-1, self.num_imgs, self.img_c, self.img_h, self.img_w)
        b, n, c, h, w = imgs.shape
        imgs_reshaped = imgs.reshape(b * n, c, h, w)
        
        # Resize images using torch.nn.functional directly
        imgs_resized = torch.nn.functional.interpolate(
            imgs_reshaped, 
            size=(self.img_h_enc, self.img_w_enc), 
            mode='bilinear', 
            align_corners=False
        )
        
        # Backbone
        features_stacked = self.backbone(imgs_resized)
        features = features_stacked.view(b, n * self.patches_per_img, self.vit_embed_dim)
        
        # Spatial aggregation + MLP
        spatial_out = self.actor_spatial(features, prop)
        mlp_in = torch.cat([spatial_out, prop], dim=-1)
        mean = self.actor_mlp(mlp_in)
        
        return mean


class _TorchPolicyExporter(torch.nn.Module):
    """Exporter of actor-critic into JIT file."""

    def __init__(self, policy, normalizer=None):
        super().__init__()
        self.is_recurrent = policy.is_recurrent
        print("policy:", policy, hasattr(policy, "backbone"), hasattr(policy, "actor_mlp"))
        # copy policy parameters
        print(f"{type(policy).__name__=}")
        if type(policy).__name__ == "ActorCriticViT":
            self.actor = _ViTInferenceWrapper(policy)
            
        elif type(policy).__name__ == "ActorCriticCNN":
            self.actor = _CNNInferenceWrapper(policy)
            self.cnn_encoder = self.actor.cnn_encoder
        elif hasattr(policy, "actor"):
            self.actor = copy.deepcopy(policy.actor)
            if self.is_recurrent:
                self.rnn = copy.deepcopy(policy.memory_a.rnn)
        elif hasattr(policy, "student"):
            self.actor = copy.deepcopy(policy.student)
            if self.is_recurrent:
                self.rnn = copy.deepcopy(policy.memory_s.rnn)

        # elif hasattr(policy, "backbone") :
        #     self.actor = copy.deepcopy(policy.actor_spatial)
        # elif hasattr(policy, "actor_spatial"):
        #     self.actor = copy.deepcopy(policy.actor_spatial)
        
        else:
            raise ValueError("Policy does not have an actor/student module.")
        # set up recurrent network
        if self.is_recurrent:
            self.rnn.cpu()
            self.rnn_type = type(self.rnn).__name__.lower()  # 'lstm' or 'gru'
            self.register_buffer("hidden_state", torch.zeros(self.rnn.num_layers, 1, self.rnn.hidden_size))
            if self.rnn_type == "lstm":
                self.register_buffer("cell_state", torch.zeros(self.rnn.num_layers, 1, self.rnn.hidden_size))
                self.forward = self.forward_lstm
                self.reset = self.reset_memory
            elif self.rnn_type == "gru":
                self.forward = self.forward_gru
                self.reset = self.reset_memory
            else:
                raise NotImplementedError(f"Unsupported RNN type: {self.rnn_type}")
        # copy normalizer if exists
        if normalizer:
            self.normalizer = copy.deepcopy(normalizer)
        else:
            self.normalizer = torch.nn.Identity()

    def forward_lstm(self, x):
        x = self.normalizer(x)
        x, (h, c) = self.rnn(x.unsqueeze(0), (self.hidden_state, self.cell_state))
        self.hidden_state[:] = h
        self.cell_state[:] = c
        x = x.squeeze(0)
        return self.actor(x)

    def forward_gru(self, x):
        x = self.normalizer(x)
        x, h = self.rnn(x.unsqueeze(0), self.hidden_state)
        self.hidden_state[:] = h
        x = x.squeeze(0)
        return self.actor(x)

    def forward(self, x):
        return self.actor(self.normalizer(x))

    @torch.jit.export
    def reset(self):
        pass

    def reset_memory(self):
        self.hidden_state[:] = 0.0
        if hasattr(self, "cell_state"):
            self.cell_state[:] = 0.0

    def export(self, path, filename):
        os.makedirs(path, exist_ok=True)
        path = os.path.join(path, filename)
        self.to("cpu")
        traced_script_module = torch.jit.script(self)
        traced_script_module.save(path)


class _OnnxPolicyExporter(torch.nn.Module):
    """Exporter of actor-critic into ONNX file."""

    def __init__(self, policy, normalizer=None, verbose=False):
        super().__init__()
        self.verbose = verbose
        self.is_recurrent = policy.is_recurrent
        # copy policy parameters
        if hasattr(policy, "actor"):
            self.actor = copy.deepcopy(policy.actor)
            if self.is_recurrent:
                self.rnn = copy.deepcopy(policy.memory_a.rnn)
        elif hasattr(policy, "student"):
            self.actor = copy.deepcopy(policy.student)
            if self.is_recurrent:
                self.rnn = copy.deepcopy(policy.memory_s.rnn)
        elif type(policy).__name__ == "ActorCriticViT":
            # For ActorCriticViT, create inference wrapper
            self.actor = _ViTInferenceWrapper(policy)
        else:
            raise ValueError("Policy does not have an actor/student module.")
        # set up recurrent network
        if self.is_recurrent:
            self.rnn.cpu()
            self.rnn_type = type(self.rnn).__name__.lower()  # 'lstm' or 'gru'
            if self.rnn_type == "lstm":
                self.forward = self.forward_lstm
            elif self.rnn_type == "gru":
                self.forward = self.forward_gru
            else:
                raise NotImplementedError(f"Unsupported RNN type: {self.rnn_type}")
        # copy normalizer if exists
        if normalizer:
            self.normalizer = copy.deepcopy(normalizer)
        else:
            self.normalizer = torch.nn.Identity()

    def forward_lstm(self, x_in, h_in, c_in):
        x_in = self.normalizer(x_in)
        x, (h, c) = self.rnn(x_in.unsqueeze(0), (h_in, c_in))
        x = x.squeeze(0)
        return self.actor(x), h, c

    def forward_gru(self, x_in, h_in):
        x_in = self.normalizer(x_in)
        x, h = self.rnn(x_in.unsqueeze(0), h_in)
        x = x.squeeze(0)
        return self.actor(x), h

    def forward(self, x):
        return self.actor(self.normalizer(x))

    def export(self, path, filename):
        self.to("cpu")
        self.eval()
        opset_version = 18  # was 11, but it caused problems with linux-aarch, and 18 worked well across all systems.
        if self.is_recurrent:
            obs = torch.zeros(1, self.rnn.input_size)
            h_in = torch.zeros(self.rnn.num_layers, 1, self.rnn.hidden_size)

            if self.rnn_type == "lstm":
                c_in = torch.zeros(self.rnn.num_layers, 1, self.rnn.hidden_size)
                torch.onnx.export(
                    self,
                    (obs, h_in, c_in),
                    os.path.join(path, filename),
                    export_params=True,
                    opset_version=opset_version,
                    verbose=self.verbose,
                    input_names=["obs", "h_in", "c_in"],
                    output_names=["actions", "h_out", "c_out"],
                    dynamic_axes={},
                )
            elif self.rnn_type == "gru":
                torch.onnx.export(
                    self,
                    (obs, h_in),
                    os.path.join(path, filename),
                    export_params=True,
                    opset_version=opset_version,
                    verbose=self.verbose,
                    input_names=["obs", "h_in"],
                    output_names=["actions", "h_out"],
                    dynamic_axes={},
                )
            else:
                raise NotImplementedError(f"Unsupported RNN type: {self.rnn_type}")
        else:
            # Determine input size based on actor type
            if isinstance(self.actor, _ViTInferenceWrapper):
                # For ViT, input size is prop_dim + image_flat_dim
                input_size = self.actor.prop_dim + (self.actor.num_imgs * self.actor.img_c * self.actor.img_h * self.actor.img_w)
            else:
                # For standard MLP actor
                input_size = self.actor[0].in_features
            
            obs = torch.zeros(1, input_size)
            torch.onnx.export(
                self,
                obs,
                os.path.join(path, filename),
                export_params=True,
                opset_version=opset_version,
                verbose=self.verbose,
                input_names=["obs"],
                output_names=["actions"],
                dynamic_axes={},
            )
