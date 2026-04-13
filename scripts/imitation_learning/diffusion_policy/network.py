"""
Diffusion Policy network components.

Contains:
1. ConditionalResidualBlock1D: Conditional residual block (FiLM conditioning)
2. ConditionalUnet1D: 1D U-Net denoising network
3. DiffusionPolicy: Complete Diffusion Policy model (ResNet18 encoder + U-Net)
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


# ============================================================================
# 1D U-Net Components
# ============================================================================

class SinusoidalPosEmb(nn.Module):
    """Sinusoidal positional encoding for encoding diffusion timestep."""

    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=x.device, dtype=torch.float32) * -emb)
        emb = x.unsqueeze(-1).float() * emb.unsqueeze(0)
        emb = torch.cat([emb.sin(), emb.cos()], dim=-1)
        return emb


class ConditionalResidualBlock1D(nn.Module):
    """Conditional 1D residual block.

    Uses FiLM (Feature-wise Linear Modulation) to inject conditioning information.
    Input: (B, C_in, T)
    Condition: (B, C_cond)
    Output: (B, C_out, T)
    """

    def __init__(self, in_channels, out_channels, cond_dim, kernel_size=3):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size, padding=kernel_size // 2)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size, padding=kernel_size // 2)
        self.norm1 = nn.GroupNorm(min(8, out_channels), out_channels)
        self.norm2 = nn.GroupNorm(min(8, out_channels), out_channels)

        # FiLM: map condition to scale and bias
        self.cond_mlp = nn.Sequential(
            nn.Mish(),
            nn.Linear(cond_dim, out_channels * 2),
        )

        # Residual connection (if channel count changes)
        self.residual_conv = (
            nn.Conv1d(in_channels, out_channels, 1)
            if in_channels != out_channels
            else nn.Identity()
        )

    def forward(self, x, cond):
        """
        Args:
            x: (B, C_in, T) input features
            cond: (B, C_cond) condition vector
        Returns:
            (B, C_out, T) output features
        """
        residual = self.residual_conv(x)

        x = self.conv1(x)
        x = self.norm1(x)

        # FiLM conditioning
        scale, bias = self.cond_mlp(cond).chunk(2, dim=-1)  # (B, C_out) each
        x = x * (scale.unsqueeze(-1) + 1) + bias.unsqueeze(-1)

        x = F.mish(x)
        x = self.conv2(x)
        x = self.norm2(x)
        x = F.mish(x)

        return x + residual


class ConditionalUnet1D(nn.Module):
    """Conditional 1D U-Net denoising network.

    Input:
      - noisy_action: (B, T_a, D_act) noisy action sequence
      - timestep: (B,) diffusion timestep
      - global_cond: (B, D_cond) global condition (observation features)

    Output:
      - noise_pred: (B, T_a, D_act) predicted noise
    """

    def __init__(
        self,
        action_dim: int = 7,
        global_cond_dim: int = 256,
        diffusion_step_embed_dim: int = 128,
        down_dims: tuple = (256, 512, 1024),
        kernel_size: int = 3,
    ):
        super().__init__()

        # Timestep embedding
        self.timestep_emb = nn.Sequential(
            SinusoidalPosEmb(diffusion_step_embed_dim),
            nn.Linear(diffusion_step_embed_dim, diffusion_step_embed_dim * 4),
            nn.Mish(),
            nn.Linear(diffusion_step_embed_dim * 4, diffusion_step_embed_dim),
        )

        # Condition dimension = timestep embed + global condition
        cond_dim = diffusion_step_embed_dim + global_cond_dim

        # Encoder (downsampling)
        in_out = [(action_dim, down_dims[0])]
        for i in range(len(down_dims) - 1):
            in_out.append((down_dims[i], down_dims[i + 1]))

        self.down_blocks = nn.ModuleList()
        self.down_samples = nn.ModuleList()
        for c_in, c_out in in_out:
            self.down_blocks.append(
                nn.ModuleList([
                    ConditionalResidualBlock1D(c_in, c_out, cond_dim, kernel_size),
                    ConditionalResidualBlock1D(c_out, c_out, cond_dim, kernel_size),
                ])
            )
            self.down_samples.append(nn.Conv1d(c_out, c_out, 3, stride=2, padding=1))

        # Bottleneck
        self.mid_block1 = ConditionalResidualBlock1D(down_dims[-1], down_dims[-1], cond_dim, kernel_size)
        self.mid_block2 = ConditionalResidualBlock1D(down_dims[-1], down_dims[-1], cond_dim, kernel_size)

        # Decoder (upsampling)
        # skip connections order (after pop): 1024, 512, 256
        # decoder each layer: up(prev_ch) -> concat(skip_ch) -> res -> next_ch
        self.up_blocks = nn.ModuleList()
        self.up_samples = nn.ModuleList()
        # Build decoder channels: [(1024, 512), (512, 256), (256, 256)]
        decoder_channels = []
        for i in reversed(range(len(in_out))):
            skip_ch = in_out[i][1]  # skip connection channel count
            if i > 0:
                out_ch = in_out[i][0]  # output reduced to previous layer's input channels
            else:
                out_ch = in_out[i][1]  # last layer keeps down_dims[0]
            decoder_channels.append((skip_ch, out_ch))

        prev_ch = down_dims[-1]  # bottleneck output channels
        for skip_ch, out_ch in decoder_channels:
            self.up_samples.append(nn.ConvTranspose1d(prev_ch, prev_ch, 4, stride=2, padding=1))
            self.up_blocks.append(
                nn.ModuleList([
                    ConditionalResidualBlock1D(prev_ch + skip_ch, out_ch, cond_dim, kernel_size),
                    ConditionalResidualBlock1D(out_ch, out_ch, cond_dim, kernel_size),
                ])
            )
            prev_ch = out_ch

        # Final output layer
        self.final_conv = nn.Sequential(
            nn.Conv1d(down_dims[0], down_dims[0], kernel_size, padding=kernel_size // 2),
            nn.Mish(),
            nn.Conv1d(down_dims[0], action_dim, 1),
        )

    def forward(self, noisy_action, timestep, global_cond):
        """
        Args:
            noisy_action: (B, T_a, D_act) noisy action sequence
            timestep: (B,) or (B, 1) diffusion timestep
            global_cond: (B, D_cond) global condition vector

        Returns:
            noise_pred: (B, T_a, D_act)
        """
        # (B, T_a, D_act) -> (B, D_act, T_a) for 1D conv
        x = noisy_action.permute(0, 2, 1)

        # Timestep embedding
        timestep = timestep.view(-1)
        t_emb = self.timestep_emb(timestep)  # (B, embed_dim)

        # Concatenate conditions
        cond = torch.cat([t_emb, global_cond], dim=-1)  # (B, cond_dim)

        # Encoder
        skip_connections = []
        for down_block, down_sample in zip(self.down_blocks, self.down_samples):
            for res_block in down_block:
                x = res_block(x, cond)
            skip_connections.append(x)
            x = down_sample(x)

        # Bottleneck
        x = self.mid_block1(x, cond)
        x = self.mid_block2(x, cond)

        # Decoder
        for up_sample, up_block in zip(self.up_samples, self.up_blocks):
            x = up_sample(x)
            skip = skip_connections.pop()
            # Handle size mismatch (due to rounding in downsampling/upsampling)
            if x.shape[-1] != skip.shape[-1]:
                x = x[..., :skip.shape[-1]]
            x = torch.cat([x, skip], dim=1)
            for res_block in up_block:
                x = res_block(x, cond)

        x = self.final_conv(x)

        # (B, D_act, T_a) -> (B, T_a, D_act)
        return x.permute(0, 2, 1)


# ============================================================================
# Vision Encoder
# ============================================================================

class ResNet18Encoder(nn.Module):
    """ResNet18 vision encoder.

    Input: (B, 3, H, W) RGB image
    Output: (B, feature_dim) feature vector
    """

    def __init__(self, feature_dim: int = 256, pretrained: bool = True):
        super().__init__()
        resnet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT if pretrained else None)
        # Remove the final fc and avgpool
        self.backbone = nn.Sequential(*list(resnet.children())[:-2])  # Output (B, 512, H/32, W/32)
        # Spatial softmax pooling
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, feature_dim)

    def forward(self, x):
        """
        Args:
            x: (B, 3, H, W) image, range [0, 1]
        Returns:
            (B, feature_dim) feature vector
        """
        feat = self.backbone(x)    # (B, 512, h, w)
        feat = self.pool(feat)     # (B, 512, 1, 1)
        feat = feat.flatten(1)     # (B, 512)
        feat = self.fc(feat)       # (B, feature_dim)
        return feat


# ============================================================================
# Diffusion Policy Model
# ============================================================================

class DiffusionPolicy(nn.Module):
    """Diffusion Policy complete model.

    Observation encoding:
      - Dual-view RGB images -> 2 x ResNet18 -> concat
      - Low-dimensional observations (qpos + qvel) -> MLP
      - Multi-frame observations flattened then concatenated

    Denoising network:
      - Conditional 1D U-Net, conditioned on observation features, predicts noise of action sequence
    """

    def __init__(
        self,
        action_dim: int = 7,
        obs_horizon: int = 2,
        pred_horizon: int = 16,
        lowdim_dim: int = 14,       # qpos(7) + qvel(7)
        img_feature_dim: int = 256,
        lowdim_feature_dim: int = 64,
        down_dims: tuple = (256, 512, 1024),
        diffusion_step_embed_dim: int = 128,
        pretrained_resnet: bool = True,
    ):
        super().__init__()
        self.action_dim = action_dim
        self.obs_horizon = obs_horizon
        self.pred_horizon = pred_horizon

        # Vision encoder (shared weights for dual views)
        self.image_encoder = ResNet18Encoder(
            feature_dim=img_feature_dim,
            pretrained=pretrained_resnet,
        )

        # Low-dimensional observation encoder
        self.lowdim_encoder = nn.Sequential(
            nn.Linear(lowdim_dim, 128),
            nn.Mish(),
            nn.Linear(128, lowdim_feature_dim),
        )

        # Global condition dimension: (2 views * img_feat + lowdim_feat) * obs_horizon
        global_cond_dim = (img_feature_dim * 2 + lowdim_feature_dim) * obs_horizon

        # Denoising U-Net
        self.noise_pred_net = ConditionalUnet1D(
            action_dim=action_dim,
            global_cond_dim=global_cond_dim,
            diffusion_step_embed_dim=diffusion_step_embed_dim,
            down_dims=down_dims,
        )

    def encode_obs(self, obs_images, obs_lowdim):
        """Encode observations into a global condition vector.

        Args:
            obs_images: (B, T_o, 2, 3, H, W) dual-view RGB
            obs_lowdim: (B, T_o, D_low) low-dimensional observations

        Returns:
            global_cond: (B, D_cond) global condition vector
        """
        B, T_o = obs_images.shape[:2]

        # Encode images for each frame
        all_img_features = []
        for t in range(T_o):
            # Encode each view separately
            front = obs_images[:, t, 0]  # (B, 3, H, W)
            wrist = obs_images[:, t, 1]  # (B, 3, H, W)
            feat_front = self.image_encoder(front)  # (B, img_feat_dim)
            feat_wrist = self.image_encoder(wrist)  # (B, img_feat_dim)
            all_img_features.append(torch.cat([feat_front, feat_wrist], dim=-1))

        # Encode low-dimensional observations
        all_lowdim_features = []
        for t in range(T_o):
            feat_low = self.lowdim_encoder(obs_lowdim[:, t])  # (B, lowdim_feat_dim)
            all_lowdim_features.append(feat_low)

        # Concatenate features from all frames
        # [(B, 2*img_feat + lowdim_feat)] * T_o -> (B, T_o * (2*img_feat + lowdim_feat))
        frame_features = []
        for t in range(T_o):
            frame_features.append(
                torch.cat([all_img_features[t], all_lowdim_features[t]], dim=-1)
            )
        global_cond = torch.cat(frame_features, dim=-1)

        return global_cond

    def forward(self, noisy_action, timestep, obs_images, obs_lowdim):
        """Forward pass: predict noise.

        Args:
            noisy_action: (B, T_a, D_act) noisy actions
            timestep: (B,) diffusion timestep
            obs_images: (B, T_o, 2, 3, H, W) observation images
            obs_lowdim: (B, T_o, D_low) low-dimensional observations

        Returns:
            noise_pred: (B, T_a, D_act) predicted noise
        """
        global_cond = self.encode_obs(obs_images, obs_lowdim)
        return self.noise_pred_net(noisy_action, timestep, global_cond)
