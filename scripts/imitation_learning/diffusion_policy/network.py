"""
Diffusion Policy 网络组件.

包含:
1. ConditionalResidualBlock1D: 条件残差块 (FiLM conditioning)
2. ConditionalUnet1D: 1D U-Net 去噪网络
3. DiffusionPolicy: 完整的 Diffusion Policy 模型 (ResNet18 encoder + U-Net)
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


# ============================================================================
# 1D U-Net 组件
# ============================================================================

class SinusoidalPosEmb(nn.Module):
    """正弦位置编码, 用于编码 diffusion timestep."""

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


class Conv1dBlock(nn.Module):
    """Conv1d -> GroupNorm -> Mish."""

    def __init__(self, in_channels, out_channels, kernel_size, n_groups=8):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size, padding=kernel_size // 2),
            nn.GroupNorm(n_groups, out_channels),
            nn.Mish(),
        )

    def forward(self, x):
        return self.block(x)


class ConditionalResidualBlock1D(nn.Module):
    """条件 1D 残差块 (与 ManiSkill 参考实现对齐).

    FiLM conditioning 在两个 conv block 之间注入.
    """

    def __init__(self, in_channels, out_channels, cond_dim, kernel_size=3, n_groups=8):
        super().__init__()
        self.blocks = nn.ModuleList([
            Conv1dBlock(in_channels, out_channels, kernel_size, n_groups=n_groups),
            Conv1dBlock(out_channels, out_channels, kernel_size, n_groups=n_groups),
        ])

        # FiLM: predicts per-channel scale and bias
        self.out_channels = out_channels
        self.cond_encoder = nn.Sequential(
            nn.Mish(),
            nn.Linear(cond_dim, out_channels * 2),
            nn.Unflatten(-1, (-1, 1)),
        )

        # 残差连接 (如果通道数变化)
        self.residual_conv = (
            nn.Conv1d(in_channels, out_channels, 1)
            if in_channels != out_channels
            else nn.Identity()
        )

    def forward(self, x, cond):
        out = self.blocks[0](x)
        embed = self.cond_encoder(cond)
        embed = embed.reshape(embed.shape[0], 2, self.out_channels, 1)
        scale = embed[:, 0, ...]
        bias = embed[:, 1, ...]
        out = scale * out + bias
        out = self.blocks[1](out)
        out = out + self.residual_conv(x)
        return out


class ConditionalUnet1D(nn.Module):
    """条件 1D U-Net 去噪网络.

    输入:
      - noisy_action: (B, T_a, D_act) 带噪声的动作序列
      - timestep: (B,) diffusion timestep
      - global_cond: (B, D_cond) 全局条件 (观测特征)

    输出:
      - noise_pred: (B, T_a, D_act) 预测的噪声
    """

    def __init__(
        self,
        action_dim: int = 7,
        global_cond_dim: int = 256,
        diffusion_step_embed_dim: int = 128,
        down_dims: tuple = (256, 512, 1024),
        kernel_size: int = 5,
        n_groups: int = 8,
    ):
        super().__init__()

        all_dims = [action_dim] + list(down_dims)
        start_dim = down_dims[0]

        # Timestep embedding
        dsed = diffusion_step_embed_dim
        self.timestep_emb = nn.Sequential(
            SinusoidalPosEmb(dsed),
            nn.Linear(dsed, dsed * 4),
            nn.Mish(),
            nn.Linear(dsed * 4, dsed),
        )

        # 条件维度 = timestep embed + global condition
        cond_dim = dsed + global_cond_dim

        in_out = list(zip(all_dims[:-1], all_dims[1:]))

        # Bottleneck
        mid_dim = all_dims[-1]
        self.mid_modules = nn.ModuleList([
            ConditionalResidualBlock1D(mid_dim, mid_dim, cond_dim, kernel_size, n_groups),
            ConditionalResidualBlock1D(mid_dim, mid_dim, cond_dim, kernel_size, n_groups),
        ])

        # Encoder (下采样) — 最后一层不 downsample
        self.down_modules = nn.ModuleList()
        for ind, (dim_in, dim_out) in enumerate(in_out):
            is_last = ind >= (len(in_out) - 1)
            self.down_modules.append(nn.ModuleList([
                ConditionalResidualBlock1D(dim_in, dim_out, cond_dim, kernel_size, n_groups),
                ConditionalResidualBlock1D(dim_out, dim_out, cond_dim, kernel_size, n_groups),
                nn.Conv1d(dim_out, dim_out, 3, 2, 1) if not is_last else nn.Identity(),
            ]))

        # Decoder (上采样)
        self.up_modules = nn.ModuleList()
        for ind, (dim_in, dim_out) in enumerate(reversed(in_out[1:])):
            is_last = ind >= (len(in_out) - 1)
            self.up_modules.append(nn.ModuleList([
                ConditionalResidualBlock1D(dim_out * 2, dim_in, cond_dim, kernel_size, n_groups),
                ConditionalResidualBlock1D(dim_in, dim_in, cond_dim, kernel_size, n_groups),
                nn.ConvTranspose1d(dim_in, dim_in, 4, 2, 1) if not is_last else nn.Identity(),
            ]))

        # 最终输出层
        self.final_conv = nn.Sequential(
            Conv1dBlock(start_dim, start_dim, kernel_size=kernel_size),
            nn.Conv1d(start_dim, action_dim, 1),
        )

    def forward(self, noisy_action, timestep, global_cond):
        """
        Args:
            noisy_action: (B, T_a, D_act) 带噪声的动作序列
            timestep: (B,) 或 int, diffusion timestep
            global_cond: (B, D_cond) 全局条件向量

        Returns:
            noise_pred: (B, T_a, D_act)
        """
        # (B, T_a, D_act) -> (B, D_act, T_a) for 1D conv
        x = noisy_action.moveaxis(-1, -2)

        # Timestep embedding
        if not torch.is_tensor(timestep):
            timestep = torch.tensor([timestep], dtype=torch.long, device=x.device)
        elif len(timestep.shape) == 0:
            timestep = timestep[None].to(x.device)
        timestep = timestep.expand(x.shape[0])
        global_feature = self.timestep_emb(timestep)

        if global_cond is not None:
            global_feature = torch.cat([global_feature, global_cond], dim=-1)

        # Encoder
        h = []
        for resnet, resnet2, downsample in self.down_modules:
            x = resnet(x, global_feature)
            x = resnet2(x, global_feature)
            h.append(x)
            x = downsample(x)

        # Bottleneck
        for mid_module in self.mid_modules:
            x = mid_module(x, global_feature)

        # Decoder
        for resnet, resnet2, upsample in self.up_modules:
            x = torch.cat((x, h.pop()), dim=1)
            x = resnet(x, global_feature)
            x = resnet2(x, global_feature)
            x = upsample(x)

        x = self.final_conv(x)

        # (B, D_act, T_a) -> (B, T_a, D_act)
        return x.moveaxis(-1, -2)


# ============================================================================
# 视觉编码器
# ============================================================================

class SpatialSoftmax(nn.Module):
    """Spatial Softmax: 将 (B, C, H, W) 特征图转为 (B, C*2) 空间坐标."""

    def __init__(self, height, width, num_channels):
        super().__init__()
        x = torch.linspace(-1, 1, width)
        y = torch.linspace(-1, 1, height)
        yy, xx = torch.meshgrid(y, x, indexing="ij")
        self.register_buffer("xx", xx.reshape(1, 1, height, width))
        self.register_buffer("yy", yy.reshape(1, 1, height, width))

    def forward(self, features):
        B, C, H, W = features.shape
        weights = F.softmax(features.reshape(B, C, -1), dim=-1).reshape(B, C, H, W)
        expected_x = (weights * self.xx).sum(dim=[2, 3])
        expected_y = (weights * self.yy).sum(dim=[2, 3])
        return torch.cat([expected_x, expected_y], dim=-1)


class ResNet18Encoder(nn.Module):
    """ResNet18 视觉编码器 (SpatialSoftmax + 部分冻结).

    输入: (B, 3, H, W) RGB 图像, 范围 [0, 1]
    输出: (B, feature_dim) 特征向量
    """

    def __init__(self, feature_dim: int = 256, pretrained: bool = True, freeze_early: bool = True):
        super().__init__()
        resnet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT if pretrained else None)

        self.conv1 = resnet.conv1
        self.bn1 = resnet.bn1
        self.relu = resnet.relu
        self.maxpool = resnet.maxpool
        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3
        self.layer4 = resnet.layer4

        # SpatialSoftmax on layer4 output (512 channels, 8x8 for 256x256 input)
        self.spatial_softmax = SpatialSoftmax(8, 8, 512)
        self.fc = nn.Linear(1024, feature_dim)

        # ImageNet normalization
        self.register_buffer("img_mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("img_std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

        # 可选: 冻结 conv1~layer2，只训练 layer3+layer4
        # 冻结早期层能大幅降低 activation 显存占用 (因为整条前向链不需要存 backward 用的中间量),
        # 也能抑制小数据量下的过拟合; 数据量足够时建议解冻以获得更好性能.
        if freeze_early:
            for param in self.parameters():
                param.requires_grad = False
            for layer in [self.layer3, self.layer4]:
                for param in layer.parameters():
                    param.requires_grad = True
            for param in self.spatial_softmax.parameters():
                param.requires_grad = True
            for param in self.fc.parameters():
                param.requires_grad = True

    def forward(self, x):
        """
        Args:
            x: (B, 3, H, W) 图像, 范围 [0, 1]
        Returns:
            (B, feature_dim) 特征向量
        """
        x = (x - self.img_mean) / self.img_std
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)          # (B, 512, 8, 8) for 256x256 input
        feat = self.spatial_softmax(x)  # (B, 1024)
        feat = self.fc(feat)            # (B, feature_dim)
        return feat


# ============================================================================
# Diffusion Policy 模型
# ============================================================================

class DiffusionPolicy(nn.Module):
    """Diffusion Policy 完整模型.

    观测编码:
      - 双视角 RGB 图像 → 2 × ResNet18 → concat
      - 低维观测 (qpos + qvel) → MLP
      - 多帧观测 flatten 后 concat

    去噪网络:
      - 条件 1D U-Net, 以观测特征为条件, 预测动作序列的噪声
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
        freeze_resnet_early: bool = True,
    ):
        super().__init__()
        self.action_dim = action_dim
        self.obs_horizon = obs_horizon
        self.pred_horizon = pred_horizon

        # 视觉编码器 (双视角共享权重)
        self.image_encoder = ResNet18Encoder(
            feature_dim=img_feature_dim,
            pretrained=pretrained_resnet,
            freeze_early=freeze_resnet_early,
        )

        # 低维观测编码器
        self.lowdim_encoder = nn.Sequential(
            nn.Linear(lowdim_dim, 128),
            nn.Mish(),
            nn.Linear(128, lowdim_feature_dim),
        )

        # 全局条件维度: (2 views * img_feat + lowdim_feat) * obs_horizon
        global_cond_dim = (img_feature_dim * 2 + lowdim_feature_dim) * obs_horizon

        # 去噪 U-Net
        self.noise_pred_net = ConditionalUnet1D(
            action_dim=action_dim,
            global_cond_dim=global_cond_dim,
            diffusion_step_embed_dim=diffusion_step_embed_dim,
            down_dims=down_dims,
        )

    def encode_obs(self, obs_images, obs_lowdim):
        """编码观测为全局条件向量.

        Args:
            obs_images: (B, T_o, 2, 3, H, W) 双视角 RGB
            obs_lowdim: (B, T_o, D_low) 低维观测

        Returns:
            global_cond: (B, D_cond) 全局条件向量
        """
        B, T_o = obs_images.shape[:2]

        # 编码每一帧的图像
        all_img_features = []
        for t in range(T_o):
            # 分别编码两个视角
            front = obs_images[:, t, 0]  # (B, 3, H, W)
            wrist = obs_images[:, t, 1]  # (B, 3, H, W)
            feat_front = self.image_encoder(front)  # (B, img_feat_dim)
            feat_wrist = self.image_encoder(wrist)  # (B, img_feat_dim)
            all_img_features.append(torch.cat([feat_front, feat_wrist], dim=-1))

        # 编码低维观测
        all_lowdim_features = []
        for t in range(T_o):
            feat_low = self.lowdim_encoder(obs_lowdim[:, t])  # (B, lowdim_feat_dim)
            all_lowdim_features.append(feat_low)

        # 拼接所有帧的特征
        # [(B, 2*img_feat + lowdim_feat)] * T_o -> (B, T_o * (2*img_feat + lowdim_feat))
        frame_features = []
        for t in range(T_o):
            frame_features.append(
                torch.cat([all_img_features[t], all_lowdim_features[t]], dim=-1)
            )
        global_cond = torch.cat(frame_features, dim=-1)

        return global_cond

    def forward(self, noisy_action, timestep, obs_images, obs_lowdim):
        """前向传播: 预测噪声.

        Args:
            noisy_action: (B, T_a, D_act) 带噪声的动作
            timestep: (B,) diffusion timestep
            obs_images: (B, T_o, 2, 3, H, W) 观测图像
            obs_lowdim: (B, T_o, D_low) 低维观测

        Returns:
            noise_pred: (B, T_a, D_act) 预测的噪声
        """
        global_cond = self.encode_obs(obs_images, obs_lowdim)
        return self.noise_pred_net(noisy_action, timestep, global_cond)
