# """
# Diffusion Policy 网络组件.

# 包含:
# 1. ConditionalResidualBlock1D: 条件残差块 (FiLM conditioning)
# 2. ConditionalUnet1D: 1D U-Net 去噪网络
# 3. DiffusionPolicy: 完整的 Diffusion Policy 模型 (ResNet18 encoder + U-Net)
# """

# import math
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# import torchvision.models as models


# # ============================================================================
# # 1D U-Net 组件
# # ============================================================================

# class SinusoidalPosEmb(nn.Module):
#     """正弦位置编码, 用于编码 diffusion timestep."""

#     def __init__(self, dim):
#         super().__init__()
#         self.dim = dim

#     def forward(self, x):
#         half_dim = self.dim // 2
#         emb = math.log(10000) / (half_dim - 1)
#         emb = torch.exp(torch.arange(half_dim, device=x.device, dtype=torch.float32) * -emb)
#         emb = x.unsqueeze(-1).float() * emb.unsqueeze(0)
#         emb = torch.cat([emb.sin(), emb.cos()], dim=-1)
#         return emb


# class ConditionalResidualBlock1D(nn.Module):
#     """条件 1D 残差块.

#     使用 FiLM (Feature-wise Linear Modulation) 将条件信息注入.
#     输入: (B, C_in, T)
#     条件: (B, C_cond)
#     输出: (B, C_out, T)
#     """

#     def __init__(self, in_channels, out_channels, cond_dim, kernel_size=3):
#         super().__init__()
#         self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size, padding=kernel_size // 2)
#         self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size, padding=kernel_size // 2)
#         self.norm1 = nn.GroupNorm(min(8, out_channels), out_channels)
#         self.norm2 = nn.GroupNorm(min(8, out_channels), out_channels)

#         # FiLM: 将条件映射到 scale 和 bias
#         self.cond_mlp = nn.Sequential(
#             nn.Mish(),
#             nn.Linear(cond_dim, out_channels * 2),
#         )

#         # 残差连接 (如果通道数变化)
#         self.residual_conv = (
#             nn.Conv1d(in_channels, out_channels, 1)
#             if in_channels != out_channels
#             else nn.Identity()
#         )

#     def forward(self, x, cond):
#         """
#         Args:
#             x: (B, C_in, T) 输入特征
#             cond: (B, C_cond) 条件向量
#         Returns:
#             (B, C_out, T) 输出特征
#         """
#         residual = self.residual_conv(x)

#         x = self.conv1(x)
#         x = self.norm1(x)

#         # FiLM conditioning
#         scale, bias = self.cond_mlp(cond).chunk(2, dim=-1)  # (B, C_out) each
#         x = x * (scale.unsqueeze(-1) + 1) + bias.unsqueeze(-1)

#         x = F.mish(x)
#         x = self.conv2(x)
#         x = self.norm2(x)
#         x = F.mish(x)

#         return x + residual


# class ConditionalUnet1D(nn.Module):
#     """条件 1D U-Net 去噪网络.

#     输入:
#       - noisy_action: (B, T_a, D_act) 带噪声的动作序列
#       - timestep: (B,) diffusion timestep
#       - global_cond: (B, D_cond) 全局条件 (观测特征)

#     输出:
#       - noise_pred: (B, T_a, D_act) 预测的噪声
#     """

#     def __init__(
#         self,
#         action_dim: int = 7,
#         global_cond_dim: int = 256,
#         diffusion_step_embed_dim: int = 128,
#         down_dims: tuple = (256, 512, 1024),
#         kernel_size: int = 3,
#     ):
#         super().__init__()

#         # Timestep embedding
#         self.timestep_emb = nn.Sequential(
#             SinusoidalPosEmb(diffusion_step_embed_dim),
#             nn.Linear(diffusion_step_embed_dim, diffusion_step_embed_dim * 4),
#             nn.Mish(),
#             nn.Linear(diffusion_step_embed_dim * 4, diffusion_step_embed_dim),
#         )

#         # 条件维度 = timestep embed + global condition
#         cond_dim = diffusion_step_embed_dim + global_cond_dim

#         # Encoder (下采样)
#         in_out = [(action_dim, down_dims[0])]
#         for i in range(len(down_dims) - 1):
#             in_out.append((down_dims[i], down_dims[i + 1]))

#         self.down_blocks = nn.ModuleList()
#         self.down_samples = nn.ModuleList()
#         for c_in, c_out in in_out:
#             self.down_blocks.append(
#                 nn.ModuleList([
#                     ConditionalResidualBlock1D(c_in, c_out, cond_dim, kernel_size),
#                     ConditionalResidualBlock1D(c_out, c_out, cond_dim, kernel_size),
#                 ])
#             )
#             self.down_samples.append(nn.Conv1d(c_out, c_out, 3, stride=2, padding=1))

#         # Bottleneck
#         self.mid_block1 = ConditionalResidualBlock1D(down_dims[-1], down_dims[-1], cond_dim, kernel_size)
#         self.mid_block2 = ConditionalResidualBlock1D(down_dims[-1], down_dims[-1], cond_dim, kernel_size)

#         # Decoder (上采样)
#         # skip connections 顺序 (pop 后): 1024, 512, 256
#         # decoder 每层: up(prev_ch) -> concat(skip_ch) -> res -> next_ch
#         self.up_blocks = nn.ModuleList()
#         self.up_samples = nn.ModuleList()
#         # 构建 decoder 通道: [(1024, 512), (512, 256), (256, 256)]
#         decoder_channels = []
#         for i in reversed(range(len(in_out))):
#             skip_ch = in_out[i][1]  # skip connection 的通道数
#             if i > 0:
#                 out_ch = in_out[i][0]  # 输出降到上一层的输入通道
#             else:
#                 out_ch = in_out[i][1]  # 最后一层保持 down_dims[0]
#             decoder_channels.append((skip_ch, out_ch))

#         prev_ch = down_dims[-1]  # bottleneck 输出通道
#         for skip_ch, out_ch in decoder_channels:
#             self.up_samples.append(nn.ConvTranspose1d(prev_ch, prev_ch, 4, stride=2, padding=1))
#             self.up_blocks.append(
#                 nn.ModuleList([
#                     ConditionalResidualBlock1D(prev_ch + skip_ch, out_ch, cond_dim, kernel_size),
#                     ConditionalResidualBlock1D(out_ch, out_ch, cond_dim, kernel_size),
#                 ])
#             )
#             prev_ch = out_ch

#         # 最终输出层
#         self.final_conv = nn.Sequential(
#             nn.Conv1d(down_dims[0], down_dims[0], kernel_size, padding=kernel_size // 2),
#             nn.Mish(),
#             nn.Conv1d(down_dims[0], action_dim, 1),
#         )

#     def forward(self, noisy_action, timestep, global_cond):
#         """
#         Args:
#             noisy_action: (B, T_a, D_act) 带噪声的动作序列
#             timestep: (B,) 或 (B, 1) diffusion timestep
#             global_cond: (B, D_cond) 全局条件向量

#         Returns:
#             noise_pred: (B, T_a, D_act)
#         """
#         # (B, T_a, D_act) -> (B, D_act, T_a) for 1D conv
#         x = noisy_action.permute(0, 2, 1)

#         # Timestep embedding
#         timestep = timestep.view(-1)
#         t_emb = self.timestep_emb(timestep)  # (B, embed_dim)

#         # 拼接条件
#         cond = torch.cat([t_emb, global_cond], dim=-1)  # (B, cond_dim)

#         # Encoder
#         skip_connections = []
#         for down_block, down_sample in zip(self.down_blocks, self.down_samples):
#             for res_block in down_block:
#                 x = res_block(x, cond)
#             skip_connections.append(x)
#             x = down_sample(x)

#         # Bottleneck
#         x = self.mid_block1(x, cond)
#         x = self.mid_block2(x, cond)

#         # Decoder
#         for up_sample, up_block in zip(self.up_samples, self.up_blocks):
#             x = up_sample(x)
#             skip = skip_connections.pop()
#             # 处理尺寸不匹配 (由于下采样/上采样的取整)
#             if x.shape[-1] != skip.shape[-1]:
#                 x = x[..., :skip.shape[-1]]
#             x = torch.cat([x, skip], dim=1)
#             for res_block in up_block:
#                 x = res_block(x, cond)

#         x = self.final_conv(x)

#         # (B, D_act, T_a) -> (B, T_a, D_act)
#         return x.permute(0, 2, 1)


# # ============================================================================
# # 视觉编码器
# # ============================================================================

# class ResNet18Encoder(nn.Module):
#     """ResNet18 视觉编码器.

#     输入: (B, 3, H, W) RGB 图像
#     输出: (B, feature_dim) 特征向量
#     """

#     def __init__(self, feature_dim: int = 256, pretrained: bool = True):
#         super().__init__()
#         resnet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT if pretrained else None)
#         # 去掉最后的 fc 和 avgpool
#         self.backbone = nn.Sequential(*list(resnet.children())[:-2])  # 输出 (B, 512, H/32, W/32)
#         # Spatial softmax pooling
#         self.pool = nn.AdaptiveAvgPool2d((1, 1))
#         self.fc = nn.Linear(512, feature_dim)

#         # ImageNet normalization
#         self.register_buffer("img_mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
#         self.register_buffer("img_std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

#     def forward(self, x):
#         """
#         Args:
#             x: (B, 3, H, W) 图像, 范围 [0, 1]
#         Returns:
#             (B, feature_dim) 特征向量
#         """
#         x = (x - self.img_mean) / self.img_std
#         feat = self.backbone(x)    # (B, 512, h, w)
#         feat = self.pool(feat)     # (B, 512, 1, 1)
#         feat = feat.flatten(1)     # (B, 512)
#         feat = self.fc(feat)       # (B, feature_dim)
#         return feat


# # ============================================================================
# # Diffusion Policy 模型
# # ============================================================================

# class DiffusionPolicy(nn.Module):
#     """Diffusion Policy 完整模型.

#     观测编码:
#       - 双视角 RGB 图像 → 2 × ResNet18 → concat
#       - 低维观测 (qpos + qvel) → MLP
#       - 多帧观测 flatten 后 concat

#     去噪网络:
#       - 条件 1D U-Net, 以观测特征为条件, 预测动作序列的噪声
#     """

#     def __init__(
#         self,
#         action_dim: int = 7,
#         obs_horizon: int = 2,
#         pred_horizon: int = 16,
#         lowdim_dim: int = 14,       # qpos(7) + qvel(7)
#         img_feature_dim: int = 256,
#         lowdim_feature_dim: int = 64,
#         down_dims: tuple = (256, 512, 1024),
#         diffusion_step_embed_dim: int = 128,
#         pretrained_resnet: bool = True,
#     ):
#         super().__init__()
#         self.action_dim = action_dim
#         self.obs_horizon = obs_horizon
#         self.pred_horizon = pred_horizon

#         # 视觉编码器 (双视角共享权重)
#         self.image_encoder = ResNet18Encoder(
#             feature_dim=img_feature_dim,
#             pretrained=pretrained_resnet,
#         )

#         # 低维观测编码器
#         self.lowdim_encoder = nn.Sequential(
#             nn.Linear(lowdim_dim, 128),
#             nn.Mish(),
#             nn.Linear(128, lowdim_feature_dim),
#         )

#         # 全局条件维度: (2 views * img_feat + lowdim_feat) * obs_horizon
#         global_cond_dim = (img_feature_dim * 2 + lowdim_feature_dim) * obs_horizon

#         # 去噪 U-Net
#         self.noise_pred_net = ConditionalUnet1D(
#             action_dim=action_dim,
#             global_cond_dim=global_cond_dim,
#             diffusion_step_embed_dim=diffusion_step_embed_dim,
#             down_dims=down_dims,
#         )

#     def encode_obs(self, obs_images, obs_lowdim):
#         """编码观测为全局条件向量.

#         Args:
#             obs_images: (B, T_o, 2, 3, H, W) 双视角 RGB
#             obs_lowdim: (B, T_o, D_low) 低维观测

#         Returns:
#             global_cond: (B, D_cond) 全局条件向量
#         """
#         B, T_o = obs_images.shape[:2]

#         # 编码每一帧的图像
#         all_img_features = []
#         for t in range(T_o):
#             # 分别编码两个视角
#             front = obs_images[:, t, 0]  # (B, 3, H, W)
#             wrist = obs_images[:, t, 1]  # (B, 3, H, W)
#             feat_front = self.image_encoder(front)  # (B, img_feat_dim)
#             feat_wrist = self.image_encoder(wrist)  # (B, img_feat_dim)
#             all_img_features.append(torch.cat([feat_front, feat_wrist], dim=-1))

#         # 编码低维观测
#         all_lowdim_features = []
#         for t in range(T_o):
#             feat_low = self.lowdim_encoder(obs_lowdim[:, t])  # (B, lowdim_feat_dim)
#             all_lowdim_features.append(feat_low)

#         # 拼接所有帧的特征
#         # [(B, 2*img_feat + lowdim_feat)] * T_o -> (B, T_o * (2*img_feat + lowdim_feat))
#         frame_features = []
#         for t in range(T_o):
#             frame_features.append(
#                 torch.cat([all_img_features[t], all_lowdim_features[t]], dim=-1)
#             )
#         global_cond = torch.cat(frame_features, dim=-1)

#         return global_cond

#     def forward(self, noisy_action, timestep, obs_images, obs_lowdim):
#         """前向传播: 预测噪声.

#         Args:
#             noisy_action: (B, T_a, D_act) 带噪声的动作
#             timestep: (B,) diffusion timestep
#             obs_images: (B, T_o, 2, 3, H, W) 观测图像
#             obs_lowdim: (B, T_o, D_low) 低维观测

#         Returns:
#             noise_pred: (B, T_a, D_act) 预测的噪声
#         """
#         global_cond = self.encode_obs(obs_images, obs_lowdim)
#         return self.noise_pred_net(noisy_action, timestep, global_cond)

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


