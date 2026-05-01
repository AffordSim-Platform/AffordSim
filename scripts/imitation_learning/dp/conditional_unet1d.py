"""
RSS Diffusion Policy 的 Conditional 1D U-Net.

严格对齐 diffusion_policy-main/diffusion_policy/model/diffusion/conditional_unet1d.py:
- Residual block: block0 -> FiLM(scale, bias) -> block1 -> + residual_conv(x)
- cond_predict_scale=True 时 out = scale * out + bias (无 +1)
- Encoder 最深层无 Downsample (Identity)
- Decoder 只有 len(in_out)-1 个 up 块, 最上层无 Upsample (Identity)
"""

from typing import Union

import math
import torch
import torch.nn as nn
import einops
from einops.layers.torch import Rearrange


# ---------------------------------------------------------------------------
# Sinusoidal timestep embedding
# ---------------------------------------------------------------------------

class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        device = x.device
        half = self.dim // 2
        emb = math.log(10000.0) / (half - 1)
        emb = torch.exp(torch.arange(half, device=device) * -emb)
        emb = x[:, None].float() * emb[None, :]
        emb = torch.cat([emb.sin(), emb.cos()], dim=-1)
        return emb


# ---------------------------------------------------------------------------
# Conv / up / down blocks
# ---------------------------------------------------------------------------

class Downsample1d(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.conv = nn.Conv1d(dim, dim, 3, stride=2, padding=1)

    def forward(self, x):
        return self.conv(x)


class Upsample1d(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.conv = nn.ConvTranspose1d(dim, dim, 4, stride=2, padding=1)

    def forward(self, x):
        return self.conv(x)


class Conv1dBlock(nn.Module):
    """Conv1d -> GroupNorm -> Mish."""

    def __init__(self, in_ch: int, out_ch: int, kernel_size: int, n_groups: int = 8):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv1d(in_ch, out_ch, kernel_size, padding=kernel_size // 2),
            nn.GroupNorm(n_groups, out_ch),
            nn.Mish(),
        )

    def forward(self, x):
        return self.block(x)


# ---------------------------------------------------------------------------
# Conditional residual block (FiLM)
# ---------------------------------------------------------------------------

class ConditionalResidualBlock1D(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        cond_dim: int,
        kernel_size: int = 3,
        n_groups: int = 8,
        cond_predict_scale: bool = True,
    ):
        super().__init__()
        self.out_channels = out_channels
        self.cond_predict_scale = cond_predict_scale

        self.blocks = nn.ModuleList([
            Conv1dBlock(in_channels, out_channels, kernel_size, n_groups=n_groups),
            Conv1dBlock(out_channels, out_channels, kernel_size, n_groups=n_groups),
        ])

        cond_channels = out_channels * 2 if cond_predict_scale else out_channels
        self.cond_encoder = nn.Sequential(
            nn.Mish(),
            nn.Linear(cond_dim, cond_channels),
            Rearrange("b c -> b c 1"),
        )

        self.residual_conv = (
            nn.Conv1d(in_channels, out_channels, 1)
            if in_channels != out_channels
            else nn.Identity()
        )

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        """
        x:    (B, C_in,  T)
        cond: (B, cond_dim)
        """
        out = self.blocks[0](x)
        embed = self.cond_encoder(cond)  # (B, cond_channels, 1)

        if self.cond_predict_scale:
            embed = embed.reshape(embed.shape[0], 2, self.out_channels, 1)
            scale = embed[:, 0]
            bias = embed[:, 1]
            out = scale * out + bias
        else:
            out = out + embed

        out = self.blocks[1](out)
        out = out + self.residual_conv(x)
        return out


# ---------------------------------------------------------------------------
# Conditional U-Net 1D
# ---------------------------------------------------------------------------

class ConditionalUnet1D(nn.Module):
    """Conditional 1D U-Net (RSS 原版结构).

    输入:
        sample:      (B, T, input_dim)
        timestep:    (B,) 或 scalar
        global_cond: (B, global_cond_dim) 可选
    输出:
        (B, T, input_dim)
    """

    def __init__(
        self,
        input_dim: int,
        global_cond_dim: int = None,
        diffusion_step_embed_dim: int = 256,
        down_dims=(256, 512, 1024),
        kernel_size: int = 5,
        n_groups: int = 8,
        cond_predict_scale: bool = True,
    ):
        super().__init__()
        all_dims = [input_dim] + list(down_dims)
        start_dim = down_dims[0]

        dsed = diffusion_step_embed_dim
        diffusion_step_encoder = nn.Sequential(
            SinusoidalPosEmb(dsed),
            nn.Linear(dsed, dsed * 4),
            nn.Mish(),
            nn.Linear(dsed * 4, dsed),
        )

        cond_dim = dsed + (global_cond_dim if global_cond_dim is not None else 0)

        in_out = list(zip(all_dims[:-1], all_dims[1:]))
        mid_dim = all_dims[-1]

        self.mid_modules = nn.ModuleList([
            ConditionalResidualBlock1D(
                mid_dim, mid_dim, cond_dim=cond_dim,
                kernel_size=kernel_size, n_groups=n_groups,
                cond_predict_scale=cond_predict_scale,
            ),
            ConditionalResidualBlock1D(
                mid_dim, mid_dim, cond_dim=cond_dim,
                kernel_size=kernel_size, n_groups=n_groups,
                cond_predict_scale=cond_predict_scale,
            ),
        ])

        down_modules = nn.ModuleList()
        for idx, (d_in, d_out) in enumerate(in_out):
            is_last = idx >= (len(in_out) - 1)
            down_modules.append(nn.ModuleList([
                ConditionalResidualBlock1D(
                    d_in, d_out, cond_dim=cond_dim,
                    kernel_size=kernel_size, n_groups=n_groups,
                    cond_predict_scale=cond_predict_scale,
                ),
                ConditionalResidualBlock1D(
                    d_out, d_out, cond_dim=cond_dim,
                    kernel_size=kernel_size, n_groups=n_groups,
                    cond_predict_scale=cond_predict_scale,
                ),
                Downsample1d(d_out) if not is_last else nn.Identity(),
            ]))

        up_modules = nn.ModuleList()
        for idx, (d_in, d_out) in enumerate(reversed(in_out[1:])):
            is_last = idx >= (len(in_out) - 1)
            up_modules.append(nn.ModuleList([
                ConditionalResidualBlock1D(
                    d_out * 2, d_in, cond_dim=cond_dim,
                    kernel_size=kernel_size, n_groups=n_groups,
                    cond_predict_scale=cond_predict_scale,
                ),
                ConditionalResidualBlock1D(
                    d_in, d_in, cond_dim=cond_dim,
                    kernel_size=kernel_size, n_groups=n_groups,
                    cond_predict_scale=cond_predict_scale,
                ),
                Upsample1d(d_in) if not is_last else nn.Identity(),
            ]))

        final_conv = nn.Sequential(
            Conv1dBlock(start_dim, start_dim, kernel_size=kernel_size),
            nn.Conv1d(start_dim, input_dim, 1),
        )

        self.diffusion_step_encoder = diffusion_step_encoder
        self.down_modules = down_modules
        self.up_modules = up_modules
        self.final_conv = final_conv

    def forward(
        self,
        sample: torch.Tensor,
        timestep: Union[torch.Tensor, float, int],
        global_cond: torch.Tensor = None,
    ) -> torch.Tensor:
        """sample: (B, T, input_dim) -> (B, T, input_dim)."""
        # (B, T, C) -> (B, C, T) for Conv1d
        x = einops.rearrange(sample, "b t c -> b c t")

        # timestep to tensor of shape (B,)
        if not torch.is_tensor(timestep):
            timestep = torch.tensor([timestep], dtype=torch.long, device=x.device)
        elif timestep.dim() == 0:
            timestep = timestep[None].to(x.device)
        timestep = timestep.expand(x.shape[0])

        global_feature = self.diffusion_step_encoder(timestep)
        if global_cond is not None:
            global_feature = torch.cat([global_feature, global_cond], dim=-1)

        # Encoder
        skips = []
        for resnet, resnet2, downsample in self.down_modules:
            x = resnet(x, global_feature)
            x = resnet2(x, global_feature)
            skips.append(x)
            x = downsample(x)

        # Bottleneck
        for mid in self.mid_modules:
            x = mid(x, global_feature)

        # Decoder
        for resnet, resnet2, upsample in self.up_modules:
            x = torch.cat((x, skips.pop()), dim=1)
            x = resnet(x, global_feature)
            x = resnet2(x, global_feature)
            x = upsample(x)

        x = self.final_conv(x)
        x = einops.rearrange(x, "b c t -> b t c")
        return x
