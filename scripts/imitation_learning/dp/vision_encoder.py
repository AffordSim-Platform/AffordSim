"""
视觉 + 低维观测编码器 (RSS 原版对齐).

要点:
- ResNet18 pretrained, BatchNorm -> GroupNorm 替换 (小 batch + EMA 更稳)
- 每个 RGB key 独立 encoder (front / wrist 权重不共享)
- ImageNet 归一化 (mean/std) 放在 encoder 内部, 外部输入 [0,1] 的 RGB
- RandomCrop / CenterCrop: 训练随机裁, 推理中心裁; 每个样本独立裁剪位置
"""

from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.models as models


IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def replace_bn_with_gn(
    root: nn.Module,
    features_per_group: int = 16,
) -> nn.Module:
    """把 ``root`` 里所有 BatchNorm2d 替换为 GroupNorm (num_groups = C // features_per_group)."""
    replace_list = []
    for name, module in root.named_modules():
        for child_name, child in module.named_children():
            if isinstance(child, nn.BatchNorm2d):
                replace_list.append((module, child_name, child))
    for parent, child_name, bn in replace_list:
        num_channels = bn.num_features
        num_groups = max(1, num_channels // features_per_group)
        setattr(parent, child_name, nn.GroupNorm(num_groups=num_groups, num_channels=num_channels))
    return root


def build_resnet18_encoder(
    pretrained: bool = True,
    use_group_norm: bool = True,
) -> nn.Module:
    """ResNet18 去掉 fc+avgpool, 输出 (B, 512, h, w). 可选 BN->GN 替换."""
    weights = models.ResNet18_Weights.DEFAULT if pretrained else None
    net = models.resnet18(weights=weights)
    backbone = nn.Sequential(*list(net.children())[:-2])  # -> (B, 512, H/32, W/32)
    if use_group_norm:
        backbone = replace_bn_with_gn(backbone)
    return backbone


class BatchRandomCrop(nn.Module):
    """每个样本独立随机裁剪 (训练); eval 时 center crop."""

    def __init__(self, crop_h: int, crop_w: int):
        super().__init__()
        self.crop_h = crop_h
        self.crop_w = crop_w

    def _center(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        y0 = (H - self.crop_h) // 2
        x0 = (W - self.crop_w) // 2
        return x[:, :, y0:y0 + self.crop_h, x0:x0 + self.crop_w]

    def _random(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        assert H >= self.crop_h and W >= self.crop_w, f"input {H}x{W} < crop {self.crop_h}x{self.crop_w}"
        if H == self.crop_h and W == self.crop_w:
            return x
        ys = torch.randint(0, H - self.crop_h + 1, (B,), device=x.device)
        xs = torch.randint(0, W - self.crop_w + 1, (B,), device=x.device)
        # Build indices per sample
        out = torch.empty(B, C, self.crop_h, self.crop_w, dtype=x.dtype, device=x.device)
        # Loop is negligible vs ResNet forward cost
        for i in range(B):
            out[i] = x[i, :, ys[i]:ys[i] + self.crop_h, xs[i]:xs[i] + self.crop_w]
        return out

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.training:
            return self._random(x)
        return self._center(x)


class SingleImageEncoder(nn.Module):
    """Resize -> (RandomCrop / CenterCrop) -> ImageNet norm -> ResNet18 -> AdaptiveAvgPool -> fc.

    image_aug=False 时关闭随机裁剪: 直接 resize 到 crop_shape 后跳过裁剪步, train/eval 行为完全一致.
    """

    def __init__(
        self,
        feature_dim: int,
        resize_shape: tuple,
        crop_shape: tuple,
        pretrained: bool = True,
        use_group_norm: bool = True,
        image_aug: bool = True,
    ):
        super().__init__()
        self.image_aug = image_aug
        if image_aug:
            # 标准路径: resize 到大尺寸, 训练时随机 crop, 推理时 center crop.
            self.resize = torchvision.transforms.Resize(resize_shape, antialias=True)
            self.randomizer = BatchRandomCrop(crop_shape[0], crop_shape[1])
        else:
            # 关掉增强: 直接 resize 到 crop_shape, 不做 cropper. 模型输入尺寸 = crop_shape.
            self.resize = torchvision.transforms.Resize(crop_shape, antialias=True)
            self.randomizer = nn.Identity()
        self.register_buffer("_mean", torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1))
        self.register_buffer("_std", torch.tensor(IMAGENET_STD).view(1, 3, 1, 1))
        self.backbone = build_resnet18_encoder(pretrained=pretrained, use_group_norm=use_group_norm)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, feature_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, 3, H, W) in [0, 1]. -> (B, feature_dim)."""
        x = self.resize(x)
        x = self.randomizer(x)
        x = (x - self._mean) / self._std
        feat = self.backbone(x)
        feat = self.pool(feat).flatten(1)
        feat = self.fc(feat)
        return feat


class MultiViewObsEncoder(nn.Module):
    """
    Obs encoder for (front, wrist) RGB + lowdim (qpos+qvel).

    forward:
        obs_images: (B, T_o, V=2, 3, H, W) in [0, 1]
        obs_lowdim: (B, T_o, D_low)
        -> (B, T_o * (V * img_feat_dim + lowdim_feat_dim))
    """

    def __init__(
        self,
        img_feature_dim: int = 128,
        lowdim_dim: int = 14,
        lowdim_feature_dim: int = 64,
        resize_shape=(240, 240),
        crop_shape=(216, 216),
        num_views: int = 2,
        pretrained: bool = True,
        use_group_norm: bool = True,
        image_aug: bool = True,
    ):
        super().__init__()
        self.num_views = num_views
        self.img_feature_dim = img_feature_dim
        self.lowdim_feature_dim = lowdim_feature_dim
        self.image_aug = image_aug

        # Per-view independent encoder (对齐原版 share_rgb_model=False)
        self.view_encoders = nn.ModuleList([
            SingleImageEncoder(
                feature_dim=img_feature_dim,
                resize_shape=resize_shape,
                crop_shape=crop_shape,
                pretrained=pretrained,
                use_group_norm=use_group_norm,
                image_aug=image_aug,
            )
            for _ in range(num_views)
        ])

        self.lowdim_encoder = nn.Sequential(
            nn.Linear(lowdim_dim, 128),
            nn.Mish(),
            nn.Linear(128, lowdim_feature_dim),
        )

    @property
    def per_frame_feature_dim(self) -> int:
        return self.num_views * self.img_feature_dim + self.lowdim_feature_dim

    def forward(self, obs_images: torch.Tensor, obs_lowdim: torch.Tensor) -> torch.Tensor:
        """Returns flat global feature (B, T_o * per_frame_feature_dim)."""
        B, T_o, V, C, H, W = obs_images.shape
        assert V == self.num_views, f"expected {self.num_views} views, got {V}"

        # Encode each view across (B*T_o) images
        view_feats: List[torch.Tensor] = []
        for v in range(V):
            imgs = obs_images[:, :, v].reshape(B * T_o, C, H, W)  # (B*T_o, C, H, W)
            feat = self.view_encoders[v](imgs)                    # (B*T_o, F_img)
            view_feats.append(feat.reshape(B, T_o, -1))           # (B, T_o, F_img)

        # Lowdim
        low = self.lowdim_encoder(obs_lowdim)  # (B, T_o, F_low)

        # Concat per-frame: (B, T_o, V*F_img + F_low)
        per_frame = torch.cat(view_feats + [low], dim=-1)
        # Flatten across time: (B, T_o * (V*F_img + F_low))
        return per_frame.reshape(B, -1)
