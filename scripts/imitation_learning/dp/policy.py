"""
Diffusion Policy (image + lowdim, global-cond, U-Net).

对齐原版 DiffusionUnetHybridImagePolicy:
- obs -> MultiViewObsEncoder -> global_cond
- ConditionalUnet1D(input_dim=action_dim) 预测噪声, epsilon 目标
- 训练用 diffusers.DDPMScheduler (cosine, clip_sample)
- 推理支持 DDPM / DDIM

注: 数据已在离线预处理里 min-max 归一化到 [-1,1] (dataset_stat.json), 所以
policy 自身不内置 normalizer; env rollout 侧反归一化 action 再下发.
"""

from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from diffusers.schedulers.scheduling_ddim import DDIMScheduler
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler

from conditional_unet1d import ConditionalUnet1D
from vision_encoder import MultiViewObsEncoder


class DiffusionPolicy(nn.Module):
    def __init__(
        self,
        # action / horizons
        action_dim: int = 7,
        obs_horizon: int = 2,
        pred_horizon: int = 16,
        # obs encoder
        lowdim_dim: int = 14,
        img_feature_dim: int = 128,
        lowdim_feature_dim: int = 64,
        resize_shape: Tuple[int, int] = (240, 240),
        crop_shape: Tuple[int, int] = (216, 216),
        num_views: int = 2,
        pretrained_resnet: bool = True,
        use_group_norm: bool = True,
        image_aug: bool = True,
        # U-Net
        down_dims: Tuple[int, ...] = (256, 512, 1024),
        kernel_size: int = 5,
        n_groups: int = 8,
        diffusion_step_embed_dim: int = 256,
        cond_predict_scale: bool = True,
        # scheduler
        num_train_timesteps: int = 100,
        beta_schedule: str = "squaredcos_cap_v2",
        prediction_type: str = "epsilon",
        clip_sample: bool = True,
        num_inference_steps: Optional[int] = None,
        inference_scheduler: str = "ddim",  # 'ddpm' | 'ddim'
    ):
        super().__init__()
        self.action_dim = action_dim
        self.obs_horizon = obs_horizon
        self.pred_horizon = pred_horizon
        self.prediction_type = prediction_type

        # ---------------- obs encoder ----------------
        self.obs_encoder = MultiViewObsEncoder(
            img_feature_dim=img_feature_dim,
            lowdim_dim=lowdim_dim,
            lowdim_feature_dim=lowdim_feature_dim,
            resize_shape=resize_shape,
            crop_shape=crop_shape,
            num_views=num_views,
            pretrained=pretrained_resnet,
            use_group_norm=use_group_norm,
            image_aug=image_aug,
        )
        global_cond_dim = obs_horizon * self.obs_encoder.per_frame_feature_dim

        # ---------------- U-Net ----------------
        self.model = ConditionalUnet1D(
            input_dim=action_dim,
            global_cond_dim=global_cond_dim,
            diffusion_step_embed_dim=diffusion_step_embed_dim,
            down_dims=down_dims,
            kernel_size=kernel_size,
            n_groups=n_groups,
            cond_predict_scale=cond_predict_scale,
        )

        # ---------------- noise schedulers ----------------
        self.noise_scheduler = DDPMScheduler(
            num_train_timesteps=num_train_timesteps,
            beta_schedule=beta_schedule,
            prediction_type=prediction_type,
            clip_sample=clip_sample,
        )
        self.num_train_timesteps = num_train_timesteps

        # Inference scheduler (共享 beta schedule 以保证分布一致)
        if inference_scheduler == "ddim":
            self.inference_noise_scheduler = DDIMScheduler(
                num_train_timesteps=num_train_timesteps,
                beta_schedule=beta_schedule,
                prediction_type=prediction_type,
                clip_sample=clip_sample,
                set_alpha_to_one=True,
                steps_offset=0,
            )
        else:
            self.inference_noise_scheduler = self.noise_scheduler
        self.num_inference_steps = (
            num_inference_steps if num_inference_steps is not None else num_train_timesteps
        )

    # ---------------------------------------------------------------
    # Training
    # ---------------------------------------------------------------
    def compute_loss(self, batch: dict) -> torch.Tensor:
        obs_images = batch["obs_images"]  # (B, T_o, V, 3, H, W) in [0,1]
        obs_lowdim = batch["obs_lowdim"]  # (B, T_o, D_low)
        actions = batch["actions"]        # (B, T_a, D_act)

        B = actions.shape[0]
        device = actions.device

        global_cond = self.obs_encoder(obs_images, obs_lowdim)  # (B, global_cond_dim)

        noise = torch.randn(actions.shape, device=device)
        timesteps = torch.randint(
            0, self.num_train_timesteps, (B,), device=device
        ).long()

        noisy = self.noise_scheduler.add_noise(actions, noise, timesteps)
        pred = self.model(noisy, timesteps, global_cond=global_cond)

        if self.prediction_type == "epsilon":
            target = noise
        elif self.prediction_type == "sample":
            target = actions
        else:
            raise ValueError(f"Unsupported prediction_type={self.prediction_type}")

        return F.mse_loss(pred, target)

    # ---------------------------------------------------------------
    # Inference
    # ---------------------------------------------------------------
    @torch.no_grad()
    def predict_action(
        self,
        obs_images: torch.Tensor,
        obs_lowdim: torch.Tensor,
        generator: Optional[torch.Generator] = None,
    ) -> torch.Tensor:
        """
        obs_images: (B, T_o, V, 3, H, W) in [0,1]
        obs_lowdim: (B, T_o, D_low)

        Returns: (B, T_a, D_act) 预测的动作序列 (仍在 [-1,1] 归一化空间).
        """
        B = obs_images.shape[0]
        device = obs_images.device

        global_cond = self.obs_encoder(obs_images, obs_lowdim)

        scheduler = self.inference_noise_scheduler
        scheduler.set_timesteps(self.num_inference_steps)

        sample = torch.randn(
            (B, self.pred_horizon, self.action_dim),
            device=device, generator=generator,
        )

        for t in scheduler.timesteps:
            model_out = self.model(sample, t, global_cond=global_cond)
            sample = scheduler.step(model_out, t, sample, generator=generator).prev_sample

        return sample
