"""
Exponential Moving Average (EMA) — 从 diffusion_policy 原版搬过来.

推理/保存时用 EMA 权重; 训练时每个 optimization step 调 step(model).
"""

import torch
from torch.nn.modules.batchnorm import _BatchNorm


class EMAModel:
    """
    EMA warmup 使用 Karras 调度:
        decay = 1 - (1 + step/inv_gamma)^(-power)
    gamma=1, power=2/3 对应 reaches 0.999 at 31.6K steps, 0.9999 at 1M steps.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        update_after_step: int = 0,
        inv_gamma: float = 1.0,
        power: float = 0.75,
        min_value: float = 0.0,
        max_value: float = 0.9999,
    ):
        self.averaged_model = model
        self.averaged_model.eval()
        self.averaged_model.requires_grad_(False)

        self.update_after_step = update_after_step
        self.inv_gamma = inv_gamma
        self.power = power
        self.min_value = min_value
        self.max_value = max_value

        self.decay = 0.0
        self.optimization_step = 0

    def get_decay(self, optimization_step: int) -> float:
        step = max(0, optimization_step - self.update_after_step - 1)
        value = 1.0 - (1.0 + step / self.inv_gamma) ** (-self.power)
        if step <= 0:
            return 0.0
        return max(self.min_value, min(value, self.max_value))

    @torch.no_grad()
    def step(self, new_model: torch.nn.Module):
        self.decay = self.get_decay(self.optimization_step)

        for module, ema_module in zip(new_model.modules(), self.averaged_model.modules()):
            for param, ema_param in zip(
                module.parameters(recurse=False),
                ema_module.parameters(recurse=False),
            ):
                if isinstance(module, _BatchNorm):
                    ema_param.copy_(param.to(dtype=ema_param.dtype).data)
                elif not param.requires_grad:
                    ema_param.copy_(param.to(dtype=ema_param.dtype).data)
                else:
                    ema_param.mul_(self.decay)
                    ema_param.add_(param.data.to(dtype=ema_param.dtype), alpha=1.0 - self.decay)

        self.optimization_step += 1
