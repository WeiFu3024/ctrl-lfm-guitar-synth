import math

import torch

from dataset.registry import Registry

optimizer_registry = Registry("optimizer")


@optimizer_registry.register("adamw")
def build_adamw(params, cfg) -> torch.optim.Optimizer:
    return torch.optim.AdamW(
        params,
        lr=cfg.lr,
        weight_decay=cfg.weight_decay,
        betas=tuple(cfg.betas),
        eps=cfg.eps,
    )


lr_scheduler_registry = Registry("lr_scheduler")


@lr_scheduler_registry.register("warmup_cosine")
def build_warmup_cosine(optimizer, cfg, total_steps: int) -> torch.optim.lr_scheduler.LambdaLR:
    warmup_steps = max(1, int(cfg.warmup_fraction * total_steps))

    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return step / warmup_steps
        progress = min((step - warmup_steps) / max(1, total_steps - warmup_steps), 1.0)
        cosine = 0.5 * (1 + math.cos(math.pi * progress))
        return cfg.min_lr_ratio + (1 - cfg.min_lr_ratio) * cosine

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
