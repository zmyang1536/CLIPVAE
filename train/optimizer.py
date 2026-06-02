import torch
import torch.optim as optim
from typing import Dict, Any


def get_optimizer(
        model: torch.nn.Module,
        config: Dict[str, Any]
) -> optim.Optimizer:
    """
    创建优化器

    Args:
        model: 模型
        config: 训练配置字典

    Returns:
        optimizer: 优化器
    """
    optimizer_name = config["optimizer"].lower()
    lr = config["learning_rate"]
    weight_decay = config["weight_decay"]

    if optimizer_name == "adam":
        return optim.Adam(
            model.parameters(),
            lr=lr,
            weight_decay=weight_decay,
            betas=config.get("betas", (0.9, 0.999))
        )
    elif optimizer_name == "adamw":
        return optim.AdamW(
            model.parameters(),
            lr=lr,
            weight_decay=weight_decay,
            betas=config.get("betas", (0.9, 0.999))
        )
    elif optimizer_name == "sgd":
        return optim.SGD(
            model.parameters(),
            lr=lr,
            momentum=config.get("momentum", 0.9),
            weight_decay=weight_decay
        )
    else:
        raise ValueError(f"不支持的优化器: {optimizer_name}")


def get_scheduler(
        optimizer: optim.Optimizer,
        config: Dict[str, Any]
) -> optim.lr_scheduler._LRScheduler:
    """
    创建学习率调度器

    Args:
        optimizer: 优化器
        config: 训练配置字典

    Returns:
        scheduler: 学习率调度器
    """
    scheduler_name = config["scheduler"].lower()
    epochs = config["epochs"]

    if scheduler_name == "cosine":
        return optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=epochs - config.get("warmup_epochs", 0),
            eta_min=config.get("min_lr", 1e-6)
        )
    elif scheduler_name == "cosine_warmup":
        from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
        return CosineAnnealingWarmRestarts(
            optimizer,
            T_0=config.get("T_0", 10),
            T_mult=config.get("T_mult", 2),
            eta_min=config.get("min_lr", 1e-6)
        )
    elif scheduler_name == "step":
        return optim.lr_scheduler.StepLR(
            optimizer,
            step_size=config.get("step_size", 30),
            gamma=config.get("gamma", 0.1)
        )
    elif scheduler_name == "reduce_on_plateau":
        return optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode=config.get("mode", "min"),
            factor=config.get("factor", 0.1),
            patience=config.get("patience", 10),
            min_lr=config.get("min_lr", 1e-6)
        )
    else:
        raise ValueError(f"不支持的学习率调度器: {scheduler_name}")