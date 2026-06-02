import torch
from typing import Dict, Any


def save_checkpoint(
        checkpoint: Dict[str, Any],
        save_path: str
) -> None:
    """
    保存检查点

    Args:
        checkpoint: 检查点字典
        save_path: 保存路径
    """
    torch.save(checkpoint, save_path)


def load_checkpoint(
        load_path: str,
        device: torch.device
) -> Dict[str, Any]:
    """
    加载检查点

    Args:
        load_path: 检查点路径
        device: 加载设备

    Returns:
        checkpoint: 检查点字典
    """
    checkpoint = torch.load(load_path, map_location=device)
    return checkpoint