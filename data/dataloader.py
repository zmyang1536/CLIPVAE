from torch.utils.data import DataLoader
from typing import Tuple

from data.dataset import SpatialTranscriptomicsDataset


def get_dataloaders(
        train_dataset: SpatialTranscriptomicsDataset,
        val_dataset: SpatialTranscriptomicsDataset,
        batch_size: int = 64,
        val_batch_size: int = 128,
        num_workers: int = 4,
        pin_memory: bool = True
) -> Tuple[DataLoader, DataLoader]:
    """
    创建训练和验证数据加载器

    Args:
        train_dataset: 训练数据集
        val_dataset: 验证数据集
        batch_size: 训练批次大小
        val_batch_size: 验证批次大小
        num_workers: 数据加载线程数
        pin_memory: 是否将数据固定在内存中

    Returns:
        train_loader: 训练数据加载器
        val_loader: 验证数据加载器
    """
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=val_batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False
    )

    return train_loader, val_loader