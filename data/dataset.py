import os
import torch
import numpy as np
import scanpy as sc
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from typing import Optional, Tuple

from utils.logger import get_logger

logger = get_logger(__name__)


class SpatialTranscriptomicsDataset(Dataset):
    """
    空间转录组数据集类

    输入:
        - AnnData对象 (.h5ad文件)，包含基因表达矩阵和空间坐标
        - H&E图像文件 (.png/.jpg)
        - 每个spot对应的图像块坐标
    """

    def __init__(
            self,
            adata_path: str,
            image_path: str,
            patch_size: int = 224,
            transform: Optional[transforms.Compose] = None,
            split: str = "train",
            train_ratio: float = 0.8,
            seed: int = 42
    ):
        super().__init__()

        self.adata_path = adata_path
        self.image_path = image_path
        self.patch_size = patch_size
        self.split = split

        # 加载数据
        logger.info(f"加载AnnData文件: {adata_path}")
        self.adata = sc.read_h5ad(adata_path)

        logger.info(f"加载H&E图像: {image_path}")
        self.image = Image.open(image_path).convert("RGB")
        self.image_width, self.image_height = self.image.size

        # 数据划分
        np.random.seed(seed)
        n_samples = len(self.adata)
        indices = np.random.permutation(n_samples)
        train_size = int(n_samples * train_ratio)

        if split == "train":
            self.indices = indices[:train_size]
        elif split == "val":
            self.indices = indices[train_size:]
        else:
            self.indices = indices

        logger.info(f"{split}集大小: {len(self.indices)}")

        # 图像变换
        if transform is None:
            self.transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                )
            ])
        else:
            self.transform = transform

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        获取一个样本

        Returns:
            gene_expr: 基因表达向量，形状 [gene_dim]
            image_patch: 图像块张量，形状 [3, 224, 224]
        """
        real_idx = self.indices[idx]

        # 获取基因表达
        gene_expr = self.adata.X[real_idx].toarray().squeeze().astype(np.float32)
        gene_expr = torch.from_numpy(gene_expr)

        # 获取空间坐标
        x, y = self.adata.obsm["spatial"][real_idx]

        # 裁剪图像块
        half_size = self.patch_size // 2
        left = max(0, x - half_size)
        top = max(0, y - half_size)
        right = min(self.image_width, x + half_size)
        bottom = min(self.image_height, y + half_size)

        image_patch = self.image.crop((left, top, right, bottom))

        # 如果裁剪的图像块大小不对，进行填充
        if image_patch.size != (self.patch_size, self.patch_size):
            new_patch = Image.new("RGB", (self.patch_size, self.patch_size), (0, 0, 0))
            new_patch.paste(image_patch, (0, 0))
            image_patch = new_patch

        # 应用变换
        image_patch = self.transform(image_patch)

        return gene_expr, image_patch