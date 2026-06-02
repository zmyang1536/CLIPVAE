import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scanpy as sc
from typing import Optional

from utils.logger import get_logger

logger = get_logger(__name__)

# 设置绘图风格
sns.set_style("white")
plt.rcParams["figure.dpi"] = 100
plt.rcParams["font.family"] = "Arial"


def plot_umap(
        embeddings: np.ndarray,
        labels: np.ndarray,
        save_path: str,
        title: str = "UMAP Visualization",
        palette: str = "tab20"
) -> None:
    """
    绘制UMAP可视化图

    Args:
        embeddings: 嵌入向量，形状 [n_samples, n_features]
        labels: 标签，形状 [n_samples]
        save_path: 保存路径
        title: 图标题
        palette: 调色板
    """
    logger.info(f"绘制UMAP图: {save_path}")

    # 计算UMAP
    import umap
    reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, random_state=42)
    umap_embeddings = reducer.fit_transform(embeddings)

    # 绘图
    plt.figure(figsize=(8, 6))
    sns.scatterplot(
        x=umap_embeddings[:, 0],
        y=umap_embeddings[:, 1],
        hue=labels,
        palette=palette,
        s=10,
        alpha=0.8
    )

    plt.title(title, fontsize=14)
    plt.xlabel("UMAP 1", fontsize=12)
    plt.ylabel("UMAP 2", fontsize=12)
    plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.tight_layout()

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()


def plot_spatial_distribution(
        adata: sc.AnnData,
        save_path: str,
        color: str = "leiden",
        title: str = "Spatial Distribution",
        spot_size: float = 1.5
) -> None:
    """
    绘制空间分布图

    Args:
        adata: AnnData对象，包含obsm["spatial"]
        save_path: 保存路径
        color: 用于着色的列名
        title: 图标题
        spot_size: 点的大小
    """
    logger.info(f"绘制空间分布图: {save_path}")

    plt.figure(figsize=(8, 8))
    sc.pl.spatial(
        adata,
        color=color,
        spot_size=spot_size,
        title=title,
        show=False
    )

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()