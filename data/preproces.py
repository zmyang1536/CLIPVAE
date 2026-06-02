import os
import scanpy as sc
import numpy as np
from PIL import Image
from typing import Optional

from utils.logger import get_logger

logger = get_logger(__name__)


def preprocess_spatial_data(
        adata_path: str,
        image_path: str,
        output_dir: str,
        n_top_genes: int = 2000,
        min_counts: int = 10,
        min_cells: int = 5,
        target_sum: int = 10000
) -> None:
    """
    预处理空间转录组数据

    Args:
        adata_path: 原始AnnData文件路径
        image_path: 原始H&E图像路径
        output_dir: 输出目录
        n_top_genes: 高变基因数量
        min_counts: 每个细胞的最小计数
        min_cells: 每个基因的最小细胞数
        target_sum: 归一化目标总和
    """
    os.makedirs(output_dir, exist_ok=True)

    # 加载数据
    logger.info(f"加载原始数据: {adata_path}")
    adata = sc.read_h5ad(adata_path)

    # 基本过滤
    logger.info("执行基本过滤...")
    sc.pp.filter_cells(adata, min_counts=min_counts)
    sc.pp.filter_genes(adata, min_cells=min_cells)
    logger.info(f"过滤后: {adata.n_obs} 个细胞, {adata.n_vars} 个基因")

    # 归一化和对数变换
    logger.info("执行归一化和对数变换...")
    sc.pp.normalize_total(adata, target_sum=target_sum)
    sc.pp.log1p(adata)

    # 选择高变基因
    logger.info(f"选择前 {n_top_genes} 个高变基因...")
    sc.pp.highly_variable_genes(adata, n_top_genes=n_top_genes, flavor="seurat_v3")
    adata = adata[:, adata.var.highly_variable]
    logger.info(f"高变基因过滤后: {adata.n_vars} 个基因")

    # 保存处理后的数据
    output_adata_path = os.path.join(output_dir, "processed_adata.h5ad")
    adata.write(output_adata_path)
    logger.info(f"处理后的AnnData已保存到: {output_adata_path}")

    # 处理图像
    logger.info(f"处理H&E图像: {image_path}")
    image = Image.open(image_path).convert("RGB")
    output_image_path = os.path.join(output_dir, "processed_image.png")
    image.save(output_image_path)
    logger.info(f"处理后的图像已保存到: {output_image_path}")

    logger.info("数据预处理完成！")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="预处理空间转录组数据")
    parser.add_argument("--adata_path", type=str, required=True, help="原始AnnData文件路径")
    parser.add_argument("--image_path", type=str, required=True, help="原始H&E图像路径")
    parser.add_argument("--output_dir", type=str, required=True, help="输出目录")
    parser.add_argument("--n_top_genes", type=int, default=2000, help="高变基因数量")
    args = parser.parse_args()

    preprocess_spatial_data(
        adata_path=args.adata_path,
        image_path=args.image_path,
        output_dir=args.output_dir,
        n_top_genes=args.n_top_genes
    )