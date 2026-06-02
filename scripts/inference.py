import os
import argparse
import torch
import numpy as np
import scanpy as sc
from torch.utils.data import DataLoader
from tqdm import tqdm

from data.dataset import SpatialTranscriptomicsDataset
from models.clipvae import CLIPVAE
from utils.logger import setup_logger, get_logger
from utils.seed import set_seed
from utils.checkpoint import load_checkpoint
from utils.visualization import plot_umap, plot_spatial_distribution

logger = get_logger(__name__)


def main():
    parser = argparse.ArgumentParser(description="CLIPVAE推理和可视化")
    parser.add_argument("--checkpoint", type=str, required=True, help="模型检查点路径")
    parser.add_argument("--adata_path", type=str, required=True, help="AnnData文件路径")
    parser.add_argument("--image_path", type=str, required=True, help="H&E图像路径")
    parser.add_argument("--output_dir", type=str, default="outputs/inference", help="输出目录")
    parser.add_argument("--batch_size", type=int, default=128, help="批次大小")
    parser.add_argument("--device", type=str, default="cuda", help="设备")
    parser.add_argument("--n_clusters", type=int, default=10, help="聚类数量")
    args = parser.parse_args()

    # 设置输出目录
    os.makedirs(args.output_dir, exist_ok=True)

    # 设置日志
    setup_logger(os.path.join(args.output_dir, "inference.log"))
    logger.info("开始CLIPVAE推理")

    # 设置随机种子
    set_seed(42)

    # 设置设备
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    logger.info(f"使用设备: {device}")

    # 加载检查点
    logger.info(f"加载检查点: {args.checkpoint}")
    checkpoint = load_checkpoint(args.checkpoint, device)
    config = checkpoint["config"]

    # 创建数据集
    logger.info("加载数据集...")
    dataset = SpatialTranscriptomicsDataset(
        adata_path=args.adata_path,
        image_path=args.image_path,
        patch_size=config["data"]["patch_size"],
        split="all"
    )

    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )

    # 创建模型
    logger.info("创建模型...")
    model = CLIPVAE(**config["model"])
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)
    model.eval()

    # 生成嵌入
    logger.info("生成多模态嵌入...")
    all_z = []

    with torch.no_grad():
        for gene_expr, image_patches in tqdm(dataloader, desc="Generating embeddings"):
            gene_expr = gene_expr.to(device)
            image_patches = image_patches.to(device)

            mu = model.encode(gene_expr, image_patches)
            all_z.append(mu.cpu())

    all_z = torch.cat(all_z).numpy()

    # 保存嵌入到AnnData
    adata = dataset.adata.copy()
    adata.obsm["X_clipvae"] = all_z

    # 聚类
    logger.info(f"执行Leiden聚类，n_clusters={args.n_clusters}")
    sc.pp.neighbors(adata, use_rep="X_clipvae", n_neighbors=15)
    sc.tl.leiden(adata, resolution=1.0)

    # 保存结果
    output_adata_path = os.path.join(args.output_dir, "clipvae_results.h5ad")
    adata.write(output_adata_path)
    logger.info(f"结果已保存到: {output_adata_path}")

    # 可视化
    logger.info("生成可视化图...")

    # UMAP可视化
    umap_path = os.path.join(args.output_dir, "umap_visualization.png")
    plot_umap(
        embeddings=all_z,
        labels=adata.obs["leiden"].values,
        save_path=umap_path,
        title="CLIPVAE UMAP Visualization"
    )

    # 空间分布可视化
    spatial_path = os.path.join(args.output_dir, "spatial_distribution.png")
    plot_spatial_distribution(
        adata=adata,
        save_path=spatial_path,
        color="leiden",
        title="CLIPVAE Spatial Distribution"
    )

    logger.info("推理和可视化完成！")


if __name__ == "__main__":
    main()