import os
import argparse
import yaml
import torch
import numpy as np
import scanpy as sc
from torch.utils.data import DataLoader
from tqdm import tqdm

from data.dataset import SpatialTranscriptomicsDataset
from models.clipvae import CLIPVAE
from train.loss import CLIPVAELoss
from utils.metrics import calculate_clustering_metrics
from utils.logger import setup_logger, get_logger
from utils.seed import set_seed
from utils.checkpoint import load_checkpoint

logger = get_logger(__name__)


def main():
    parser = argparse.ArgumentParser(description="评估CLIPVAE模型")
    parser.add_argument("--checkpoint", type=str, required=True, help="模型检查点路径")
    parser.add_argument("--adata_path", type=str, required=True, help="AnnData文件路径")
    parser.add_argument("--image_path", type=str, required=True, help="H&E图像路径")
    parser.add_argument("--output_dir", type=str, default="outputs/evaluation", help="输出目录")
    parser.add_argument("--batch_size", type=int, default=128, help="批次大小")
    parser.add_argument("--device", type=str, default="cuda", help="设备")
    args = parser.parse_args()

    # 设置输出目录
    os.makedirs(args.output_dir, exist_ok=True)

    # 设置日志
    setup_logger(os.path.join(args.output_dir, "evaluate.log"))
    logger.info("开始评估CLIPVAE模型")

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

    # 损失函数
    criterion = CLIPVAELoss(**config["loss"])

    # 评估
    logger.info("开始评估...")
    total_losses = {
        "total_loss": 0.0,
        "recon_loss": 0.0,
        "kl_loss": 0.0,
        "contrastive_loss": 0.0
    }

    all_z = []
    all_labels = []

    with torch.no_grad():
        for gene_expr, image_patches in tqdm(dataloader, desc="Evaluating"):
            gene_expr = gene_expr.to(device)
            image_patches = image_patches.to(device)

            recon_gene, mu, logvar, z, contrastive_logits = model(gene_expr, image_patches)
            loss, loss_dict = criterion(recon_gene, gene_expr, mu, logvar, contrastive_logits)

            # 累积损失
            for k, v in loss_dict.items():
                total_losses[k] += v

            # 收集嵌入
            all_z.append(mu.cpu())

    # 计算平均损失
    avg_losses = {k: v / len(dataloader) for k, v in total_losses.items()}

    # 计算聚类指标
    all_z = torch.cat(all_z).numpy()
    metrics = {}

    if "label" in dataset.adata.obs:
        all_labels = dataset.adata.obs["label"].values
        metrics = calculate_clustering_metrics(all_z, all_labels)

    # 保存结果
    logger.info("评估结果:")
    for k, v in avg_losses.items():
        logger.info(f"  {k}: {v:.4f}")

    for k, v in metrics.items():
        logger.info(f"  {k}: {v:.4f}")

    # 保存结果到文件
    result_path = os.path.join(args.output_dir, "evaluation_results.txt")
    with open(result_path, "w") as f:
        f.write("CLIPVAE 评估结果\n")
        f.write("=" * 30 + "\n\n")
        f.write("损失函数:\n")
        for k, v in avg_losses.items():
            f.write(f"  {k}: {v:.4f}\n")

        if metrics:
            f.write("\n聚类指标:\n")
            for k, v in metrics.items():
                f.write(f"  {k}: {v:.4f}\n")

    logger.info(f"评估结果已保存到: {result_path}")

    # 保存嵌入
    adata = dataset.adata.copy()
    adata.obsm["X_clipvae"] = all_z
    embedding_path = os.path.join(args.output_dir, "clipvae_embeddings.h5ad")
    adata.write(embedding_path)
    logger.info(f"嵌入结果已保存到: {embedding_path}")

    logger.info("评估完成！")


if __name__ == "__main__":
    main()