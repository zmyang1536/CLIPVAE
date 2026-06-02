import os
import argparse
import yaml
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from data.dataset import SpatialTranscriptomicsDataset
from models.clipvae import CLIPVAE
from train.trainer import Trainer
from train.loss import CLIPVAELoss
from train.optimizer import get_optimizer, get_scheduler
from utils.logger import setup_logger, get_logger
from utils.seed import set_seed


def main():
    parser = argparse.ArgumentParser(description="训练CLIPVAE模型")
    parser.add_argument("--config", type=str, required=True, help="配置文件路径")
    parser.add_argument("--resume", type=str, default=None, help="恢复训练的检查点路径")
    parser.add_argument("--output_dir", type=str, default="outputs", help="输出目录")
    args = parser.parse_args()

    # 加载配置
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    # 设置输出目录
    config["logging"]["output_dir"] = os.path.join(
        args.output_dir,
        config["experiment_name"]
    )
    os.makedirs(config["logging"]["output_dir"], exist_ok=True)

    # 设置日志
    setup_logger(os.path.join(config["logging"]["output_dir"], "train.log"))
    logger = get_logger(__name__)

    # 设置随机种子
    set_seed(config["seed"])

    # 设置设备
    device = torch.device(config["device"] if torch.cuda.is_available() else "cpu")
    logger.info(f"使用设备: {device}")

    # 加载数据集
    logger.info("加载数据集...")
    train_dataset = SpatialTranscriptomicsDataset(
        adata_path=config["data"]["adata_path"],
        image_path=config["data"]["image_path"],
        patch_size=config["data"]["patch_size"],
        split="train",
        train_ratio=config["data"]["train_ratio"],
        seed=config["seed"]
    )

    val_dataset = SpatialTranscriptomicsDataset(
        adata_path=config["data"]["adata_path"],
        image_path=config["data"]["image_path"],
        patch_size=config["data"]["patch_size"],
        split="val",
        train_ratio=config["data"]["train_ratio"],
        seed=config["seed"]
    )

    # 创建数据加载器
    train_loader = DataLoader(
        train_dataset,
        batch_size=config["data"]["batch_size"],
        shuffle=True,
        num_workers=config["num_workers"],
        pin_memory=config["pin_memory"]
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config["data"]["val_batch_size"],
        shuffle=False,
        num_workers=config["num_workers"],
        pin_memory=config["pin_memory"]
    )

    logger.info(f"训练集大小: {len(train_dataset)}, 验证集大小: {len(val_dataset)}")

    # 创建模型
    logger.info("创建CLIPVAE模型...")
    model = CLIPVAE(**config["model"])
    model = model.to(device)

    # 多GPU支持
    if torch.cuda.device_count() > 1:
        logger.info(f"使用 {torch.cuda.device_count()} 个GPU进行训练")
        model = nn.DataParallel(model)

    # 损失函数
    criterion = CLIPVAELoss(**config["loss"])

    # 优化器和学习率调度器
    optimizer = get_optimizer(model, config["training"])
    scheduler = get_scheduler(optimizer, config["training"])

    # 创建训练器
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        config=config,
        device=device
    )

    # 恢复训练
    if args.resume is not None:
        logger.info(f"从检查点恢复训练: {args.resume}")
        trainer.load_checkpoint(args.resume)

    # 开始训练
    trainer.train()


if __name__ == "__main__":
    main()