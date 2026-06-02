import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from typing import Dict, Optional, Any

from utils.logger import get_logger
from utils.checkpoint import save_checkpoint, load_checkpoint
from utils.metrics import calculate_clustering_metrics

logger = get_logger(__name__)


class Trainer:
    """CLIPVAE模型训练器"""

    def __init__(
            self,
            model: nn.Module,
            train_loader: DataLoader,
            val_loader: DataLoader,
            criterion: nn.Module,
            optimizer: optim.Optimizer,
            scheduler: Optional[optim.lr_scheduler._LRScheduler],
            config: Dict[str, Any],
            device: torch.device
    ):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.config = config
        self.device = device

        # 训练状态
        self.start_epoch = 0
        self.best_metric = float("-inf") if config["logging"]["mode"] == "max" else float("inf")
        self.global_step = 0

        # 日志和保存目录
        self.log_dir = os.path.join(config["logging"]["output_dir"], "logs")
        self.checkpoint_dir = os.path.join(config["logging"]["output_dir"], "checkpoints")
        os.makedirs(self.log_dir, exist_ok=True)
        os.makedirs(self.checkpoint_dir, exist_ok=True)

        self.writer = SummaryWriter(self.log_dir)

        # 分布式训练支持
        self.is_distributed = torch.distributed.is_available() and torch.distributed.is_initialized()
        self.rank = torch.distributed.get_rank() if self.is_distributed else 0

        logger.info(f"初始化训练器: 设备={device}, 分布式={self.is_distributed}")

    def train(self) -> None:
        """开始训练循环"""
        logger.info("开始训练...")

        for epoch in range(self.start_epoch, self.config["training"]["epochs"]):
            logger.info(f"Epoch {epoch + 1}/{self.config['training']['epochs']}")

            # 训练一个epoch
            train_losses = self._train_epoch(epoch)

            # 验证
            if (epoch + 1) % self.config["logging"]["val_interval"] == 0:
                val_losses, val_metrics = self._validate_epoch(epoch)

                # 更新学习率
                if self.scheduler is not None:
                    if isinstance(self.scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                        self.scheduler.step(val_losses["total_loss"])
                    else:
                        self.scheduler.step()

                # 保存检查点
                self._save_checkpoint(epoch, val_losses, val_metrics)

            logger.info("-" * 60)

        logger.info(f"训练完成！最佳 {self.config['logging']['monitor']}: {self.best_metric:.4f}")
        self.writer.close()

    def _train_epoch(self, epoch: int) -> Dict[str, float]:
        """训练一个epoch"""
        self.model.train()
        total_losses = {
            "total_loss": 0.0,
            "recon_loss": 0.0,
            "kl_loss": 0.0,
            "contrastive_loss": 0.0
        }

        if self.rank == 0:
            pbar = tqdm(self.train_loader, desc=f"Training Epoch {epoch + 1}")
        else:
            pbar = self.train_loader

        for batch_idx, (gene_expr, image_patches) in enumerate(pbar):
            gene_expr = gene_expr.to(self.device)
            image_patches = image_patches.to(self.device)

            # 前向传播
            self.optimizer.zero_grad()
            recon_gene, mu, logvar, z, contrastive_logits = self.model(gene_expr, image_patches)

            # 计算损失
            loss, loss_dict = self.criterion(recon_gene, gene_expr, mu, logvar, contrastive_logits)

            # 反向传播
            loss.backward()

            # 梯度裁剪
            if self.config["training"]["gradient_clip_val"] > 0:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.config["training"]["gradient_clip_val"]
                )

            self.optimizer.step()

            # 累积损失
            for k, v in loss_dict.items():
                total_losses[k] += v

            self.global_step += 1

            # 日志
            if self.rank == 0 and batch_idx % self.config["logging"]["log_interval"] == 0:
                self.writer.add_scalar("train/total_loss", loss_dict["total_loss"], self.global_step)
                self.writer.add_scalar("train/recon_loss", loss_dict["recon_loss"], self.global_step)
                self.writer.add_scalar("train/kl_loss", loss_dict["kl_loss"], self.global_step)
                self.writer.add_scalar("train/contrastive_loss", loss_dict["contrastive_loss"], self.global_step)
                self.writer.add_scalar(
                    "train/lr",
                    self.optimizer.param_groups[0]["lr"],
                    self.global_step
                )
                pbar.set_postfix({"loss": f"{loss_dict['total_loss']:.4f}"})

        # 计算平均损失
        avg_losses = {k: v / len(self.train_loader) for k, v in total_losses.items()}

        logger.info("训练结果:")
        for k, v in avg_losses.items():
            logger.info(f"  {k}: {v:.4f}")

        return avg_losses

    def _validate_epoch(self, epoch: int) -> tuple[Dict[str, float], Dict[str, float]]:
        """验证一个epoch"""
        self.model.eval()
        total_losses = {
            "total_loss": 0.0,
            "recon_loss": 0.0,
            "kl_loss": 0.0,
            "contrastive_loss": 0.0
        }

        all_z = []
        all_labels = []

        with torch.no_grad():
            if self.rank == 0:
                pbar = tqdm(self.val_loader, desc=f"Validation Epoch {epoch + 1}")
            else:
                pbar = self.val_loader

            for gene_expr, image_patches in pbar:
                gene_expr = gene_expr.to(self.device)
                image_patches = image_patches.to(self.device)

                recon_gene, mu, logvar, z, contrastive_logits = self.model(gene_expr, image_patches)
                loss, loss_dict = self.criterion(recon_gene, gene_expr, mu, logvar, contrastive_logits)

                # 累积损失
                for k, v in loss_dict.items():
                    total_losses[k] += v

                # 收集嵌入用于聚类评估
                all_z.append(mu.cpu())

                # 如果有真实标签，收集标签
                if "label" in self.val_loader.dataset.adata.obs:
                    labels = self.val_loader.dataset.adata.obs["label"].values[
                        self.val_loader.dataset.indices[
                        pbar.n - len(gene_expr): pbar.n
                        ]
                    ]
                    all_labels.extend(labels)

        # 计算平均损失
        avg_losses = {k: v / len(self.val_loader) for k, v in total_losses.items()}

        # 计算聚类指标
        all_z = torch.cat(all_z).numpy()
        metrics = {}

        if len(all_labels) > 0:
            all_labels = np.array(all_labels)
            metrics = calculate_clustering_metrics(all_z, all_labels)

        # 记录日志
        logger.info("验证结果:")
        for k, v in avg_losses.items():
            logger.info(f"  {k}: {v:.4f}")
            self.writer.add_scalar(f"val/{k}", v, epoch)

        for k, v in metrics.items():
            logger.info(f"  {k}: {v:.4f}")
            self.writer.add_scalar(f"val/{k}", v, epoch)

        return avg_losses, metrics

    def _save_checkpoint(
            self,
            epoch: int,
            val_losses: Dict[str, float],
            val_metrics: Dict[str, float]
    ) -> None:
        """保存检查点"""
        if self.rank != 0:
            return

        # 确定监控指标
        if self.config["logging"]["monitor"] in val_losses:
            monitor_value = val_losses[self.config["logging"]["monitor"]]
        elif self.config["logging"]["monitor"] in val_metrics:
            monitor_value = val_metrics[self.config["logging"]["monitor"]]
        else:
            logger.warning(f"监控指标 {self.config['logging']['monitor']} 不存在，使用总损失")
            monitor_value = val_losses["total_loss"]

        is_best = False

        # 判断是否是最佳模型
        if self.config["logging"]["mode"] == "max":
            if monitor_value > self.best_metric:
                self.best_metric = monitor_value
                is_best = True
        else:
            if monitor_value < self.best_metric:
                self.best_metric = monitor_value
                is_best = True

        # 准备检查点
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict() if self.scheduler else None,
            "best_metric": self.best_metric,
            "config": self.config
        }

        # 保存最新模型
        save_checkpoint(
            checkpoint,
            os.path.join(self.checkpoint_dir, "latest_checkpoint.pth")
        )

        # 保存最佳模型
        if is_best:
            logger.info(f"发现新的最佳模型，{self.config['logging']['monitor']} = {monitor_value:.4f}")
            save_checkpoint(
                checkpoint,
                os.path.join(self.checkpoint_dir, "best_model.pth")
            )

        # 定期保存
        if (epoch + 1) % self.config["logging"]["save_interval"] == 0:
            save_checkpoint(
                checkpoint,
                os.path.join(self.checkpoint_dir, f"checkpoint_epoch_{epoch + 1}.pth")
            )

    def load_checkpoint(self, checkpoint_path: str) -> None:
        """从检查点恢复训练"""
        logger.info(f"加载检查点: {checkpoint_path}")
        checkpoint = load_checkpoint(checkpoint_path, self.device)

        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

        if self.scheduler is not None and checkpoint["scheduler_state_dict"] is not None:
            self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

        self.start_epoch = checkpoint["epoch"] + 1
        self.best_metric = checkpoint["best_metric"]

        logger.info(f"恢复训练从 epoch {self.start_epoch} 开始，最佳指标: {self.best_metric:.4f}")