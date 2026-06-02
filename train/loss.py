import torch
import torch.nn as nn
import torch.nn.functional as F


class CLIPVAELoss(nn.Module):
    """
    CLIPVAE总损失函数 = 重构损失 + KL散度 + 对比学习损失

    论文中使用的权重:
    - kl_weight: 0.001
    - contrastive_weight: 0.1
    """

    def __init__(
            self,
            recon_loss_type: str = "mse",
            kl_weight: float = 0.001,
            contrastive_weight: float = 0.1
    ):
        super().__init__()

        self.recon_loss_type = recon_loss_type
        self.kl_weight = kl_weight
        self.contrastive_weight = contrastive_weight

        if recon_loss_type == "mse":
            self.recon_loss_fn = nn.MSELoss(reduction="mean")
        elif recon_loss_type == "poisson":
            self.recon_loss_fn = nn.PoissonNLLLoss(log_input=False, reduction="mean")
        else:
            raise ValueError(f"不支持的重构损失类型: {recon_loss_type}")

        self.contrastive_loss_fn = nn.CrossEntropyLoss()

    def forward(
            self,
            recon_gene: torch.Tensor,
            true_gene: torch.Tensor,
            mu: torch.Tensor,
            logvar: torch.Tensor,
            contrastive_logits: torch.Tensor
    ) -> tuple[torch.Tensor, dict[str, float]]:
        """
        计算总损失

        Args:
            recon_gene: 重构的基因表达
            true_gene: 真实的基因表达
            mu: 潜在分布均值
            logvar: 潜在分布对数方差
            contrastive_logits: 对比学习对数几率

        Returns:
            total_loss: 总损失
            loss_dict: 各损失分量的字典
        """
        batch_size = recon_gene.shape[0]

        # 1. 重构损失
        recon_loss = self.recon_loss_fn(recon_gene, true_gene)

        # 2. KL散度损失
        kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / batch_size

        # 3. 对比学习损失 (双向: 图像到基因 和 基因到图像)
        labels = torch.arange(batch_size, device=contrastive_logits.device)
        loss_i2g = self.contrastive_loss_fn(contrastive_logits, labels)
        loss_g2i = self.contrastive_loss_fn(contrastive_logits.T, labels)
        contrastive_loss = (loss_i2g + loss_g2i) / 2.0

        # 总损失
        total_loss = (
                recon_loss +
                self.kl_weight * kl_loss +
                self.contrastive_weight * contrastive_loss
        )

        # 损失字典
        loss_dict = {
            "total_loss": total_loss.item(),
            "recon_loss": recon_loss.item(),
            "kl_loss": kl_loss.item(),
            "contrastive_loss": contrastive_loss.item()
        }

        return total_loss, loss_dict