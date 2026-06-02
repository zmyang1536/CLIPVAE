import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple

from models.layers.clip_encoder import CLIPImageEncoder
from models.layers.gene_encoder import GeneEncoder
from models.layers.decoder import GeneDecoder
from utils.logger import get_logger

logger = get_logger(__name__)


class CLIPVAE(nn.Module):
    """
    CLIPVAE: 多模态空间转录组分析框架

    论文原文: "CLIPVAE: Contrastive Learning-Informed Variational Autoencoder
              for Multi-Modal Spatial Transcriptomics Analysis"

    核心思想: 结合CLIP的图像-文本对齐能力和VAE的生成能力，
             学习空间转录组数据的统一多模态表示。
    """

    def __init__(
            self,
            gene_dim: int,
            image_embedding_dim: int = 768,
            hidden_dim: int = 512,
            latent_dim: int = 128,
            num_gene_layers: int = 3,
            dropout: float = 0.2,
            activation: str = "gelu",
            freeze_clip: bool = True
    ):
        super().__init__()

        self.gene_dim = gene_dim
        self.image_embedding_dim = image_embedding_dim
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.freeze_clip = freeze_clip

        # 1. 图像编码器 (使用预训练的CLIP ViT-B/32)
        self.image_encoder = CLIPImageEncoder(
            model_name="ViT-B/32",
            output_dim=hidden_dim,
            freeze=freeze_clip
        )

        # 2. 基因表达编码器
        self.gene_encoder = GeneEncoder(
            input_dim=gene_dim,
            hidden_dim=hidden_dim,
            num_layers=num_gene_layers,
            dropout=dropout,
            activation=activation
        )

        # 3. 多模态融合层
        self.fusion_layer = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            self._get_activation(activation),
            nn.Dropout(dropout)
        )

        # 4. VAE 均值和对数方差投影
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)

        # 5. 基因表达解码器
        self.decoder = GeneDecoder(
            latent_dim=latent_dim,
            hidden_dim=hidden_dim,
            output_dim=gene_dim,
            num_layers=num_gene_layers,
            dropout=dropout,
            activation=activation
        )

        # 6. 对比学习投影头
        self.image_proj = nn.Linear(hidden_dim, latent_dim)
        self.gene_proj = nn.Linear(hidden_dim, latent_dim)

        # 初始化权重
        self._init_weights()

        logger.info(
            f"初始化 CLIPVAE: 基因维度={gene_dim}, 图像嵌入维度={image_embedding_dim}, "
            f"隐藏维度={hidden_dim}, 潜在维度={latent_dim}, 冻结CLIP={freeze_clip}"
        )

    def _get_activation(self, name: str):
        if name == "gelu":
            return nn.GELU()
        elif name == "relu":
            return nn.ReLU()
        elif name == "leaky_relu":
            return nn.LeakyReLU(0.1)
        else:
            raise ValueError(f"不支持的激活函数: {name}")

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """重参数化技巧"""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(
            self,
            gene_expr: torch.Tensor,
            image_patches: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        前向传播

        Args:
            gene_expr: 基因表达矩阵，形状 [batch_size, gene_dim]
            image_patches: H&E图像块，形状 [batch_size, 3, 224, 224]

        Returns:
            recon_gene: 重构的基因表达
            mu: 潜在分布均值
            logvar: 潜在分布对数方差
            z: 采样的潜在向量
            contrastive_logits: 对比学习对数几率
        """
        batch_size = gene_expr.shape[0]

        # 编码图像和基因
        image_emb = self.image_encoder(image_patches)  # [batch_size, hidden_dim]
        gene_emb = self.gene_encoder(gene_expr)  # [batch_size, hidden_dim]

        # 多模态融合
        fused_emb = torch.cat([image_emb, gene_emb], dim=1)
        fused_emb = self.fusion_layer(fused_emb)

        # 计算VAE参数
        mu = self.fc_mu(fused_emb)
        logvar = self.fc_logvar(fused_emb)
        z = self.reparameterize(mu, logvar)

        # 解码重构基因表达
        recon_gene = self.decoder(z)

        # 对比学习投影
        image_proj = F.normalize(self.image_proj(image_emb), dim=1)
        gene_proj = F.normalize(self.gene_proj(gene_emb), dim=1)

        # 计算对比学习对数几率 (温度系数=0.07)
        temperature = 0.07
        contrastive_logits = torch.matmul(image_proj, gene_proj.T) / temperature

        return recon_gene, mu, logvar, z, contrastive_logits

    def encode(
            self,
            gene_expr: torch.Tensor,
            image_patches: torch.Tensor,
            return_distribution: bool = False
    ) -> torch.Tensor:
        """仅编码，用于推理阶段"""
        image_emb = self.image_encoder(image_patches)
        gene_emb = self.gene_encoder(gene_expr)

        fused_emb = torch.cat([image_emb, gene_emb], dim=1)
        fused_emb = self.fusion_layer(fused_emb)

        mu = self.fc_mu(fused_emb)
        logvar = self.fc_logvar(fused_emb)

        if return_distribution:
            return mu, logvar
        else:
            return mu  # 推理时使用均值作为嵌入