import torch
import torch.nn as nn
import clip
from typing import Optional

from utils.logger import get_logger

logger = get_logger(__name__)


class CLIPImageEncoder(nn.Module):
    """CLIP图像编码器，使用预训练的ViT模型"""

    def __init__(
            self,
            model_name: str = "ViT-B/32",
            output_dim: int = 512,
            freeze: bool = True
    ):
        super().__init__()

        logger.info(f"加载预训练的CLIP模型: {model_name}")
        self.clip_model, self.preprocess = clip.load(model_name, device="cpu")

        # 冻结CLIP参数
        if freeze:
            for param in self.clip_model.parameters():
                param.requires_grad = False
            logger.info("已冻结CLIP模型的所有参数")

        # 投影层，将CLIP输出映射到目标维度
        self.projection = nn.Linear(self.clip_model.visual.output_dim, output_dim)
        self.norm = nn.LayerNorm(output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播

        Args:
            x: 输入图像，形状 [batch_size, 3, 224, 224]

        Returns:
            image_emb: 图像嵌入，形状 [batch_size, output_dim]
        """
        # CLIP图像编码
        with torch.set_grad_enabled(not self.freeze):
            image_features = self.clip_model.encode_image(x)
            image_features = image_features.float()  # CLIP默认使用float16

        # 投影和归一化
        image_emb = self.projection(image_features)
        image_emb = self.norm(image_emb)

        return image_emb