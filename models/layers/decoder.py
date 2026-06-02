import torch
import torch.nn as nn
import torch.nn.functional as F


class GeneDecoder(nn.Module):
    """基因表达解码器，基于多层感知机"""

    def __init__(
            self,
            latent_dim: int,
            hidden_dim: int,
            output_dim: int,
            num_layers: int = 3,
            dropout: float = 0.2,
            activation: str = "gelu"
    ):
        super().__init__()

        self.layers = nn.ModuleList()

        # 输入层
        self.layers.append(nn.Linear(latent_dim, hidden_dim))
        self.layers.append(nn.LayerNorm(hidden_dim))
        self.layers.append(self._get_activation(activation))
        self.layers.append(nn.Dropout(dropout))

        # 隐藏层
        for _ in range(num_layers - 1):
            self.layers.append(nn.Linear(hidden_dim, hidden_dim))
            self.layers.append(nn.LayerNorm(hidden_dim))
            self.layers.append(self._get_activation(activation))
            self.layers.append(nn.Dropout(dropout))

        # 输出层 (使用ReLU确保非负基因表达)
        self.layers.append(nn.Linear(hidden_dim, output_dim))
        self.layers.append(nn.ReLU())

    def _get_activation(self, name: str):
        if name == "gelu":
            return nn.GELU()
        elif name == "relu":
            return nn.ReLU()
        elif name == "leaky_relu":
            return nn.LeakyReLU(0.1)
        else:
            raise ValueError(f"不支持的激活函数: {name}")

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            z = layer(z)
        return z