import torch
import torch.nn as nn
import torch.nn.functional as F


class GeneEncoder(nn.Module):
    """基因表达编码器，基于多层感知机"""

    def __init__(
            self,
            input_dim: int,
            hidden_dim: int,
            num_layers: int = 3,
            dropout: float = 0.2,
            activation: str = "gelu"
    ):
        super().__init__()

        self.layers = nn.ModuleList()

        # 输入层
        self.layers.append(nn.Linear(input_dim, hidden_dim))
        self.layers.append(nn.LayerNorm(hidden_dim))
        self.layers.append(self._get_activation(activation))
        self.layers.append(nn.Dropout(dropout))

        # 隐藏层
        for _ in range(num_layers - 1):
            self.layers.append(nn.Linear(hidden_dim, hidden_dim))
            self.layers.append(nn.LayerNorm(hidden_dim))
            self.layers.append(self._get_activation(activation))
            self.layers.append(nn.Dropout(dropout))

    def _get_activation(self, name: str):
        if name == "gelu":
            return nn.GELU()
        elif name == "relu":
            return nn.ReLU()
        elif name == "leaky_relu":
            return nn.LeakyReLU(0.1)
        else:
            raise ValueError(f"不支持的激活函数: {name}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x)
        return x