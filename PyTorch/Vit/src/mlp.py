import torch
import torch.nn as nn

class MLP(nn.Module):
    def __init__(self, vec_length, hidden_dim):
        """ [input]
            - vec_length (int) : パッチのベクトルが変換されたベクトルの長さ
            - hidden_dim (int) : 隠れ層のノード数
        """
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(vec_length, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, vec_length)
        )

    def forward(self, x):
        return self.net(x)