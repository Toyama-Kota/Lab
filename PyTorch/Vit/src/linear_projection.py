import torch
import torch.nn as nn

class LinearProjection(nn.Module):
    def __init__(self, patch_dim, vec_length):
        """ [input]
            - patch_dim (int) : 一枚あたりのパッチのベクトルの長さ（= channels * (patch_size ** 2)）
            - vec_length (int) : パッチのベクトルが変換されたベクトルの長さ 
        """
        super().__init__()
        self.net = nn.Linear(patch_dim, vec_length)

    def forward(self, x):
        return self.net(x)