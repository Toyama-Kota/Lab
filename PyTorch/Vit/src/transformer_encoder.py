import torch
import torch.nn as nn

# ◆ ---------- オリジナル ---------- ◆
from mlp import MLP
from multi_head_attention import MultiHeadAttention

class TransformerEncoder(nn.Module):
    def __init__(self, vec_length, n_heads, mlp_dim, depth):
        """ [input]
            - vec_length (int) : 各パッチのベクトルが変換されたベクトルの長さ（参考[1] (1)式 D）
            - depth (int) : Transformer Encoder の層の深さ（参考[1] (2)式 L）
            - n_heads (int) : Multi-Head Attention の head の数
            - mlp_dim (int) : MLP の隠れ層のノード数
        """
        super().__init__()

        # Layers
        self.norm = nn.LayerNorm(vec_length)
        self.multi_head_attention = MultiHeadAttention(vec_length = vec_length, n_heads = n_heads)
        self.mlp = MLP(vec_length = vec_length, hidden_dim = mlp_dim)
        self.depth = depth

    def forward(self, x):
        for _ in range(self.depth):
            x = self.multi_head_attention(self.norm(x)) + x
            x = self.mlp(self.norm(x)) + x

        return x