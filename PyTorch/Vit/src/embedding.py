import torch
import torch.nn as nn

from einops import repeat
from einops.layers.torch import Rearrange

class Embedding(nn.Module):
    def __init__(self, vec_length, patche_num):
        """ [input]
            - vec_length (int) : パッチのベクトルが変換されたベクトルの長さ
            - patche_num (int) : パッチの枚数
        """
        super().__init__()
        # class token
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        # position embedding
        self.pos_embedding = nn.Parameter(torch.randn(1, patche_num + 1, dim))
    
    def forward(self, x):
        # バッチサイズを抽出
        batch_size, _, __ = x.shape

        # クラストークン付加
        # x.shape : [batch_size, patche_num, patch_dim] -> [batch_size, patche_num + 1, patch_dim]
        cls_tokens = repeat(self.cls_token, "1 1 d -> b 1 d", b = batch_size)
        x = torch.concat([cls_tokens, x], dim = 1)

        # 位置エンコーディング
        x += self.pos_embedding

        return x