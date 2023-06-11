import torch
import torch.nn as nn

from einops import repeat
from einops.layers.torch import Rearrange

class MultiHeadAttention(nn.Module):
    def __init__(self, vec_length, n_heads):
        """ [input]
            - vec_length (int) : パッチのベクトルが変換されたベクトルの長さ
            - n_heads (int) : heads の数
        """
        super().__init__()
        self.n_heads = n_heads
        self.dim_heads = vec_length // n_heads

        self.Weight_q = nn.Linear(vec_length, vec_length)
        self.Weight_k = nn.Linear(vec_length, vec_length)
        self.Weight_v = nn.Linear(vec_length, vec_length)

        self.split_into_heads = Rearrange("b n (h d) -> b h n d", h = self.n_heads)
        self.softmax = nn.Softmax(vec_length = -1)
        self.concat = Rearrange("b h n d -> b n (h d)", h = self.n_heads)

    def forward(self, x):
        """[input]
            - x (torch.Tensor)
                - x.shape = torch.Size([batch_size, n_patches + 1, vec_length])
        """
        q = self.Weight_q(x)
        k = self.Weight_k(x)
        v = self.Weight_v(x)

        q = self.split_into_heads(q)
        k = self.split_into_heads(k)
        v = self.split_into_heads(v)

        # Logit[i] = Q[i] * tK[i] / sqrt(D) (i = 1, ... , n_heads)
        # AttentionWeight[i] = Softmax(Logit[i]) (i = 1, ... , n_heads)
        logit = torch.matmul(q, k.transpose(-1, -2)) * (self.dim_heads ** -0.5)
        attention_weight = self.softmax(logit)

        output = torch.matmul(attention_weight, v)
        output = self.concat(output)
        return output