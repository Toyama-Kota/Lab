import torch
import torch.nn as nn

from einops import repeat
from einops.layers.torch import Rearrange

class Patching(nn.Module):
    def __init__(self, patch_size):
        """ [input]
            - patch_size (int) : パッチの縦の長さ（=横の長さ）
        """
        super().__init__()
        self.net = Rearrange("b c (h ph) (w pw) -> b (h w) (ph pw c)", ph = patch_size, pw = patch_size)
    
    def forward(self, x):
        return self.net(x)