import torch
import torch.nn as nn

class MLPHead(nn.Module):
    def __init__(self, vec_length, out_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(vec_length),
            nn.Linear(vec_length, out_dim)
        )
    
    def forward(self, x):
        x = self.net(x)
        return x