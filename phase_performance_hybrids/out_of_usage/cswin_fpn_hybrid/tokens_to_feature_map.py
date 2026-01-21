import torch
import torch.nn as nn
from einops.layers.torch import Rearrange

class TokensToFeatureMap(nn.Module):
    """
    Converts a sequence of tokens of shape [B, N, C] into a 2D feature map [B, C, H, W]
    given the expected spatial resolution (reso = H = W).
    """
    def __init__(self, reso: int):
        super().__init__()
        self.reso = reso
        self.rearrange = Rearrange('b (h w) c -> b c h w')

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, N, C] with N = reso * reso
        B, N, C = x.shape
        h = w = self.reso
        assert N == h * w, f"TokensToFeatureMap: N={N} does not match HxW={h*w}."
        return self.rearrange(x, h=h, w=w)
