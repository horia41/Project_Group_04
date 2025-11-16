import torch.nn as nn
from einops.layers.torch import Rearrange
from timm.layers import to_2tuple

class HPEM(nn.Module):
    """
    Hybrid Patch Embedding Module (HPEM)
    This replaces the original 'stage1_conv_embed' from CSWinTransformer.
    It uses a "CNN-style" stem of two 3x3 convolutions with BatchNorm
    to perform the 4x downsampling.
    Based on the idea from Lu et al.
    """
    def __init__(self, img_size=256, patch_size=4, in_chans=3, embed_dim=64):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)

        # Calculate the output H/W (e.g., 256 / 4 = 64)
        self.img_size = img_size
        self.patch_size = patch_size
        self.patches_resolution = (img_size[0] // patch_size[0], img_size[1] // patch_size[1])
        self.num_patches = self.patches_resolution[0] * self.patches_resolution[1]

        # The "CNN Part"
        # First 2x downsampling
        self.conv1 = nn.Conv2d(in_chans, embed_dim // 2, kernel_size=3, stride=2, padding=1)
        self.bn1 = nn.BatchNorm2d(embed_dim // 2)
        self.relu = nn.GELU()

        # Second 2x downsampling (total 4x)
        self.conv2 = nn.Conv2d(embed_dim // 2, embed_dim, kernel_size=3, stride=2, padding=1)
        self.bn2 = nn.BatchNorm2d(embed_dim)

        # The "ViT Part" (Tokenization)
        # Rearrange to tokens: [B, C, H, W] -> [B, H*W, C]
        self.rearrange = Rearrange('b c h w -> b (h w) c',
                                   h=self.patches_resolution[0],
                                   w=self.patches_resolution[1])
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        # Pass through the CNN stem
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)

        # Convert to tokens
        x = self.rearrange(x)
        x = self.norm(x)
        return x