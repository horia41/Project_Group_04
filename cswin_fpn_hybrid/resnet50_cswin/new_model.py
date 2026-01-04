import torch
import torch.nn as nn
import timm
from einops.layers.torch import Rearrange
import math


class ECA(nn.Module):
    """
    Efficient Channel Attention (ECA)
    Paper: https://arxiv.org/abs/1910.03151
    """

    def __init__(self, channels, gamma=2, b=1):
        super().__init__()
        # Dynamic kernel size based on channel dimension
        t = int(abs((math.log(channels, 2) + b) / gamma))
        k = t if t % 2 else t + 1

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k, padding=k // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # x: [B, C, H, W]
        y = self.avg_pool(x)  # [B, C, 1, 1]
        y = y.squeeze(-1).transpose(-1, -2)  # [B, 1, C]
        y = self.conv(y)  # 1D Conv acts on channels
        y = y.transpose(-1, -2).unsqueeze(-1)  # [B, C, 1, 1]
        y = self.sigmoid(y)
        return x * y.expand_as(x)


class GatedECABridge(nn.Module):
    """
    - GroupNorm instead of BatchNorm
    - Residual connection
    - Softer gating
    """

    def __init__(self, in_ch=1024, out_ch=256, drop_rate=0.0):
        super().__init__()

        # 1. Feature Branch
        self.proj = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=1, bias=False),
            nn.GroupNorm(32, out_ch)  # Changed from BatchNorm
        )

        # 2. Gating Branch
        self.gate = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=1, bias=False),
            nn.GroupNorm(32, out_ch),  # Changed from BatchNorm
            nn.Tanh()  # Softer than Sigmoid
        )

        # 3. ECA
        self.eca = ECA(out_ch)

        # 4. Residual connection
        self.residual = nn.Conv2d(in_ch, out_ch, kernel_size=1, bias=False) if in_ch != out_ch else nn.Identity()

        # 5. Activation & Dropout
        self.act = nn.GELU()
        self.drop = nn.Dropout(drop_rate)

    def forward(self, x):
        residual = self.residual(x)

        features = self.proj(x)
        mask = self.gate(x)

        x = features * (0.5 + 0.5 * mask)

        x = self.eca(x)
        x = self.act(x)


        x = x + residual
        x = self.drop(x)

        return x


class ResNetCSWinHybrid(nn.Module):
    def __init__(self, num_classes=2, resnet_pretrained=True, cswin_pretrained=True, drop_rate=0.2, drop_path_rate=0.2):
        super().__init__()

        # 1. ResNet50 pretrained
        self.resnet_stem = timm.create_model(
            'resnet50',
            pretrained=resnet_pretrained,
            features_only=True,
            out_indices=(3,)
        )

        # 2. CSWin-Tiny pretrained
        cswin_tiny = timm.create_model(
            'CSWin_64_12211_tiny_224',
            pretrained=cswin_pretrained,
            drop_path_rate=drop_path_rate
        )

        self.cswin_input_dim = cswin_tiny.stage3[0].dim  # 256

        # 3. Gated Bridge
        self.bridge = GatedECABridge(
            in_ch=1024,
            out_ch=self.cswin_input_dim,
            drop_rate=drop_rate
        )

        # 4. Tokenizer
        self.tokenizer = Rearrange('b c h w -> b (h w) c')
        self.pos_embed = nn.Parameter(torch.zeros(1, 14 * 14, self.cswin_input_dim))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        self.pos_drop = nn.Dropout(drop_rate)

        # 5. CSWin Body
        self.stage3 = cswin_tiny.stage3
        self.merge3 = cswin_tiny.merge3
        self.stage4 = cswin_tiny.stage4
        self.norm = cswin_tiny.norm

        # 6. Classifier Head
        self.head_in_features = cswin_tiny.norm.normalized_shape[0]

        # Added Dropout before the final classifier
        self.head_drop = nn.Dropout(drop_rate)
        self.head = nn.Linear(self.head_in_features, num_classes)

        del cswin_tiny

    def forward(self, x):
        # 1. ResNet Stem
        x = self.resnet_stem(x)[0]  # [B, 1024, 14, 14]
        # print(f"After ResNet: {x.shape}")

        # 2. Gated Bridge (Gate + ECA + Dropout happen here)
        x = self.bridge(x)  # [B, 256, 14, 14]
        # print(f"After bridge: {x.shape}")

        # 3. Tokenizer
        x = self.tokenizer(x)
        x = x + self.pos_embed
        x = self.pos_drop(x)
        # print(f"After tokenizer: {x.shape}")

        # 4. CSWin Stages (Now with DropPath active inside them)
        for blk in self.stage3:
            x = blk(x)
        x = self.merge3(x)

        for blk in self.stage4:
            x = blk(x)
        x = self.norm(x)

        # 5. Classifier Head
        x = torch.mean(x, dim=1)
        x = self.head_drop(x)  # Added dropout here
        out = self.head(x)

        return out