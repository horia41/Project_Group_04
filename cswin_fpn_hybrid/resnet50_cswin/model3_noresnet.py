import torch
import torch.nn as nn
import timm
from cswin_fpn_hybrid.cswin import models
from cswin_fpn_hybrid.cswin.models import CSWin_64_12211_tiny_224


class SelfAttention(nn.Module):
    """
    Self-Attention layer operating on a single token stream.
    Matches the parameterization depth of the previous cross-attention
    (separate projections) for fair comparison.
    """

    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        
    def forward(self, x):
        """
        Args:
            x: Tokens [B, N, C]
        """
        B, N, C = x.shape

        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)

        return x


class ResNetCSWinHybridV3(nn.Module):
    """
    CSWin-only variant with a fusion block replaced by self-attention.

    Architecture:
    1. CSWin processes the image through stages 1 & 2 to 14x14 tokens
    2. Self-attention + FFN applied at the fusion point
    3. Continue through CSWin stages 3 & 4
    """

    def __init__(self, num_classes=2, resnet_pretrained=True, cswin_pretrained=True,
                 drop_rate=0.2, drop_path_rate=0.2, num_fusion_heads=8):
        super().__init__()

        # CSWin transformer
        cswin_tiny = timm.create_model(
            'CSWin_64_12211_tiny_224',
            pretrained=cswin_pretrained,
            drop_path_rate=drop_path_rate,
            num_classes=num_classes  # Will be replaced anyway
        )

        # CSWin transformer (pre-fusion stages)
        # cswin_transformer.py exposes stage1_conv_embed instead of patch_embed
        self.conv_embed = cswin_tiny.stage1_conv_embed
        self.stage1 = cswin_tiny.stage1
        self.merge1 = cswin_tiny.merge1
        self.stage2 = cswin_tiny.stage2
        self.merge2 = cswin_tiny.merge2  # After this: 14x14 spatial resolution
        # Get dimension after merge2
        # After merge2, we have 4*embed_dim channels = 256 for CSWin_tiny
        self.cswin_merged_dim = cswin_tiny.stage3[0].dim  # 256

        # Fusion (self-attention + FFN)
        self.self_attn = SelfAttention(
            dim=self.cswin_merged_dim,
            num_heads=num_fusion_heads,
            attn_drop=drop_rate,
            proj_drop=drop_rate
        )
        self.fusion_norm = nn.LayerNorm(self.cswin_merged_dim)
        self.fusion_ffn = nn.Sequential(
            nn.Linear(self.cswin_merged_dim, self.cswin_merged_dim * 4),
            nn.GELU(),
            nn.Dropout(drop_rate),
            nn.Linear(self.cswin_merged_dim * 4, self.cswin_merged_dim),
            nn.Dropout(drop_rate),
        )

        # CSWin transformer (post-fusion stages)
        self.stage3 = cswin_tiny.stage3
        self.merge3 = cswin_tiny.merge3
        self.stage4 = cswin_tiny.stage4
        self.norm = cswin_tiny.norm

        # Classification head
        self.head_in_features = cswin_tiny.norm.normalized_shape[0]
        self.head_drop = nn.Dropout(drop_rate)
        self.head = nn.Linear(self.head_in_features, num_classes)

        del cswin_tiny

    def forward(self, x):
        # B = x.shape[0]

        # CSWin transformer (pre-fusion stages)
        cswin_x = self.conv_embed(x)  # [B, 56*56=3136, embed_dim]

        # Process through early stages (1 & 2) to reach 14x14
        for blk in self.stage1:
            cswin_x = blk(cswin_x)
        cswin_x = self.merge1(cswin_x)  # [B, 28*28, 2*embed_dim]

        for blk in self.stage2:
            cswin_x = blk(cswin_x)
        cswin_x = self.merge2(cswin_x)  # [B, 14*14=196, 4*embed_dim=256]

        cswin_tokens = cswin_x  # [B, 196, 256]

        # Fusion (self-attention + FFN)
        fused_tokens = self.self_attn(cswin_tokens)  # [B, 196, 256]
        fused_tokens = fused_tokens + cswin_tokens
        fused_tokens_norm = self.fusion_norm(fused_tokens)
        fused_tokens = fused_tokens + self.fusion_ffn(fused_tokens_norm)

        # CSWin transformer (post-fusion stages)
        for blk in self.stage3:
            fused_tokens = blk(fused_tokens)
        fused_tokens = self.merge3(fused_tokens)  # [B, 7*7=49, 512]
        for blk in self.stage4:
            fused_tokens = blk(fused_tokens)
        fused_tokens = self.norm(fused_tokens)  # [B, 49, 512]

        # Classification head
        pooled = torch.mean(fused_tokens, dim=1)  # [B, 512]
        pooled = self.head_drop(pooled)
        out = self.head(pooled)  # [B, num_classes]

        return out
