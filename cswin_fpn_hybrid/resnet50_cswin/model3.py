import torch
import torch.nn as nn
import timm
from einops.layers.torch import Rearrange
import math
from cswin_fpn_hybrid.cswin import models
from cswin_fpn_hybrid.cswin.models import CSWin_64_12211_tiny_224


class Bridge(nn.Module):
    # TODO: instead of this "simple bridge", we can try a "gated bridge" (but still without
    #  attention at this point)
    """
    Simple bridge for dimension reduction and feature normalization.
    Prepares ResNet features for tokenization and cross-attention.
    """

    def __init__(self, in_ch=512, out_ch=256, drop_rate=0.0):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.GELU(),
            nn.Dropout(drop_rate)
        )

    def forward(self, x):
        return self.proj(x)


class CrossAttention(nn.Module):
    """
    Cross-Attention layer for fusing two feature streams.
    Query from one stream attends to Key/Value from another stream.
    """

    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.kv = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x, context):
        """
        Args:
            x: Query tokens [B, N, C]
            context: Key/Value tokens [B, M, C]
        """
        B, N, C = x.shape

        q = self.q(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        kv = self.kv(context).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        k, v = kv[0], kv[1]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)

        return x


class ResNetCSWinHybridV3(nn.Module):
    """
    Hybrid model with parallel ResNet and CSWin processing.
    
    Architecture:
    1. ResNet processes image -> [B, 1024, 14, 14]
    2. CSWin processes raw image in parallel -> reaches 14x14 spatial resolution after merge2
    3. Cross-attention fuses the two token streams
    4. Continue through CSWin stages 3 & 4
    """

    def __init__(self, num_classes=2, resnet_pretrained=True, cswin_pretrained=True, 
                 drop_rate=0.2, drop_path_rate=0.2, num_fusion_heads=8):
        super().__init__()

        # ResNet
        self.resnet_stem = timm.create_model(
            'resnet50',
            pretrained=resnet_pretrained,
            features_only=True,
            out_indices=(2,)  # Layer 2: [B, 512, 28, 28]
            # TODO: I am using layer 2 features so that CSWin can cross-attend more low-level, local
            #  features; but if this does not perform as well, we can change it to layer 3 again 
        )

        # CSWin transformer
        cswin_tiny = timm.create_model(
            'CSWin_64_12211_tiny_224',
            pretrained=cswin_pretrained,
            drop_path_rate=drop_path_rate,
            num_classes=num_classes  # Will be replaced anyway
        )

        # Dimensions
        self.embed_dim = cswin_tiny.embed_dim  # 64 for CSWin_tiny
        self.bridge_dim = 256  # Target dimension for ResNet features

        # Bridge (basically to tokenize ResNet features)
        self.bridge = Bridge(
            in_ch=512,
            out_ch=self.bridge_dim,
            drop_rate=drop_rate
        )
        self.resnet_tokenizer = Rearrange('b c h w -> b (h w) c')
        self.resnet_pos_embed = nn.Parameter(torch.zeros(1, 28 * 28, self.bridge_dim))
        nn.init.trunc_normal_(self.resnet_pos_embed, std=0.02)
        # Downsampling layer to match CSWin resolution (28*28 -> 14*14)
        self.resnet_downsample = nn.AdaptiveAvgPool1d(14 * 14)

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
        if self.cswin_merged_dim != self.bridge_dim:
            self.cswin_proj = nn.Linear(self.cswin_merged_dim, self.bridge_dim)
        else:
            self.cswin_proj = nn.Identity()

        # Fusion (cross-attention + FFN)
        self.cross_attn = CrossAttention(
            dim=self.bridge_dim,
            num_heads=num_fusion_heads,
            attn_drop=drop_rate,
            proj_drop=drop_rate
        )
        self.fusion_norm = nn.LayerNorm(self.bridge_dim)
        self.fusion_ffn = nn.Sequential(
            nn.Linear(self.bridge_dim, self.bridge_dim * 4),
            nn.GELU(),
            nn.Dropout(drop_rate),
            nn.Linear(self.bridge_dim * 4, self.bridge_dim),
            nn.Dropout(drop_rate),
        )

        # CSWin transformer (post-fusion stages)
        # Project from ResNet bridge dimension back to CSWin dimension for stage3
        self.fused_to_cswin_proj = nn.Linear(self.bridge_dim, self.cswin_merged_dim)

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
        B = x.shape[0]

        # ResNet
        resnet_feat = self.resnet_stem(x)[0]  # [B, 512, 28, 28]
        resnet_feat = self.bridge(resnet_feat)  # [B, 256, 28, 28]
        resnet_tokens = self.resnet_tokenizer(resnet_feat)  # [B, 784, 256]
        resnet_tokens = resnet_tokens + self.resnet_pos_embed  # [B, 784, 256]
        # Downsample from 28x28 (784 tokens) to 14x14 (196 tokens)
        resnet_tokens = self.resnet_downsample(resnet_tokens.transpose(1, 2)).transpose(1, 2)  # [B, 196, 256]

        # CSWin transformer (pre-fusion stages)
        cswin_x = self.conv_embed(x)  # [B, 56*56=3136, embed_dim]

        # Process through early stages (1 & 2) to reach 14x14
        for blk in self.stage1:
            cswin_x = blk(cswin_x)
        cswin_x = self.merge1(cswin_x)  # [B, 28*28, 2*embed_dim]

        for blk in self.stage2:
            cswin_x = blk(cswin_x)
        cswin_x = self.merge2(cswin_x)  # [B, 14*14=196, 4*embed_dim=256]

        # Project CSWin features to match ResNet bridge dimension
        cswin_tokens = self.cswin_proj(cswin_x)  # [B, 196, 256]

        # Fusion (cross-attention + FFN)
        fused_tokens = self.cross_attn(cswin_tokens, resnet_tokens)  # [B, 196, 256]
        fused_tokens = fused_tokens + cswin_tokens
        fused_tokens_norm = self.fusion_norm(fused_tokens)
        fused_tokens = fused_tokens + self.fusion_ffn(fused_tokens_norm)
        fused_tokens = self.fused_to_cswin_proj(fused_tokens)  # [B, 196, 256]

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
