# ------------------------------------------
# CSWin Transformer
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
# written By Xiaoyi Dong
# ------------------------------------------
# Modified 
# ------------------------------------------

import torch
import torch.utils.checkpoint as checkpoint
from einops import rearrange
from cswin_fpn_hybrid.cswin import CSWinTransformer

class CSWinBackbone(CSWinTransformer):
    """
    CSWin transformer, modified to expose feature maps which are passed to a FPN 
      S2: stride 4   (reso = img_size // 4,   channels = embed_dim)
      S3: stride 8   (reso = img_size // 8,   channels = 2*embed_dim)
      S4: stride 16  (reso = img_size // 16,  channels = 4*embed_dim)
      S5: stride 32  (reso = img_size // 32,  channels = 8*embed_dim)
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Stage resolutions 
        img_size = kwargs.get("img_size", 224)
        self.reso1 = img_size // 4   # S1
        self.reso2 = img_size // 8   # S2
        self.reso3 = img_size // 16  # S3
        self.reso4 = img_size // 32  # S4

        # Channel dimensions per stage
        embed_dim = kwargs.get("embed_dim", 96)
        self.s1_channels = embed_dim
        self.s2_channels = embed_dim * 2
        self.s3_channels = embed_dim * 4
        self.s4_channels = embed_dim * 8

    def forward_features_tokens(self, x: torch.Tensor):
        """
        Forward that returns intermediate token sequences at each stage to build feature maps:
        returns dict { 's1_tokens', 's2_tokens', 's3_tokens', 's4_tokens' }
        """
        # Stem -> tokens at reso1
        x = self.stage1_conv_embed(x)  # [B, N2, C2]
        
        # Stage1
        for blk in self.stage1:
            if self.use_chk:
                x = checkpoint.checkpoint(blk, x)
            else:
                x = blk(x)
        s1_tokens = x  # [B, N2, C2]

        # Merge1 
        x = self.merge1(x)  # [B, N3, C3]
        
        # Stage2
        for blk in self.stage2:
            if self.use_chk:
                x = checkpoint.checkpoint(blk, x)
            else:
                x = blk(x)
        s2_tokens = x  # [B, N3, C3]

        # Merge2
        x = self.merge2(x)  # [B, N4, C4]
        
        # Stage3
        for blk in self.stage3:
            if self.use_chk:
                x = checkpoint.checkpoint(blk, x)
            else:
                x = blk(x)
        s3_tokens = x  # [B, N4, C4]

        # Merge3
        x = self.merge3(x)  # [B, N5, C5]
        
        # Stage4
        for blk in self.stage4:
            if self.use_chk:
                x = checkpoint.checkpoint(blk, x)
            else:
                x = blk(x)
        x = self.norm(x)  # [B, N5, C5]
        s4_tokens = x

        return {
            "s1_tokens": s1_tokens,
            "s2_tokens": s2_tokens,
            "s3_tokens": s3_tokens,
            "s4_tokens": s4_tokens,
        }

    def forward_features(self, x: torch.Tensor):
        """
        Returns 2D feature maps for FPN: dict { 's1', 's2','s3','s4' } with shapes:
          s1: [B, C2, H/4,  W/4]
          s2: [B, C3, H/8,  W/8]
          s3: [B, C4, H/16, W/16]
          s4: [B, C5, H/32, W/32]
        Assumes input H=W=img_size used at init.
        """
        toks = self.forward_features_tokens(x)
        s1 = rearrange(toks["s1_tokens"], 'b (h w) c -> b c h w', h=self.reso1, w=self.reso1)
        s2 = rearrange(toks["s2_tokens"], 'b (h w) c -> b c h w', h=self.reso2, w=self.reso2)
        s3 = rearrange(toks["s3_tokens"], 'b (h w) c -> b c h w', h=self.reso3, w=self.reso3)
        s4 = rearrange(toks["s4_tokens"], 'b (h w) c -> b c h w', h=self.reso4, w=self.reso4)
        return {"s1": s1, "s2": s2, "s3": s3, "s4": s4}

    def forward(self, x: torch.Tensor):
        """
        Classification forward kept for compatibility.
        """
        toks = self.forward_features_tokens(x)
        # Perhaps turn into tensor rather than dict?
        return toks
