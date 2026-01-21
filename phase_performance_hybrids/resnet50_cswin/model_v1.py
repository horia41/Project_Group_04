import torch
import torch.nn as nn
import timm
from einops.layers.torch import Rearrange

class ResNetCSWinHybrid(nn.Module):
    def __init__(self, num_classes=2, resnet_pretrained=True, cswin_pretrained=True):
        super().__init__()

        # ResNet50 Stem
        # Stop at layer3 (output: [B, 1024, 14, 14])
        self.resnet_stem = timm.create_model(
            'resnet50',
            pretrained=resnet_pretrained,
            features_only=True,
            out_indices=(3,)  # (0,1,2,3) for layers 1,2,3,4
        )

        # CSWin-Tiny model
        # use stage3, merge3, stage4, final norm from it
        cswin_tiny = timm.create_model(
            'CSWin_64_12211_tiny_224',
            pretrained=cswin_pretrained,
        )

        # Input stage3 is embed_dim * 4 = 64 * 4 = 256
        self.cswin_input_dim = cswin_tiny.stage3[0].dim

        # Bridge Layer
        # project down ResNet 1024 channels down to CSWin 256
        self.bridge = nn.Sequential(
            nn.Conv2d(1024, self.cswin_input_dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(self.cswin_input_dim),
            nn.GELU()
        )

        # Tokenizer
        # flatten the 2D feature map from the bridge into tokens
        # input should be [B, 256, 14, 14], output should be [B, 196, 256]
        self.tokenizer = Rearrange('b c h w -> b (h w) c')

        # addition for positional embedding because rearrange destroys 2d structure
        self.pos_embed = nn.Parameter(torch.zeros(1, 14 * 14, self.cswin_input_dim))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

        # CSWin Body
        # pre-built stages from the CSWin instance
        self.stage3 = cswin_tiny.stage3
        self.merge3 = cswin_tiny.merge3
        self.stage4 = cswin_tiny.stage4
        self.norm = cswin_tiny.norm

        # Classifier Head
        self.head_in_features = cswin_tiny.norm.normalized_shape[0]
        self.head = nn.Linear(self.head_in_features, num_classes)

        # free up memory
        del cswin_tiny

    def forward(self, x):
        # ResNet Stem
        # [B, 3, 224, 224]
        features = self.resnet_stem(x)
        x_stem = features[0] # sanity check for shape : [B, 1024, 14, 14]

        # Bridge
        x_bridge = self.bridge(x_stem) # sanity check for shape : [B, 256, 14, 14]

        # Tokenizer
        x_tokens = self.tokenizer(x_bridge) # sanity check for shape : [B, 196, 256]

        # Add learnable position info so the Transformer knows spatial context
        x_tokens = x_tokens + self.pos_embed

        # Manually iterate through Stage 3 & CSWin part
        x_s3 = x_tokens
        for blk in self.stage3:
            x_s3 = blk(x_s3)

        x_m3 = self.merge3(x_s3)         # sanity check for shape : [B, 49, 512]

        # Manually iterate through Stage 4
        x_s4 = x_m3
        for blk in self.stage4:
            x_s4 = blk(x_s4)

        x_norm = self.norm(x_s4)         # sanity check for shape : [B, 49, 512]

        # Classifier Head
        x_pooled = torch.mean(x_norm, dim=1) # Global Avg Pool -> [B, 512]
        out = self.head(x_pooled)            # sanity check for shape : [B, 2]

        return out