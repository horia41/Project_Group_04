import torch
import torch.nn as nn
import timm
from einops.layers.torch import Rearrange
from cswin_fpn_hybrid.cswin.models import CSWin_64_12211_tiny_224

class ResNetCSWinHybrid(nn.Module):
    def __init__(self, num_classes=2, resnet_pretrained=True):
        super().__init__()

        # --- 1. ResNet50 Stem ---
        # Load a ResNet50, pre-trained on ImageNet
        # We will stop at layer3 (output: [B, 1024, 14, 14])
        self.resnet_stem = timm.create_model(
            'resnet50',
            pretrained=resnet_pretrained,
            features_only=True,
            out_indices=(3,)  # (0,1,2,3) for layers 1,2,3,4
        )

        # --- 2. CSWin-Tiny Body (for stealing parts) ---
        # We create a CSWin-Tiny model just to borrow its building blocks.
        # We will use its stage3, merge3, stage4, and final norm.
        # This model is NOT trained (its weights are random).
        cswin_tiny = CSWin_64_12211_tiny_224()

        # Input to CSWin stage3 is embed_dim * 4 = 64 * 4 = 256
        self.cswin_input_dim = cswin_tiny.stage3[0].dim

        # --- 3. Bridge Layer ---
        # We need to project ResNet's 1024 channels down to CSWin's 256
        self.bridge = nn.Conv2d(1024, self.cswin_input_dim, kernel_size=1)

        # --- 4. Tokenizer ---
        # Flatten the 2D feature map from the bridge into tokens
        # Input: [B, 256, 14, 14] -> Output: [B, 196, 256]
        self.tokenizer = Rearrange('b c h w -> b (h w) c')

        # --- 5. CSWin Body (Borrowed) ---
        # We "steal" the pre-built stages from the CSWin instance
        self.stage3 = cswin_tiny.stage3
        self.merge3 = cswin_tiny.merge3
        self.stage4 = cswin_tiny.stage4
        self.norm = cswin_tiny.norm

        # --- 6. Classifier Head ---
        # The output of self.norm will be [B, 49, 512]
        self.head_in_features = cswin_tiny.norm.normalized_shape[0]
        self.head = nn.Linear(self.head_in_features, num_classes)

        # Don't need this anymore, free up memory
        del cswin_tiny

    def forward(self, x):
        # 1. ResNet Stem
        # Input: [B, 3, 224, 224]
        features = self.resnet_stem(x)
        x_stem = features[0] # Shape: [B, 1024, 14, 14]

        # 2. Bridge
        x_bridge = self.bridge(x_stem) # Shape: [B, 256, 14, 14]

        # 3. Tokenizer
        x_tokens = self.tokenizer(x_bridge) # Shape: [B, 196, 256]

        # --- START CORRECTED CODE ---
        # 4. CSWin Body

        # Manually iterate through Stage 3
        x_s3 = x_tokens
        for blk in self.stage3:
            # Note: We are not using checkpointing here for simplicity
            x_s3 = blk(x_s3)

        x_m3 = self.merge3(x_s3)         # Shape: [B, 49, 512]

        # Manually iterate through Stage 4
        x_s4 = x_m3
        for blk in self.stage4:
            x_s4 = blk(x_s4)

        x_norm = self.norm(x_s4)         # Shape: [B, 49, 512]
        # --- END CORRECTED CODE ---

        # 5. Classifier Head
        x_pooled = torch.mean(x_norm, dim=1) # Global Avg Pool -> [B, 512]
        out = self.head(x_pooled)            # Shape: [B, 2]

        return out