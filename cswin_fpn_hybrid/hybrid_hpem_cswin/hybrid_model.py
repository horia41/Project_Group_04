import torch.nn as nn
from cswin_fpn_hybrid.cswin.cswin_transformer import CSWinTransformer
from .hpem import HPEM

class HybridCSWinClassifier(CSWinTransformer):
    """
    It gets from the base CSWinTransformer and does two things:
    1. Replaces the 'stage1_conv_embed' with HPEM (CNN part)
    2. Replaces the 'head' with a new classifier for the 2-class problem
    """
    def __init__(self, num_classes=2, **kwargs):

        # 1. Initialize the parent CSWinTransformer
        #    This builds the full CSWin model (stages, blocks, etc.)
        super().__init__(**kwargs)

        # 2. OVERWRITE the patch embedding
        #    Replace 'self.stage1_conv_embed' with our new HPEM
        self.stage1_conv_embed = HPEM(
            img_size=kwargs.get('img_size', 256),
            patch_size=4, # CSWin default
            in_chans=kwargs.get('in_chans', 3),
            embed_dim=kwargs.get('embed_dim', 64)
        )

        # 3. OVERWRITE the classifier head
        #    Get the feature dim from the final 'norm' layer
        num_features = self.norm.normalized_shape[0]
        self.head = nn.Linear(num_features, num_classes) if num_classes > 0 else nn.Identity()

        # Re-apply weight initialization for the new head
        self.head.apply(self._init_weights)

    def forward(self, x):
        # This uses the parent's 'forward_features' method, which
        # automatically uses our new 'self.stage1_conv_embed' (HPEM)
        # and applies the Global Average Pooling (torch.mean).
        x = self.forward_features(x)

        # Pass through our new 2-class head
        x = self.head(x)
        return x