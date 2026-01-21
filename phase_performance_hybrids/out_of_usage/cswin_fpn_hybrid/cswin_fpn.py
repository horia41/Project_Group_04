import torch
import torch.nn as nn
from phase_performance_hybrids.out_of_usage.cswin_fpn_hybrid.cswin_backbone import CSWinBackbone
from collections import OrderedDict
from torchvision.ops import FeaturePyramidNetwork

class CSWinFPN(nn.Module):
    
    def __init__(self, 
                 cswin_kwargs: dict = {},
                 fpn_out_channels: int = 256):
        super().__init__()
        cswin_kwargs = cswin_kwargs 
        self.backbone = CSWinBackbone(**cswin_kwargs)
        # Map input feature names to their channel sizes
        in_channels_list = [
            self.backbone.s1_channels,
            self.backbone.s2_channels,
            self.backbone.s3_channels,
            self.backbone.s4_channels,
        ]
        self.fpn = FeaturePyramidNetwork(
            in_channels_list=in_channels_list, 
            out_channels=fpn_out_channels
        )

    def forward(self, x: torch.Tensor):
        # Get S1..S4 from CSWin
        s_feats = self.backbone.forward_features(x)  # conv features from backbone
        s_feats_dict = OrderedDict([
            ("s1", s_feats["s1"]),
            ("s2", s_feats["s2"]),
            ("s3", s_feats["s3"]),
            ("s4", s_feats["s4"]),
        ])
        # Fuse with FPN to get f1...f4
        f_feats = self.fpn(s_feats_dict)  # dict with keys matching input (s1..s4), now f-levels
        return f_feats
