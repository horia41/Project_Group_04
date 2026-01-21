# ------------------------------------------
# CSWin Transformer
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
# written By Xiaoyi Dong
# ------------------------------------------

from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from timm.models import register_model

from phase_performance_hybrids.cswin_transformer.cswin_transformer import CSWinTransformer

def _cfg(url='', **kwargs):
    return {
        'url': url,
        'num_classes': 1000, 
        'input_size': (3, 224, 224), 
        'pool_size': None,
        'crop_pct': .9, 
        'interpolation': 'bicubic',
        'mean': IMAGENET_DEFAULT_MEAN, 
        'std': IMAGENET_DEFAULT_STD,
        'first_conv': 'patch_embed.proj', 
        'classifier': 'head',
        **kwargs
    }

default_cfgs = {
    'cswin_224': _cfg(),
    'cswin_384': _cfg(crop_pct=1.0),
}

### 224 models

@register_model
def CSWin_64_12211_tiny_224(pretrained=False, **kwargs):
    model = CSWinTransformer(
        patch_size=4,
        embed_dim=64, 
        depth=[1, 2, 21, 1],
        split_size=[1, 2, 7, 7], 
        num_heads=[2, 4, 8, 16], 
        mlp_ratio=4., 
        **kwargs
    )
    model.default_cfg = default_cfgs['cswin_224']
    return model

@register_model
def CSWin_64_24322_small_224(pretrained=False, **kwargs):
    model = CSWinTransformer(
        patch_size=4, 
        embed_dim=64, 
        depth=[2, 4, 32, 2],
        split_size=[1, 2, 7, 7], 
        num_heads=[2, 4, 8, 16], 
        mlp_ratio=4., 
        **kwargs
    )
    model.default_cfg = default_cfgs['cswin_224']
    return model

@register_model
def CSWin_96_24322_base_224(pretrained=False, **kwargs):
    model = CSWinTransformer(
        patch_size=4, 
        embed_dim=96, 
        depth=[2, 4, 32, 2],
        split_size=[1, 2, 7, 7], 
        num_heads=[4, 8, 16, 32], 
        mlp_ratio=4., 
        **kwargs
    )
    model.default_cfg = default_cfgs['cswin_224']
    return model

@register_model
def CSWin_144_24322_large_224(pretrained=False, **kwargs):
    model = CSWinTransformer(
        patch_size=4, 
        embed_dim=144, 
        depth=[2, 4, 32, 2],
        split_size=[1, 2, 7, 7], 
        num_heads=[6, 12, 24, 24], 
        mlp_ratio=4., 
        **kwargs
    )
    model.default_cfg = default_cfgs['cswin_224']
    return model

### 384 models

@register_model
def CSWin_96_24322_base_384(pretrained=False, **kwargs):
    model = CSWinTransformer(
        patch_size=4, 
        embed_dim=96, 
        depth=[2, 4, 32, 2],
        split_size=[1, 2, 12, 12], 
        num_heads=[4, 8, 16, 32], 
        mlp_ratio=4., 
        **kwargs
    )
    model.default_cfg = default_cfgs['cswin_384']
    return model

@register_model
def CSWin_144_24322_large_384(pretrained=False, **kwargs):
    model = CSWinTransformer(
        patch_size=4, 
        embed_dim=144, 
        depth=[2, 4, 32, 2],
        split_size=[1, 2, 12, 12], 
        num_heads=[6, 12, 24, 24], 
        mlp_ratio=4., 
        **kwargs
    )
    model.default_cfg = default_cfgs['cswin_384']
    return model
