
from functools import partial
import torch
import torch.nn as nn
import timm.models.vision_transformer


class VisionTransformer(timm.models.vision_transformer.VisionTransformer):
    """ *** Vision Transformer with support for global average pooling ***
    """
    def __init__(self, global_pool='avg', pre_logits=True, **kwargs):
        super(VisionTransformer, self).__init__(**kwargs)
        self.pre_logits = pre_logits
        self.global_pool = global_pool

    def forward_head(self, x):
        x = self.pool(x)
        x = self.fc_norm(x)
        x = self.head_drop(x)
        return x if self.pre_logits else self.head(x)


def vit_large_patch16(**kwargs):
    model = VisionTransformer(
        patch_size=16, embed_dim=1024, depth=24, num_heads=16, mlp_ratio=4, qkv_bias=True, fc_norm=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model

