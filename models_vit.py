
from functools import partial

import timm.models.vision_transformer
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from timm.models.layers import trunc_normal_

class VisionTransformer(timm.models.vision_transformer.VisionTransformer):
    """ Vision Transformer with support for global average pooling
    """
    def __init__(self, global_pool=False, **kwargs):
        super(VisionTransformer, self).__init__(**kwargs)

        self.global_pool = global_pool
        if self.global_pool:
            norm_layer = kwargs['norm_layer']
            embed_dim = kwargs['embed_dim']
            self.fc_norm = norm_layer(embed_dim)

            del self.norm  # remove the original norm

    def forward_features(self, x):
        B = x.shape[0]
        x = self.patch_embed(x)

        cls_tokens = self.cls_token.expand(B, -1, -1)  # stole cls_tokens impl from Phil Wang, thanks
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embed
        x = self.pos_drop(x)

        for blk in self.blocks:
            x = blk(x)

        if self.global_pool:
            x = x[:, 1:, :].mean(dim=1,keepdim=True)  # global pool without cls token
            outcome = self.fc_norm(x)
        else:
            x = self.norm(x)
            outcome = x[:, 0]

        return outcome


class PixioVisionTransformer(timm.models.vision_transformer.VisionTransformer):
    """ Vision Transformer with support for global average pooling
    """
    def __init__(self, global_pool=False, n_cls_tokens=8, **kwargs):
        super(PixioVisionTransformer, self).__init__(**kwargs)

        self.n_cls_tokens = n_cls_tokens
        self.cls_token = nn.Parameter(
            torch.zeros(1, n_cls_tokens, kwargs['embed_dim'])
        )
        self.pos_embed = nn.Parameter(
            torch.zeros(1, self.patch_embed.num_patches + n_cls_tokens, kwargs['embed_dim'])
        )
        self.global_pool = global_pool
        if self.global_pool:
            norm_layer = kwargs['norm_layer']
            embed_dim = kwargs['embed_dim']
            self.fc_norm = norm_layer(embed_dim)

            del self.norm  # remove the original norm


    def _interpolate_pos_emb(
        self, 
        x: torch.Tensor
    ):
        """Interpolate the pre-trained positional embeddings to match the input x"""
        assert x.shape[-2] % self.patch_embed.patch_size[0] == 0, \
            f'height {x.shape[-2]} must be divisible by patch size {self.patch_embed.patch_size[0]}'
        assert x.shape[-1] % self.patch_embed.patch_size[1] == 0, \
            f'width {x.shape[-1]} must be divisible by patch size {self.patch_embed.patch_size[1]}'
        
        H = x.shape[-2] // self.patch_embed.patch_size[0]
        W = x.shape[-1] // self.patch_embed.patch_size[1]
        
        cls_pos_embed = self.pos_embed[:, :self.n_cls_tokens]
        patch_pos_embed = self.pos_embed[:, self.n_cls_tokens:]
        
        pt_size = int(patch_pos_embed.shape[1] ** 0.5)
        
        if pt_size == H == W:
            return self.pos_embed
        
        patch_pos_embed = patch_pos_embed.reshape(1, pt_size, pt_size, -1).permute(0, 3, 1, 2)
        patch_pos_embed = torch.nn.functional.interpolate(
            patch_pos_embed, size=(H, W), mode='bicubic', align_corners=False
        )
        patch_pos_embed = patch_pos_embed.permute(0, 2, 3, 1).reshape(1, H * W, -1)
        
        new_pos_embed = torch.cat((cls_pos_embed, patch_pos_embed), dim=1)
        
        return new_pos_embed 


    def forward_features(self, x):
        B = x.shape[0]
        pos_embed = self._interpolate_pos_emb(x)
        x = self.patch_embed(x)
        x += pos_embed[:, self.n_cls_tokens:, :]

        cls_token = self.cls_token + pos_embed[:, :self.n_cls_tokens, :]
        cls_tokens = self.cls_token.expand(B, -1, -1)  # stole cls_tokens impl from Phil Wang, thanks
        x = torch.cat((cls_tokens, x), dim=1)
        # x = x + self.pos_embed
        x = self.pos_drop(x)

        for blk in self.blocks:
            x = blk(x)

        if self.global_pool:
            x = x[:, self.n_cls_tokens:, :].mean(dim=1,keepdim=True)  # global pool without cls token
            outcome = self.fc_norm(x)
        else:
            x = self.norm(x)
            outcome = x[:, :self.n_cls_tokens].mean(dim=1)

        return outcome


def Pixio(**kwargs):
    model = PixioVisionTransformer(
        patch_size=16, embed_dim=1024, depth=24, num_heads=16, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


def RETFound_mae(**kwargs):
    model = VisionTransformer(
        patch_size=16, embed_dim=1024, depth=24, num_heads=16, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model

def GastroNet(**kwargs):
    model = VisionTransformer(
        patch_size=16, 
        embed_dim=768,     # 從 1024 降至 768
        depth=12,         # 從 24 層減半至 12 層
        num_heads=12,      # 從 16 頭降至 12 頭
        mlp_ratio=4, 
        qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), 
        **kwargs
    )
    return model


def Dinov2(args, **kwargs):
    
    if args.model_arch == 'dinov2_vits14':
        arch = 'vit_small_patch14_dinov2.lvd142m'
    elif args.model_arch == 'dinov2_vitb14':
        arch = 'vit_base_patch14_dinov2.lvd142m'
    elif args.model_arch == 'dinov2_vitl14':
        arch = 'vit_large_patch14_dinov2.lvd142m'
    elif args.model_arch == 'dinov2_vitg14':
        arch = 'vit_giant_patch14_dinov2.lvd142m'
    else:
        raise ValueError(f"Unknown model_arch '{args.model_arch}'. "
                         f"Expected one of: dinov2_vits14, dinov2_vitb14, dinov2_vitl14, dinov2_vitg14")
        
    model = timm.create_model(
        arch,
        pretrained=True,
        img_size=224,
        **kwargs
    )
    return model



def RETFound_dinov2(args, **kwargs):
    model = timm.create_model(
        'vit_large_patch14_dinov2.lvd142m',
        pretrained=True,
        img_size=224,
        **kwargs
    )
    return model


def Dinov3(args, **kwargs):
    # Load ViT-L/16 backbone (hub model has `head = Identity` by default)
    model = torch.hub.load(
        repo_or_dir="facebookresearch/dinov3",
        model=args.model_arch,
        pretrained=False,   # main() will load your checkpoint
        trust_repo=True,
    )

    # Figure out feature dimension for the probe
    feat_dim = getattr(model, "embed_dim", None) or getattr(model, "num_features", None)
    model.head = nn.Linear(feat_dim, args.nb_classes)
    trunc_normal_(model.head.weight, std=2e-5)
    if model.head.bias is not None:
        nn.init.zeros_(model.head.bias)

    return model
