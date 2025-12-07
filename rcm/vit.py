import os
import math
import einops
import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
import logging

from functools import reduce
from operator import mul

try:
    import flash_attn
    from flash_attn import flash_attn_func, flash_attn_qkvpacked_func
    ATTENTION_MODE = 'flash'
except:
    try:
        import xformers
        import xformers.ops
        ATTENTION_MODE = 'xformers'
    except:
        ATTENTION_MODE = 'math'

print(f'attention mode is {ATTENTION_MODE}')


def timestep_embedding(timesteps, dim, max_period=10000):
    """
    Create sinusoidal timestep embeddings.

    :param timesteps: a 1-D Tensor of N indices, one per batch element.
                      These may be fractional.
    :param dim: the dimension of the output.
    :param max_period: controls the minimum frequency of the embeddings.
    :return: an [N x dim] Tensor of positional embeddings.
    """
    half = dim // 2
    freqs = torch.exp(
        -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
    ).to(device=timesteps.device)
    args = timesteps[:, None].float() * freqs[None]
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
    return embedding


class LayerNorm(nn.LayerNorm):
    def forward(self, x):
        return super().forward(x.float()).type(x.dtype)


class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., qk_norm=False):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.attn_drop_rate = attn_drop
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, L, C = x.shape
        # print(self.qkv.weight.dtype, self.qkv.weight.shape)
        qkv = self.qkv(x)
        if ATTENTION_MODE == 'flash':
            qkv = einops.rearrange(qkv, 'B L (K H D) -> B L K H D', K=3, H=self.num_heads)
            x = flash_attn_qkvpacked_func(qkv, dropout_p=self.attn_drop_rate)
            x = einops.rearrange(x, 'B L H D -> B L (H D)')

        elif ATTENTION_MODE == 'xformers':    
            qkv = einops.rearrange(qkv, 'B L (K H D) -> K B L H D', K=3, H=self.num_heads)
            q, k, v = qkv[0], qkv[1], qkv[2]  # B L H D
            if self.qk_norm:
                q = self.qnorm(q).to(v)
                k = self.knorm(k).to(v)
            x = xformers.ops.memory_efficient_attention(q, k, v, p=self.attn_drop_rate)
            x = einops.rearrange(x, 'B L H D -> B L (H D)', H=self.num_heads)
        elif ATTENTION_MODE == 'math':
            qkv = einops.rearrange(qkv, 'B L (K H D) -> K B H L D', K=3, H=self.num_heads)
            q, k, v = qkv[0], qkv[1], qkv[2]  # B H L D
            if self.qk_norm:
                q = self.qnorm(q).to(v)
                k = self.knorm(k).to(v)
            attn = (q @ k.transpose(-2, -1)) * self.scale
            attn = attn.softmax(dim=-1)
            attn = self.attn_drop(attn)
            x = (attn @ v).transpose(1, 2).reshape(B, L, C)
        else:
            raise NotImplemented

        x = self.proj(x)
        x = self.proj_drop(x)
        return x



class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Block(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=LayerNorm, use_checkpoint=False, skip=False, noise_inject=False, qk_norm=False):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.skip_linear = nn.Linear(2 * dim, dim) if skip else None

        if noise_inject:
            self.to_noise = nn.Linear(dim, dim)
            self.to_noise_norm = None if self.skip_norm else nn.LayerNorm(dim)
        else:
            self.to_noise = None

        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
        self.use_checkpoint = use_checkpoint

    def _forward(self, x, skip=None, noise=None):

        if skip is not None:
            # cls_token = x[:, 0].unsqueeze(dim=1)
            x = self.skip_linear(torch.cat([x, skip], dim=2))
            # x = torch.cat([cls_token, x], dim=1)
        if noise is not None:
            high_freq = self.to_noise(noise)
            x += high_freq

        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x

    def forward(self, x, skip=None, noise=None):
        if self.use_checkpoint:
            return torch.utils.checkpoint.checkpoint(self._forward, x, skip, noise)
        else:
            return self._forward(x, skip, noise)


def modulate(x, shift, scale):
    return x * (1 + scale) + shift


class DecoderBlock(nn.Module):

    def __init__(self, dim, encoder_dim, num_heads=12, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0., proj_drop=0.0,
                 drop_path=0., act_layer=nn.GELU, norm_layer=LayerNorm, use_checkpoint=False, skip=False, skip_post_norm=False):
        super().__init__()
        self.norm1 = norm_layer(dim)

        self.attn = Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=proj_drop)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.skip_linear = nn.Linear(encoder_dim + dim, dim) if skip else None
        self.skip_norm = nn.LayerNorm(dim) if (skip_post_norm and skip) else nn.Identity()

        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(encoder_dim, 6 * dim, bias=True)
        )

        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
        self.use_checkpoint = use_checkpoint

    def _forward(self, x, t=None, skip=None):
        """
            t.shape: B, D
        """
        if skip is not None and self.skip_linear is not None:
            x = self.skip_linear(torch.cat([x, skip], dim=2))
            x = self.skip_norm(x)

        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(t).chunk(6, dim=2)
        x = x + self.drop_path( gate_msa * self.attn(modulate(self.norm1(x), shift_msa, scale_msa)) )
        x = x + self.drop_path( gate_mlp * self.mlp(modulate(self.norm2(x), shift_mlp, scale_mlp)) )

        return x

    def forward(self, x, t, skip=None):
        if self.use_checkpoint:
            return torch.utils.checkpoint.checkpoint(self._forward, x, t, skip)
        else:
            return self._forward(x, t, skip)


class PatchEmbed(nn.Module):
    """ 2D Image to Patch Embedding
    """
    def __init__(
            self,
            img_size=224,
            patch_size=16,
            in_chans=3,
            embed_dim=768,
            norm_layer=None,
            flatten=True,
            bias=True,
    ):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        self.img_size = img_size
        self.patch_size = patch_size
        self.grid_size = (img_size[0] // patch_size[0], img_size[1] // patch_size[1])
        self.num_patches = self.grid_size[0] * self.grid_size[1]
        self.flatten = flatten

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size, bias=bias)
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x):
        x = self.proj(x)
        if self.flatten:
            x = x.flatten(2).transpose(1, 2)  # BCHW -> BNC
        # NOTE: U-ViT/DiT do not use normalization layer after the patch embedding
        # x = self.norm(x)
        return x


class RCMViT(nn.Module):

    def __init__(self, 
            image_size=224,
            patch_size=16,
            in_channels=3,
            embed_dim=768,
            hidden_dim=4096,
            output_dim=256,
            depth=12,
            num_heads=12,
            mlp_ratio=4., qkv_bias=False, qk_scale=None,
            norm_layer=LayerNorm,
            mlp_time_embed=False,
            use_checkpoint=False,
            tokens = 1,

            stop_grad_conv1 = False,
            moco_initialization = False,
            proj_layers=3,
            output_format = "clstoken",

            qk_norm=False,
            drop=0., attn_drop=0., drop_path=0.,
            **kwargs, # reserved for training rcm-uvit model
        ):
        super().__init__()
        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models
        self.in_chans = in_channels
        self.dtype= torch.float32 # by default use float32
        self.patch_size = patch_size
        self.patch_embed = PatchEmbed(img_size=image_size, patch_size=patch_size, in_chans=in_channels, embed_dim=embed_dim)

        self.num_patches = num_patches = (image_size // patch_size) ** 2
        self.patch_dim = patch_size ** 2 * in_channels
        self.kwargs = kwargs # reserved for training rcm-uvit model
        self.output_format = output_format

        self.time_embed = nn.Sequential(
            nn.Linear(embed_dim, 4 * embed_dim),
            nn.SiLU(),
            nn.Linear(4 * embed_dim, embed_dim),
        ) if mlp_time_embed else nn.Identity()

        self.extras = 1 + tokens
        self.blocks = nn.ModuleList([
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                norm_layer=norm_layer, use_checkpoint=use_checkpoint, drop=drop, attn_drop=attn_drop, drop_path=drop_path, qk_norm=qk_norm)
            for _ in range(depth)])

        self.tokens = tokens
        self.cls_token = nn.Parameter(torch.zeros(1, tokens, embed_dim))
        nn.init.normal_(self.cls_token, std=1e-6)
        self.norm = norm_layer(embed_dim)
        self.pos_embed = nn.Parameter(torch.zeros(1, self.extras + num_patches, embed_dim))
        trunc_normal_(self.pos_embed, std=.02)

    def get_num_param(self):
        n_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        adaln_params = 0
        for n, p in self.named_parameters():
            if "adaLN" in n:
                # print(n, p.shape)
                adaln_params += p.numel()
        print("adaLN parameters:", adaln_params)
        return n_params

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed', 'cls_token'}


class DMViT(RCMViT):

    def __init__(
        self,
        image_size=224,
        patch_size=16,
        in_channels=3,
        embed_dim=768,
        qk_norm=False,
        depth=12,
        num_heads=12,
        mlp_ratio=4., qkv_bias=False, qk_scale=None,
        norm_layer=LayerNorm,
        mlp_time_embed=False,
        use_checkpoint=False,
        tokens = 1,

        decoder_depth = 12,
        decoder_embed_dim=768,
        decoder_num_heads=12,

        num_classes = 1000,
        skip_post_norm=True,
        zero_init_out=True,

        **kwargs,
    ):
        super().__init__()

        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models
        self.in_chans = in_channels
        self.dtype= torch.float32 # by default use float32
        self.patch_size = patch_size
        self.patch_embed = PatchEmbed(img_size=image_size, patch_size=patch_size, in_chans=in_channels, embed_dim=embed_dim)

        self.num_patches = num_patches = (image_size // patch_size) ** 2
        self.patch_dim = patch_size ** 2 * in_channels
        self.kwargs = kwargs # reserved for training rcm-uvit model

        self.time_embed = nn.Sequential(
            nn.Linear(embed_dim, 4 * embed_dim),
            nn.SiLU(),
            nn.Linear(4 * embed_dim, embed_dim),
        ) if mlp_time_embed else nn.Identity()

        self.extras = 1 + tokens
        self.blocks = nn.ModuleList([
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                norm_layer=norm_layer, use_checkpoint=use_checkpoint, qk_norm=qk_norm)
            for _ in range(depth)
        ])

        self.tokens = tokens
        self.cls_token = nn.Parameter(torch.zeros(1, tokens, embed_dim))
        self.norm = norm_layer(embed_dim)
        self.pos_embed = nn.Parameter(torch.zeros(1, self.extras + num_patches, embed_dim))

        self.decoder_blocks = nn.ModuleList([
            DecoderBlock(
                dim=decoder_embed_dim, encoder_dim=embed_dim, num_heads=decoder_num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                norm_layer=norm_layer, use_checkpoint=use_checkpoint, 
                skip=True, 
                skip_post_norm=skip_post_norm)
            for i in range(decoder_depth)
        ])

        self.encoder_to_decoder = nn.Linear(embed_dim, decoder_embed_dim) if decoder_embed_dim != embed_dim else None
        self.out_norm = nn.LayerNorm(decoder_embed_dim)
        self.out = nn.Linear(decoder_embed_dim, (self.patch_size**2)*in_channels)
        self.class_embed = nn.Embedding(num_classes + 1, embed_dim) if num_classes > 0 else None # the final token is unconditional token
        self.zero_init_out = zero_init_out
        self.init_weights()

        n_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"total num parameters: {n_params}")

    def init_weights(self):

        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        self.apply(_basic_init)

        nn.init.normal_(self.cls_token, std=1e-6)
        trunc_normal_(self.pos_embed, std=.02)

        # zero-init final output layer
        if self.zero_init_out:
            self.out.weight.detach().zero_()
            self.out.bias.detach().zero_()            
            print("zero-init output layer")

        for block in self.decoder_blocks:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)

        # init class embedding layer
        if self.class_embed:
            nn.init.normal_(self.class_embed.weight, std=0.02)

    def get_num_param(self):
        n_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        adaln_params = 0
        for n, p in self.named_parameters():
            if "adaLN" in n:
                # print(n, p.shape)
                adaln_params += p.numel()
        print("adaLN parameters:", adaln_params)
        return n_params

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed', 'cls_token'}

    def unpatchify(self, x):
        # b, n, c
        num_patch = int(math.sqrt(self.num_patches))
        x = einops.rearrange(
                                x, "b (m n) (p1 p2 c) -> b c (m p1) (n p2)", 
                                m=num_patch, n=num_patch,
                                p1=self.patch_size, p2=self.patch_size,
                                c=self.in_chans
                            )
        return x

    def forward(self, x, t, y=None, mask=None):

        x = self.patch_embed(x)
        B, L, D = x.shape

        time_token = self.time_embed(timestep_embedding(t, self.embed_dim))
        time_token = time_token.unsqueeze(dim=1) # B, 1, D
        x = torch.cat((time_token, x), dim=1)

        cls_tokens = self.cls_token.expand(B, -1, -1).to(self.dtype)

        cond = cls_tokens

        if y is not None:
            y = self.class_embed(y)
            y = y.unsqueeze(dim=1) # B, 1, D
            cond = cls_tokens +  y # broadcast to all class tokens

        x = torch.cat((cond, x), dim=1)

        x = x + self.pos_embed
        x = x.to(self.dtype)

        skip = []
        for blk in self.blocks:
            x = blk(x)
            skip.append(x)

        x = self.norm(x)

        if self.encoder_to_decoder is not None:
            x = self.encoder_to_decoder(x)

        if y is not None:
            time_token += y

        for blk in self.decoder_blocks:
            x = blk(x, time_token, skip.pop() if len(skip)!=0 else None)

        x = self.out_norm(x[:, self.extras:])
        x = self.out(x)
        x = self.unpatchify(x) # B, C=3, H, W

        return x