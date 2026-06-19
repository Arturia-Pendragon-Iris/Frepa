# -*- coding: utf-8 -*-
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from einops import rearrange
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Type

try:
    from torchvision.models.swin_transformer import swin_v2_b, Swin_V2_B_Weights
    _TORCHVISION_AVAILABLE = True
except (ImportError, RuntimeError):
    _TORCHVISION_AVAILABLE = False


# ---------------------------------------------------------------------------
# Basic building blocks
# ---------------------------------------------------------------------------

class DropPath(nn.Module):
    """Stochastic depth per sample (drop entire residual branch)."""

    def __init__(self, drop_prob: float = 0.):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.drop_prob == 0. or not self.training:
            return x
        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        noise = torch.rand(shape, dtype=x.dtype, device=x.device).floor_() + keep_prob
        return x * noise / keep_prob

class MLPBlock(nn.Module):
    def __init__(
        self,
        embedding_dim: int,
        mlp_dim: int,
        act: Type[nn.Module] = nn.GELU,
    ) -> None:
        super().__init__()
        self.lin1 = nn.Linear(embedding_dim, mlp_dim)
        self.lin2 = nn.Linear(mlp_dim, embedding_dim)
        self.act = act()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.lin2(self.act(self.lin1(x)))


class LayerNorm2d(nn.Module):
    def __init__(self, num_channels: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(num_channels))
        self.bias = nn.Parameter(torch.zeros(num_channels))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        u = x.mean(1, keepdim=True)
        s = (x - u).pow(2).mean(1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        x = self.weight[:, None, None] * x + self.bias[:, None, None]
        return x


class LayerNorm(nn.Module):
    """LayerNorm supporting channels_last (default) and channels_first formats."""

    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError
        self.normalized_shape = (normalized_shape,)

    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        u = x.mean(1, keepdim=True)
        s = (x - u).pow(2).mean(1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        x = self.weight[:, None, None] * x + self.bias[:, None, None]
        return x


class ConvNeXtBlock(nn.Module):
    """ConvNeXt block: DwConv -> LayerNorm -> Linear -> GELU -> Linear (channels_last path)."""

    def __init__(self, dim, drop_path=0., layer_scale_init_value=1e-6):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim)
        self.norm = LayerNorm(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(dim, 4 * dim)
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(4 * dim, dim)
        self.gamma = (
            nn.Parameter(layer_scale_init_value * torch.ones(dim), requires_grad=True)
            if layer_scale_init_value > 0 else None
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        residual = x
        x = self.dwconv(x)
        x = x.permute(0, 2, 3, 1)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        if self.gamma is not None:
            x = self.gamma * x
        x = x.permute(0, 3, 1, 2)
        return residual + self.drop_path(x)


class ConvBlock(nn.Module):
    """Conv -> BN -> ReLU."""

    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.conv(x)


class UpConv(nn.Module):
    """Upsample (x2) -> Conv -> BN -> ReLU."""

    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.up(x)


# ---------------------------------------------------------------------------
# Patch embeddings
# ---------------------------------------------------------------------------

class PatchEmbed(nn.Module):
    """Image to patch embedding via a single strided convolution."""

    def __init__(
        self,
        kernel_size: Tuple[int, int] = (16, 16),
        stride: Tuple[int, int] = (16, 16),
        padding: Tuple[int, int] = (0, 0),
        in_chans: int = 3,
        embed_dim: int = 768,
    ) -> None:
        super().__init__()
        self.proj = nn.Conv2d(
            in_chans, embed_dim, kernel_size=kernel_size, stride=stride, padding=padding
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x).permute(0, 2, 3, 1)  # B C H W -> B H W C


class PatchEmbed_2(nn.Module):
    """Hierarchical patch embedding via two ConvNeXt stages (4x4 + 4x4 = 16x total stride)."""

    def __init__(self, in_chans: int = 3, embed_dim: int = 768) -> None:
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_chans, embed_dim // 2, kernel_size=4, stride=4),
            LayerNorm(embed_dim // 2, eps=1e-6, data_format="channels_first"),
        )
        self.block1 = nn.Sequential(*[ConvNeXtBlock(dim=embed_dim // 2) for _ in range(3)])

        self.conv2 = nn.Sequential(
            nn.Conv2d(embed_dim // 2, embed_dim, kernel_size=4, stride=4),
            LayerNorm(embed_dim, eps=1e-6, data_format="channels_first"),
        )
        self.block2 = nn.Sequential(*[ConvNeXtBlock(dim=embed_dim) for _ in range(3)])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.block1(x) + x
        x = self.conv2(x)
        x = self.block2(x) + x
        return x.permute(0, 2, 3, 1)  # B C H W -> B H W C


# ---------------------------------------------------------------------------
# ViT encoder components (lightly adapted from ViTDet / SAM)
# ---------------------------------------------------------------------------

# https://github.com/facebookresearch/detectron2/blob/main/detectron2/modeling/backbone/vit.py
class ImageEncoderViT(nn.Module):
    def __init__(
        self,
        img_size: int = 512,
        patch_size: int = 16,
        in_chans: int = 3,
        embed_dim: int = 768,
        depth: int = 12,
        num_heads: int = 12,
        mlp_ratio: float = 4.0,
        out_chans: int = 256,
        qkv_bias: bool = True,
        norm_layer: Type[nn.Module] = nn.LayerNorm,
        act_layer: Type[nn.Module] = nn.GELU,
        use_abs_pos: bool = True,
        use_rel_pos: bool = False,
        rel_pos_zero_init: bool = True,
        window_size: int = 0,
        global_attn_indexes: Tuple[int, ...] = (),
        use_hierarchical_embed: bool = False,
    ) -> None:
        """
        Args:
            img_size: Input image size.
            patch_size: Patch size (ignored when use_hierarchical_embed=True).
            in_chans: Number of input image channels.
            embed_dim: Patch embedding dimension.
            depth: Depth of ViT.
            num_heads: Number of attention heads.
            mlp_ratio: Ratio of MLP hidden dim to embedding dim.
            qkv_bias: Add learnable bias to q, k, v.
            norm_layer: Normalization layer.
            act_layer: Activation layer.
            use_abs_pos: Use absolute positional embeddings.
            use_rel_pos: Add relative positional embeddings to attention map.
            rel_pos_zero_init: Zero-initialize relative positional parameters.
            window_size: Window size for window attention blocks.
            global_attn_indexes: Block indices using global (non-windowed) attention.
            use_hierarchical_embed: Replace standard PatchEmbed with ConvNeXt hierarchy.
        """
        super().__init__()
        self.img_size = img_size

        if use_hierarchical_embed:
            self.patch_embed = PatchEmbed_2(in_chans=in_chans, embed_dim=embed_dim)
        else:
            self.patch_embed = PatchEmbed(
                kernel_size=(patch_size, patch_size),
                stride=(patch_size, patch_size),
                in_chans=in_chans,
                embed_dim=embed_dim,
            )

        self.pos_embed: Optional[nn.Parameter] = None
        if use_abs_pos:
            self.pos_embed = nn.Parameter(
                torch.zeros(1, img_size // patch_size, img_size // patch_size, embed_dim)
            )

        self.blocks = nn.ModuleList([
            Block(
                dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                norm_layer=norm_layer,
                act_layer=act_layer,
                use_rel_pos=use_rel_pos,
                rel_pos_zero_init=rel_pos_zero_init,
                window_size=window_size if i not in global_attn_indexes else 0,
                input_size=(img_size // patch_size, img_size // patch_size),
            )
            for i in range(depth)
        ])

        self.neck = nn.Sequential(
            nn.Conv2d(embed_dim, out_chans, kernel_size=1, bias=False),
            LayerNorm2d(out_chans),
            nn.Conv2d(out_chans, out_chans, kernel_size=3, padding=1, bias=False),
            LayerNorm2d(out_chans),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.patch_embed(x)
        if self.pos_embed is not None:
            x = x + self.pos_embed
        for blk in self.blocks:
            x = blk(x)
        x = self.neck(x.permute(0, 3, 1, 2))
        return x


class Block(nn.Module):
    """Transformer block with optional window attention."""

    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        norm_layer: Type[nn.Module] = nn.LayerNorm,
        act_layer: Type[nn.Module] = nn.GELU,
        use_rel_pos: bool = False,
        rel_pos_zero_init: bool = True,
        window_size: int = 0,
        input_size: Optional[Tuple[int, int]] = None,
    ) -> None:
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            use_rel_pos=use_rel_pos,
            rel_pos_zero_init=rel_pos_zero_init,
            input_size=input_size if window_size == 0 else (window_size, window_size),
        )
        self.norm2 = norm_layer(dim)
        self.mlp = MLPBlock(embedding_dim=dim, mlp_dim=int(dim * mlp_ratio), act=act_layer)
        self.window_size = window_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shortcut = x
        x = self.norm1(x)
        if self.window_size > 0:
            H, W = x.shape[1], x.shape[2]
            x, pad_hw = window_partition(x, self.window_size)
        x = self.attn(x)
        if self.window_size > 0:
            x = window_unpartition(x, self.window_size, pad_hw, (H, W))
        x = shortcut + x
        x = x + self.mlp(self.norm2(x))
        return x


class Attention(nn.Module):
    """Multi-head attention with optional decomposed relative positional embeddings."""

    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = True,
        use_rel_pos: bool = False,
        rel_pos_zero_init: bool = True,
        input_size: Optional[Tuple[int, int]] = None,
    ) -> None:
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)

        self.use_rel_pos = use_rel_pos
        if self.use_rel_pos:
            assert input_size is not None, \
                "Input size must be provided if using relative positional encoding."
            self.rel_pos_h = nn.Parameter(torch.zeros(2 * input_size[0] - 1, head_dim))
            self.rel_pos_w = nn.Parameter(torch.zeros(2 * input_size[1] - 1, head_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, H, W, _ = x.shape
        qkv = self.qkv(x).reshape(B, H * W, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.reshape(3, B * self.num_heads, H * W, -1).unbind(0)

        attn = (q * self.scale) @ k.transpose(-2, -1)
        if self.use_rel_pos:
            attn = add_decomposed_rel_pos(attn, q, self.rel_pos_h, self.rel_pos_w, (H, W), (H, W))
        attn = attn.softmax(dim=-1)

        x = (
            (attn @ v)
            .view(B, self.num_heads, H, W, -1)
            .permute(0, 2, 3, 1, 4)
            .reshape(B, H, W, -1)
        )
        return self.proj(x)


# ---------------------------------------------------------------------------
# Window partition utilities
# ---------------------------------------------------------------------------

def window_partition(
    x: torch.Tensor, window_size: int
) -> Tuple[torch.Tensor, Tuple[int, int]]:
    """Partition into non-overlapping windows, padding if needed.

    Args:
        x: Input tokens [B, H, W, C].
        window_size: Window size.
    Returns:
        windows: [B * num_windows, window_size, window_size, C].
        (Hp, Wp): Padded height and width.
    """
    B, H, W, C = x.shape
    pad_h = (window_size - H % window_size) % window_size
    pad_w = (window_size - W % window_size) % window_size
    if pad_h > 0 or pad_w > 0:
        x = F.pad(x, (0, 0, 0, pad_w, 0, pad_h))
    Hp, Wp = H + pad_h, W + pad_w
    x = x.view(B, Hp // window_size, window_size, Wp // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    return windows, (Hp, Wp)


def window_unpartition(
    windows: torch.Tensor,
    window_size: int,
    pad_hw: Tuple[int, int],
    hw: Tuple[int, int],
) -> torch.Tensor:
    """Reverse window partition and remove padding.

    Args:
        windows: [B * num_windows, window_size, window_size, C].
        window_size: Window size.
        pad_hw: Padded (Hp, Wp).
        hw: Original (H, W) before padding.
    Returns:
        x: [B, H, W, C].
    """
    Hp, Wp = pad_hw
    H, W = hw
    B = windows.shape[0] // (Hp * Wp // window_size // window_size)
    x = windows.view(B, Hp // window_size, Wp // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, Hp, Wp, -1)
    if Hp > H or Wp > W:
        x = x[:, :H, :W, :].contiguous()
    return x


def get_rel_pos(q_size: int, k_size: int, rel_pos: torch.Tensor) -> torch.Tensor:
    max_rel_dist = int(2 * max(q_size, k_size) - 1)
    if rel_pos.shape[0] != max_rel_dist:
        rel_pos_resized = F.interpolate(
            rel_pos.reshape(1, rel_pos.shape[0], -1).permute(0, 2, 1),
            size=max_rel_dist,
            mode="linear",
        )
        rel_pos_resized = rel_pos_resized.reshape(-1, max_rel_dist).permute(1, 0)
    else:
        rel_pos_resized = rel_pos

    q_coords = torch.arange(q_size)[:, None] * max(k_size / q_size, 1.0)
    k_coords = torch.arange(k_size)[None, :] * max(q_size / k_size, 1.0)
    relative_coords = (q_coords - k_coords) + (k_size - 1) * max(q_size / k_size, 1.0)
    return rel_pos_resized[relative_coords.long()]


def add_decomposed_rel_pos(
    attn: torch.Tensor,
    q: torch.Tensor,
    rel_pos_h: torch.Tensor,
    rel_pos_w: torch.Tensor,
    q_size: Tuple[int, int],
    k_size: Tuple[int, int],
) -> torch.Tensor:
    """Add decomposed relative positional embeddings (from MViTv2)."""
    q_h, q_w = q_size
    k_h, k_w = k_size
    Rh = get_rel_pos(q_h, k_h, rel_pos_h)
    Rw = get_rel_pos(q_w, k_w, rel_pos_w)

    B, _, dim = q.shape
    r_q = q.reshape(B, q_h, q_w, dim)
    rel_h = torch.einsum("bhwc,hkc->bhwk", r_q, Rh)
    rel_w = torch.einsum("bhwc,wkc->bhwk", r_q, Rw)

    attn = (
        attn.view(B, q_h, q_w, k_h, k_w)
        + rel_h[:, :, :, :, None]
        + rel_w[:, :, :, None, :]
    ).view(B, q_h * q_w, k_h * k_w)
    return attn


# ---------------------------------------------------------------------------
# Decoder and top-level models
# ---------------------------------------------------------------------------

class ImageDecoder(nn.Module):
    def __init__(self, in_chans: int = 256, out_chans: int = 1):
        super().__init__()
        self.Up5 = UpConv(in_chans, in_chans // 2)
        self.Up_conv5 = ConvBlock(in_chans // 2, in_chans // 2)
        self.Up4 = UpConv(in_chans // 2, in_chans // 4)
        self.Up_conv4 = ConvBlock(in_chans // 4, in_chans // 4)
        self.Up3 = UpConv(in_chans // 4, in_chans // 8)
        self.Up_conv3 = ConvBlock(in_chans // 8, in_chans // 8)
        self.Up2 = UpConv(in_chans // 8, in_chans // 16)
        self.Up_conv2 = ConvBlock(in_chans // 16, in_chans // 16)
        self.Conv = nn.Conv2d(in_chans // 16, out_chans, kernel_size=3, stride=1, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        d5 = self.Up_conv5(self.Up5(x))
        d4 = self.Up_conv4(self.Up4(d5))
        d3 = self.Up_conv3(self.Up3(d4))
        d2 = self.Up_conv2(self.Up2(d3))
        return self.Conv(d2)


class Frepa_ViT(nn.Module):
    def __init__(self, in_chans=1, mid_chans=256, out_chans=3, use_hierarchical_embed=False):
        super().__init__()
        self.encoder = ImageEncoderViT(
            img_size=512,
            in_chans=in_chans,
            out_chans=mid_chans,
            use_hierarchical_embed=use_hierarchical_embed,
        )
        self.decoder = ImageDecoder(in_chans=mid_chans, out_chans=out_chans)

    def forward(self, x):
        return self.decoder(self.encoder(x))


class Frepa_SwinT(nn.Module):
    def __init__(self, in_chans=1):
        super().__init__()
        if not _TORCHVISION_AVAILABLE:
            raise ImportError("torchvision is required for Frepa_SwinT but could not be imported.")
        self.encoder = swin_v2_b(weights=Swin_V2_B_Weights.DEFAULT)
        self.encoder.features[0][0] = nn.Conv2d(in_chans, 128, kernel_size=4, stride=4)
        self.encoder = self.encoder.features
        self.decoder = ImageDecoder(in_chans=1024, out_chans=3)

    def forward(self, x):
        x = self.encoder(x)
        x = rearrange(x, 'b h w c -> b c h w')
        return self.decoder(x)
