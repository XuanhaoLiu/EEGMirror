# --------------------------------------------------------
# EEGMirror
# Shared Transformer backbone used by all three training stages:
#   Stage 1 (modeling_vqnsp.py)   -- encoder & decoder of the neural codebook
#   Stage 2 (modeling_pretrain.py)-- the masked brain encoder E_brain
#   Stage 3 (modeling_align.py)   -- fine-tuned E_brain for multimodal alignment
# --------------------------------------------------------
import torch
import torch.nn as nn
from timm.models.layers import trunc_normal_

from .layers import Block, init_weights_trunc_normal
from .mape import MontageAgnosticPositionEmbedding


class NeuralPatchEmbed(nn.Module):
    """Neural patch embedder E_e, paper Sec 3.1.1.

    Normalization + a small stack of 1-D conv + GELU layers, followed by
    global average pooling so that it accepts an *arbitrary* patch length
    W (the sliding-window length is a hyper-parameter in EEGMirror, unlike
    LaBraM which hard-codes W = 200 samples = 1s).

    Input:  (B, C, K, W)  raw EEG patches (C channels, K patches/channel)
    Output: (B, C, K, D)  continuous patch embeddings q_{c,k}
    """

    def __init__(self, embed_dim, hidden_chans=8, out_chans=16):
        super().__init__()
        self.norm_in = nn.InstanceNorm1d(1, affine=False)
        self.conv1 = nn.Conv1d(1, hidden_chans, kernel_size=15, stride=2, padding=7)
        self.gn1 = nn.GroupNorm(4, hidden_chans)
        self.conv2 = nn.Conv1d(hidden_chans, hidden_chans, kernel_size=3, padding=1)
        self.gn2 = nn.GroupNorm(4, hidden_chans)
        self.conv3 = nn.Conv1d(hidden_chans, out_chans, kernel_size=3, padding=1)
        self.gn3 = nn.GroupNorm(4, out_chans)
        self.act = nn.GELU()
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.proj = nn.Linear(out_chans, embed_dim)

    def forward(self, x):
        B, C, K, W = x.shape
        x = x.reshape(B * C * K, 1, W)
        x = self.norm_in(x)
        x = self.act(self.gn1(self.conv1(x)))
        x = self.act(self.gn2(self.conv2(x)))
        x = self.act(self.gn3(self.conv3(x)))
        x = self.pool(x).squeeze(-1)          # (B*C*K, out_chans)
        x = self.proj(x)                      # (B*C*K, D)
        return x.reshape(B, C, K, -1)


class CodeEmbed(nn.Module):
    """Linear projection used by the Stage-1 *decoder*, which consumes
    already-quantized codebook vectors (dim = code_dim) rather than raw
    EEG samples -- analogous to LaBraM's `PatchEmbed` (in_chans != 1
    branch) reused for the VQNSP decoder."""

    def __init__(self, in_dim, embed_dim):
        super().__init__()
        self.proj = nn.Linear(in_dim, embed_dim)

    def forward(self, x):
        # x: (B, C, K, in_dim) -> (B, C, K, embed_dim)
        return self.proj(x)


class EEGMirrorTransformer(nn.Module):
    """Montage-agnostic Transformer encoder.

    `patch_embed` turns raw/quantized patches into continuous embeddings;
    MAPE injects spatial + temporal position information; masked positions
    (Stage 2 only) are replaced by a learned `mask_token`.
    """

    def __init__(self, patch_embed: nn.Module, embed_dim=200, depth=12, num_heads=10,
                 mlp_ratio=4., qkv_bias=True, qk_norm=None, drop_rate=0., attn_drop_rate=0.,
                 drop_path_rate=0., norm_layer=nn.LayerNorm, init_values=0.1,
                 use_mask_token=False, omega=10000.0, init_std=0.02):
        super().__init__()
        self.embed_dim = embed_dim
        self.init_std = init_std
        self.patch_embed = patch_embed
        self.mape = MontageAgnosticPositionEmbedding(embed_dim, omega=omega)

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.mask_token = nn.Parameter(torch.zeros(1, 1, embed_dim)) if use_mask_token else None

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
        self.blocks = nn.ModuleList([
            Block(dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias,
                  qk_norm=qk_norm, drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i],
                  norm_layer=norm_layer, init_values=init_values)
            for i in range(depth)
        ])
        self.norm = norm_layer(embed_dim)
        self.pos_drop = nn.Dropout(p=drop_rate)

        trunc_normal_(self.cls_token, std=init_std)
        if self.mask_token is not None:
            trunc_normal_(self.mask_token, std=init_std)
        self.apply(lambda m: init_weights_trunc_normal(m, std=init_std))

    @torch.jit.ignore
    def no_weight_decay(self):
        names = {'cls_token'}
        if self.mask_token is not None:
            names.add('mask_token')
        return names

    def forward(self, patches, ch_names, bool_masked_pos=None,
                return_patch_tokens=False, return_all_tokens=False):
        """
        patches: (B, C, K, W) raw EEG (or (B, C, K, code_dim) for the Stage-1 decoder)
        ch_names: list[str] of length C -- channel names for THIS batch (one montage/batch)
        bool_masked_pos: optional (B, C*K) bool tensor, True = masked
        """
        x = self.patch_embed(patches)                    # (B, C, K, D)
        B, C, K, D = x.shape
        x = x.reshape(B, C * K, D)

        if bool_masked_pos is not None:
            assert self.mask_token is not None, "This encoder was built without a mask_token."
            mask_token = self.mask_token.expand(B, C * K, -1)
            w = bool_masked_pos.unsqueeze(-1).type_as(mask_token)
            x = x * (1 - w) + mask_token * w

        mape = self.mape(ch_names, K, device=x.device, dtype=x.dtype).reshape(1, C * K, D)
        x = x + mape

        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = self.pos_drop(x)

        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)

        if return_all_tokens:
            return x
        if return_patch_tokens:
            return x[:, 1:]
        return x[:, 0]  # cls token (pooled representation)
