# --------------------------------------------------------
# EEGMirror -- Stage 2: Masked brain-encoder pre-training (paper Sec 3.1.4, Fig. 1(b) & Fig. 2).
# --------------------------------------------------------
import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import trunc_normal_

from .encoder import EEGMirrorTransformer, NeuralPatchEmbed
from .masking import MASKING_FNS

DEFAULT_MASK_RATIOS = {'random': 0.5, 'channel': 0.5, 'frame': 0.75}


class EEGMirrorForMaskedModeling(nn.Module):
    def __init__(self, patch_len, embed_dim=200, depth=12, num_heads=10, mlp_ratio=4.,
                 vocab_size=8192, drop_path_rate=0., init_std=0.02):
        super().__init__()
        self.patch_len = patch_len
        self.encoder = EEGMirrorTransformer(
            patch_embed=NeuralPatchEmbed(embed_dim),
            embed_dim=embed_dim, depth=depth, num_heads=num_heads, mlp_ratio=mlp_ratio,
            drop_path_rate=drop_path_rate, use_mask_token=True, init_std=init_std,
        )
        self.lm_head = nn.Linear(embed_dim, vocab_size)
        trunc_normal_(self.lm_head.weight, std=init_std)

    def get_encoder(self):
        """The encoder E_brain to be reused (and fine-tuned) in Stage 3."""
        return self.encoder

    def forward_one_pass(self, x, ch_names, bool_masked_pos):
        feats = self.encoder(x, ch_names, bool_masked_pos=bool_masked_pos, return_patch_tokens=True)
        return self.lm_head(feats[bool_masked_pos])  # (num_masked_tokens_in_batch, vocab_size)

    def forward(self, x, ch_names, target_ids, mask_ratios=None):
        """
        x: (B, C, K, W) raw EEG patches.
        target_ids: (B, C*K) int64 codebook indices from the frozen Stage-1 VQNSP,
                    computed on the UNMASKED patches.
        Returns: total_loss, {per-strategy detached losses}
        """
        mask_ratios = mask_ratios or DEFAULT_MASK_RATIOS
        B, C, K, W = x.shape
        device = x.device

        logs = {}
        total_loss = 0.0
        # three masking strategies
        for name, ratio in mask_ratios.items():
            bool_masked_pos = MASKING_FNS[name](B, C, K, ratio, device=device)
            logits = self.forward_one_pass(x, ch_names, bool_masked_pos)
            targets = target_ids[bool_masked_pos]
            loss = F.cross_entropy(logits, targets)
            logs[f'{name}_loss'] = loss.detach()
            logs[f'{name}_acc'] = (logits.argmax(dim=-1) == targets).float().mean().detach()
            total_loss = total_loss + loss

        logs['total_loss'] = total_loss.detach()
        return total_loss, logs


def build_pretrain_model(patch_len, size='base', vocab_size=8192, **kwargs):
    presets = {
        'base': dict(embed_dim=200, depth=12, num_heads=10),
        'small': dict(embed_dim=128, depth=6, num_heads=8),
    }
    cfg = presets[size]
    cfg.update(kwargs)
    return EEGMirrorForMaskedModeling(patch_len=patch_len, vocab_size=vocab_size, **cfg)
