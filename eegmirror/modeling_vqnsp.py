# --------------------------------------------------------
# EEGMirror -- Stage 1: Neural Codebook pre-training (paper Sec 3.1.3,Fig. 1(a)).
# --------------------------------------------------------
import torch
import torch.nn as nn
import torch.nn.functional as F

from .encoder import EEGMirrorTransformer, NeuralPatchEmbed, CodeEmbed
from .quantizer import NormEMAVectorQuantizer


class EEGMirrorVQNSP(nn.Module):
    def __init__(self, patch_len, embed_dim=200, encoder_depth=12, decoder_depth=3,
                 num_heads=10, mlp_ratio=4., code_dim=32, n_embed=8192, decay=0.99,
                 kmeans_init=True, drop_path_rate=0., smooth_l1_loss=False):
        super().__init__()
        self.patch_len = patch_len
        self.code_dim = code_dim

        self.encoder = EEGMirrorTransformer(
            patch_embed=NeuralPatchEmbed(embed_dim),
            embed_dim=embed_dim, depth=encoder_depth, num_heads=num_heads, mlp_ratio=mlp_ratio,
            drop_path_rate=drop_path_rate, use_mask_token=False,
        )
        self.decoder = EEGMirrorTransformer(
            patch_embed=CodeEmbed(code_dim, embed_dim),
            embed_dim=embed_dim, depth=decoder_depth, num_heads=num_heads, mlp_ratio=mlp_ratio,
            drop_path_rate=0., use_mask_token=False,
        )
        self.quantize = NormEMAVectorQuantizer(
            n_embed=n_embed, embedding_dim=code_dim, beta=1.0, decay=decay, kmeans_init=kmeans_init,
        )

        self.encode_task_layer = nn.Sequential(
            nn.Linear(embed_dim, embed_dim), nn.Tanh(), nn.Linear(embed_dim, code_dim),
        )
        # amplitude-only reconstruction head (LaBraM additionally has a
        # `decode_task_layer_angle` for phase -- intentionally removed here)
        self.decode_task_layer = nn.Sequential(
            nn.Linear(embed_dim, embed_dim), nn.Tanh(), nn.Linear(embed_dim, patch_len),
        )
        self.loss_fn = F.smooth_l1_loss if smooth_l1_loss else F.mse_loss

    def get_number_of_tokens(self):
        return self.quantize.num_tokens

    @staticmethod
    def std_norm(x):
        # normalize each sample's amplitude spectrum over (channel, patch, freq)
        mean = torch.mean(x, dim=(1, 2, 3), keepdim=True)
        std = torch.std(x, dim=(1, 2, 3), keepdim=True) + 1e-6
        return (x - mean) / std

    def encode(self, x, ch_names):
        """x: (B, C, K, W) raw EEG patches -> quantized (B, code_dim, C, K), loss, indices (B*C*K,)"""
        B, C, K, W = x.shape
        feats = self.encoder(x, ch_names, return_patch_tokens=True)      # (B, C*K, D)
        with torch.amp.autocast(device_type=feats.device.type, enabled=False):
            z = self.encode_task_layer(feats.float())                   # (B, C*K, code_dim)
        z = z.reshape(B, C, K, self.code_dim).permute(0, 3, 1, 2)        # (B, code_dim, C, K)
        quantize, loss, embed_ind = self.quantize(z)
        return quantize, embed_ind, loss

    def decode(self, quantize, ch_names):
        """quantize: (B, code_dim, C, K) -> reconstructed amplitude (B, C, K, W)"""
        B, code_dim, C, K = quantize.shape
        q = quantize.permute(0, 2, 3, 1)                                 # (B, C, K, code_dim)
        feats = self.decoder(q, ch_names, return_patch_tokens=True)      # (B, C*K, D)
        rec = self.decode_task_layer(feats)                              # (B, C*K, W)
        return rec.reshape(B, C, K, -1)

    @torch.no_grad()
    def get_codebook_indices(self, x, ch_names):
        """Frozen inference path used by Stage 2 to build target token ids."""
        _, embed_ind, _ = self.encode(x, ch_names)
        B, C, K, _ = x.shape
        return embed_ind.view(B, C * K)

    def forward(self, x, ch_names):
        """
        x: (B, C, K, W) raw EEG patches (already sliding-window segmented).
        Returns (loss, log_dict).
        """
        x_fft = torch.fft.fft(x, dim=-1)
        amplitude = self.std_norm(torch.abs(x_fft))                      # target, phase intentionally dropped

        quantize, _, emb_loss = self.encode(x, ch_names)
        rec_amplitude = self.decode(quantize, ch_names)
        rec_loss = self.loss_fn(rec_amplitude, amplitude)

        loss = emb_loss + rec_loss
        log = {
            'quant_loss': emb_loss.detach(),
            'rec_amplitude_loss': rec_loss.detach(),
            'total_loss': loss.detach(),
        }
        return loss, log


def build_vqnsp(patch_len, size='base', **kwargs):
    presets = {
        'base': dict(embed_dim=200, encoder_depth=12, decoder_depth=3, num_heads=10),
        'small': dict(embed_dim=128, encoder_depth=6, decoder_depth=2, num_heads=8),
    }
    cfg = presets[size]
    cfg.update(kwargs)
    return EEGMirrorVQNSP(patch_len=patch_len, **cfg)
