# --------------------------------------------------------
# EEGMirror -- Stage 3: Brain Modality Alignment (paper Sec 3.2, Fig. 3(a)).
#
# Fine-tunes the pretrained encoder E_brain (Stage 2) with two heads:
#   - Semantic Predictor D_s : MLP, pooled EEG feature -> CLIP text
#     embedding (77 x 768), trained with SoftCLIP (MindEye-style) + MSE.
#   - Seq2Seq low-level decoder D_l : auto-regressive Transformer decoder,
#     EEG patch-token sequence -> per-frame Stable-Diffusion VAE latents
#     H_hat = {h_hat_1, ..., h_hat_M}, trained with MSE.
# --------------------------------------------------------
import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import trunc_normal_


def soft_clip_loss(target_embed, pred_embed, temperature=0.1):
    """SoftCLIP loss (MindEye, Scotti et al. NeurIPS'23), paper Eq. in Sec 3.2.1.

    target_embed / pred_embed: (N, T, D) CLIP text token embeddings
    (T=77, D=768). The contrastive dot-products need a single vector per
    sample, so we mean-pool over the token dimension and L2-normalize
    before computing the (N, N) similarity matrices; the raw (T, D)
    tensors are instead compared directly by the separate MSE term
    (L_text = L_SoftCLIP + L_MSE, see modeling_align.py forward()).
    """
    N = target_embed.shape[0]
    lt = F.normalize(target_embed.mean(dim=1), dim=-1)  # (N, D)
    lp = F.normalize(pred_embed.mean(dim=1), dim=-1)     # (N, D)

    logits_tt = lt @ lt.t() / temperature   # target-target similarity
    logits_pt = lp @ lt.t() / temperature   # prediction-target similarity

    target_probs = F.softmax(logits_tt, dim=1)
    log_pred_probs = F.log_softmax(logits_pt, dim=1)
    loss = -(target_probs * log_pred_probs).sum(dim=1).mean()
    return loss


class SemanticPredictor(nn.Module):
    """MLP decoder D_s: pooled brain feature -> CLIP text embedding (77, 768)."""

    def __init__(self, embed_dim, num_tokens=77, text_dim=768, hidden_dim=None):
        super().__init__()
        hidden_dim = hidden_dim or embed_dim
        self.num_tokens = num_tokens
        self.text_dim = text_dim
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim), nn.GELU(),
            nn.Linear(hidden_dim, num_tokens * text_dim),
        )

    def forward(self, pooled_feat):
        out = self.mlp(pooled_feat)
        return out.reshape(pooled_feat.shape[0], self.num_tokens, self.text_dim)


class Seq2SeqLowLevelDecoder(nn.Module):
    """Auto-regressive Transformer decoder D_l: EEG patch tokens -> per-frame
    VAE latent embeddings H_hat = {h_hat_1, ..., h_hat_M}.
    """

    def __init__(self, embed_dim, num_frames, latent_shape, depth=4, num_heads=8, mlp_ratio=4., dropout=0.1):
        super().__init__()
        self.num_frames = num_frames
        self.latent_shape = latent_shape  # (c, h, w)
        out_dim = 1
        for d in latent_shape:
            out_dim *= d
        self.out_dim = out_dim

        self.query_embed = nn.Parameter(torch.zeros(1, num_frames, embed_dim))
        trunc_normal_(self.query_embed, std=0.02)

        layer = nn.TransformerDecoderLayer(
            d_model=embed_dim, nhead=num_heads, dim_feedforward=int(embed_dim * mlp_ratio),
            dropout=dropout, batch_first=True, activation='gelu',
        )
        self.decoder = nn.TransformerDecoder(layer, num_layers=depth)
        self.head = nn.Linear(embed_dim, out_dim)

    def forward(self, memory):
        """memory: (B, N_patch_tokens, D) -- EEG patch tokens from E_brain."""
        B = memory.shape[0]
        tgt = self.query_embed.expand(B, -1, -1)
        causal_mask = nn.Transformer.generate_square_subsequent_mask(self.num_frames).to(memory.device)
        out = self.decoder(tgt, memory, tgt_mask=causal_mask)  # (B, M, D)
        out = self.head(out)                                   # (B, M, out_dim)
        return out.reshape(B, self.num_frames, *self.latent_shape)


class EEGMirrorAlignModel(nn.Module):
    def __init__(self, encoder, embed_dim, num_text_tokens=77, text_dim=768,
                 num_frames=6, latent_shape=(4, 36, 64), seq2seq_depth=4, seq2seq_heads=8,
                 softclip_temperature=0.1):
        super().__init__()
        self.encoder = encoder  # pretrained EEGMirrorTransformer (E_brain), fine-tuned here
        self.semantic_predictor = SemanticPredictor(embed_dim, num_text_tokens, text_dim)
        self.seq2seq = Seq2SeqLowLevelDecoder(embed_dim, num_frames, latent_shape, depth=seq2seq_depth, num_heads=seq2seq_heads)
        self.softclip_temperature = softclip_temperature

    def forward(self, x, ch_names, text_target=None, frame_target=None):
        """
        x: (B, C, K, W) raw EEG patches.
        text_target: (B, 77, 768) CLIP text embedding of the BLIP caption (optional).
        frame_target: (B, M, c, h, w) ground-truth VAE latents of the video clip (optional).
        """
        all_tokens = self.encoder(x, ch_names, return_all_tokens=True)  # (B, 1+C*K, D)
        pooled, patch_tokens = all_tokens[:, 0], all_tokens[:, 1:]

        pred_text = self.semantic_predictor(pooled)
        pred_frames = self.seq2seq(patch_tokens)

        losses = {}
        if text_target is not None:
            losses['softclip_loss'] = soft_clip_loss(text_target, pred_text, self.softclip_temperature)
            losses['text_mse_loss'] = F.mse_loss(pred_text, text_target)
        if frame_target is not None:
            losses['low_level_mse_loss'] = F.mse_loss(pred_frames, frame_target)

        total_loss = sum(losses.values()) if losses else None
        return {
            'pred_text_embed': pred_text,
            'pred_frame_latents': pred_frames,
            'losses': losses,
            'total_loss': total_loss,
        }
