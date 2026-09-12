# --------------------------------------------------------
# EEGMirror -- Stage 4: Co-training with an inflated video-diffusion model
# (paper Sec 3.3, Fig. 3(b)).
#
# EEGMirror uses a Stable-Diffusion UNet inflated with sparse temporal
# attention (network inflation, Tune-A-Video / Wu et al. ICCV'23),
# fine-tuned to denoise a GT video's VAE latents conditioned on the
# semantic (pred_text_embed) and low-level (pred_frame_latents) signals
# decoded from EEG in Stage 3. Two UNet implementations are provided,
# both sharing the exact same `forward(noisy_latents, t, text_embed,
# frame_cond)` -> noise_pred contract so `DiffusionCoTrainer` (below)
# doesn't care which one it was handed:
#
#   - `TinyInflatedUNet`: a lightweight stand-in (plain 2D convs per frame
#     + SparseTemporalAttention across frames), no internet access or
#     pretrained weights needed -- used by scripts/test_pipeline.py and as
#     the --co_train_diffusion default when no --unet_ckpt is given.
#   - `TuneAVideoUNetWrapper` / `build_tuneavideo_unet(...)`: loads the
#     REAL inflated UNet from Tune-A-Video's own implementation
#     (/work1/xuanhao/Tune-A-Video, `tuneavideo.models.unet
#     .UNet3DConditionModel` -- a `diffusers`-compatible 3D UNet with
#     `InflatedConv3d` + sparse-temporal cross-frame attention baked in),
#     restored from a fine-tuned checkpoint directory via
#     `UNet3DConditionModel.from_pretrained(ckpt_dir, subfolder='unet')`
#     (the layout `diffusers`' `DiffusionPipeline.save_pretrained` writes,
#     e.g. Tune-A-Video's own `train_tuneavideo.py`).
#
# `SparseTemporalAttention` implements the "each frame attends to the
# first frame and the previous frame" inflation rule (Sec 3.3) for the toy
# UNet; the real Tune-A-Video UNet has its own (richer) sparse-causal
# attention built into `Transformer3DModel` (see
# /work1/xuanhao/Tune-A-Video/tuneavideo/models/attention.py), so nothing
# from this file needs to reimplement it for the real path.
# --------------------------------------------------------
import math
import os
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F


class SparseTemporalAttention(nn.Module):
    """Frame j attends only to {frame 0, frame j-1, frame j} (Sec 3.3)."""

    def __init__(self, dim, num_heads=4):
        super().__init__()
        self.attn = nn.MultiheadAttention(dim, num_heads, batch_first=True)

    def forward(self, x):
        # x: (B, M, N, D) -- N spatial tokens per frame
        B, M, N, D = x.shape
        outs = []
        for j in range(M):
            key_frames = sorted(set([0, max(0, j - 1), j]))
            kv = x[:, key_frames].reshape(B, len(key_frames) * N, D)
            q = x[:, j]
            out, _ = self.attn(q, kv, kv)
            outs.append(out)
        return torch.stack(outs, dim=1)


def timestep_embedding(t, dim):
    half = dim // 2
    freqs = torch.exp(-math.log(10000) * torch.arange(half, device=t.device).float() / half)
    args = t.float().unsqueeze(-1) * freqs.unsqueeze(0)
    emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
    if dim % 2:
        emb = F.pad(emb, (0, 1))
    return emb


class TinyInflatedUNet(nn.Module):
    """Toy stand-in for the real inflated Stable-Diffusion UNet.

    Same contract: predicts the noise added to `noisy_latents`, given the
    diffusion timestep and (semantic, low-level) EEG-decoded conditioning.
    """

    def __init__(self, latent_shape, text_dim=768, hidden=64, num_heads=4):
        super().__init__()
        c, h, w = latent_shape
        self.latent_shape = latent_shape
        self.time_dim = hidden

        self.in_conv = nn.Conv2d(c, hidden, kernel_size=3, padding=1)
        self.text_proj = nn.Linear(text_dim, hidden)
        self.frame_cond_proj = nn.Conv2d(c, hidden, kernel_size=3, padding=1)
        self.time_proj = nn.Linear(hidden, hidden)

        self.spatial_block = nn.Sequential(
            nn.GroupNorm(8, hidden), nn.SiLU(), nn.Conv2d(hidden, hidden, 3, padding=1),
            nn.GroupNorm(8, hidden), nn.SiLU(), nn.Conv2d(hidden, hidden, 3, padding=1),
        )
        self.temporal_attn = SparseTemporalAttention(hidden, num_heads=num_heads)
        self.out_conv = nn.Conv2d(hidden, c, kernel_size=3, padding=1)

    def forward(self, noisy_latents, t, text_embed, frame_cond):
        """
        noisy_latents: (B, M, c, h, w)
        t: (B,) diffusion timesteps
        text_embed: (B, 77, 768) semantic conditioning (pred_text_embed)
        frame_cond: (B, M, c, h, w) low-level conditioning (pred_frame_latents)
        """
        B, M, c, h, w = noisy_latents.shape
        x = noisy_latents.reshape(B * M, c, h, w)
        x = self.in_conv(x)

        t_emb = self.time_proj(timestep_embedding(t, self.time_dim)).to(x.dtype)      # (B, hidden)
        t_emb = t_emb.repeat_interleave(M, dim=0)[:, :, None, None]
        text_emb = self.text_proj(text_embed.mean(dim=1))                             # (B, hidden)
        text_emb = text_emb.repeat_interleave(M, dim=0)[:, :, None, None]
        frame_emb = self.frame_cond_proj(frame_cond.reshape(B * M, c, h, w))

        x = x + t_emb + text_emb + frame_emb
        x = x + self.spatial_block(x)

        x = x.reshape(B, M, -1, h, w).permute(0, 1, 3, 4, 2).reshape(B, M, h * w, -1)
        x = self.temporal_attn(x)
        x = x.reshape(B, M, h, w, -1).permute(0, 1, 4, 2, 3).reshape(B * M, -1, h, w)

        noise_pred = self.out_conv(x).reshape(B, M, c, h, w)
        return noise_pred


TUNEAVIDEO_ROOT = '/work1/xuanhao/Tune-A-Video'


def _import_tuneavideo_unet():
    """Lazy import so `diffusers`/`tuneavideo` are only required when the
    real UNet is actually requested (TinyInflatedUNet needs neither)."""
    if TUNEAVIDEO_ROOT not in sys.path:
        sys.path.insert(0, TUNEAVIDEO_ROOT)

    # Tune-A-Video pins diffusers==0.11.1, whose `dynamic_modules_utils.py`
    # imports `cached_download`/`HfFolder` from `huggingface_hub` -- both
    # removed from recent huggingface_hub releases. We never hit any
    # actual dynamic-module-download or Hub-auth code path (checkpoints
    # are always loaded from a local directory here), so a no-op shim is
    # enough to satisfy the import without downgrading huggingface_hub
    # (and risking breaking anything else in a shared conda env).
    import huggingface_hub
    if not hasattr(huggingface_hub, 'cached_download'):
        huggingface_hub.cached_download = huggingface_hub.hf_hub_download
    if not hasattr(huggingface_hub, 'HfFolder'):
        class _HfFolder:
            @staticmethod
            def get_token():
                return None
        huggingface_hub.HfFolder = _HfFolder

    from tuneavideo.models.unet import UNet3DConditionModel
    return UNet3DConditionModel


class TuneAVideoUNetWrapper(nn.Module):
    """Wraps a REAL (Tune-A-Video-inflated) `UNet3DConditionModel` so it
    exposes the same `forward(noisy_latents, t, text_embed, frame_cond)`
    contract as `TinyInflatedUNet`, translating between EEGMirror's
    frame-major `(B, M, c, h, w)` tensor layout (used throughout
    modeling_align.py / datasets.py) and diffusers'/Tune-A-Video's
    channel-major `(B, c, M, h, w)` layout (`InflatedConv3d` expects
    `b c f h w`, see Tune-A-Video's resnet.py).

    Low-level conditioning: the paper guides generation with BOTH the
    predicted text embedding and the predicted frame embeddings (Sec 3.3 /
    Fig. 3(b)). A stock Tune-A-Video/SD UNet only has a text
    cross-attention conditioning pathway, so `frame_cond` (pred_frame_
    latents) is threaded in the standard, architecture-preserving way used
    by e.g. SD-Inpainting/InstructPix2Pix: concatenated with the noisy
    latents along the channel dimension before `conv_in`. This is detected
    automatically from the loaded checkpoint's `unet.config.in_channels`:
      - `in_channels == out_channels`      : vanilla text-to-video UNet;
        `frame_cond` is NOT fed to the UNet (it is still supervised via
        Stage 3's `low_level_mse_loss`, just not used as UNet conditioning).
      - `in_channels == 2 * out_channels`  : `frame_cond` is concatenated
        with `noisy_latents` before `conv_in`, so the checkpoint must have
        been (fine-)tuned with this doubled input (e.g. by initializing
        `conv_in` with 2x its channels before Tune-A-Video training) to
        actually make use of the low-level guidance signal.
    """

    def __init__(self, unet):
        super().__init__()
        self.unet = unet
        self.latent_channels = unet.config.out_channels
        in_ch = unet.config.in_channels
        if in_ch == self.latent_channels:
            self.use_frame_cond = False
        elif in_ch == 2 * self.latent_channels:
            self.use_frame_cond = True
        else:
            raise ValueError(
                f"unet.config.in_channels={in_ch} matches neither a vanilla text-to-video UNet "
                f"(in_channels == out_channels == {self.latent_channels}) nor a low-level-conditioned "
                f"one (in_channels == 2 * out_channels == {2 * self.latent_channels})."
            )

    @classmethod
    def from_pretrained(cls, ckpt_dir, subfolder='unet', torch_dtype=None):
        """ckpt_dir: a `diffusers.DiffusionPipeline.save_pretrained(...)`
        output directory (e.g. Tune-A-Video's own `train_tuneavideo.py`
        writes one after fine-tuning) -- `<ckpt_dir>/<subfolder>/
        config.json` + weights."""
        UNet3DConditionModel = _import_tuneavideo_unet()
        unet = UNet3DConditionModel.from_pretrained(ckpt_dir, subfolder=subfolder)
        if torch_dtype is not None:
            unet = unet.to(torch_dtype)
        return cls(unet)

    def forward(self, noisy_latents, t, text_embed, frame_cond):
        """
        noisy_latents: (B, M, c, h, w)
        t: (B,) diffusion timesteps
        text_embed: (B, 77, cross_attention_dim) semantic conditioning (pred_text_embed)
        frame_cond: (B, M, c, h, w) low-level conditioning (pred_frame_latents)
        Returns noise_pred: (B, M, c, h, w)
        """
        dtype = next(self.unet.parameters()).dtype
        x = noisy_latents.permute(0, 2, 1, 3, 4).to(dtype)  # (B, c, M, h, w)
        if self.use_frame_cond:
            cond = frame_cond.permute(0, 2, 1, 3, 4).to(dtype)
            x = torch.cat([x, cond], dim=1)  # (B, 2c, M, h, w)
        out = self.unet(x, t, encoder_hidden_states=text_embed.to(dtype)).sample  # (B, c, M, h, w)
        return out.permute(0, 2, 1, 3, 4).float()  # (B, M, c, h, w)


def build_tuneavideo_unet(ckpt_dir, subfolder='unet', torch_dtype=None):
    """Convenience factory mirroring this project's other `build_*`
    helpers (build_vqnsp, build_pretrain_model): loads the real,
    fine-tuned inflated UNet used by Stage 4's DiffusionCoTrainer.

    Example: build_tuneavideo_unet('/work1/xuanhao/EEGMirror/output/finetuned_t2v_ckpt')
    """
    return TuneAVideoUNetWrapper.from_pretrained(ckpt_dir, subfolder=subfolder, torch_dtype=torch_dtype)


class DiffusionCoTrainer(nn.Module):
    """DDPM-style epsilon-prediction wrapper used to co-train the
    alignment module and the (inflated) video-diffusion UNet together."""

    def __init__(self, align_model, unet, num_timesteps=1000, beta_start=1e-4, beta_end=2e-2):
        super().__init__()
        self.align_model = align_model
        self.unet = unet
        betas = torch.linspace(beta_start, beta_end, num_timesteps)
        alphas_cumprod = torch.cumprod(1.0 - betas, dim=0)
        self.register_buffer('sqrt_alphas_cumprod', torch.sqrt(alphas_cumprod))
        self.register_buffer('sqrt_one_minus_alphas_cumprod', torch.sqrt(1.0 - alphas_cumprod))
        self.num_timesteps = num_timesteps

    def forward(self, x, ch_names, video_latents, text_target=None, frame_target=None):
        """video_latents: (B, M, c, h, w) GT VAE latents of the target video clip."""
        align_out = self.align_model(x, ch_names, text_target=text_target, frame_target=frame_target)
        text_cond = align_out['pred_text_embed']
        frame_cond = align_out['pred_frame_latents']

        B = video_latents.shape[0]
        device = video_latents.device
        t = torch.randint(0, self.num_timesteps, (B,), device=device)
        noise = torch.randn_like(video_latents)

        sqrt_ac = self.sqrt_alphas_cumprod[t].view(B, 1, 1, 1, 1)
        sqrt_1mac = self.sqrt_one_minus_alphas_cumprod[t].view(B, 1, 1, 1, 1)
        noisy_latents = sqrt_ac * video_latents + sqrt_1mac * noise

        noise_pred = self.unet(noisy_latents, t, text_cond, frame_cond)
        diffusion_loss = F.mse_loss(noise_pred, noise)

        losses = dict(align_out['losses'])
        losses['diffusion_loss'] = diffusion_loss.detach()
        total_loss = diffusion_loss + (align_out['total_loss'] if align_out['total_loss'] is not None else 0.0)
        return {'total_loss': total_loss, 'losses': losses}
