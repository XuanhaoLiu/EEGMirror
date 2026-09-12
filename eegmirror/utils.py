# --------------------------------------------------------
# EEGMirror -- misc training utilities.
# --------------------------------------------------------
import json
import os
import random

import numpy as np
import torch


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class AverageMeter:
    def __init__(self):
        self.reset()

    def reset(self):
        self.sum = 0.0
        self.count = 0

    def update(self, val, n=1):
        self.sum += float(val) * n
        self.count += n

    @property
    def avg(self):
        return self.sum / max(self.count, 1)


def save_checkpoint(path, model, optimizer=None, epoch=None, extra=None):
    os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
    payload = {'model': model.state_dict()}
    if optimizer is not None:
        payload['optimizer'] = optimizer.state_dict()
    if epoch is not None:
        payload['epoch'] = epoch
    if extra:
        payload.update(extra)
    torch.save(payload, path)


def load_checkpoint(path, model, optimizer=None, map_location='cpu'):
    ckpt = torch.load(path, map_location=map_location)
    state_dict = ckpt['model'] if 'model' in ckpt else ckpt
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing:
        print(f"[load_checkpoint] missing keys: {missing}")
    if unexpected:
        print(f"[load_checkpoint] unexpected keys: {unexpected}")
    if optimizer is not None and 'optimizer' in ckpt:
        optimizer.load_state_dict(ckpt['optimizer'])
    return ckpt


def save_args(path, args):
    os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
    with open(path, 'w') as f:
        json.dump(vars(args) if not isinstance(args, dict) else args, f, indent=2)


def sd_vae_latent_shape(height: int, width: int, latent_channels: int = 4, downsample_factor: int = 8):
    """Stable-Diffusion's frozen VAE (AutoencoderKL) downsamples spatial
    resolution by `downsample_factor` (8 for SD 1.x/2.x) and maps to
    `latent_channels` (4) channels. Given a pixel-space video frame of
    size (height, width), this returns the (C, H, W) shape of one VAE
    latent frame h_j = E_v(f_j) used as the low-level alignment target
    y2 in EEG2Video, see paper Sec 3.2.2.

    Example (this project's default): 512x288 video -> (4, 36, 64).
    """
    if height % downsample_factor != 0 or width % downsample_factor != 0:
        raise ValueError(
            f"height={height} and width={width} must both be divisible by "
            f"the VAE downsample factor ({downsample_factor})."
        )
    return latent_channels, height // downsample_factor, width // downsample_factor


def video_latent_shape(num_frames: int, height: int, width: int,
                        latent_channels: int = 4, downsample_factor: int = 8):
    """Full y2 shape for an M-frame clip: (M, C, H, W)."""
    c, h, w = sd_vae_latent_shape(height, width, latent_channels, downsample_factor)
    return num_frames, c, h, w
