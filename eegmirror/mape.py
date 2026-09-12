# --------------------------------------------------------
# EEGMirror
# Montage-Agnostic Position Embeddings (MAPE), paper Sec 3.1.2.
#   SE_{x,y} = concat[ PE(x_p), PE(y_p) ]                 (spatial)
#   TE(t_r)  = sinusoidal PE of the relative patch index    (temporal)
#   e = patch_embedding + SE(channel) + TE(patch_index)
# --------------------------------------------------------
import torch
import torch.nn as nn
import torch.nn.functional as F

from .channels import channels_to_xy


def _sincos_pe(pos: torch.Tensor, dim: int, omega: float = 10000.0) -> torch.Tensor:
    """Standard Transformer sinusoidal position encoding.

    pos: (...,) arbitrary-shaped tensor of scalar positions.
    returns: (..., dim)
    """
    device = pos.device
    dtype = torch.float32
    i = torch.arange(dim, device=device, dtype=dtype)
    div = torch.pow(torch.tensor(float(omega), device=device, dtype=dtype), (2 * (i // 2)) / max(dim, 1))
    angles = pos.to(dtype).unsqueeze(-1) / div  # (..., dim)
    pe = torch.zeros(*pos.shape, dim, device=device, dtype=dtype)
    pe[..., 0::2] = torch.sin(angles[..., 0::2])
    pe[..., 1::2] = torch.cos(angles[..., 1::2])
    return pe


class MontageAgnosticPositionEmbedding(nn.Module):
    """Computes MAPE = Spatial Embedding (SE) + Temporal Embedding (TE).

    Both are parameter-free (closed-form sinusoidal) so the module can be
    evaluated on any channel montage / any number of patches at inference
    time without re-training or padding.
    """

    def __init__(self, embed_dim: int, omega: float = 10000.0):
        super().__init__()
        self.embed_dim = embed_dim
        # SE concatenates a PE(x) half and a PE(y) half -> total embed_dim.
        self.half_dim = embed_dim - embed_dim // 2  # ceil half, so concat covers embed_dim exactly
        self.other_half = embed_dim // 2
        self.omega = omega
        self._se_cache = {}

    def spatial_embedding(self, ch_names, device=None, dtype=torch.float32):
        """Returns (C, embed_dim) spatial embedding for a list of channel names."""
        key = (tuple(ch_names), str(device))
        cached = self._se_cache.get(key)
        if cached is not None:
            return cached.to(dtype)
        xy = channels_to_xy(ch_names)  # list of (x, y) in [0, 100]
        xy = torch.tensor(xy, dtype=torch.float32, device=device)  # (C, 2)
        pe_x = _sincos_pe(xy[:, 0], self.half_dim, self.omega)     # (C, half_dim)
        pe_y = _sincos_pe(xy[:, 1], self.other_half, self.omega)   # (C, other_half)
        se = torch.cat([pe_x, pe_y], dim=-1)  # (C, embed_dim)
        self._se_cache[key] = se.detach()
        return se.to(dtype)

    def temporal_embedding(self, num_patches: int, device=None, dtype=torch.float32):
        """Returns (num_patches, embed_dim) relative temporal embedding."""
        t_r = torch.arange(num_patches, dtype=torch.float32, device=device)
        te = _sincos_pe(t_r, self.embed_dim, self.omega)
        return te.to(dtype)

    def forward(self, ch_names, num_patches: int, device=None, dtype=torch.float32):
        """Build the full (C, K, embed_dim) MAPE grid to add to patch embeddings.

        ch_names: list[str] of length C (channels present in this sample/batch)
        num_patches: K, number of temporal patches per channel
        """
        se = self.spatial_embedding(ch_names, device=device, dtype=dtype)   # (C, D)
        te = self.temporal_embedding(num_patches, device=device, dtype=dtype)  # (K, D)
        mape = se.unsqueeze(1) + te.unsqueeze(0)  # (C, K, D)
        return mape
