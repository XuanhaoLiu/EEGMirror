# --------------------------------------------------------
# EEGMirror
# Masking strategies for Stage-2 pre-training (paper Sec 3.1.4, Fig. 2).
# All three return a (B, C*K) bool tensor (True = masked), flattened in
# the same (channel-major, then time) order used by EEGMirrorTransformer:
# index = channel_idx * K + patch_idx.
# --------------------------------------------------------
import torch


def random_masking(B, C, K, ratio, device=None):
    """Random masking: every (channel, patch) token treated i.i.d."""
    L = C * K
    len_mask = max(1, min(L - 1, round(L * ratio)))
    noise = torch.rand(B, L, device=device)
    ids_sorted = torch.argsort(noise, dim=1)  # ascending: first `len_mask` get masked
    mask = torch.zeros(B, L, dtype=torch.bool, device=device)
    mask.scatter_(1, ids_sorted[:, :len_mask], True)
    return mask


def channel_masking(B, C, K, ratio, device=None):
    """Mask a subset of channels across ALL timestamps (spatial masking).

    Number of masked channels = floor(C * ratio), at least 1, so the model
    is forced to infer a channel's signal purely from its neighbours.
    """
    n_mask = max(1, min(C - 1, int(C * ratio)))  # floor via int() on a non-negative ratio*C
    mask = torch.zeros(B, C, K, dtype=torch.bool, device=device)
    for b in range(B):
        chan_idx = torch.randperm(C, device=device)[:n_mask]
        mask[b, chan_idx, :] = True
    return mask.reshape(B, C * K)


def frame_masking(B, C, K, ratio, device=None):
    """Mask a contiguous-or-scattered subset of timestamps across ALL
    channels (temporal masking).

    Number of masked timestamps = floor(K * ratio), at least 1, and at
    least one masked *contiguous span* is guaranteed by masking a single
    random contiguous block of that size (rather than K independent
    coin-flips), matching "mask at least one temporal segment".
    """
    n_mask = max(1, min(K - 1, int(K * ratio)))  # floor(K*ratio), e.g. 10 patches @0.75 -> 7
    mask = torch.zeros(B, C, K, dtype=torch.bool, device=device)
    for b in range(B):
        start = torch.randint(0, K - n_mask + 1, (1,), device=device).item()
        mask[b, :, start:start + n_mask] = True
    return mask.reshape(B, C * K)


MASKING_FNS = {
    'random': random_masking,
    'channel': channel_masking,
    'frame': frame_masking,
}
