# --------------------------------------------------------
# EEGMirror
# EMA-updated, L2-normalized vector quantizer (the "neural codebook" of
# paper Sec 3.1.3), adapted from LaBraM's
# norm_ema_quantizer.py (itself adapted from BEiT-v2 / VQ-VAE-2),
#(amplitude only, see modeling_vqnsp.py).
# --------------------------------------------------------
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as distributed
from einops import rearrange, repeat


def l2norm(t):
    return F.normalize(t, p=2, dim=-1)


def ema_inplace(moving_avg, new, decay):
    moving_avg.data.mul_(decay).add_(new, alpha=(1 - decay))


def sample_vectors(samples, num):
    num_samples, device = samples.shape[0], samples.device
    if num_samples >= num:
        indices = torch.randperm(num_samples, device=device)[:num]
    else:
        indices = torch.randint(0, num_samples, (num,), device=device)
    return samples[indices]


def kmeans(samples, num_clusters, num_iters=10, use_cosine_sim=False):
    dim, dtype, device = samples.shape[-1], samples.dtype, samples.device
    means = sample_vectors(samples, num_clusters)

    for _ in range(num_iters):
        if use_cosine_sim:
            dists = samples @ means.t()
        else:
            diffs = rearrange(samples, 'n d -> n () d') - rearrange(means, 'c d -> () c d')
            dists = -(diffs ** 2).sum(dim=-1)

        buckets = dists.max(dim=-1).indices
        bins = torch.bincount(buckets, minlength=num_clusters)
        zero_mask = bins == 0
        bins_min_clamped = bins.masked_fill(zero_mask, 1)

        new_means = buckets.new_zeros(num_clusters, dim, dtype=dtype)
        new_means.scatter_add_(0, repeat(buckets, 'n -> n d', d=dim), samples)
        new_means = new_means / bins_min_clamped[..., None]

        if use_cosine_sim:
            new_means = l2norm(new_means)
        means = torch.where(zero_mask[..., None], means, new_means)

    return means, bins


class EmbeddingEMA(nn.Module):
    def __init__(self, num_tokens, codebook_dim, decay=0.99, eps=1e-5, kmeans_init=True):
        super().__init__()
        self.num_tokens = num_tokens
        self.codebook_dim = codebook_dim
        self.decay = decay
        self.eps = eps
        if not kmeans_init:
            weight = l2norm(torch.randn(num_tokens, codebook_dim))
        else:
            weight = torch.zeros(num_tokens, codebook_dim)
        self.register_buffer('initted', torch.Tensor([not kmeans_init]))
        self.weight = nn.Parameter(weight, requires_grad=False)
        self.cluster_size = nn.Parameter(torch.zeros(num_tokens), requires_grad=False)
        self.embed_avg = nn.Parameter(weight.clone(), requires_grad=False)
        self.update = True

    @torch.jit.ignore
    def init_embed_(self, data):
        if self.initted:
            return
        embed, cluster_size = kmeans(data, self.num_tokens, 10, use_cosine_sim=True)
        self.weight.data.copy_(embed)
        self.cluster_size.data.copy_(cluster_size)
        self.initted.data.copy_(torch.Tensor([True]))

    def forward(self, embed_id):
        return F.embedding(embed_id, self.weight)


def norm_ema_inplace(moving_avg, new, decay):
    moving_avg.data.mul_(decay).add_(new, alpha=(1 - decay))
    moving_avg.data.copy_(l2norm(moving_avg.data))


class NormEMAVectorQuantizer(nn.Module):
    def __init__(self, n_embed, embedding_dim, beta=1.0, decay=0.99, eps=1e-5,
                 statistic_code_usage=True, kmeans_init=True):
        super().__init__()
        self.codebook_dim = embedding_dim
        self.num_tokens = n_embed
        self.beta = beta
        self.decay = decay

        self.embedding = EmbeddingEMA(self.num_tokens, self.codebook_dim, decay, eps, kmeans_init)
        self.statistic_code_usage = statistic_code_usage
        if statistic_code_usage:
            self.register_buffer('cluster_size', torch.zeros(n_embed))
        if distributed.is_available() and distributed.is_initialized():
            self.all_reduce_fn = distributed.all_reduce
        else:
            self.all_reduce_fn = nn.Identity()

    def forward(self, z):
        # z: (B, C, H, W) -> flatten to (B*H*W, C) style tokens
        z = rearrange(z, 'b c h w -> b h w c')
        z = l2norm(z)
        z_flattened = z.reshape(-1, self.codebook_dim)
        self.embedding.init_embed_(z_flattened)

        d = z_flattened.pow(2).sum(dim=1, keepdim=True) + \
            self.embedding.weight.pow(2).sum(dim=1) - 2 * \
            torch.einsum('bd,nd->bn', z_flattened, self.embedding.weight)

        encoding_indices = torch.argmin(d, dim=1)
        z_q = self.embedding(encoding_indices).view(z.shape)
        encodings = F.one_hot(encoding_indices, self.num_tokens).type(z.dtype)

        if not self.training:
            with torch.no_grad():
                cluster_size = encodings.sum(0)
                self.all_reduce_fn(cluster_size)
                ema_inplace(self.cluster_size, cluster_size, self.decay)

        if self.training and self.embedding.update:
            bins = encodings.sum(0)
            self.all_reduce_fn(bins)
            ema_inplace(self.cluster_size, bins, self.decay)

            zero_mask = (bins == 0)
            bins = bins.masked_fill(zero_mask, 1.)

            embed_sum = z_flattened.t() @ encodings
            self.all_reduce_fn(embed_sum)

            embed_normalized = l2norm((embed_sum / bins.unsqueeze(0)).t())
            embed_normalized = torch.where(zero_mask[..., None], self.embedding.weight, embed_normalized)
            norm_ema_inplace(self.embedding.weight, embed_normalized, self.decay)

        loss = self.beta * F.mse_loss(z_q.detach(), z)
        z_q = z + (z_q - z).detach()  # straight-through estimator
        z_q = rearrange(z_q, 'b h w c -> b c h w')
        return z_q, loss, encoding_indices
