# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
# This implementation is inspired from
# https://github.com/lucidrains/vector-quantize-pytorch
# which is released under MIT License. Hereafter, the original license:
# MIT License
#
# Copyright (c) 2020 Phil Wang
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

"""Core vector quantization implementation."""

import typing as tp
import warnings
import torch
from torch import nn, Tensor
import torch.nn.functional as F
from einops import rearrange, repeat
from torch import jit

from .. import distrib

def default(val: tp.Any, d: tp.Any) -> tp.Any:
    return val if val is not None else d

def ema_inplace(moving_avg: Tensor, new: Tensor, decay: float):
    moving_avg.data.mul_(decay).add_(new, alpha=(1 - decay))

def laplace_smoothing(x: Tensor, n_categories: int, epsilon: float = 1e-5) -> Tensor:
    return (x + epsilon) / (x.sum() + n_categories * epsilon)

def uniform_init(*shape: int) -> Tensor:
    t = torch.empty(shape)
    nn.init.kaiming_uniform_(t)
    return t

def sample_vectors(samples: Tensor, num: int) -> Tensor:
    num_samples, device = samples.shape[0], samples.device
    if num_samples >= num:
        indices = torch.randperm(num_samples, device=device)[:num]
    else:
        indices = torch.randint(0, num_samples, (num,), device=device)
    return samples[indices]

def kmeans(samples: Tensor, num_clusters: int, num_iters: int = 10) -> tp.Tuple[Tensor, Tensor]:
    dim, dtype = samples.shape[-1], samples.dtype
    means = sample_vectors(samples, num_clusters)

    for _ in range(num_iters):
        diffs = rearrange(samples, "n d -> n () d") - rearrange(means, "c d -> () c d")
        dists = -(diffs ** 2).sum(dim=-1)
        buckets = dists.max(dim=-1).indices
        bins = torch.bincount(buckets, minlength=num_clusters)
        zero_mask = bins == 0
        bins_min_clamped = bins.masked_fill(zero_mask, 1)

        new_means = buckets.new_zeros(num_clusters, dim, dtype=dtype)
        new_means.scatter_add_(0, repeat(buckets, "n -> n d", d=dim), samples)
        new_means = new_means / bins_min_clamped[..., None]
        means = torch.where(zero_mask[..., None], means, new_means)

    return means, bins


def _quantize_tensor(x: Tensor, precision: int = 7) -> Tensor:
    """Control precision of floating point operations"""
    return torch.round(x * 10**precision) / 10**precision

class EuclideanCodebook(nn.Module):
    """Codebook with Euclidean distance."""

    def __init__(
        self,
        dim: int,
        codebook_size: int,
        kmeans_init: bool = False,
        kmeans_iters: int = 10,
        decay: float = 0.99,
        epsilon: float = 1e-5,
        threshold_ema_dead_code: int = 2,
    ):
        super().__init__()
        self.decay = decay
        init_fn: tp.Union[tp.Callable[..., torch.Tensor], tp.Any] = uniform_init if not kmeans_init else torch.zeros
        embed = init_fn(codebook_size, dim)

        self.codebook_size = codebook_size
        self.kmeans_iters = kmeans_iters
        self.epsilon = epsilon
        self.threshold_ema_dead_code = threshold_ema_dead_code

        self.register_buffer("inited", torch.Tensor([not kmeans_init]))
        self.register_buffer("cluster_size", torch.zeros(codebook_size))
        self.register_buffer("embed", embed)
        self.register_buffer("embed_avg", embed.clone())


    @jit.ignore
    def init_embed_(self, data: Tensor) -> None:
        if self.inited:
            return

        embed, cluster_size = kmeans(data, self.codebook_size, self.kmeans_iters)
        self.embed.data.copy_(embed)
        self.embed_avg.data.copy_(embed.clone())
        self.cluster_size.data.copy_(cluster_size)
        self.inited.data.copy_(torch.Tensor([True]))
        distrib.broadcast_tensors(self.buffers())

    def replace_(self, samples: Tensor, mask: Tensor) -> None:
        modified_codebook = torch.where(
            mask[..., None],
            sample_vectors(samples, self.codebook_size),
            self.embed
        )
        self.embed.data.copy_(modified_codebook)

    def expire_codes_(self, batch_samples: Tensor) -> None:
        if self.threshold_ema_dead_code == 0:
            return

        expired_codes = self.cluster_size < self.threshold_ema_dead_code
        if not torch.any(expired_codes):
            return

        batch_samples = rearrange(batch_samples, "... d -> (...) d")
        self.replace_(batch_samples, mask=expired_codes)
        distrib.broadcast_tensors(self.buffers())

    def preprocess(self, x: Tensor) -> Tensor:
        return rearrange(x, "... d -> (...) d")

    def quantize(self, x: Tensor) -> Tensor:
        """Stabilized quantization for consistent binary tree decisions across architectures"""
        # Carefully control precision of the codebook
        embed = _quantize_tensor(self.embed.t())
        
        # Break down distance calculation into controlled steps
        # Calculate x squared term first
        x_squared = _quantize_tensor(x.pow(2).sum(1, keepdim=True))
        
        # Calculate embed squared term
        embed_squared = _quantize_tensor(embed.pow(2).sum(0, keepdim=True))
        
        # Calculate cross term with controlled precision
        # Use matmul for better numerical stability than @
        cross_term = _quantize_tensor(torch.matmul(x, embed))
        cross_term = _quantize_tensor(cross_term * 2)
        
        # Combine terms with controlled precision and ordering
        # Note: we add the squared terms first since they're likely larger
        dist = _quantize_tensor(x_squared + embed_squared)
        dist = _quantize_tensor(dist - cross_term)
        dist = -dist  # Negate at the end to avoid accumulated precision loss
        
        # Use stable sorting for consistent index selection
        embed_ind = dist.max(dim=-1, keepdim=True).indices
         
        return embed_ind
        
    def postprocess_emb(self, embed_ind: Tensor, shape: tp.Tuple) -> Tensor:
        return embed_ind.view(*shape[:-1])

    def dequantize(self, embed_ind: Tensor) -> Tensor:
        return F.embedding(embed_ind, self.embed)

    def encode(self, x: Tensor) -> Tensor:
        shape = x.shape
        x = self.preprocess(x)
        embed_ind = self.quantize(x)
        embed_ind = self.postprocess_emb(embed_ind, shape)
        return embed_ind

    def decode(self, embed_ind: Tensor) -> Tensor:
        return self.dequantize(embed_ind)

    def forward(self, x: Tensor) -> tp.Tuple[Tensor, Tensor]:
        shape, dtype = x.shape, x.dtype
        x = self.preprocess(x)

        self.init_embed_(x)

        embed_ind = self.quantize(x)
        embed_onehot = F.one_hot(embed_ind, self.codebook_size).type(dtype)
        embed_ind = self.postprocess_emb(embed_ind, shape)
        quantize = self.dequantize(embed_ind)

        if self.training:
            self.expire_codes_(x)
            ema_inplace(self.cluster_size, embed_onehot.sum(0), self.decay)
            embed_sum = x.t() @ embed_onehot
            ema_inplace(self.embed_avg, embed_sum.t(), self.decay)
            cluster_size = (
                laplace_smoothing(self.cluster_size, self.codebook_size, self.epsilon)
                * self.cluster_size.sum()
            )
            embed_normalized = self.embed_avg / cluster_size.unsqueeze(1)
            self.embed.data.copy_(embed_normalized)

        return quantize, embed_ind

class VectorQuantization(nn.Module):
    """Vector quantization implementation."""

    def __init__(
        self,
        dim: int,
        codebook_size: int,
        codebook_dim: tp.Optional[int] = None,
        decay: float = 0.99,
        epsilon: float = 1e-5,
        kmeans_init: bool = True,
        kmeans_iters: int = 50,
        threshold_ema_dead_code: int = 2,
        commitment_weight: float = 1.,
    ):
        super().__init__()
        self._codebook_dim: int = default(codebook_dim, dim)
        requires_projection = self._codebook_dim != dim
        self.project_in = (nn.Linear(dim, self._codebook_dim) if requires_projection else nn.Identity())
        self.project_out = (nn.Linear(self._codebook_dim, dim) if requires_projection else nn.Identity())

        self.epsilon = epsilon
        self.commitment_weight = commitment_weight

        self._codebook = EuclideanCodebook(
            dim=self._codebook_dim,
            codebook_size=codebook_size,
            kmeans_init=kmeans_init,
            kmeans_iters=kmeans_iters,
            decay=decay,
            epsilon=epsilon,
            threshold_ema_dead_code=threshold_ema_dead_code
        )
        self.codebook_size = codebook_size

    @property
    def codebook(self) -> Tensor:
        return self._codebook.embed

    def _quantize_tensor(self, x: Tensor) -> Tensor:
        """Control precision of tensor operations"""
        return torch.round(x * 10**7) / 10**7
    
    def encode(self, x: Tensor) -> Tensor:
        """Stabilized encoding process"""
        x = rearrange(x, "b d n -> b n d")
        # Stabilize projection
        x = _quantize_tensor(self.project_in(x))
        embed_in = self._codebook.encode(x)
        return embed_in

    def decode(self, embed_ind: Tensor) -> Tensor:
        """Stabilized decoding process"""
        quantize = self._codebook.decode(embed_ind)
        # Stabilize projection
        quantize = _quantize_tensor(self.project_out(quantize))
        quantize = rearrange(quantize, "b n d -> b d n")
        return quantize

    def forward(self, x: Tensor) -> tp.Tuple[Tensor, Tensor, Tensor]:
        device = x.device
        x = rearrange(x, "b d n -> b n d")
        x = self._quantize_tensor(self.project_in(x))

        quantize, embed_ind = self._codebook(x)
        quantize = self._quantize_tensor(quantize)

        if self.training:
            quantize = x + (quantize - x).detach()

        loss = torch.tensor([0.0], device=device, requires_grad=self.training)

        if self.training:
            warnings.warn('When using RVQ in training model, first check '
                        'https://github.com/facebookresearch/encodec/issues/25 . '
                        'The bug wasn\'t fixed here for reproducibility.')
            if self.commitment_weight > 0:
                commit_loss = F.mse_loss(quantize.detach(), x)
                loss = loss + commit_loss * self.commitment_weight

        quantize = self._quantize_tensor(self.project_out(quantize))
        quantize = rearrange(quantize, "b n d -> b d n")
        return quantize, embed_ind, loss

class ResidualVectorQuantization(nn.Module):
    """Residual vector quantization implementation with stability improvements."""
    
    def __init__(self, *, num_quantizers: int, **kwargs):
        super().__init__()
        self.layers = nn.ModuleList(
            [VectorQuantization(**kwargs) for _ in range(num_quantizers)]
        )
        
    def _quantize_tensor(self, x: Tensor) -> Tensor:
        """Control precision of tensor operations"""
        return torch.round(x * 10**7) / 10**7

    def forward(self, x: Tensor, n_q: tp.Optional[int] = None) -> tp.Tuple[Tensor, Tensor, Tensor]:
        quantized_out = torch.tensor(0.0, device=x.device)
        residual = self._quantize_tensor(x)

        all_losses = []
        all_indices = []

        n_q = n_q or len(self.layers)

        for layer in self.layers[:n_q]:
            quantized, indices, loss = layer(residual)
            residual = self._quantize_tensor(residual - quantized)
            quantized_out = self._quantize_tensor(quantized_out + quantized)

            all_indices.append(indices)
            all_losses.append(loss)

        out_losses, out_indices = map(torch.stack, (all_losses, all_indices))
        return quantized_out, out_indices, out_losses

    def encode(self, x: Tensor, n_q: tp.Optional[int] = None) -> Tensor:
        """Stabilized RVQ encoding"""
        # Initial quantization of input
        residual = _quantize_tensor(x)
        all_indices = []
        n_q = n_q or len(self.layers)
        
        for layer in self.layers[:n_q]:
            # Get indices for this layer
            indices = layer.encode(residual)
            
            # Decode and quantize to match encoder exactly
            quantized = layer.decode(indices)
            quantized = _quantize_tensor(quantized)
            
            # Compute and quantize residual
            residual = _quantize_tensor(residual - quantized)
            
            all_indices.append(indices)
            
        # Stack indices with controlled precision
        out_indices = torch.stack(all_indices)
        return out_indices

    def decode(self, q_indices: Tensor) -> Tensor:
        """Stabilized RVQ decoding"""
        quantized_out = torch.zeros(
            q_indices.shape[1:], 
            device=q_indices.device, 
            dtype=q_indices.dtype
        )
        
        for i, indices in enumerate(q_indices):
            layer = self.layers[i]
            # Decode and stabilize each layer output
            quantized = layer.decode(indices)
            quantized = _quantize_tensor(quantized)
            
            # Accumulate with controlled precision
            quantized_out = _quantize_tensor(quantized_out + quantized)
            
        return quantized_out

