"""RADAR policy adapted to the MARS CVRPState interface.

RADAR (Learning to Route with Asymmetry-aware Distance Representations)
uses two key innovations:
  1. SVD-lowrank embeddings of the (normalized) pairwise distance matrix
     to initialise node representations with structural position information.
  2. Sinkhorn-normalised Mixed-Score Multi-Head Attention in the encoder,
     replacing softmax attention to model asymmetric distance relationships.

In the MARS Euclidean-CVRP setting the distance matrix is symmetric, so
Sinkhorn reduces to a doubly-stochastic softmax — but the architecture is
preserved faithfully.

Default hyper-parameters (embed_dim=64, encoder_layers=6, num_heads=8,
ff_dim=256, ms_hidden_dim=16, svd_rank=10) give ~325 k trainable parameters,
matching the MARS AttentionModel policy backbone at embed_dim=64.

Interface:
    prepare(state)  — encode all nodes once per problem batch
    logits(state)   — return pre-softmax logit scores (B, pomo, n+1)
                      compatible with the shared rollout() in baseline_cvrp.py
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


# ──────────────────────────────────────────────────────────────
# Sinkhorn normalisation
# ──────────────────────────────────────────────────────────────

def _sinkhorn(scores: torch.Tensor, n_iter: int = 10) -> torch.Tensor:
    """Doubly-stochastic normalisation via Sinkhorn iterations (log-space)."""
    scores = scores - scores.amax(dim=-1, keepdim=True)
    for _ in range(n_iter):
        scores = scores - scores.logsumexp(dim=-1, keepdim=True)
        scores = scores - scores.logsumexp(dim=-2, keepdim=True)
    return scores.exp()


# ──────────────────────────────────────────────────────────────
# Shared building blocks
# ──────────────────────────────────────────────────────────────

class _AddAndInstanceNorm(nn.Module):
    """Residual add + Instance Normalisation (RADAR uses InstanceNorm, not LayerNorm)."""

    def __init__(self, dim: int):
        super().__init__()
        self.norm = nn.InstanceNorm1d(dim, affine=True, track_running_stats=False)

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        added = x1 + x2                                   # (B, n, dim)
        return self.norm(added.transpose(1, 2)).transpose(1, 2)


class _FeedForward(nn.Module):
    def __init__(self, dim: int, ff_dim: int):
        super().__init__()
        self.w1 = nn.Linear(dim, ff_dim)
        self.w2 = nn.Linear(ff_dim, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w2(F.relu(self.w1(x)))


def _reshape_by_heads(x: torch.Tensor, h: int) -> torch.Tensor:
    """(B, n, H*qkv) → (B, H, n, qkv)."""
    B, n, _ = x.shape
    return x.reshape(B, n, h, -1).transpose(1, 2)


# ──────────────────────────────────────────────────────────────
# Mixed-Score MHA with Sinkhorn normalisation
# ──────────────────────────────────────────────────────────────

class _MixedScoreMHA(nn.Module):
    """Mixed-Score Multi-Head Attention with Sinkhorn normalisation.

    Combines the standard QK dot-product score with cost-matrix scores
    via a small per-head MLP, then normalises with Sinkhorn instead of
    softmax.  Parameters are initialised from uniform distributions as
    in the original RADAR code.
    """

    def __init__(self, head_num: int, qkv_dim: int, ms_hidden: int,
                 sinkhorn_iters: int = 10):
        super().__init__()
        self.head_num = head_num
        self.sqrt_qkv = math.sqrt(qkv_dim)
        self.sinkhorn_iters = sinkhorn_iters
        mix1_init = math.sqrt(0.5)
        mix2_init = math.sqrt(1.0 / 16.0)
        self.mix1_weight = nn.Parameter(
            torch.empty(head_num, 3, ms_hidden).uniform_(-mix1_init, mix1_init)
        )
        self.mix1_bias = nn.Parameter(
            torch.empty(head_num, ms_hidden).uniform_(-mix1_init, mix1_init)
        )
        self.mix2_weight = nn.Parameter(
            torch.empty(head_num, ms_hidden, 1).uniform_(-mix2_init, mix2_init)
        )
        self.mix2_bias = nn.Parameter(
            torch.empty(head_num, 1).uniform_(-mix2_init, mix2_init)
        )

    def forward(
        self,
        q: torch.Tensor,        # (B, H, n, qkv)
        k: torch.Tensor,        # (B, H, n, qkv)
        v: torch.Tensor,        # (B, H, n, qkv)
        cost_mat: torch.Tensor, # (B, n, n)
    ) -> torch.Tensor:
        B, H, n, qkv = q.shape
        dot = torch.matmul(q, k.transpose(2, 3)) / self.sqrt_qkv  # (B,H,n,n)
        c   = cost_mat[:, None].expand(B, H, n, n)
        # Three-channel input: [dot-product, cost, cost^T]
        two_scores = torch.stack([dot, c, c.transpose(2, 3)], dim=4)  # (B,H,n,n,3)
        ts = two_scores.transpose(1, 2)                                 # (B,n,H,n,3)
        ms1 = torch.matmul(ts, self.mix1_weight)                        # (B,n,H,n,ms)
        ms1 = ms1 + self.mix1_bias[None, None, :, None, :]
        ms1 = F.relu(ms1)
        ms2 = torch.matmul(ms1, self.mix2_weight)                       # (B,n,H,n,1)
        ms2 = ms2 + self.mix2_bias[None, None, :, None, :]
        mixed = ms2.squeeze(-1).transpose(1, 2)                         # (B,H,n,n)
        weights = _sinkhorn(mixed, self.sinkhorn_iters)                 # (B,H,n,n)
        out = torch.matmul(weights, v)                                  # (B,H,n,qkv)
        return out.transpose(1, 2).reshape(B, n, H * qkv)


# ──────────────────────────────────────────────────────────────
# Encoder
# ──────────────────────────────────────────────────────────────

class _EncoderLayer(nn.Module):
    def __init__(self, dim: int, head_num: int, qkv_dim: int,
                 ff_dim: int, ms_hidden: int, sinkhorn_iters: int):
        super().__init__()
        self.head_num = head_num
        self.Wq = nn.Linear(dim, head_num * qkv_dim, bias=True)
        self.Wk = nn.Linear(dim, head_num * qkv_dim, bias=True)
        self.Wv = nn.Linear(dim, head_num * qkv_dim, bias=False)
        self.ms_mha = _MixedScoreMHA(head_num, qkv_dim, ms_hidden, sinkhorn_iters)
        self.combine = nn.Linear(head_num * qkv_dim, dim)
        self.norm1 = _AddAndInstanceNorm(dim)
        self.ff    = _FeedForward(dim, ff_dim)
        self.norm2 = _AddAndInstanceNorm(dim)

    def forward(self, x: torch.Tensor, cost_mat: torch.Tensor) -> torch.Tensor:
        q = _reshape_by_heads(self.Wq(x), self.head_num)
        k = _reshape_by_heads(self.Wk(x), self.head_num)
        v = _reshape_by_heads(self.Wv(x), self.head_num)
        attn_out = self.ms_mha(q, k, v, cost_mat)
        x = self.norm1(x, self.combine(attn_out))
        x = self.norm2(x, self.ff(x))
        return x


class _RADAREncoder(nn.Module):
    def __init__(self, dim: int, head_num: int, qkv_dim: int,
                 ff_dim: int, ms_hidden: int, num_layers: int, sinkhorn_iters: int):
        super().__init__()
        self.layers = nn.ModuleList([
            _EncoderLayer(dim, head_num, qkv_dim, ff_dim, ms_hidden, sinkhorn_iters)
            for _ in range(num_layers)
        ])

    def forward(self, x: torch.Tensor, cost_mat: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x, cost_mat)
        return x


# ──────────────────────────────────────────────────────────────
# Decoder  (returns pre-softmax logits, NOT probs)
# ──────────────────────────────────────────────────────────────

class _RADARDecoder(nn.Module):
    """POMO-style decoder returning raw logit scores before masking.

    Compatible with the MARS ``rollout()`` interface which applies the
    mask externally and then calls Categorical / argmax on the logits.
    """

    def __init__(self, dim: int, head_num: int, qkv_dim: int, logit_clipping: float):
        super().__init__()
        self.head_num = head_num
        self.sqrt_qkv = math.sqrt(qkv_dim)
        self.sqrt_dim  = math.sqrt(dim)
        self.logit_clipping = logit_clipping
        # +1 dim for vehicle load
        self.Wq0 = nn.Linear(dim + 1, head_num * qkv_dim, bias=False)
        self.Wk  = nn.Linear(dim, head_num * qkv_dim, bias=False)
        self.Wv  = nn.Linear(dim, head_num * qkv_dim, bias=False)
        self.combine = nn.Linear(head_num * qkv_dim, dim)
        # Cached keys/values set by set_kv()
        self._k:   torch.Tensor | None = None
        self._v:   torch.Tensor | None = None
        self._shk: torch.Tensor | None = None  # single-head key (B, dim, n+1)

    def set_kv(self, encoded: torch.Tensor) -> None:
        """Cache keys & values from the encoded node embeddings."""
        self._k   = _reshape_by_heads(self.Wk(encoded), self.head_num)
        self._v   = _reshape_by_heads(self.Wv(encoded), self.head_num)
        self._shk = encoded.transpose(1, 2)   # (B, dim, n+1)

    def forward(
        self,
        cur_emb: torch.Tensor,  # (B, pomo, dim) — current node embedding
        load:    torch.Tensor,  # (B, pomo)      — remaining vehicle capacity
        mask:    torch.Tensor,  # (B, pomo, n+1) — True means unavailable
    ) -> torch.Tensor:
        """Return raw logit scores (B, pomo, n+1) before masking."""
        B, pomo, dim = cur_emb.shape
        H = self.head_num
        # Query: concat current embedding + load scalar
        q_in = torch.cat([cur_emb, load[:, :, None]], dim=2)  # (B, pomo, dim+1)
        q = _reshape_by_heads(self.Wq0(q_in), H)              # (B, H, pomo, qkv)
        # Match the original RADAR decoder: unavailable nodes are excluded
        # from both the multi-head context and the final action distribution.
        score = torch.matmul(q, self._k.transpose(2, 3)) / self.sqrt_qkv
        score = score.masked_fill(mask[:, None], float("-inf"))
        weights = torch.softmax(score, dim=-1)                 # (B, H, pomo, n+1)
        ctx = torch.matmul(weights, self._v)                   # (B, H, pomo, qkv)
        ctx = ctx.transpose(1, 2).reshape(B, pomo, H * (dim // H))
        ctx = self.combine(ctx)                                # (B, pomo, dim)
        # Single-head attention logits
        logits = torch.matmul(ctx, self._shk) / self.sqrt_dim # (B, pomo, n+1)
        return self.logit_clipping * torch.tanh(logits)


# ──────────────────────────────────────────────────────────────
# Full policy  (prepare / logits interface for MARS rollout)
# ──────────────────────────────────────────────────────────────

class RADARPolicy(nn.Module):
    """RADAR policy adapted to the CVRPState / MARS rollout() interface.

    ``prepare(state)`` encodes all nodes once using SVD embeddings +
    Sinkhorn-MHA encoder.  ``logits(state)`` decodes one step and returns
    **raw pre-softmax logit scores** (B, pomo, n+1) compatible with the
    shared ``rollout()`` helper in ``baseline_cvrp.py``.

    Args:
        embed_dim: Node embedding dimension.  Default 64 (≈325 k params,
            matching MARS AM policy at embed_dim=64).
        encoder_layers: Number of Sinkhorn-MHA encoder layers.  Default 6.
        num_heads: Attention heads (must divide embed_dim).  Default 8.
        ff_dim: Feed-forward hidden dimension.  Default 256.
        ms_hidden_dim: Hidden size of per-head MLP in MixedScoreMHA.  Default 16.
        svd_rank: Rank k for ``torch.svd_lowrank``.  Initial embedding size
            is ``2k + 1`` (Q-factor, K-factor, normalised demand).  Default 10.
        logit_clipping: Tanh clipping coefficient C.  Default 10.0.
        sinkhorn_iters: Sinkhorn iterations.  Default 10.
    """

    def __init__(
        self,
        embed_dim:      int   = 64,
        encoder_layers: int   = 6,
        num_heads:      int   = 8,
        ff_dim:         int   = 256,
        ms_hidden_dim:  int   = 16,
        svd_rank:       int   = 10,
        logit_clipping: float = 10.0,
        sinkhorn_iters: int   = 10,
    ):
        super().__init__()
        assert embed_dim % num_heads == 0, (
            f"embed_dim ({embed_dim}) must be divisible by num_heads ({num_heads})"
        )
        qkv_dim = embed_dim // num_heads
        self.svd_rank = svd_rank
        # SVD features (2k) + normalised demand (1) → embed_dim
        self.projection = nn.Linear(2 * svd_rank + 1, embed_dim)
        self.encoder = _RADAREncoder(
            embed_dim, num_heads, qkv_dim, ff_dim, ms_hidden_dim,
            encoder_layers, sinkhorn_iters,
        )
        self.decoder = _RADARDecoder(embed_dim, num_heads, qkv_dim, logit_clipping)
        self._encoded: torch.Tensor | None = None

    # ── Encoding ─────────────────────────────────────────────────

    def prepare(self, state) -> None:
        """Encode all nodes.  Called once per problem batch by rollout().

        Builds the Euclidean pairwise distance matrix from ``state.xy``,
        normalises it per instance, runs SVD-lowrank to obtain initial
        node features, appends the normalised demand, then applies the
        Sinkhorn-MHA encoder.
        """
        xy     = state.xy.float()      # (B, n+1, 2)   depot at index 0
        demand = state.demand.float()  # (B, n+1)       depot demand == 0

        # ── Pairwise Euclidean distance matrix ───────────────────
        dist = torch.cdist(xy, xy)     # (B, n+1, n+1)

        # ── Per-instance z-score normalisation ───────────────────
        mean = dist.mean(dim=(1, 2), keepdim=True)
        std  = dist.std(dim=(1, 2), keepdim=True).clamp_min(1e-9)
        dist_norm = (dist - mean) / std

        # ── SVD-lowrank embedding ─────────────────────────────────
        # Q ≈ U * sqrt(S),  K ≈ V * sqrt(S)  →  X = [Q | K]
        U, S, V = torch.svd_lowrank(dist_norm, q=self.svd_rank)
        sqrt_S   = S.sqrt().unsqueeze(1)           # (B, 1, k)
        Q        = U * sqrt_S                      # (B, n+1, k)
        K        = V * sqrt_S                      # (B, n+1, k)
        svd_feat = torch.cat([Q, K], dim=-1)       # (B, n+1, 2k)

        # ── Normalised demand feature ─────────────────────────────
        total_dem  = demand.sum(dim=1, keepdim=True).clamp_min(1e-6)
        demand_feat = (demand / total_dem).unsqueeze(-1)  # (B, n+1, 1)

        # ── Project and encode ────────────────────────────────────
        x = torch.cat([svd_feat, demand_feat], dim=-1)  # (B, n+1, 2k+1)
        x = self.projection(x)                          # (B, n+1, d)
        self._encoded = self.encoder(x, dist_norm)      # (B, n+1, d)
        self.decoder.set_kv(self._encoded)

    # ── Decoding ─────────────────────────────────────────────────

    def logits(self, state) -> torch.Tensor:
        """Return pre-softmax logit scores (B, pomo, n+1).

        The decoder uses the feasibility mask while building its attention
        context. The shared rollout applies the same mask once more to the
        returned action logits before sampling.
        """
        B, pomo = state.current.shape
        enc = self._encoded                           # (B, n+1, d)
        cur_emb = enc.gather(
            1, state.current[:, :, None].expand(B, pomo, enc.size(-1))
        )                                             # (B, pomo, d)
        return self.decoder(cur_emb, state.load, state.mask())  # (B, pomo, n+1)
