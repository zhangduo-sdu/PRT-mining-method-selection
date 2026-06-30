"""Physics-regularized transformer model for tabular geomechanical decision data."""

from __future__ import annotations

from typing import Dict, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from attention_mask import build_feature_types


class TypeAwareEmbedding(nn.Module):
    """Embed each scalar feature using type-specific linear projections."""

    def __init__(self, feature_order: List[str], d_model: int):
        super().__init__()
        self.feature_order = feature_order
        self.feature_types = build_feature_types(feature_order)
        self.d_model = d_model

        self.type_layers = nn.ModuleDict(
            {
                "quantitative": nn.Linear(1, d_model),
                "qualitative": nn.Linear(1, d_model),
                "simulation": nn.Linear(1, d_model),
            }
        )
        self.feature_embedding = nn.Embedding(len(feature_order), d_model)
        self.layer_norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Convert [batch, n_features] into [batch, n_features, d_model]."""
        tokens = []
        for j, name in enumerate(self.feature_order):
            ftype = self.feature_types[name]
            scalar = x[:, j : j + 1]
            token = self.type_layers[ftype](scalar)
            tokens.append(token)

        out = torch.stack(tokens, dim=1)
        pos_ids = torch.arange(len(self.feature_order), device=x.device)
        out = out + self.feature_embedding(pos_ids).unsqueeze(0)
        out = self.layer_norm(out)
        return out


class MaskedMultiHeadSelfAttention(nn.Module):
    """Manual multi-head self-attention with a shared physics mask."""

    def __init__(self, d_model: int, n_heads: int, dropout: float):
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError("d_model must be divisible by n_heads.")

        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads

        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        attn_block_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Apply masked self-attention.

        Parameters
        ----------
        x:
            Token tensor, shape [batch, n_tokens, d_model].
        attn_block_mask:
            Boolean mask, shape [n_tokens, n_tokens]. True means blocked.
        """
        bsz, n_tokens, _ = x.shape

        q = self.q_proj(x).view(bsz, n_tokens, self.n_heads, self.d_head).transpose(1, 2)
        k = self.k_proj(x).view(bsz, n_tokens, self.n_heads, self.d_head).transpose(1, 2)
        v = self.v_proj(x).view(bsz, n_tokens, self.n_heads, self.d_head).transpose(1, 2)

        scores = torch.matmul(q, k.transpose(-2, -1)) / (self.d_head ** 0.5)

        # True entries are not allowed.
        mask = attn_block_mask.unsqueeze(0).unsqueeze(0)
        scores = scores.masked_fill(mask, -1.0e9)

        attn = torch.softmax(scores, dim=-1)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).contiguous().view(bsz, n_tokens, self.d_model)
        out = self.out_proj(out)

        return out, attn


class TransformerBlock(nn.Module):
    """One transformer block with masked self-attention and FFN."""

    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float):
        super().__init__()
        self.attn = MaskedMultiHeadSelfAttention(d_model, n_heads, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
        )

    def forward(self, x: torch.Tensor, attn_block_mask: torch.Tensor):
        attn_out, attn = self.attn(x, attn_block_mask)
        x = self.norm1(x + self.dropout(attn_out))
        ffn_out = self.ffn(x)
        x = self.norm2(x + self.dropout(ffn_out))
        return x, attn


class PhysicsRegularizedTransformer(nn.Module):
    """Physics-regularized transformer for superiority-score prediction."""

    def __init__(
        self,
        feature_order: List[str],
        d_model: int = 64,
        n_heads: int = 4,
        d_ff: int = 128,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.feature_order = feature_order
        self.embedding = TypeAwareEmbedding(feature_order, d_model)
        self.block = TransformerBlock(d_model, n_heads, d_ff, dropout)
        self.predictor = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, 1),
        )

    def forward(self, x: torch.Tensor, attn_block_mask: torch.Tensor):
        tokens = self.embedding(x)
        encoded, attn = self.block(tokens, attn_block_mask)
        pooled = encoded.mean(dim=1)
        score = torch.sigmoid(self.predictor(pooled)).squeeze(-1)
        return score, attn
