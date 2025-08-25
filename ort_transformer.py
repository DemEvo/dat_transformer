# ort_transformer.py
# Encoder-only Transformer with decoupled head dim and head-orthogonality regularization.
#
# Features:
# - No adaptive depth/width/attention. Plain Transformer encoder blocks.
# - Decoupled head dimension: d_head is independent from d_model / n_heads.
# - Optional orthogonality/diversity regularization across heads (Q/K/V projections).
# - Additive attention mask (broadcastable to [B, H, T, T]) with large negatives on masked keys.
#
# Example usage:
#   cfg = EncoderConfig(n_layers=6, d_model=256, n_heads=4, d_head=64, d_ff=1024,
#                       ortho_q=1e-4, ortho_k=1e-4, ortho_v=0.0)
#   enc = TransformerEncoder(cfg)
#   clf = SequenceClassifier(enc, vocab_size=100, d_model=cfg.d_model, n_classes=2)
#   logits, aux = clf(tokens, return_aux=True)
#   loss = CE(logits, labels) + aux['ortho_loss']
#
from dataclasses import dataclass
from typing import Optional, Dict, Any, Tuple
import math
import torch
from torch import nn
import torch.nn.functional as F

# -----------------------------
# Configs
# -----------------------------

@dataclass
class LayerConfig:
    d_model: int
    n_heads: int
    d_head: int                 # decoupled head dimension
    d_ff: int
    attn_dropout: float = 0.0
    resid_dropout: float = 0.0
    norm_eps: float = 1e-5
    # Orthogonality penalties (L2 on inter-head overlap); set to 0.0 to disable
    ortho_q: float = 0.0
    ortho_k: float = 0.0
    ortho_v: float = 0.0

@dataclass
class EncoderConfig:
    n_layers: int
    d_model: int
    n_heads: int
    d_head: int
    d_ff: int
    attn_dropout: float = 0.0
    resid_dropout: float = 0.0
    norm_eps: float = 1e-5
    ortho_q: float = 0.0
    ortho_k: float = 0.0
    ortho_v: float = 0.0

    def layer(self) -> "LayerConfig":
        return LayerConfig(
            d_model=self.d_model,
            n_heads=self.n_heads,
            d_head=self.d_head,
            d_ff=self.d_ff,
            attn_dropout=self.attn_dropout,
            resid_dropout=self.resid_dropout,
            norm_eps=self.norm_eps,
            ortho_q=self.ortho_q,
            ortho_k=self.ortho_k,
            ortho_v=self.ortho_v,
        )

# -----------------------------
# Orthogonality regularizer
# -----------------------------

def _orthogonal_heads_loss(linear: nn.Linear, n_heads: int, d_head: int, eps: float = 1e-8) -> torch.Tensor:
    """
    Scale-invariant diversity penalty between head subspaces for a projection Linear.

    Split weight matrix W: [out = H*Dh, in = D] into H blocks [Dh, D]. Flatten each block to v_h and L2-normalize;
    encourage off-diagonal Gram entries to be small:
        L = mean_{h!=h'} <v_h, v_{h'}>^2
    """
    W = linear.weight  # [H*Dh, D]
    H = n_heads
    assert W.shape[0] == H * d_head, f"Linear out_features={W.shape[0]} must equal n_heads*d_head"
    V = W.view(H, d_head, -1).reshape(H, -1)            # [H, Dh*D]
    V = V / (V.norm(dim=1, keepdim=True) + eps)         # scale-invariant
    G = V @ V.t()                                       # [H, H]
    off = G - torch.diag_embed(torch.diag(G))
    return (off.pow(2).sum() / (H * (H - 1) + eps))

# -----------------------------
# Core modules
# -----------------------------

class MultiHeadSelfAttention(nn.Module):
    """MHA with decoupled head_dim + optional head-orthogonality regularization."""
    def __init__(self, cfg: LayerConfig):
        super().__init__()
        self.cfg = cfg
        H, D, Dh = cfg.n_heads, cfg.d_model, cfg.d_head
        self.q_proj = nn.Linear(D, H * Dh, bias=False)
        self.k_proj = nn.Linear(D, H * Dh, bias=False)
        self.v_proj = nn.Linear(D, H * Dh, bias=False)
        self.out_proj = nn.Linear(H * Dh, D, bias=False)
        self.attn_drop = nn.Dropout(cfg.attn_dropout)

    def forward(self, x: torch.Tensor, attn_mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, Dict[str, Any]]:
        B, T, D = x.shape
        H, Dh = self.cfg.n_heads, self.cfg.d_head
        q = self.q_proj(x).view(B, T, H, Dh).transpose(1, 2)   # [B,H,T,Dh]
        k = self.k_proj(x).view(B, T, H, Dh).transpose(1, 2)   # [B,H,T,Dh]
        v = self.v_proj(x).view(B, T, H, Dh).transpose(1, 2)   # [B,H,T,Dh]

        # Память-эффективное внимание (без явного T×T)
        y = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=attn_mask,  # broadcastable к [B,H,T,T] или [B,1,1,T]
            dropout_p=self.cfg.attn_dropout if self.training else 0.0,
            is_causal=False,
        )  # [B,H,T,Dh]

        y = y.transpose(1, 2).contiguous().view(B, T, H * Dh)   # [B,T,H*Dh]
        y = self.out_proj(y)                                    # [B,T,D]
        # return y, {"attn_probs": attn}
        # Чтобы не держать T×T в aux, по умолчанию НЕ возвращаем карты внимания:
        return y, {}

    def orthogonality_loss(self) -> torch.Tensor:
        loss = 0.0
        if self.cfg.ortho_q > 0:
            loss = loss + self.cfg.ortho_q * _orthogonal_heads_loss(self.q_proj, self.cfg.n_heads, self.cfg.d_head)
        if self.cfg.ortho_k > 0:
            loss = loss + self.cfg.ortho_k * _orthogonal_heads_loss(self.k_proj, self.cfg.n_heads, self.cfg.d_head)
        if self.cfg.ortho_v > 0:
            loss = loss + self.cfg.ortho_v * _orthogonal_heads_loss(self.v_proj, self.cfg.n_heads, self.cfg.d_head)
        return torch.as_tensor(loss, device=self.q_proj.weight.device, dtype=self.q_proj.weight.dtype)

class FeedForward(nn.Module):
    def __init__(self, d_model: int, d_ff: int, resid_dropout: float = 0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Linear(d_ff, d_model),
            nn.Dropout(resid_dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

class EncoderLayer(nn.Module):
    def __init__(self, cfg: LayerConfig):
        super().__init__()
        self.cfg = cfg
        self.ln1 = nn.LayerNorm(cfg.d_model, eps=cfg.norm_eps)
        self.ln2 = nn.LayerNorm(cfg.d_model, eps=cfg.norm_eps)
        self.mha = MultiHeadSelfAttention(cfg)
        self.ffn = FeedForward(cfg.d_model, cfg.d_ff, resid_dropout=cfg.resid_dropout)
        self.drop = nn.Dropout(cfg.resid_dropout)

    def forward(self, x: torch.Tensor, attn_mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, Dict[str, Any]]:
        h = self.ln1(x)
        sa, aux_sa = self.mha(h, attn_mask=attn_mask)
        x = x + self.drop(sa)
        h2 = self.ln2(x)
        ff = self.ffn(h2)
        x = x + self.drop(ff)
        ortho = self.mha.orthogonality_loss()
        return x, {"attn_probs": aux_sa.get("attn_probs"), "ortho_loss": ortho}

class TransformerEncoder(nn.Module):
    def __init__(self, cfg: EncoderConfig):
        super().__init__()
        self.cfg = cfg
        self.layers = nn.ModuleList([EncoderLayer(cfg.layer()) for _ in range(cfg.n_layers)])
        self.final_ln = nn.LayerNorm(cfg.d_model, eps=cfg.norm_eps)

    def forward(self, x: torch.Tensor, attn_mask: Optional[torch.Tensor] = None, return_aux: bool = False):
        ortho_total = torch.zeros([], device=x.device, dtype=x.dtype)
        last_attn = None
        for layer in self.layers:
            x, aux = layer(x, attn_mask=attn_mask)
            last_attn = aux.get("attn_probs")
            ortho_total = ortho_total + aux.get("ortho_loss", torch.zeros([], device=x.device, dtype=x.dtype))
        x = self.final_ln(x)
        if return_aux:
            return x, {"attn_probs": last_attn, "ortho_loss": ortho_total}
        return x

# -----------------------------
# Simple sequence classifier (CLS at position 0)
# -----------------------------

class SequenceClassifier(nn.Module):
    def __init__(self, encoder: TransformerEncoder, vocab_size: int, d_model: int, n_classes: int, pad_token: int = 0):
        super().__init__()
        self.encoder = encoder
        self.pad_token = pad_token
        self.embed = nn.Embedding(vocab_size, d_model, padding_idx=pad_token)
        self.head = nn.Linear(d_model, n_classes)

    def forward(self, tokens: torch.Tensor, return_aux: bool = False):
        x = self.embed(tokens)  # [B,T,D]
        padding = (tokens == self.pad_token)  # [B,T]
        if padding.any():
            # broadcastable additive mask to [B, H, Tq, Tk]
            mask = padding.unsqueeze(1).unsqueeze(2)  # [B,1,1,T]
            attn_mask = mask * (-1e9)
        else:
            attn_mask = None
        feats, aux = self.encoder(x, attn_mask=attn_mask, return_aux=True)
        cls = feats[:, 0, :]
        logits = self.head(cls)
        return (logits, aux) if return_aux else logits
