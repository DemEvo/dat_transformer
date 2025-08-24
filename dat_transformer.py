"""
DAT: Dynamic Adaptive Transformer
---------------------------------

This module implements three improvements outlined in the DAT note:
  1) Adaptive Depth (per‑token halting)
  2) Adaptive Width (per‑head gating in MHA)
  3) Memory‑Augmented + Parametric (non‑linear) Attention

PyTorch implementation with clean, modular components you can mix & match.

Key classes
-----------
- GatingMLP: generic sigmoid gate (optionally 2‑layer)
- ParametricScoreMLP: MLP scoring f([q;k]) -> scalar for attention
- MemoryBank: simple external K/V store with top‑k retrieval (+ optional writes)
- MemoryAugmentedAttention: attention over (context + retrieved memory)
- AdaptiveWidthMultiheadAttention: MHA with per‑head gates (trainable)
- AdaptiveLayer: Transformer layer with Add&Norm + FFN + halting gate
- DynamicEncoder: stack that fuses layers with probabilistic halting weights

Notes
-----
- Parametric attention is much heavier than dot‑product. A chunked path is provided
  to control memory. For large sequences use `score_chunk_size`.
- During training we compute all heads/layers and use gates/halting only as weights
  (fully differentiable). At inference you may enable `prune_heads_threshold` and
  `early_exit` for actual compute savings.
- MemoryBank is a simple torch tensor store; plug your vector DB/Faiss backend by
  swapping `retrieve`/`write`.

Author: ChatGPT (PyTorch ≥ 1.12)
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Tuple, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F


# -----------------------------
# Utilities
# -----------------------------

def _shape_proj(x: torch.Tensor, n_heads: int, d_head: int) -> torch.Tensor:
    """[B, T, n_heads*d_head] -> [B, n_heads, T, d_head]"""
    B, T, _ = x.shape
    return x.view(B, T, n_heads, d_head).permute(0, 2, 1, 3).contiguous()


def _merge_heads(x: torch.Tensor) -> torch.Tensor:
    """[B, n_heads, T, d_head] -> [B, T, n_heads*d_head]"""
    B, H, T, Dh = x.shape
    return x.permute(0, 2, 1, 3).contiguous().view(B, T, H * Dh)


# -----------------------------
# Gating blocks
# -----------------------------
class GatingMLP(nn.Module):
    """Generic sigmoid gate. Returns values in [0, 1].

    If hidden_dim is provided, uses a 2-layer MLP with GELU; else a single Linear.
    """

    def __init__(self, d_in: int, d_out: int = 1, hidden_dim: Optional[int] = None):
        super().__init__()
        if hidden_dim is None:
            self.pre = None
            self.net = nn.Linear(d_in, d_out)           # <- всегда Linear
        else:
            self.pre = nn.Sequential(
                nn.Linear(d_in, hidden_dim),
                nn.GELU(),
            )
            self.net = nn.Linear(hidden_dim, d_out)     # <- финальный Linear c .weight

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = x if getattr(self, "pre", None) is None else self.pre(x)
        return torch.sigmoid(self.net(h))



class ParametricScoreMLP(nn.Module):
    """MLP scoring function f([q;k]) -> scalar, shared across heads.

    Supports chunked key processing to save memory at long sequence lengths.
    """

    def __init__(self, d_q: int, d_k: int, hidden: int = 128):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(d_q + d_k, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, 1),
        )

    def pairwise_scores(
        self,
        q: torch.Tensor,  # [B, H, Tq, Dq]
        k: torch.Tensor,  # [B, H, Tk, Dk]
        score_chunk_size: Optional[int] = None,
    ) -> torch.Tensor:  # returns [B, H, Tq, Tk]
        B, H, Tq, Dq = q.shape
        _, _, Tk, Dk = k.shape
        if score_chunk_size is None:
            # Build all concatenations at once (may be memory heavy)
            q_exp = q.unsqueeze(3).expand(B, H, Tq, Tk, Dq)
            k_exp = k.unsqueeze(2).expand(B, H, Tq, Tk, Dk)
            x = torch.cat([q_exp, k_exp], dim=-1)  # [B,H,Tq,Tk,Dq+Dk]
            s = self.mlp(x).squeeze(-1)  # [B,H,Tq,Tk]
            return s
        # Chunked along keys
        outs = []
        for start in range(0, Tk, score_chunk_size):
            end = min(start + score_chunk_size, Tk)
            k_slice = k[:, :, start:end, :]  # [B,H,C,Dk]
            q_exp = q.unsqueeze(3).expand(B, H, Tq, end - start, Dq)
            k_exp = k_slice.unsqueeze(2).expand(B, H, Tq, end - start, Dk)
            x = torch.cat([q_exp, k_exp], dim=-1)
            s = self.mlp(x).squeeze(-1)  # [B,H,Tq,C]
            outs.append(s)
        return torch.cat(outs, dim=-1)


# -----------------------------
# External Memory
# -----------------------------
class MemoryBank(nn.Module):
    """Simple in‑memory K/V store on torch tensors.

    Keys: [N, Dk], Values: [N, Dv]. Methods are differentiable wrt new writes,
    but retrieval is a top‑k gather (non‑diff indices; okay for inference/use).
    """

    def __init__(self, d_key: int, d_value: int, device: Optional[torch.device] = None):
        super().__init__()
        self.register_buffer("keys", torch.empty(0, d_key, device=device))
        self.register_buffer("values", torch.empty(0, d_value, device=device))

    @torch.no_grad()
    def write(self, k: torch.Tensor, v: torch.Tensor):
        """Append entries to memory. k:[M,Dk], v:[M,Dv]."""
        if self.keys.numel() == 0:
            self.keys = k.detach().clone()
            self.values = v.detach().clone()
        else:
            self.keys = torch.cat([self.keys, k.detach()], dim=0)
            self.values = torch.cat([self.values, v.detach()], dim=0)

    def size(self) -> int:
        return self.keys.shape[0]

    def retrieve(self, q: torch.Tensor, topk: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Retrieve top‑k memory rows per query.

        q: [B, T, Dk]
        returns:
          K_mem: [B, T, topk, Dk]
          V_mem: [B, T, topk, Dv]
        """
        if self.size() == 0 or topk <= 0:
            B, T, Dk = q.shape
            d_v = self.values.shape[1] if self.values.numel() else Dk
            device = q.device
            return (
                torch.zeros(B, T, 0, Dk, device=device),
                torch.zeros(B, T, 0, d_v, device=device),
            )
        # [B,T,N]
        scores = torch.einsum("btd,nd->btn", q, self.keys)
        idx = scores.topk(min(topk, self.size()), dim=-1).indices  # [B,T,K]
        # Gather keys/values
        K_mem = self.keys[idx]  # [B,T,K,Dk]
        V_mem = self.values[idx]  # [B,T,K,Dv]
        return K_mem, V_mem


# -----------------------------
# Attention blocks
# -----------------------------
@dataclass
class AttentionConfig:
    n_heads: int
    d_model: int
    d_head: int
    dropout_p: float = 0.0
    use_parametric_scores: bool = False
    score_hidden: int = 128
    score_chunk_size: Optional[int] = None      # ← добавили
    mem_topk: int = 0                            # ← добавили
    head_gate_hidden: Optional[int] = 64
    prune_heads_threshold: Optional[float] = None  # inference only

class MemoryAugmentedAttention(nn.Module):
    """Scaled attention over (context + retrieved memory).

    If `use_parametric_scores` is True, replaces q·k with MLP([q;k]).
    """

    def __init__(self, cfg: AttentionConfig):
        super().__init__()
        self.cfg = cfg
        if cfg.use_parametric_scores:
            self.score = ParametricScoreMLP(cfg.d_head, cfg.d_head, cfg.score_hidden)
        else:
            self.score = None
        self.dropout = nn.Dropout(cfg.dropout_p)
        # ленивые проекции для памяти (на случай Dk/Dv != Dh)
        self.mem_k_proj = None
        self.mem_v_proj = None

    def forward(
        self,
        q: torch.Tensor,  # [B,H,Tq,Dh]
        k_ctx: torch.Tensor,  # [B,H,Tk,Dh]
        v_ctx: torch.Tensor,  # [B,H,Tk,Dh]
        memory_bank: Optional[MemoryBank] = None,
        q_for_mem: Optional[torch.Tensor] = None,  # [B,Tq,Dh] (aggregated across heads)
        attn_mask: Optional[torch.Tensor] = None,  # [B,1,Tq,Tk_aug] (additive -inf mask)
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if attn_mask is not None and attn_mask.device != q.device:
            attn_mask = attn_mask.to(q.device)

        B, H, Tq, Dh = q.shape
        # Tk = k_ctx.shape[2]

        # Retrieve + augment
        if memory_bank is not None and self.cfg.mem_topk > 0 and q_for_mem is not None:
            K_mem, V_mem = memory_bank.retrieve(q_for_mem, self.cfg.mem_topk)  # [B,Tq,K,Dk],[B,Tq,K,Dv]
            # ── Приводим размеры памяти к Dh, если нужно ─────────────────────
            Dk_in = K_mem.shape[-1]
            Dv_in = V_mem.shape[-1]
            if Dk_in != Dh:
                if self.mem_k_proj is None or self.mem_k_proj.in_features != Dk_in:
                    self.mem_k_proj = nn.Linear(Dk_in, Dh, bias=False).to(q.device)
                K_mem = self.mem_k_proj(K_mem)  # [B,Tq,K,Dh]
            if Dv_in != Dh:
                if self.mem_v_proj is None or self.mem_v_proj.in_features != Dv_in:
                    self.mem_v_proj = nn.Linear(Dv_in, Dh, bias=False).to(q.device)
                V_mem = self.mem_v_proj(V_mem)  # [B,Tq,K,Dh]
            # ─────────────────────────────────────────────────────────────────
            # Tile across heads
            K_mem = K_mem.unsqueeze(1).expand(B, H, Tq, self.cfg.mem_topk, Dh)
            V_mem = V_mem.unsqueeze(1).expand(B, H, Tq, self.cfg.mem_topk, Dh)
            # Перейдём в per-query путь ниже (корректный для памяти)
            use_per_query_path = True
        else:
            K_mem = V_mem = None
            use_per_query_path = False

        if self.score is None and not use_per_query_path:
            # Standard scaled dot product attention on augmented ctx only
            scores = torch.matmul(q, k_ctx.transpose(-2, -1)) / math.sqrt(Dh)  # [B,H,Tq,Tk]
            if attn_mask is not None:
                scores = scores + attn_mask
            weights = torch.softmax(scores, dim=-1)
            weights = self.dropout(weights)
            out = torch.matmul(weights, v_ctx)  # [B,H,Tq,Dh]
            return out, weights

        # General / per‑query path (handles parametric scores and/or memory per query)
        outs = []
        weights_out = []
        for t in range(Tq):
            q_t = q[:, :, t : t + 1, :]  # [B,H,1,Dh]
            k_cat = k_ctx
            v_cat = v_ctx
            if K_mem is not None:
                # For this query t, append its memory neighbors
                k_mem_t = K_mem[:, :, t, :, :]  # [B,H,K,Dh]
                v_mem_t = V_mem[:, :, t, :, :]  # [B,H,K,Dh]
                k_cat = torch.cat([k_cat, k_mem_t], dim=2)  # [B,H,Tk+K,Dh]
                v_cat = torch.cat([v_cat, v_mem_t], dim=2)

            if self.score is None:
                scores = torch.matmul(q_t, k_cat.transpose(-2, -1)) / math.sqrt(Dh)  # [B,H,1,Tk_aug]
            else:
                # Правильно: считаем пары для одного запроса q_t (Tq=1)
                scores = self.score.pairwise_scores(
                    q_t,  # [B,H,1,Dh]
                    k_cat,  # [B,H,Tk_aug,Dh]
                    self.cfg.score_chunk_size,
                )  # -> [B,H,1,Tk_aug]

            if attn_mask is not None:
                # Slice mask for current query timestep if provided per sequence
                scores = scores + attn_mask[:, :, t : t + 1, : scores.shape[-1]]
            w = torch.softmax(scores, dim=-1)
            w = self.dropout(w)
            out_t = torch.matmul(w, v_cat)  # [B,H,1,Dh]
            outs.append(out_t)
            weights_out.append(w)

        out = torch.cat(outs, dim=2)  # [B,H,Tq,Dh]
        weights = torch.cat(weights_out, dim=2)  # [B,H,Tq,Tk_aug]
        return out, weights


class AdaptiveWidthMultiheadAttention(nn.Module):
    """MHA with per‑head gating (token‑wise).

    Gates are in [0,1] and are applied to each head's output. Optionally, during
    inference, heads below `prune_heads_threshold` can be skipped (compute saving).

    Supports MemoryAugmentedAttention + Parametric scores under the hood.
    """

    def __init__(self, cfg: AttentionConfig):
        super().__init__()
        self.cfg = cfg
        H, Dm, Dh = cfg.n_heads, cfg.d_model, cfg.d_head
        self.q_proj = nn.Linear(Dm, H * Dh, bias=False)
        self.k_proj = nn.Linear(Dm, H * Dh, bias=False)
        self.v_proj = nn.Linear(Dm, H * Dh, bias=False)
        self.o_proj = nn.Linear(H * Dh, Dm, bias=False)
        self.attn = MemoryAugmentedAttention(cfg)
        # Переключатель гейтинга голов
        self.use_head_gate = (cfg.head_gate_hidden is not None)
        # Per-token per-head gate: g = sigmoid(MLP(x)) -> [B,T,H]
        # Если head_gate_hidden задан (в т.ч. по умолчанию), создаём гейт;
        # если None — гейтинг отключён.
        self.head_gate = GatingMLP(Dm, d_out=H, hidden_dim=cfg.head_gate_hidden) if self.use_head_gate else None
        self.dropout = nn.Dropout(cfg.dropout_p)

    def forward(
        self,
        x: torch.Tensor,  # [B,T,Dm]
        attn_mask: Optional[torch.Tensor] = None,  # [B,1,T,Tk_aug] additive mask if used
        memory_bank: Optional[MemoryBank] = None,
        inference_prune: bool = False,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        # device guard
        if self.q_proj.weight.device != x.device:
            self.to(x.device)
        if attn_mask is not None and attn_mask.device != x.device:
            attn_mask = attn_mask.to(x.device)

        B, T, Dm = x.shape
        H, Dh = self.cfg.n_heads, self.cfg.d_head

        q = _shape_proj(self.q_proj(x), H, Dh)
        k = _shape_proj(self.k_proj(x), H, Dh)
        v = _shape_proj(self.v_proj(x), H, Dh)

        # по умолчанию гейтинга нет
        gates = None
        if self.head_gate is None:  # или: if not self.use_head_gate:
            gates_h = torch.ones(B, H, T, 1, device=x.device, dtype=x.dtype)
        else:
            gates = self.head_gate(x).clamp(0.0, 1.0)  # [B,T,H]
            gates_h = gates.permute(0, 2, 1).unsqueeze(-1)  # [B,H,T,1]

        # Aggregate queries for memory retrieval (mean across heads)
        q_for_mem = q.mean(dim=1).contiguous()  # [B,T,Dh]

        if inference_prune and self.head_gate is not None and (self.cfg.prune_heads_threshold is not None):
            # Only compute heads whose avg gate across tokens >= threshold
            active = (gates.mean(dim=1) >= self.cfg.prune_heads_threshold)  # [B,H]
            # Fallback: if all pruned for a batch item, keep at least one head
            if active.sum(dim=1).min() == 0:
                # ensure at least head 0 stays
                active[:, 0] = True
            outs = []
            weights_all = []
            for h in range(H):
                if bool(active[:, h].any()):
                    qh, kh, vh = q[:, h : h + 1], k[:, h : h + 1], v[:, h : h + 1]
                    out_h, w_h = self.attn(qh, kh, vh, memory_bank, q_for_mem, attn_mask)
                else:
                    out_h = torch.zeros(B, 1, T, Dh, device=x.device, dtype=x.dtype)
                    w_h = torch.zeros(B, 1, T, k.shape[2], device=x.device, dtype=x.dtype)
                # Apply gate for this head
                g_h = gates_h[:, h : h + 1]  # [B,1,T,1]
                outs.append(out_h * g_h)
                weights_all.append(w_h)
            out = torch.cat(outs, dim=1)  # [B,H,T,Dh]
            weights = torch.cat(weights_all, dim=1)
        else:
            out, weights = self.attn(q, k, v, memory_bank, q_for_mem, attn_mask)
            out = out * gates_h  # gate heads

        y = self.o_proj(_merge_heads(out))  # [B,T,Dm]
        y = self.dropout(y)
        return y, {"head_gates": gates, "attn_weights": weights}


# -----------------------------
# Transformer Layer with halting
# -----------------------------
@dataclass
class LayerConfig:
    d_model: int
    n_heads: int
    d_head: int
    d_ff: int
    attn_dropout: float = 0.0
    resid_dropout: float = 0.0
    act: str = "gelu"
    head_gate_hidden: Optional[int] = None
    use_parametric_scores: bool = False
    score_hidden: int = 128
    score_chunk_size: Optional[int] = None
    mem_topk: int = 0
    prune_heads_threshold: Optional[float] = None
    halt_gate_hidden: Optional[int] = None
    halt_bias_init: float = 1.5  # sigmoid(1.5)≈0.82, помогает раннему выходу
    halt_from_cls: bool = False

class AdaptiveLayer(nn.Module):
    def __init__(self, cfg: LayerConfig):
        super().__init__()
        self.cfg = cfg
        attn_cfg = AttentionConfig(
            n_heads=cfg.n_heads,
            d_model=cfg.d_model,
            d_head=cfg.d_head,
            dropout_p=cfg.attn_dropout,
            head_gate_hidden=cfg.head_gate_hidden,
            use_parametric_scores=cfg.use_parametric_scores,
            score_hidden=cfg.score_hidden,
            score_chunk_size=cfg.score_chunk_size,
            mem_topk=cfg.mem_topk,
            prune_heads_threshold=cfg.prune_heads_threshold,
        )
        self.attn = AdaptiveWidthMultiheadAttention(attn_cfg)
        self.ln1 = nn.LayerNorm(cfg.d_model)
        self.ln2 = nn.LayerNorm(cfg.d_model)
        self.ff = nn.Sequential(
            nn.Linear(cfg.d_model, cfg.d_ff),
            nn.GELU() if cfg.act == "gelu" else nn.ReLU(),
            nn.Dropout(cfg.resid_dropout),
            nn.Linear(cfg.d_ff, cfg.d_model),
        )
        self.resid_drop = nn.Dropout(cfg.resid_dropout)
        self.halt_gate = GatingMLP(cfg.d_model, d_out=1, hidden_dim=cfg.halt_gate_hidden)
        self.halt_ln = nn.LayerNorm(cfg.d_model)
        # Инициализация: высокий bias → p_halt стартует ~0.8
        nn.init.constant_(self.halt_gate.net.bias, cfg.halt_bias_init)
        nn.init.zeros_(self.halt_gate.net.weight)

    def forward(
        self,
        x: torch.Tensor,  # [B,T,D]
        attn_mask: Optional[torch.Tensor] = None,
        memory_bank: Optional[MemoryBank] = None,
        inference_prune: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
        # MHA block
        attn_out, attn_info = self.attn(x, attn_mask=attn_mask, memory_bank=memory_bank, inference_prune=inference_prune)
        x = x + self.resid_drop(attn_out)
        x = self.ln1(x)
        # FFN block
        ff_out = self.ff(x)
        x = x + self.resid_drop(ff_out)
        x = self.ln2(x)
        # Halting per token
        p_halt = self.halt_gate(self.halt_ln(x))  # [B,T,1]
        return x, p_halt, attn_info


# -----------------------------
# Dynamic Encoder (Adaptive Depth)
# -----------------------------
@dataclass
class EncoderConfig:
    n_layers: int
    layer: LayerConfig
    early_exit: bool = False
    exit_threshold: float = 0.9  # cumulative halting threshold at inference
    min_layers_before_exit: int = 1
    exit_patience: int = 0
    ponder_epsilon: float = 0.0  # optional ACT‑style regularizer weight

class DynamicEncoder(nn.Module):
    """Stack with probabilistic halting fusion.

    Training: computes all layers, combines outputs with weights w_i = p_i * Π_{j<i}(1 - p_j) per token.
    Inference with early_exit=True: stops updating tokens whose cumulative halting ≥ threshold.
    """

    def __init__(self, cfg: EncoderConfig):
        super().__init__()
        self.cfg = cfg
        self.layers = nn.ModuleList([AdaptiveLayer(cfg.layer) for _ in range(cfg.n_layers)])

    def forward(
        self,
        x: torch.Tensor,  # [B,T,D]
        attn_mask: Optional[torch.Tensor] = None,
        memory_bank: Optional[MemoryBank] = None,
        return_aux: bool = True,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        B, T, D = x.shape
        remaining = torch.ones(B, T, 1, device=x.device, dtype=x.dtype)
        weighted_sum = torch.zeros_like(x)
        ponder_cost = torch.zeros(B, T, 1, device=x.device, dtype=x.dtype)

        tokens_done = torch.zeros(B, T, 1, device=x.device, dtype=torch.bool)

        # patience/min-layers counters
        patience = int(getattr(self.cfg, "exit_patience", 0) or 0)
        min_layers = int(getattr(self.cfg, "min_layers_before_exit", 0) or 0)
        pat_ctr = torch.zeros(B, T, 1, device=x.device, dtype=torch.long)
        last_decision = torch.zeros(B, T, 1, device=x.device, dtype=torch.bool)

        aux = {"halting_ps": [], "head_gates": [], "attn_weights": [], "feats_per_layer": []}

        for i, layer in enumerate(self.layers):
            inference_prune = self.cfg.early_exit
            x, p_i, attn_info = layer(x, attn_mask=attn_mask, memory_bank=memory_bank, inference_prune=inference_prune)
            w_i = remaining * p_i  # [B,T,1]
            weighted_sum = weighted_sum + w_i * x
            remaining = remaining * (1.0 - p_i)
            ponder_cost = ponder_cost + p_i

            aux["halting_ps"].append(p_i)
            aux["head_gates"].append(attn_info.get("head_gates"))
            aux["attn_weights"].append(attn_info.get("attn_weights"))
            aux["feats_per_layer"].append(x)  # <— сохраняем признаки слоя для глубокой супервизии
            aux["feats_per_layer"].append(x)  # <— сохраняем признаки слоя для глубокой супервизии

            if self.cfg.early_exit:
                cum = (1.0 - remaining)
                decision = (cum >= self.cfg.exit_threshold)
                same = (decision == last_decision)
                pat_ctr = torch.where(same, pat_ctr + 1, torch.zeros_like(pat_ctr))
                last_decision = decision
                can_exit = (i + 1) >= min_layers
                patience_ok = (pat_ctr >= patience) if patience > 0 else torch.ones_like(pat_ctr, dtype=torch.bool)
                tokens_done = tokens_done | (decision & can_exit & patience_ok)
                if bool(tokens_done.all()):
                    break

        # If some probability mass remains (didn't halt), add the residual contribution
        if remaining.max() > 0:
            weighted_sum = weighted_sum + remaining * x
            ponder_cost = ponder_cost + remaining  # ACT‑style remainder term

        if self.cfg.ponder_epsilon > 0:
            # You can add this to loss: L += epsilon * mean(ponder_cost)
            aux["ponder_cost"] = ponder_cost.mean()

        aux["halting_ps"] = torch.stack(aux["halting_ps"], dim=0)  # [L,B,T,1]
        return (weighted_sum, aux) if return_aux else (weighted_sum, {})


# -----------------------------
# Memory Writer (optional)
# -----------------------------
class MemoryWriter(nn.Module):
    """Learns to summarize a span and write (key,value) to memory.

    Given X:[B,T,D], produces K:[B,M,Dk], V:[B,M,Dv]. Here we emit a single summary (M=1)
    per batch item by default. You can call after each paragraph or N tokens.
    """

    def __init__(self, d_in: int, d_key: int, d_value: int, hidden: int = 256, entries: int = 1):
        super().__init__()
        self.entries = entries
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.to_k = nn.Sequential(nn.Linear(d_in, hidden), nn.GELU(), nn.Linear(hidden, d_key))
        self.to_v = nn.Sequential(nn.Linear(d_in, hidden), nn.GELU(), nn.Linear(hidden, d_value))
        self.importance = GatingMLP(d_in, d_out=entries, hidden_dim=hidden)  # write gate in [0,1]

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # Global average over time (token) dimension as a simple summary
        # x:[B,T,D] -> [B,D]
        x_pooled = self.pool(x.transpose(1, 2)).squeeze(-1)
        k = self.to_k(x_pooled).unsqueeze(1).repeat(1, self.entries, 1)
        v = self.to_v(x_pooled).unsqueeze(1).repeat(1, self.entries, 1)
        imp = self.importance(x_pooled).unsqueeze(-1)  # [B,entries,1]
        return k, v, imp


# -----------------------------
# Minimal usage example (shape test)
# -----------------------------
if __name__ == "__main__":
    torch.manual_seed(0)
    B, T, Dm = 2, 12, 256
    H, Dh = 8, 32

    layer_cfg = LayerConfig(
        d_model=Dm, n_heads=H, d_head=Dh, d_ff=4 * Dm,
        attn_dropout=0.1, resid_dropout=0.1,
        head_gate_hidden=128,
        use_parametric_scores=True,  # turn on MLP scoring
        score_hidden=64,
        score_chunk_size=16,  # reduce peak mem for long seqs
        mem_topk=4,  # use external memory
        prune_heads_threshold=0.2,
        halt_gate_hidden=64,
    )
    enc_cfg = EncoderConfig(n_layers=6, layer=layer_cfg, early_exit=True, exit_threshold=0.9, ponder_epsilon=0.01)

    model = DynamicEncoder(enc_cfg)

    x = torch.randn(B, T, Dm)
    mem = MemoryBank(d_key=Dh, d_value=Dh)
    # seed memory with random facts
    mem.write(torch.randn(128, Dh), torch.randn(128, Dh))

    y, aux = model(x, memory_bank=mem)
    print("y:", y.shape)
    print("ponder_cost:", float(aux.get("ponder_cost", torch.tensor(0.0))))
    print("halting_ps shape:", aux["halting_ps"].shape)