# baseline_fast.py
from dataclasses import dataclass
from typing import Optional, Dict, Any, Tuple
import torch
import torch.nn.functional as F
from torch import nn

@dataclass
class BaseCfg:
    n_layers: int
    d_model: int
    n_heads: int
    d_ff: int
    attn_dropout: float = 0.0
    resid_dropout: float = 0.0
    norm_eps: float = 1e-5

class MHA_SDPA(nn.Module):
    def __init__(self, d_model: int, n_heads: int, attn_dropout: float):
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head  = d_model // n_heads
        H, D, Dh = n_heads, d_model, self.d_head
        self.q = nn.Linear(D, H * Dh, bias=False)
        self.k = nn.Linear(D, H * Dh, bias=False)
        self.v = nn.Linear(D, H * Dh, bias=False)
        self.o = nn.Linear(H * Dh, D, bias=False)
        self.attn_dropout = attn_dropout

    def forward(self, x: torch.Tensor, attn_mask: Optional[torch.Tensor]) -> Tuple[torch.Tensor, Dict[str,Any]]:
        B, T, D = x.shape
        H, Dh = self.n_heads, self.d_head
        q = self.q(x).view(B, T, H, Dh).transpose(1, 2)  # [B,H,T,Dh]
        k = self.k(x).view(B, T, H, Dh).transpose(1, 2)
        v = self.v(x).view(B, T, H, Dh).transpose(1, 2)
        y = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=attn_mask,  # float mask broadcastable к [B,H,T,T] или [B,1,1,T]
            dropout_p=self.attn_dropout if self.training else 0.0,
            is_causal=False,
        )                         # [B,H,T,Dh]
        y = y.transpose(1, 2).contiguous().view(B, T, H * Dh)
        y = self.o(y)             # [B,T,D]
        return y, {}              # карты внимания не возвращаем (экономим память)

class FFN(nn.Module):
    def __init__(self, d_model: int, d_ff: int, resid_dropout: float):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Linear(d_ff, d_model),
            nn.Dropout(resid_dropout),
        )
    def forward(self, x): return self.net(x)

class BaselineEncoderFast(nn.Module):
    def __init__(self, cfg: BaseCfg):
        super().__init__()
        self.cfg = cfg
        self.layers = nn.ModuleList([
            nn.ModuleDict(dict(
                ln1=nn.LayerNorm(cfg.d_model, eps=cfg.norm_eps),
                ln2=nn.LayerNorm(cfg.d_model, eps=cfg.norm_eps),
                attn=MHA_SDPA(cfg.d_model, cfg.n_heads, cfg.attn_dropout),
                ffn=FFN(cfg.d_model, cfg.d_ff, cfg.resid_dropout),
                drop=nn.Dropout(cfg.resid_dropout),
            )) for _ in range(cfg.n_layers)
        ])
        self.final_ln = nn.LayerNorm(cfg.d_model, eps=cfg.norm_eps)

    def forward(self, x: torch.Tensor, attn_mask: Optional[torch.Tensor] = None, return_aux: bool=False):
        h = x
        for L in self.layers:
            z = L['ln1'](h)
            sa, _ = L['attn'](z, attn_mask)
            h = h + L['drop'](sa)
            z = L['ln2'](h)
            ff = L['ffn'](z)
            h = h + L['drop'](ff)
        y = self.final_ln(h)
        return (y, {}) if return_aux else y
