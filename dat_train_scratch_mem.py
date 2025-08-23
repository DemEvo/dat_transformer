"""
DAT from-scratch training with external MemoryBank
==================================================

This trainer extends the pure backprop setup by wiring in:
  - External K/V memory (`MemoryBank`) with per-token top‑k retrieval
  - Optional `MemoryWriter` to append compressed entries during training
  - Attention masking for variable-length sequences (classification & CTC)
  - Robust head‑balance regularization (entropy- or variance-based)
  - Compute regularization (expected depth, L1 on gates/halts)

Author: ChatGPT (PyTorch >= 1.12)
"""
from __future__ import annotations

import math, os, time
from dataclasses import dataclass
from typing import Optional, Dict, Tuple, List

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR

# Import DAT building blocks from canvas module
from dat_transformer import (
    DynamicEncoder, EncoderConfig, LayerConfig,
    MemoryBank, MemoryWriter,
)


# -----------------------------
# Heads
# -----------------------------
class TokenClassifierHead(nn.Module):
    def __init__(self, d_in: int, n_classes: int):
        super().__init__()
        self.proj = nn.Linear(d_in, n_classes)
    def forward(self, x):
        return self.proj(x)

class CTCHead(nn.Module):
    def __init__(self, d_in: int, n_classes: int):
        super().__init__()
        self.proj = nn.Linear(d_in, n_classes)
    def forward(self, x):
        return self.proj(x)


# -----------------------------
# Helpers
# -----------------------------

def cosine_with_warmup(optimizer, warmup_steps: int, total_steps: int):
    def f(step):
        if step < warmup_steps:
            return max(1e-8, step / max(1, warmup_steps))
        p = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        p = min(1.0, max(0.0, p))
        return 0.5 * (1 + math.cos(math.pi * p))
    return LambdaLR(optimizer, lr_lambda=f)


def expected_depth(halting_ps: torch.Tensor) -> torch.Tensor:
    # halting_ps: [L,B,T,1]
    return halting_ps.squeeze(-1).sum(dim=0).mean()


def l1_head_gates(head_gates) -> torch.Tensor:
    if isinstance(head_gates, list):
        vals = [g.abs().mean() for g in head_gates if g is not None]
        return torch.stack(vals).mean() if vals else torch.tensor(0.0)
    return head_gates.abs().mean()


def head_balance_loss(head_gates, variant: str = "entropy") -> torch.Tensor:
    """Encourage diverse head usage across the batch and time.

    Args:
        head_gates: list([B,T,H]) or tensor [L,B,T,H]
        variant: "entropy" (default) or "variance"
    """
    if isinstance(head_gates, list):
        # Average per head across layers, batch, time
        avgs = torch.stack([g.mean(dim=(0,1)) for g in head_gates if g is not None])  # [L,H]
        usage = avgs.mean(dim=0)  # [H]
    else:
        usage = head_gates.mean(dim=(0,1,2))  # [H]
    # Normalize to a distribution over heads
    p = usage.clamp_min(1e-8)
    p = p / p.sum()
    if variant == "variance":
        return p.var()
    # default: maximize entropy -> minimize negative entropy
    ent = -(p * p.log()).sum()
    # convert to loss by taking negative entropy (penalize low entropy)
    return -ent / math.log(p.numel())  # normalized to [−1,0]


def make_attn_mask(padding_mask: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
    """Build additive attention mask [B,1,1,T] (broadcast to [B,1,Tq,Tk]).
    padding_mask: [B,T] where 1=real, 0=pad
    """
    if padding_mask is None:
        return None
    am = padding_mask.unsqueeze(1).unsqueeze(2).float()
    neg = torch.finfo(am.dtype).min  # large negative for stability with AMP
    return (1.0 - am) * neg


# -----------------------------
# Configs
# -----------------------------
@dataclass
class TrainConfig:
    task: str = "classification"  # "classification" | "ctc"
    n_classes: int = 100
    lr: float = 2e-4
    weight_decay: float = 0.01
    max_steps: int = 10000
    warmup_steps: int = 500
    grad_clip: float = 1.0
    amp: bool = True
    ema_decay: Optional[float] = 0.999
    log_every: int = 50
    ckpt_dir: str = "./checkpoints"

@dataclass
class RegularizationConfig:
    depth_w: float = 0.05
    head_l1_w: float = 0.001
    halt_l1_w: float = 0.001
    head_balance_w: float = 0.01
    head_balance_variant: str = "entropy"  # "entropy" | "variance"
    memory_write_w: float = 0.0  # optional penalty to limit writes

@dataclass
class MemoryConfig:
    enabled: bool = True
    topk: int = 4           # set LayerConfig.mem_topk to match
    writer_hidden: int = 256
    entries_per_write: int = 1
    write_threshold: float = 0.5  # threshold on writer importance gate


# -----------------------------
# Trainer with memory
# -----------------------------
class DATMemoryTrainer:
    def __init__(
        self,
        encoder: DynamicEncoder,
        layer_dim: int,
        train_cfg: TrainConfig,
        reg_cfg: RegularizationConfig,
        mem_cfg: MemoryConfig,
        n_classes: Optional[int] = None,
    ):
        self.encoder = encoder
        self.cfg = train_cfg
        self.reg = reg_cfg
        self.mem_cfg = mem_cfg
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Heads
        assert self.cfg.task in {"classification", "ctc"}
        n_cls = n_classes if n_classes is not None else self.cfg.n_classes
        if self.cfg.task == "classification":
            self.head = TokenClassifierHead(layer_dim, n_cls)
            self.ce = nn.CrossEntropyLoss(reduction="none")
        else:
            self.head = CTCHead(layer_dim, n_cls)
            self.ctc = nn.CTCLoss(blank=0, zero_infinity=True)

        # Memory
        self.memory = MemoryBank(d_key=self.encoder.cfg.layer.d_head, d_value=layer_dim)
        self.writer = MemoryWriter(d_in=layer_dim, d_key=self.encoder.cfg.layer.d_head,
                                   d_value=layer_dim, hidden=self.mem_cfg.writer_hidden,
                                   entries=self.mem_cfg.entries_per_write) if self.mem_cfg.enabled else None

        # Opt
        params = list(self.encoder.parameters()) + list(self.head.parameters())
        self.opt = AdamW(params, lr=self.cfg.lr, weight_decay=self.cfg.weight_decay)
        self.sched = cosine_with_warmup(self.opt, self.cfg.warmup_steps, self.cfg.max_steps)
        self.scaler = torch.cuda.amp.GradScaler(enabled=self.cfg.amp)

        # EMA
        self.ema = None
        if self.cfg.ema_decay is not None:
            self.ema = {k: v.detach().clone() for k, v in self.encoder.state_dict().items()}
            self.ema_head = {k: v.detach().clone() for k, v in self.head.state_dict().items()}

        self.encoder.to(self.device)
        self.head.to(self.device)
        self.global_step = 0
        self.encoder.cfg.early_exit = False
        os.makedirs(self.cfg.ckpt_dir, exist_ok=True)

    @torch.no_grad()
    def _update_ema(self):
        if self.ema is None:
            return
        d = self.cfg.ema_decay
        for k, v in self.encoder.state_dict().items():
            self.ema[k].mul_(d).add_(v, alpha=1 - d)
        for k, v in self.head.state_dict().items():
            self.ema_head[k].mul_(d).add_(v, alpha=1 - d)

    def _apply_ema(self):
        if self.ema is None:
            return
        self.encoder.load_state_dict(self.ema, strict=False)
        self.head.load_state_dict(self.ema_head, strict=False)

    def _classification_loss(self, logits: torch.Tensor, labels: torch.Tensor, mask: Optional[torch.Tensor]) -> torch.Tensor:
        B, T, C = logits.shape
        loss = self.ce(logits.view(B*T, C), labels.view(B*T)).view(B, T)
        if mask is not None:
            m = mask.float()
            loss = (loss*m).sum() / (m.sum() + 1e-9)
        else:
            loss = loss.mean()
        return loss

    def _ctc_loss(self, logits: torch.Tensor, targets: List[torch.Tensor], input_lengths: torch.Tensor, target_lengths: torch.Tensor) -> torch.Tensor:
        log_probs = F.log_softmax(logits, dim=-1).transpose(0,1).contiguous()
        flat_targets = torch.cat(targets).to(self.device)
        return self.ctc(log_probs, flat_targets, input_lengths.to(self.device), target_lengths.to(self.device))

    def _regularization(self, aux: Dict[str, torch.Tensor], writes: Optional[torch.Tensor] = None) -> torch.Tensor:
        reg = torch.tensor(0.0, device=self.device)
        if self.reg.depth_w > 0 and "halting_ps" in aux:
            reg = reg + self.reg.depth_w * expected_depth(aux["halting_ps"])
        if self.reg.head_l1_w > 0 and "head_gates" in aux:
            reg = reg + self.reg.head_l1_w * l1_head_gates(aux["head_gates"])
        if self.reg.halt_l1_w > 0 and "halting_ps" in aux:
            reg = reg + self.reg.halt_l1_w * aux["halting_ps"].abs().mean()
        if self.reg.head_balance_w > 0 and "head_gates" in aux and aux["head_gates"] is not None:
            reg = reg + self.reg.head_balance_w * head_balance_loss(aux["head_gates"], self.reg.head_balance_variant)
        if self.reg.memory_write_w > 0 and writes is not None:
            reg = reg + self.reg.memory_write_w * writes.mean()
        return reg

    def _forward_encoder(self, x: torch.Tensor, padding_mask: Optional[torch.Tensor]):
        attn_mask = make_attn_mask(padding_mask)
        # Pass memory_bank if enabled
        if self.mem_cfg.enabled and self.encoder.cfg.layer.mem_topk > 0:
            feats, aux = self.encoder(x, attn_mask=attn_mask, memory_bank=self.memory, return_aux=True)
        else:
            feats, aux = self.encoder(x, attn_mask=attn_mask, return_aux=True)
        return feats, aux

    def step_classification(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        x = batch["inputs"].to(self.device)
        y = batch["labels"].to(self.device)
        padding_mask = batch.get("mask")
        if padding_mask is not None:
            padding_mask = padding_mask.to(self.device)

        self.encoder.train(); self.head.train()
        self.opt.zero_grad(set_to_none=True)

        writes_gate = None
        with torch.cuda.amp.autocast(enabled=self.cfg.amp):
            feats, aux = self._forward_encoder(x, padding_mask)
            logits = self.head(feats)
            loss_main = self._classification_loss(logits, y, padding_mask)

            # Optional memory write
            if self.mem_cfg.enabled and self.writer is not None:
                k, v, imp = self.writer(feats.detach())  # summarize current batch
                # imp: [B,entries,1] in [0,1]
                writes_gate = (imp > self.mem_cfg.write_threshold).float()
                # write only selected entries
                sel = writes_gate.squeeze(-1)  # [B,entries]
                if sel.numel() > 0 and sel.sum() > 0:
                    # collapse batch x entries to rows to write
                    k_w = k.reshape(-1, k.shape[-1])[sel.reshape(-1) > 0]
                    v_w = v.reshape(-1, v.shape[-1])[sel.reshape(-1) > 0]
                    with torch.no_grad():
                        self.memory.write(k_w, v_w)

            loss_reg = self._regularization(aux, writes_gate)
            loss = loss_main + loss_reg

        self.scaler.scale(loss).backward()
        if self.cfg.grad_clip is not None:
            self.scaler.unscale_(self.opt)
            nn.utils.clip_grad_norm_(list(self.encoder.parameters()) + list(self.head.parameters()), self.cfg.grad_clip)
        self.scaler.step(self.opt)
        self.scaler.update()
        self.sched.step()
        self._update_ema()

        with torch.no_grad():
            pred = logits.argmax(dim=-1)
            if padding_mask is not None:
                correct = ((pred == y) * padding_mask).sum().item()
                total = padding_mask.sum().item()
            else:
                correct = (pred == y).sum().item()
                total = y.numel()
            acc = correct / max(1, total)

        return {"loss": float(loss.detach().cpu()), "acc": float(acc), "mem_size": float(self.memory.size())}

    def step_ctc(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        x = batch["inputs"].to(self.device)
        targets = [t.to(self.device) for t in batch["targets"]]
        input_lengths = batch["input_lengths"].to(self.device)
        target_lengths = batch["target_lengths"].to(self.device)
        # Build padding mask from input_lengths if provided as maxlen T
        padding_mask = None
        if x.dim() == 3:
            B, T, _ = x.shape
            # mask[i, t] = 1 if t < input_lengths[i]
            idx = torch.arange(T, device=self.device).unsqueeze(0)
            padding_mask = (idx < input_lengths.unsqueeze(1)).to(torch.float32)

        self.encoder.train(); self.head.train()
        self.opt.zero_grad(set_to_none=True)

        with torch.cuda.amp.autocast(enabled=self.cfg.amp):
            feats, aux = self._forward_encoder(x, padding_mask)
            logits = self.head(feats)
            loss_main = self._ctc_loss(logits, targets, input_lengths, target_lengths)
            loss_reg = self._regularization(aux)
            loss = loss_main + loss_reg

        self.scaler.scale(loss).backward()
        if self.cfg.grad_clip is not None:
            self.scaler.unscale_(self.opt)
            nn.utils.clip_grad_norm_(list(self.encoder.parameters()) + list(self.head.parameters()), self.cfg.grad_clip)
        self.scaler.step(self.opt)
        self.scaler.update()
        self.sched.step()
        self._update_ema()

        return {"loss": float(loss.detach().cpu()), "mem_size": float(self.memory.size())}

    def save_ckpt(self, tag: str = "latest"):
        path = os.path.join(self.cfg.ckpt_dir, f"dat_mem_{tag}.pt")
        torch.save({
            "step": self.global_step,
            "encoder": self.encoder.state_dict(),
            "head": self.head.state_dict(),
            "opt": self.opt.state_dict(),
            "sched": self.sched.state_dict(),
            "ema": self.ema,
            "ema_head": getattr(self, "ema_head", None),
            "cfg": {
                "train": self.cfg.__dict__,
                "reg": self.reg.__dict__,
            }
        }, path)
        return path

    def train(self, train_loader):
        self.encoder.train(); self.head.train()
        for step, batch in enumerate(train_loader, start=1):
            self.global_step = step
            if self.cfg.task == "classification":
                logs = self.step_classification(batch)
            else:
                logs = self.step_ctc(batch)
            if step % self.cfg.log_every == 0:
                print({k: round(v, 5) if isinstance(v, float) else v for k, v in logs.items()}, "step", step)
            if step >= self.cfg.max_steps:
                break
        ckpt = self.save_ckpt("final")
        print("Saved checkpoint:", ckpt)

    @torch.no_grad()
    def evaluate(self, batch, early_exit_threshold: float = 0.9):
        self.encoder.cfg.early_exit = True
        self.encoder.cfg.exit_threshold = early_exit_threshold
        self.encoder.eval(); self.head.eval()
        x = batch["inputs"].to(self.device)
        padding_mask = batch.get("mask")
        if padding_mask is not None: padding_mask = padding_mask.to(self.device)
        feats, aux = self._forward_encoder(x, padding_mask)
        logits = self.head(feats)
        ed = expected_depth(aux["halting_ps"]).item()
        self.encoder.cfg.early_exit = False
        return {"expected_depth": ed, "logits_shape": list(logits.shape), "mem_size": self.memory.size()}


# -----------------------------
# Minimal synthetic demo
# -----------------------------
if __name__ == "__main__":
    torch.manual_seed(0)
    # Backbone with memory enabled (mem_topk>0)
    Dm = 128
    layer_cfg = LayerConfig(
        d_model=Dm, n_heads=4, d_head=32, d_ff=4*Dm,
        head_gate_hidden=64, halt_gate_hidden=64,
        use_parametric_scores=False,
        mem_topk=4,  # IMPORTANT: enable memory retrieval in attention
    )
    enc_cfg = EncoderConfig(n_layers=4, layer=layer_cfg, ponder_epsilon=0.0, early_exit=False)
    encoder = DynamicEncoder(enc_cfg)

    # Trainer configs
    train_cfg = TrainConfig(task="classification", n_classes=20, max_steps=200, log_every=50)
    reg_cfg = RegularizationConfig(depth_w=0.05, head_l1_w=0.001, halt_l1_w=0.001, head_balance_w=0.01,
                                   head_balance_variant="entropy", memory_write_w=0.0)
    mem_cfg = MemoryConfig(enabled=True, topk=4, writer_hidden=256, entries_per_write=1, write_threshold=0.6)

    trainer = DATMemoryTrainer(encoder, Dm, train_cfg, reg_cfg, mem_cfg)

    # Tiny dataset
    B, T = 8, 32
    x = torch.randn(B, T, Dm)
    y = torch.randint(0, 20, (B, T))
    mask = torch.ones(B, T, dtype=torch.bool)
    ds = [{"inputs": x, "labels": y, "mask": mask}] * 200
    loader = torch.utils.data.DataLoader(ds, batch_size=1, shuffle=True)

    trainer.train(loader)
    report = trainer.evaluate({"inputs": x, "mask": mask})
    print("Eval report:", report)
