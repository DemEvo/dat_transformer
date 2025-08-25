
"""
edu_experiment_ort_vs_baseline.py — Baseline (классический Transformer) vs ORT
=============================================================================

- Задача: классификация «скобочная последовательность сбалансирована / нет».
- Baseline: классический TransformerEncoder (из вашего edu_experiment_min.py).
- ORT: наш ort_transformer с развязкой head_dim и ортогонализацией голов.
- Результат: markdown-таблица с точностями, скоростью, глубиной (для ORT глубина = число слоёв).

Запуск:
    python edu_experiment_ort_vs_baseline.py

Зависимости:
    pip install torch tqdm
    файлы: ort_transformer.py, edu_experiment_min.py (для BaselineEncoder)
"""
import random
import time
from dataclasses import dataclass
from types import SimpleNamespace
from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader

# --- Импортируем Baseline из вашего мини-эксперимента ---
# from edu_experiment_min import BaselineEncoder  # классическую реализацию держим «как эталон»
from baseline_fast import BaselineEncoderFast, BaseCfg

# --- Импортируем ORT ---
from ort_transformer import TransformerEncoder as OrtEncoder, EncoderConfig as OrtCfg

from torch.backends.cuda import sdp_kernel
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
sdp_kernel(enable_flash=True, enable_mem_efficient=True, enable_math=False)

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(it, **kwargs):
        return it

# =========================
# Константы эксперимента
# =========================
SEED = 42
random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

PAD_TOKEN = 0
CLS_TOKEN = 1
VOCAB = {PAD_TOKEN:"<pad>", CLS_TOKEN:"<cls>", 2:"(", 3:")"}
VOCAB_SIZE = max(VOCAB)+1

MAX_LEN = 512
N_CLASSES = 2

# размеры маленьких сетей
MODEL_D = 16
MODEL_HEADS = 2
MODEL_LAYERS = 10
D_FF = 3 * MODEL_D

# ORT-специфика
ORT_D_HEAD = MODEL_D         # независимый dim головы (например, 32 или 64)
ORT_ORTHO_Q = 1e-4
ORT_ORTHO_K = 1e-4
ORT_ORTHO_V = 0.0 # 0.0

# =========================
# Данные
# =========================
def make_balanced_parens(length: int, rng: random.Random) -> str:
    half = length // 2
    open_left, close_left = half, half
    out = []
    depth = 0
    while open_left > 0 or close_left > 0:
        choices = []
        if open_left > 0:
            choices.append("(")
        if close_left > 0 and depth > 0:
            choices.append(")")
        c = rng.choice(choices)
        out.append(c)
        if c == "(":
            open_left -= 1
            depth += 1
        else:
            close_left -= 1
            depth -= 1
    return "".join(out)

def corrupt_parens(s: str, rng: random.Random) -> str:
    arr = list(s)
    if rng.random() < 0.5:
        i = rng.randrange(len(arr))
        arr[i] = "(" if arr[i] == ")" else ")"
    else:
        if rng.random() < 0.5 and len(arr) > 2:
            i = rng.randrange(len(arr))
            del arr[i]
        else:
            i = rng.randrange(len(arr)+1)
            arr.insert(i, "(" if rng.random() < 0.5 else ")")
    return "".join(arr)

def max_nesting_depth(s: str) -> int:
    d = 0
    mx = 0
    for ch in s:
        if ch == "(":
            d += 1
            mx = max(mx, d)
        else:
            d -= 1
    return mx

def tokenize(s: str) -> List[int]:
    return [2 if ch == "(" else 3 for ch in s]

def pad_to_len(ids: List[int], L: int) -> List[int]:
    if len(ids) >= L: return ids[:L]
    return ids + [PAD_TOKEN] * (L - len(ids))

def build_dataset(n_train=6000, n_val=2000, n_test=2000, min_len=10, max_len=40, seed=SEED):
    rng = random.Random(seed)
    def make_split(n):
        seqs, labels, depths = [], [], []
        for _ in tqdm(range(n), desc="Генерация", mininterval=0.1):
            L_body = rng.randint(min_len, max_len)
            if L_body % 2 == 1:
                L_body += 1
            s_good = make_balanced_parens(L_body, rng)
            if rng.random() < 0.5:
                s = s_good; y = 1
            else:
                s = corrupt_parens(s_good, rng); y = 0
            d = max_nesting_depth(s)
            ids = [CLS_TOKEN] + tokenize(s)
            ids = pad_to_len(ids, MAX_LEN)
            seqs.append(ids); labels.append(y); depths.append(d)
        return torch.tensor(seqs), torch.tensor(labels), torch.tensor(depths)
    return make_split(n_train), make_split(n_val), make_split(n_test)

# =========================
# Классификаторы (разные маски)
# =========================
class OrtSequenceClassifier(nn.Module):
    """Для ORT: additive mask (логит-маска)"""
    def __init__(self, encoder, d_model, n_classes):
        super().__init__()
        self.encoder = encoder
        self.embed = nn.Embedding(VOCAB_SIZE, d_model, padding_idx=PAD_TOKEN)
        self.pos = nn.Embedding(MAX_LEN, d_model)
        self.drop = nn.Dropout(0.1)
        self.head = nn.Linear(d_model, n_classes)

    def forward(self, tokens, return_aux=False):
        pad_ok = (tokens != PAD_TOKEN)
        B, T = tokens.shape
        pos = torch.arange(T, device=tokens.device).unsqueeze(0).expand(B, T)
        x = self.embed(tokens) + self.pos(pos)
        x = self.drop(x)
        attn_mask = (1.0 - pad_ok.float()).unsqueeze(1).unsqueeze(2) * -1e9  # [B,1,1,T]
        feats, aux = self.encoder(x, attn_mask=attn_mask, return_aux=True)
        logits = self.head(feats[:, 0, :])
        return (logits, aux) if return_aux else logits

class BaselineSequenceClassifier(nn.Module):
    """Для классического Baseline: key_padding_mask (булева [B,T], True=игнорировать)"""
    def __init__(self, encoder, d_model, n_classes):
        super().__init__()
        self.encoder = encoder
        self.embed = nn.Embedding(VOCAB_SIZE, d_model, padding_idx=PAD_TOKEN)
        self.pos = nn.Embedding(MAX_LEN, d_model)
        self.drop = nn.Dropout(0.1)
        self.head = nn.Linear(d_model, n_classes)

    def forward(self, tokens, return_aux=False):
        pad_ok = (tokens != PAD_TOKEN)  # [B,T]
        B, T = tokens.shape
        pos = torch.arange(T, device=tokens.device).unsqueeze(0).expand(B, T)
        x = self.embed(tokens) + self.pos(pos)
        x = self.drop(x)
        # Теперь: АДдитивная float-маска, как у ORT (для SDPA/Flash)
        attn_mask = (1.0 - pad_ok.float()).unsqueeze(1).unsqueeze(2) * -1e9  # [B,1,1,T]
        feats, aux = self.encoder(x, attn_mask=attn_mask, return_aux=True)

        logits = self.head(feats[:, 0, :])
        return (logits, aux) if return_aux else logits

# =========================
# Тренинг / Оценка
# =========================
@dataclass
class TrainerConfig:
    max_steps: int = 1500
    batch_size: int = 64
    learning_rate: float = 3e-4

def batch_acc(logits, y): return float((logits.argmax(-1) == y).float().mean().item())

@torch.no_grad()
def evaluate_split(model, data, criterion, device):
    model.eval()
    X, Y, D = data
    total_loss = total_correct = total = 0
    for i in range(0, len(Y), 512):
        xb = X[i:i+512].to(device); yb = Y[i:i+512].to(device)
        logits, _ = model(xb, return_aux=True)
        loss = criterion(logits, yb)
        total_loss += float(loss.item()) * yb.size(0)
        total_correct += int((logits.argmax(-1) == yb).sum().item())
        total += int(yb.size(0))
    return dict(loss=total_loss/max(1,total), acc=total_correct/max(1,total))

def train_model(model, name, cfg: TrainerConfig, train_data, val_data):
    model.to(DEVICE).train()
    X, Y, D = train_data
    ds = TensorDataset(X, Y, D)
    kwargs = dict(shuffle=True, batch_size=cfg.batch_size)
    if DEVICE.type == "cuda": kwargs.update(num_workers=2, pin_memory=True)
    dl = DataLoader(ds, **kwargs)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.learning_rate, weight_decay=0.01)
    crit = nn.CrossEntropyLoss()

    step = 0
    print(f"\n[{name}] device={DEVICE.type} | params ~ {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")
    log_every = 100  # как часто логировать шаги
    t_train_start = time.perf_counter()
    fwd_time_sum = 0.0
    bwd_time_sum = 0.0
    step_count_sum = 0
    while step < cfg.max_steps:
        for xb, yb, db in dl:
            xb = xb.to(DEVICE, non_blocking=True); yb = yb.to(DEVICE, non_blocking=True)
            opt.zero_grad(set_to_none=True)

            # включаем mixed precision на forward+loss
            if DEVICE.type == "cuda": torch.cuda.synchronize()
            t0 = time.perf_counter()
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                logits, _ = model(xb, return_aux=True)
                loss = crit(logits, yb)
            if DEVICE.type == "cuda": torch.cuda.synchronize()
            t1 = time.perf_counter()

            loss.backward()
            if DEVICE.type == "cuda": torch.cuda.synchronize()
            t2 = time.perf_counter()

            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            step += 1
            # агрегируем тайминги и периодически печатаем
            fwd_time_sum += (t1 - t0)
            bwd_time_sum += (t2 - t1)
            step_count_sum += 1
            if step % log_every == 0:
                fwd_time_sum = bwd_time_sum = 0.0
                step_count_sum = 0
            if step >= cfg.max_steps: break

    if DEVICE.type == "cuda": torch.cuda.synchronize()
    t_total = time.perf_counter() - t_train_start
    print(f"[{name}] Training time: {t_total:.2f} s")

    # финальная валидация
    # финальная валидация
    model.eval()

    if DEVICE.type == "cuda": torch.cuda.synchronize()
    t0 = time.perf_counter()
    train_m = evaluate_split(model, train_data, crit, DEVICE)
    if DEVICE.type == "cuda": torch.cuda.synchronize()
    t1 = time.perf_counter()
    val_m = evaluate_split(model, val_data, crit, DEVICE)
    if DEVICE.type == "cuda": torch.cuda.synchronize()
    t2 = time.perf_counter()

    print(f"{name} | TRAIN acc {train_m['acc']:.4f} | VAL acc {val_m['acc']:.4f} "
          f"| eval train {t1 - t0:.2f}s | eval val {t2 - t1:.2f}s")


@torch.no_grad()
def evaluate_speed_and_acc(model, data):
    model.to(DEVICE).eval()
    X, Y, D = data
    ds = TensorDataset(X, Y)
    kwargs = dict(batch_size=128)
    if DEVICE.type == "cuda": kwargs.update(num_workers=2, pin_memory=True)
    dl = DataLoader(ds, **kwargs)

    times = []; correct=0; total=0
    for xb, yb in dl:
        xb = xb.to(DEVICE, non_blocking=True); yb = yb.to(DEVICE, non_blocking=True)
        if DEVICE.type == "cuda": torch.cuda.synchronize()
        t0 = time.perf_counter()
        logits, _ = model(xb, return_aux=True)
        if DEVICE.type == "cuda": torch.cuda.synchronize()
        dt = (time.perf_counter() - t0)/xb.size(0)
        times.append(dt); correct += (logits.argmax(-1) == yb).sum().item(); total += yb.numel()
    return dict(acc=correct/max(1,total), avg_ms=(sum(times)/len(times))*1000.0)

def print_markdown_summary(base_res, ort_res):
    print("| Модель       | Общая точность | Скорость, мс |")
    print("|:-------------|:---------------|-------------:|")
    print(f"| **Baseline** | {base_res['acc']:.4f} | {base_res['avg_ms']:.3f} |")
    print(f"| **ORT**      | {ort_res['acc']:.4f} | {ort_res['avg_ms']:.3f} |")

def print_run_hparams(title: str, *, d_model, n_heads, n_layers, d_ff, max_len, vocab_size, d_head=None, ortho_q=None, ortho_k=None, ortho_v=None):
    print()
    print(f"### {title}")
    print("| Параметр   | Значение |")
    print("|:-----------|---------:|")
    print(f"| d_model    | {d_model} |")
    print(f"| n_heads    | {n_heads} |")
    print(f"| n_layers   | {n_layers} |")
    print(f"| d_ff       | {d_ff} |")
    print(f"| max_len    | {max_len} |")
    print(f"| vocab_size | {vocab_size} |")
    if d_head is not None: print(f"| d_head     | {d_head} |")
    if ortho_q is not None: print(f"| ortho_q    | {ortho_q} |")
    if ortho_k is not None: print(f"| ortho_k    | {ortho_k} |")
    if ortho_v is not None: print(f"| ortho_v    | {ortho_v} |")

def main():
    # Данные
    train_data, val_data, test_data = build_dataset()

    base_cfg = BaseCfg(
        n_layers=MODEL_LAYERS,
        d_model=MODEL_D,
        n_heads=MODEL_HEADS,
        d_ff=D_FF,
        attn_dropout=0.0,
        resid_dropout=0.0,
    )
    baseline_encoder = BaselineEncoderFast(base_cfg)
    baseline_model = BaselineSequenceClassifier(baseline_encoder, d_model=MODEL_D, n_classes=N_CLASSES)

    # ORT
    ort_cfg = OrtCfg(
        n_layers=MODEL_LAYERS, d_model=MODEL_D, n_heads=MODEL_HEADS, d_head=ORT_D_HEAD,
        d_ff=D_FF, attn_dropout=0.0, resid_dropout=0.0,
        ortho_q=ORT_ORTHO_Q, ortho_k=ORT_ORTHO_K, ortho_v=ORT_ORTHO_V
    )
    ort_encoder = OrtEncoder(ort_cfg)
    ort_model   = OrtSequenceClassifier(ort_encoder, d_model=MODEL_D, n_classes=N_CLASSES)

    # Обучение
    TRAIN = TrainerConfig()
    train_model(baseline_model, "Baseline", TRAIN, train_data, val_data)
    train_model(ort_model,      "ORT",      TRAIN, train_data, val_data)

    # Оценка
    base_res = evaluate_speed_and_acc(baseline_model, test_data)
    ort_res  = evaluate_speed_and_acc(ort_model, test_data)

    # Отчёт

    print_run_hparams("Настройки сети (run)",
                      d_model=MODEL_D, n_heads=MODEL_HEADS, n_layers=MODEL_LAYERS, d_ff=D_FF,
                      max_len=MAX_LEN, vocab_size=VOCAB_SIZE)
    print_run_hparams("Настройки сети (ORT)",
                      d_model=MODEL_D, n_heads=MODEL_HEADS, n_layers=MODEL_LAYERS, d_ff=D_FF,
                      max_len=MAX_LEN, vocab_size=VOCAB_SIZE,
                      d_head=ORT_D_HEAD, ortho_q=ORT_ORTHO_Q, ortho_k=ORT_ORTHO_K, ortho_v=ORT_ORTHO_V)
    print("")
    print_markdown_summary(base_res, ort_res)

if __name__ == "__main__":
    main()
