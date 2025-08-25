"""
edu_experiment_min.py — минимальный, самодостаточный эксперимент
с маленькими сетками: Baseline Transformer vs DAT (DynamicEncoder).

- Задача: классификация «скобочная последовательность сбалансирована / нет».
- Размеры сетей существенно уменьшены, чтобы baseline не «забыдовал» всё на 100% сразу.
- Никаких свипов и сложных режимов: один прогон, простой отчёт.

Запуск:
    python edu_experiment_min.py
"""

import math
import random
import time
from types import SimpleNamespace
from dataclasses import dataclass
from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader

# --- импортируем DAT из вашего файла ---
from dat_transformer import DynamicEncoder, EncoderConfig, LayerConfig

try:
    from tqdm import tqdm
except ImportError:
    # простая заглушка
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

# словарь (минимальный)
PAD_TOKEN = 0
CLS_TOKEN = 1
VOCAB = {
    PAD_TOKEN: "<pad>",
    CLS_TOKEN: "<cls>",
    2: "(",
    3: ")",
}
VOCAB_SIZE = max(VOCAB.keys()) + 1
MAX_LEN = 64             # включая CLS
N_CLASSES = 2            # сбалансировано/нет


# размеры маленьких сетей
MODEL_D = 64
MODEL_HEADS = 2
MODEL_LAYERS = 4
D_FF = 3 * MODEL_D

# обучение
@dataclass
class TrainerConfig:
    max_steps: int = 2000
    batch_size: int = 64
    learning_rate: float = 3e-4
    log_every: int = 100
    eval_every: int = 400


TRAIN_CFG = TrainerConfig()


# =========================
# Генерация данных
# =========================
def make_balanced_parens(length: int, rng: random.Random) -> str:
    """Генерим корректную скобочную последовательность длиной length (только '(' и ')')."""
    # генерация по стеку: равное число '(' и ')'
    half = length // 2
    s = "(" * half + ")" * half
    # перемешаем с ограничением корректности — простой подход:
    # строим через случайный проход, гарантируя что не закрываем раньше
    open_left = half
    close_left = half
    out = []
    depth = 0
    while open_left > 0 or close_left > 0:
        # можем поставить '(' всегда, если есть
        # можем поставить ')' только если depth>0
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
    """Портим корректную последовательность так, чтобы стала некорректной."""
    arr = list(s)
    # 50/50: либо переворачиваем один символ, либо выкидываем/добавляем скобку
    if rng.random() < 0.5:
        i = rng.randrange(len(arr))
        arr[i] = "(" if arr[i] == ")" else ")"
    else:
        if rng.random() < 0.5 and len(arr) > 2:
            # удалить символ
            i = rng.randrange(len(arr))
            del arr[i]
        else:
            # вставить случайную скобку в случайное место
            i = rng.randrange(len(arr)+1)
            arr.insert(i, "(" if rng.random() < 0.5 else ")")
    return "".join(arr)


def max_nesting_depth(s: str) -> int:
    depth = 0
    max_d = 0
    for ch in s:
        if ch == "(":
            depth += 1
            max_d = max(max_d, depth)
        elif ch == ")":
            depth -= 1
            # если ушли <0 — это тоже информация о сложности, но глубину не увеличиваем
    return max_d


def tokenize(s: str) -> List[int]:
    return [2 if ch == "(" else 3 for ch in s]


def pad_to_len(ids: List[int], L: int) -> List[int]:
    if len(ids) >= L:
        return ids[:L]
    return ids + [PAD_TOKEN] * (L - len(ids))

def _get_n_layers(encoder, default_layers: int = MODEL_LAYERS) -> int:
    if hasattr(encoder, "n_layers"):
        return int(encoder.n_layers)
    cfg = getattr(encoder, "cfg", None)
    if cfg is not None and hasattr(cfg, "n_layers"):
        return int(cfg.n_layers)
    return int(default_layers)

def build_dataset(n_train=8000, n_val=2000, n_test=2000, min_len=10, max_len=40, seed=SEED):
    rng = random.Random(seed)
    def make_split(n):
        seqs = []
        labels = []
        depths = []
        for _ in tqdm(range(n), desc="Генерация данных", mininterval=0.1):
            L_body = rng.randint(min_len, max_len)  # длина без CLS
            if L_body % 2 == 1:
                L_body += 1  # чётная
            s_good = make_balanced_parens(L_body, rng)
            if rng.random() < 0.5:
                s = s_good
                y = 1  # balanced
            else:
                s = corrupt_parens(s_good, rng)
                y = 0  # unbalanced
            d = max_nesting_depth(s)
            ids = [CLS_TOKEN] + tokenize(s)
            ids = pad_to_len(ids, MAX_LEN)
            seqs.append(ids)
            labels.append(y)
            depths.append(d)
        return (
            torch.tensor(seqs, dtype=torch.long),
            torch.tensor(labels, dtype=torch.long),
            torch.tensor(depths, dtype=torch.long),
        )
    print("--- Фаза 1: Подготовка ---")
    print("Генерация нового набора данных...")
    train = make_split(n_train)
    val   = make_split(n_val)
    test  = make_split(n_test)
    return train, val, test


# =========================
# Baseline Encoder (маленький)
# =========================
class BaselineEncoder(nn.Module):
    """Простой TransformerEncoder без всяких адаптаций."""
    def __init__(self, n_layers: int, d_model: int, n_heads: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=d_ff,
            dropout=dropout, batch_first=True, norm_first=True
        )
        self.enc = nn.TransformerEncoder(layer, num_layers=n_layers)
        self.ln = nn.LayerNorm(d_model)
        self.n_layers = n_layers
        self.cfg = SimpleNamespace(n_layers=n_layers)

    def forward(self, x: torch.Tensor, key_padding_mask: torch.Tensor = None, return_aux: bool = True):
        # key_padding_mask: [B, T] (True=игнорировать)
        feats = self.enc(x, src_key_padding_mask=key_padding_mask)
        feats = self.ln(feats)
        aux = {}
        return (feats, aux) if return_aux else feats


# =========================
# Классификатор поверх энкодера
# =========================
class SequenceClassifier(nn.Module):
    def __init__(self, encoder, d_model, n_classes):
        super().__init__()
        self.encoder = encoder
        self.embedding = nn.Embedding(VOCAB_SIZE, d_model, padding_idx=PAD_TOKEN)
        self.head = nn.Linear(d_model, n_classes)
        self.pos_emb = nn.Embedding(MAX_LEN, d_model)
        self.drop_in = nn.Dropout(0.1)

    def forward(self, tokens, return_aux=False):
        # True там, где НЕ паддинг
        padding_mask = (tokens != PAD_TOKEN)
        B, T = tokens.size()
        pos = torch.arange(T, device=tokens.device).unsqueeze(0).expand(B, T)
        x = self.embedding(tokens) + self.pos_emb(pos)
        x = self.drop_in(x)

        if isinstance(self.encoder, DynamicEncoder):
            # Аддитивная маска для паддингов: [B,1,1,T]
            # (Без expand; этого достаточно. Число -1e9 стабильнее finfo.min)
            attn_mask = (1.0 - padding_mask.float()).unsqueeze(1).unsqueeze(2) * -1e9
            feats, aux = self.encoder(x, attn_mask=attn_mask, return_aux=True)
        else:
            # для baseline: булева маска [B,T], где True=игнорировать (то есть паддинг)
            key_padding_mask = ~padding_mask
            feats, aux = self.encoder(x, key_padding_mask=key_padding_mask, return_aux=True)

        cls = feats[:, 0, :]  # берём CLS

        logits = self.head(cls)
        return (logits, aux) if return_aux else logits


# =========================
# Обучение / Оценка
# =========================
@torch.no_grad()
def _avg_depth_from_aux(aux, L: int) -> float:
    """Ожидаемая глубина по батчу. Для Baseline вернём L."""
    if not isinstance(aux, dict):
        return float(L)
    hp = aux.get("halting_ps", None)  # ожидаем [L,B,T,1] или список длины L
    if isinstance(hp, torch.Tensor) and hp.dim() >= 3:
        # [L,B,T,1] -> суммируем по слоям и усредняем по B,T
        exp_layers = hp.sum(dim=0)  # [B,T,1]
        return float(exp_layers.mean().item())
    if isinstance(hp, (list, tuple)) and len(hp) > 0 and isinstance(hp[0], torch.Tensor):
        # список p_i [B,T,1]
        exp_layers = torch.stack(hp, dim=0).sum(dim=0).mean()
        return float(exp_layers.item())
    return float(L)

def _batch_acc(logits: torch.Tensor, y: torch.Tensor) -> float:
    return float((logits.argmax(dim=-1) == y).float().mean().item())

@torch.no_grad()
def evaluate_split(model, data_tuple, criterion, batch_size=512, device=None):
    """Вычисляем val-loss/acc и среднюю глубину на датасете (tuple из (seqs, labels, complexity))."""
    device = device or next(model.parameters()).device
    model.eval()
    X, Y, _ = data_tuple
    N = Y.size(0)
    total_loss = 0.0
    total_correct = 0
    total_count = 0
    depth_sum = 0.0
    for i in range(0, N, batch_size):
        xb = X[i:i+batch_size].to(device)
        yb = Y[i:i+batch_size].to(device)
        logits, aux = model(xb, return_aux=True)
        loss = criterion(logits, yb)
        total_loss += float(loss.item()) * yb.size(0)
        total_correct += int((logits.argmax(-1) == yb).sum().item())
        total_count += int(yb.size(0))
        depth_sum += _avg_depth_from_aux(aux, _get_n_layers(model.encoder)) * yb.size(0)

    return {
        "loss": total_loss / max(1, total_count),
        "acc": total_correct / max(1, total_count),
        "avg_depth": depth_sum / max(1, total_count),
    }

def train_model(model: nn.Module, name: str, cfg: TrainerConfig, train_data, val_data):
    model.to(DEVICE).train()
    train_x, train_y, train_d = train_data
    val_x, val_y, _ = val_data

    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.learning_rate, weight_decay=0.01)
    criterion = nn.CrossEntropyLoss()

    # DataLoader (фикс. длина, так что можно без кастомного collate); включаем сложность в батч
    kwargs = {}
    if DEVICE.type == "cuda":
        kwargs = dict(num_workers=2, pin_memory=True)
    # включим сложность в батч (для бюджет-регуляции глубины только в DAT)
    train_ds = TensorDataset(train_x, train_y, train_d)
    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True, **kwargs)

    print(f"\n--- Обучение модели: {name} ---")
    step = 0
    last_log = 0
    pbar = tqdm(total=cfg.max_steps, mininterval=0.1)
    while step < cfg.max_steps:
        for xb, yb, db in train_loader:
            xb = xb.to(DEVICE, non_blocking=True)
            yb = yb.to(DEVICE, non_blocking=True)
            db = db.to(DEVICE, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            # просим aux для DAT
            logits, aux = model(xb, return_aux=True)
            loss = criterion(logits, yb)

            # --- Deep Supervision: учим каждый слой классифицировать (только для DAT) ---
            if isinstance(aux, dict) and "feats_per_layer" in aux and "halting_ps" in aux:
                try:
                    ps = aux["halting_ps"]
                    if isinstance(ps, list):
                        ps = torch.stack(ps, dim=0)   # [L,B,T,1]
                    # Берём вероятность остановки по CLS-токену
                    p_cls = ps[..., 0, 0]            # [L,B]
                    L, B = p_cls.shape
                    one_minus = (1.0 - p_cls)
                    # w_i = p_i * Π_{j<i}(1-p_j)
                    prefix = torch.ones(1, B, device=p_cls.device, dtype=p_cls.dtype)
                    prod_prev = torch.cumprod(torch.cat([prefix, one_minus[:-1]], dim=0), dim=0)  # [L,B]
                    w = p_cls * prod_prev
                    w = w / (w.sum(dim=0, keepdim=True) + 1e-8)  # нормировка по батчу

                    ds = 0.0
                    for li, feats_i in enumerate(aux["feats_per_layer"]):
                        logits_i = model.head(feats_i[:, 0, :])  # CLS
                        ce_i = F.cross_entropy(logits_i, yb, reduction="none")  # [B]
                        ds += (ce_i * w[li]).mean()
                    # Смешиваем с основным лоссом (легко крутить 0.2–0.4)
                    loss = 0.5 * loss + 0.5 * ds
                except Exception:
                    # на всякий — не роняем обучение, если что-то не так с aux
                    pass
            # ---------------------------------------------------------------------------
            # --- Difficulty-aware бюджет глубины ---
            # exp_depth ~= sum_i p_halt_i; целевую глубину зададим линейно от сложности.
            if isinstance(aux, dict) and "halting_ps" in aux:
                ps = aux["halting_ps"]
                if isinstance(ps, list):
                    ps = torch.stack(ps, dim=0)  # [L,B,T,1]
                # ожидаемая глубина по сэмплу (средняя по токенам)
                exp_depth_b = ps.sum(dim=0).mean(dim=1).squeeze(-1)  # [B]
                L = _get_n_layers(model.encoder)
                # сложность db (макс. вложенность) приводим к [1..L]
                tgt = 1.0 + (db.float().clamp(0, 8) / 8.0) * max(L - 1, 1)  # [B]
                loss_depth = F.mse_loss(exp_depth_b, tgt)
                loss = loss + 0.05 * loss_depth

            # тёплый старт: первые 20% шагов выход практически запрещён
            if isinstance(getattr(model, "encoder", None), DynamicEncoder) and model.encoder.cfg.early_exit:
                warm = int(cfg.max_steps * 0.2)
                # минимум слоёв до выхода + терпение по стабильности логитов (если поддерживается)
                if hasattr(model.encoder.cfg, "min_layers_before_exit"):
                    model.encoder.cfg.min_layers_before_exit = 2
                if hasattr(model.encoder.cfg, "exit_patience"):
                    model.encoder.cfg.exit_patience = 1
                model.encoder.cfg.exit_threshold = 1.5 if step < warm else 0.98

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            # до цикла
            log_every = 501  # как часто печатать свод
            run_loss = run_correct = run_count = 0
            run_depth_sum = 0.0

            # внутри батч-цикла ПОСЛЕ optimizer.step()
            run_loss += float(loss.item()) * yb.size(0)
            run_correct += int((logits.argmax(-1) == yb).sum().item())
            run_count += int(yb.size(0))

            # средняя глубина на батч
            L_enc = _get_n_layers(model.encoder)
            if isinstance(getattr(model, "encoder", None), DynamicEncoder):
                run_depth_sum += _avg_depth_from_aux(aux, L_enc) * yb.size(0)
            else:
                run_depth_sum += float(L_enc) * yb.size(0)

            # каждые log_every шагов — валидационный замер + консоль
            if (step + 1) % log_every == 0:
                train_loss = run_loss / max(1, run_count)
                train_acc = run_correct / max(1, run_count)
                train_depth = run_depth_sum / max(1, run_count)

                model.eval()
                val_metrics = evaluate_split(model, val_data, criterion,
                                             batch_size=cfg.batch_size, device=DEVICE)
                model.train()

                print(f"{name} | step {step + 1:>5} | "
                      f"TRAIN loss {train_loss:.4f} acc {train_acc:.4f} depth {train_depth:.2f} | "
                      f"VAL loss {val_metrics['loss']:.4f} acc {val_metrics['acc']:.4f} depth {val_metrics['avg_depth']:.2f}")

                # сбросить окна усреднения
                run_loss = run_correct = run_count = 0
                run_depth_sum = 0.0

            step += 1
            if step - last_log >= cfg.log_every or step == cfg.max_steps:
                pbar.set_description(f"{name} | step {step} | loss {loss.item():.3f}")
                last_log = step
            pbar.update(1)
            if step >= cfg.max_steps:
                break
    pbar.close()

    # --- Финальная сводка train vs val ---------------------------------------
    model_was_train = model.training
    model.eval()  # корректный режим для Dropout/Norm во время валидации
    # (eval и no_grad вместе — так и принято: отключаем Autograd и слой/dropout-режимы)
    # см. PyTorch best practice: использовать и eval(), и torch.no_grad() при валидации. [1][2]

    # Для DAT: на оценке можно явно включить ранний выход, если ты так решил.
    was_early = None
    if isinstance(getattr(model, "encoder", None), DynamicEncoder):
        was_early = model.encoder.cfg.early_exit
        model.encoder.cfg.early_exit = True  # или оставь как есть — зависит от твоего протокола оценки

    with torch.no_grad():
        train_metrics = evaluate_split(model, train_data, criterion,
                                       batch_size=cfg.batch_size, device=DEVICE)
        val_metrics   = evaluate_split(model, val_data,   criterion,
                                       batch_size=cfg.batch_size, device=DEVICE)

    # Вернуть флаг раннего выхода как был
    if isinstance(getattr(model, "encoder", None), DynamicEncoder) and was_early is not None:
        model.encoder.cfg.early_exit = was_early

    # Вернуть режим обучения, если нужно продолжать тренинг где-то дальше
    if model_was_train:
        model.train()

    print(f"{name} | FINAL | "
          f"TRAIN loss {train_metrics['loss']:.4f} acc {train_metrics['acc']:.4f} depth {train_metrics['avg_depth']:.2f} || "
          f"VAL loss {val_metrics['loss']:.4f} acc {val_metrics['acc']:.4f} depth {val_metrics['avg_depth']:.2f}")
    # --------------------------------------------------------------------------



@torch.no_grad()
def evaluate_model(model: nn.Module, test_data):
    model.to(DEVICE).eval()
    test_x, test_y, test_depth = test_data

    # эвристика: «легкие» и «сложные» по глубине
    easy_mask = test_depth <= 2
    hard_mask = test_depth >= 4

    def eval_split(mask: torch.Tensor):
        x = test_x[mask]
        y = test_y[mask]
        if len(x) == 0:
            return dict(acc=None, avg_time_ms=None, depth=None)

        # батчим
        ds = TensorDataset(x, y)
        kwargs = {}
        if DEVICE.type == "cuda":
            kwargs = dict(num_workers=2, pin_memory=True)
        dl = DataLoader(ds, batch_size=128, **kwargs)

        correct = 0
        total = 0
        times = []
        depth_sum = 0.0
        depth_count = 0


        for xb, yb in dl:
            xb = xb.to(DEVICE, non_blocking=True)
            yb = yb.to(DEVICE, non_blocking=True)
            if DEVICE.type == "cuda": torch.cuda.synchronize()
            t0 = time.perf_counter()
            logits, aux = model(xb, return_aux=True)
            if DEVICE.type == "cuda": torch.cuda.synchronize()
            dt = time.perf_counter() - t0
            times.append(dt / xb.size(0))

            pred = logits.argmax(dim=-1)
            correct += (pred == yb).sum().item()
            total += yb.numel()
            # накапливаем ожидаемую глубину по батчам (только для DAT)
            if isinstance(model.encoder, DynamicEncoder) and isinstance(aux, dict):
                ps = aux.get("halting_ps")
                if isinstance(ps, list):
                    ps = torch.stack(ps, dim=0)
                if isinstance(ps, torch.Tensor):
                    depth_sum += float(ps.sum(dim=0).mean().item()) * yb.size(0)
                    depth_count += int(yb.size(0))

        avg_depth = (depth_sum / max(1, depth_count)) if depth_count > 0 else None
        acc = correct / max(1, total)
        avg_ms = (sum(times) / len(times)) * 1000.0
        return dict(acc=acc, avg_time_ms=avg_ms, depth=avg_depth)

    all_res  = eval_split(torch.ones_like(test_y, dtype=torch.bool))
    easy_res = eval_split(easy_mask)
    hard_res = eval_split(hard_mask)

    return {
        "Общая точность": all_res["acc"],
        "Легкие":  {"acc": easy_res["acc"], "avg_time_ms": easy_res["avg_time_ms"], "avg_depth": easy_res["depth"]},
        "Сложные":{"acc": hard_res["acc"], "avg_time_ms": hard_res["avg_time_ms"], "avg_depth": hard_res["depth"]},
    }


# =========================
# MAIN
# =========================
def _fmt_pct(x: float) -> str:
    return f"{100.0 * x:.1f}%"

def _fmt_ms(x_ms: float) -> str:
    # у тебя уже миллисекунды в словаре → просто печатаем
    return f"{x_ms:.0f} мс"

def _fmt_depth(d: float, L: int) -> str:
    return f"{d:.1f} / {L}"

def _bold(s: str, cond: bool) -> str:
    return f"**{s}**" if cond else s

def print_markdown_summary(baseline: dict, dat: dict, L: int):
    def acc(x):   return f"{float(x):.4f}"
    def depth(x): return "—" if x is None else f"{float(x):.3f}"
    def ms(x):    return f"{float(x):.3f}"

    row_base = [
        "**Baseline**",
        acc(baseline["Общая точность"]),
        acc(baseline["Легкие"]["acc"]),
        acc(baseline["Сложные"]["acc"]),
        "—",
        "—",
        ms(baseline["Легкие"]["avg_time_ms"]),
        ms(baseline["Сложные"]["avg_time_ms"]),
    ]
    row_dat = [
        "**DAT**",
        acc(dat["Общая точность"]),
        acc(dat["Легкие"]["acc"]),
        acc(dat["Сложные"]["acc"]),
        depth(dat["Легкие"].get("avg_depth")),
        depth(dat["Сложные"].get("avg_depth")),
        ms(dat["Легкие"]["avg_time_ms"]),
        ms(dat["Сложные"]["avg_time_ms"]),
    ]

    print("| Модель       | Общая точность | Точность (Легкие) | Точность (Сложные) | Глубина (Легкие) | Глубина (Сложные) | Скорость (Легкие), мс | Скорость (Сложные), мс |")
    print("|:-------------|:---------------|:------------------|:-------------------|:-----------------|:------------------|----------------------:|-----------------------:|")
    print(f"| {' | '.join(row_base)} |")
    print(f"| {' | '.join(row_dat)} |")

def print_run_hparams(
    d_model: int,
    n_heads: int,
    n_layers: int,
    d_ff: int,
    max_len: int,
    vocab_size: int,
    title: str = "Настройки сети (run)"
):
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


def main():
    # 1) Данные
    train_data, val_data, test_data = build_dataset(
        n_train=8000, n_val=2000, n_test=2000, min_len=10, max_len=40, seed=SEED
    )

    # 2) Baseline (маленький)
    baseline_encoder = BaselineEncoder(
        n_layers=MODEL_LAYERS, d_model=MODEL_D, n_heads=MODEL_HEADS, d_ff=D_FF
    )
    baseline_model = SequenceClassifier(baseline_encoder, d_model=MODEL_D, n_classes=N_CLASSES)

    # 3) DAT (маленький, без «дорогих» фич)
    dat_layer_cfg = LayerConfig(
        d_model=MODEL_D,
        n_heads=MODEL_HEADS,
        d_head=MODEL_D // MODEL_HEADS,
        d_ff=D_FF,
        use_parametric_scores=False,
        mem_topk=0,
        head_gate_hidden=None,  # ← отключили «ширину», чтобы не ломать маленькую сеть
        halt_gate_hidden=16,  # ← помягче для D=8
        halt_bias_init=-1.5,
        halt_from_cls=True,   # ← p_halt от CLS
        halt_temp=1.3,  # ← чуть сгладим сигмоиду останова
    )

    dat_enc_cfg = EncoderConfig(
        n_layers=MODEL_LAYERS,
        layer=dat_layer_cfg,
        early_exit=True,
        exit_threshold=0.98,
        # → тонкая настройка выхода (если поля поддерживаются в твоём EncoderConfig):
        min_layers_before_exit=2,
        exit_patience=1
    )
    dat_encoder = DynamicEncoder(dat_enc_cfg)
    dat_model   = SequenceClassifier(dat_encoder, d_model=MODEL_D, n_classes=N_CLASSES)

    # 4) Обучение
    train_model(baseline_model, "Baseline", TRAIN_CFG, train_data, val_data)
    # Отключаем ранний выход на обучении
    if isinstance(dat_model.encoder, DynamicEncoder):
        dat_model.encoder.cfg.early_exit = False
    train_model(dat_model, "DAT", TRAIN_CFG, train_data, val_data)

    # 5) Оценка (включим ранний выход только на инференсе)
    if isinstance(dat_model.encoder, DynamicEncoder):
        dat_model.encoder.cfg.early_exit = True # Флаг раннего выхода True False

    base_res = evaluate_model(baseline_model, test_data)
    dat_res  = evaluate_model(dat_model, test_data)

    # 6) Отчёт
    print_markdown_summary(base_res, dat_res, L=MODEL_LAYERS)
    print_run_hparams(
        d_model=MODEL_D,
        n_heads=MODEL_HEADS,
        n_layers=MODEL_LAYERS,
        d_ff=D_FF,
        max_len=MAX_LEN,
        vocab_size=VOCAB_SIZE,
    )


if __name__ == "__main__":
    main()
