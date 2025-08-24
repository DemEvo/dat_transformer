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
MODEL_D = 32
MODEL_HEADS = 2
MODEL_LAYERS = 4
D_FF = 2 * MODEL_D

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
def train_model(model: nn.Module, name: str, cfg: TrainerConfig, train_data, val_data):
    model.to(DEVICE).train()
    train_x, train_y, _ = train_data
    val_x, val_y, _ = val_data

    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.learning_rate, weight_decay=0.01)
    criterion = nn.CrossEntropyLoss()

    # DataLoader (фикс. длина, так что можно без кастомного collate)
    kwargs = {}
    if DEVICE.type == "cuda":
        kwargs = dict(num_workers=2, pin_memory=True)
    train_ds = TensorDataset(train_x, train_y)
    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True, **kwargs)

    print(f"\n--- Обучение модели: {name} ---")
    step = 0
    last_log = 0
    pbar = tqdm(total=cfg.max_steps, mininterval=0.1)
    while step < cfg.max_steps:
        for xb, yb in train_loader:
            xb = xb.to(DEVICE, non_blocking=True)
            yb = yb.to(DEVICE, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            # просим aux для DAT
            logits, aux = model(xb, return_aux=True)
            loss = criterion(logits, yb)

            # тёплый старт: первые 20% шагов выход практически запрещён
            if isinstance(getattr(model, "encoder", None), DynamicEncoder) and model.encoder.cfg.early_exit:
                warm = int(cfg.max_steps * 0.2)
                model.encoder.cfg.exit_threshold = 1.2 if step < warm else 0.98

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            step += 1
            if step - last_log >= cfg.log_every or step == cfg.max_steps:
                pbar.set_description(f"{name} | step {step} | loss {loss.item():.3f}")
                last_log = step
            pbar.update(1)
            if step >= cfg.max_steps:
                break
    pbar.close()


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

        # глубина (только для DAT; если нет — None)
        avg_depth = None
        if isinstance(model.encoder, DynamicEncoder) and isinstance(aux, dict) and ("halting_ps" in aux):
            # ожидаемая глубина = сумма p_halt по слоям
            ps = aux["halting_ps"]  # ожидаем [L,B,T,1] или [L,B,1,1]
            if isinstance(ps, list):  # иногда список по слоям
                ps = torch.stack(ps, dim=0)
            avg_depth = float(ps.sum(dim=0).mean().item())

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
        d_ff=4 * MODEL_D,
        use_parametric_scores=False,
        mem_topk=0,
        head_gate_hidden=None,  # ← отключили «ширину», чтобы не ломать маленькую сеть
        halt_gate_hidden=16,  # ← помягче для D=8
        halt_bias_init=0.0,
        halt_from_cls=True,  # ← только для DAT на этой задаче
    )

    dat_enc_cfg = EncoderConfig(
        n_layers=MODEL_LAYERS,
        layer=dat_layer_cfg,
        early_exit=True,
        exit_threshold=0.98,
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
        dat_model.encoder.cfg.early_exit = False # Флаг раннего выхода True False

    base_res = evaluate_model(baseline_model, test_data)
    dat_res  = evaluate_model(dat_model, test_data)

    # 6) Отчёт
    # print("MODEL_D:",MODEL_D)
    # print("MODEL_HEADS:",MODEL_HEADS)
    # print("MODEL_LAYERS:",MODEL_LAYERS)
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
