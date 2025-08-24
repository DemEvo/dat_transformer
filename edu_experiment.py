"""
Эксперимент: Проверка баланса скобок (Корректная постановка)
=============================================================

Цель: Корректно сравнить производительность и эффективность классического
Трансформера (Baseline) и DAT на задаче, которую обе модели способны решить.

Задача: Классификация последовательности на основе сбалансированности скобок.
Сложность варьируется в зависимости от структуры последовательности.

Гипотеза: DAT покажет схожее качество, но будет значительно эффективнее
на "лёгких" примерах (например, с ошибкой в начале последовательности).

Как запустить:
1. Убедитесь, что файл dat_transformer.py находится в той же директории.
2. Запустите скрипт: python experiment_bracket_balance.py
3. Скрипт автоматически сгенерирует данные, обучит обе модели,
   проведет оценку и выведет итоговую сравнительную таблицу.
"""
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import DataLoader, TensorDataset

import random
import time
import numpy as np
from dataclasses import dataclass
from tqdm import tqdm
import os

# --- Импорт архитектуры DAT ---
try:
    from dat_transformer import DynamicEncoder, EncoderConfig, LayerConfig
except ImportError:
    print("Ошибка: файл dat_transformer.py не найден. Пожалуйста, поместите его в ту же директорию.")
    exit()

# === Константы и сиды ===
N_CLASSES = 2
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

# --- 1. Генерация данных ---

# Словарь: 0='(', 1=')', 2-27=шум (буквы), 28=[CLS], 29=[SEP], 30=[PAD]
VOCAB_SIZE = 31
OPEN_BR, CLOSE_BR = 0, 1
CLS_TOKEN, SEP_TOKEN, PAD_TOKEN = 28, 29, 30


def is_balanced(seq):
    stack = []
    for token in seq:
        if token == OPEN_BR:
            stack.append(token)
        elif token == CLOSE_BR:
            if not stack:
                return False
            stack.pop()
    return not stack


def generate_balanced_sequence(length):
    seq = []
    balance = 0
    for _ in range(length):
        # Добавляем шум с вероятностью 50%
        if random.random() < 0.5:
            seq.append(random.randint(2, 27))
            continue

        if balance == 0 or (random.random() < 0.6 and balance < length / 2):
            seq.append(OPEN_BR)
            balance += 1
        else:
            seq.append(CLOSE_BR)
            balance -= 1

    while balance > 0:
        seq.append(CLOSE_BR)
        balance -= 1
    return seq


def generate_unbalanced_sequence(length):
    seq = generate_balanced_sequence(length)
    # Гарантированно делаем последовательность несбалансированной
    if random.random() < 0.5 and len(seq) > 0:
        # Удаляем случайную скобку
        bracket_indices = [i for i, t in enumerate(seq) if t in (OPEN_BR, CLOSE_BR)]
        if bracket_indices:
            del seq[random.choice(bracket_indices)]
    else:
        # Добавляем лишнюю скобку
        seq.insert(random.randint(0, len(seq)), random.choice([OPEN_BR, CLOSE_BR]))
    return seq


def generate_dataset(num_samples, min_len=15, max_len=60):
    sequences, labels, complexities = [], [], []
    for _ in tqdm(range(num_samples), desc="Генерация данных"):
        length = random.randint(min_len, max_len)
        if random.random() < 0.5:
            seq_tokens = generate_balanced_sequence(length)
            label = 1
        else:
            seq_tokens = generate_unbalanced_sequence(length)
            # Перепроверяем, т.к. случайное удаление/добавление могло сбалансировать
            label = 1 if is_balanced(seq_tokens) else 0

        # Оцениваем сложность: 1.0 - самая простая (ошибка в начале), 0.0 - самая сложная
        complexity = 0.5  # Нейтральное значение для сбалансированных
        if label == 0:
            balance = 0
            for i, token in enumerate(seq_tokens):
                if token == OPEN_BR:
                    balance += 1
                elif token == CLOSE_BR:
                    balance -= 1
                if balance < 0:  # Нашли ошибку
                    complexity = 1.0 - (i / len(seq_tokens))
                    break
            if balance != 0 and complexity == 0.5:  # Ошибка в конце
                complexity = 0.0

        sequence = [CLS_TOKEN] + seq_tokens + [SEP_TOKEN]
        sequences.append(torch.tensor(sequence, dtype=torch.long))
        labels.append(torch.tensor(label, dtype=torch.long))
        complexities.append(complexity)

    return sequences, torch.stack(labels), torch.tensor(complexities)


# --- 2. Определение Моделей ---

class StandardTransformerLayer(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, dropout=0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(embed_dim=d_model, num_heads=n_heads, dropout=dropout, batch_first=True)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model)
        )
        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, key_padding_mask=None):
        attn_output, _ = self.attn(x, x, x, key_padding_mask=key_padding_mask)
        x = self.ln1(x + self.dropout(attn_output))
        ffn_output = self.ffn(x)
        x = self.ln2(x + self.dropout(ffn_output))
        return x


class BaselineEncoder(nn.Module):
    def __init__(self, n_layers, d_model, n_heads, d_ff):
        super().__init__()
        self.layers = nn.ModuleList([
            StandardTransformerLayer(d_model, n_heads, d_ff) for _ in range(n_layers)
        ])
        self.n_layers = n_layers

    def forward(self, x, attn_mask=None, return_aux=False):
        key_padding_mask = (attn_mask == 0) if attn_mask is not None else None
        for layer in self.layers:
            x = layer.forward(x, key_padding_mask=key_padding_mask)
        return (x, {}) if return_aux else x


class SequenceClassifier(nn.Module):
    def __init__(self, encoder, d_model, n_classes):
        super().__init__()
        self.encoder = encoder
        self.head = nn.Linear(d_model, n_classes)
        self.embedding = nn.Embedding(VOCAB_SIZE, d_model, padding_idx=PAD_TOKEN)

    def forward(self, tokens, return_aux=False):
        padding_mask = (tokens != PAD_TOKEN)
        x = self.embedding(tokens)

        if isinstance(self.encoder, DynamicEncoder):
            neg_inf = torch.finfo(x.dtype).min
            # [B,1,1,T] → расширяем до [B,1,T,T], чтобы маска соответствовала [Tq,Tk]
            attn_mask_4d = (1.0 - padding_mask.float()).unsqueeze(1).unsqueeze(2) * neg_inf  # [B,1,1,T]
            attn_mask_4d = attn_mask_4d.expand(tokens.size(0), 1, tokens.size(1), tokens.size(1))  # [B,1,T,T]
            feats, aux = self.encoder(x, attn_mask=attn_mask_4d, return_aux=True)
        else:
            feats, aux = self.encoder(x, attn_mask=padding_mask, return_aux=True)

        cls_token_feats = feats[:, 0, :]
        logits = self.head(cls_token_feats)

        return (logits, aux) if return_aux else logits


# --- 3. Цикл Обучения и Оценки ---

@dataclass
class TrainerConfig:
    max_steps: int = 6000
    batch_size: int = 128
    learning_rate: float = 3e-4
    log_every: int = 100
    eval_every: int = 500


def pad_sequences(sequences, max_len=None):
    if max_len is None:
        max_len = max(len(s) for s in sequences)
    padded = torch.full((len(sequences), max_len), fill_value=PAD_TOKEN, dtype=torch.long)
    for i, seq in enumerate(sequences):
        padded[i, :len(seq)] = seq
    return padded


def train(model, model_name, config, train_data, val_data):
    print(f"\n--- Обучение модели: {model_name} ---")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    # Для честной тренировки: ранний выход только на инференсе
    if isinstance(model.encoder, DynamicEncoder):
        model.encoder.cfg.early_exit = False
    optimizer = AdamW(model.parameters(), lr=config.learning_rate)
    loss_fn = nn.CrossEntropyLoss()

    train_seqs, train_labels, _ = train_data
    val_seqs, val_labels, _ = val_data

    train_padded = pad_sequences(train_seqs)
    val_padded = pad_sequences(val_seqs)

    train_dataset = TensorDataset(train_padded, train_labels)
    val_dataset = TensorDataset(val_padded, val_labels)

    loader_kwargs = {}
    if device.type == "cuda":
        loader_kwargs = dict(num_workers=2, pin_memory=True)

    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, **loader_kwargs)
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size, **loader_kwargs)

    best_val_acc = 0.0

    pbar = tqdm(range(config.max_steps), desc=f"Обучение {model_name}")
    step = 0
    done = False
    while not done:
        for batch in train_loader:
            if step >= config.max_steps:
                done = True
                break

            model.train()
            tokens, labels = [t.to(device) for t in batch]

            if isinstance(model.encoder, DynamicEncoder):
                logits, aux = model(tokens, return_aux=True)
                loss = loss_fn(logits, labels)
                if model.encoder.cfg.layer.halt_gate_hidden is not None:
                    depth_penalty = 0.005 * aux['halting_ps'].sum(dim=0).mean()
                    loss += depth_penalty
            else:
                logits = model(tokens)
                loss = loss_fn(logits, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if step % config.log_every == 0: pbar.set_postfix({"loss": loss.item()})

            if step % config.eval_every == 0:
                model.eval()
                correct, total = 0, 0
                with torch.no_grad():
                    for val_batch in val_loader:
                        val_tokens, val_labels = [t.to(device) for t in val_batch]
                        val_logits = model(val_tokens)
                        preds = torch.argmax(val_logits, dim=1)
                        correct += (preds == val_labels).sum().item()
                        total += val_labels.size(0)
                val_acc = correct / total
                if val_acc > best_val_acc:
                    best_val_acc = val_acc
                    torch.save(model.state_dict(), f"{model_name}_best.pt")
                    pbar.set_postfix({"loss": loss.item(), "best_val_acc": best_val_acc})

            step += 1
            pbar.update(1)

    model.load_state_dict(torch.load(f"{model_name}_best.pt"))
    return model


@torch.no_grad()
def evaluate(model, test_data):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_cuda = (device.type == "cuda")
    model.to(device)
    model.eval()
    # Включаем ранний выход только на время оценки, если это DAT
    was_early = False
    if isinstance(model.encoder, DynamicEncoder):
        was_early = model.encoder.cfg.early_exit
        model.encoder.cfg.early_exit = True

    test_seqs, test_labels, test_complexity = test_data

    easy_indices = torch.where(test_complexity >= 0.8)[0]
    hard_indices = torch.where(test_complexity <= 0.2)[0]

    results = {}
    total_correct, total_count = 0, 0

    for name, indices in [("Легкие", easy_indices), ("Сложные", hard_indices)]:
        if len(indices) == 0: continue

        subset_seqs = [test_seqs[i] for i in indices]
        subset_labels = test_labels[indices]
        padded_seqs = pad_sequences(subset_seqs).to(device)
        labels = subset_labels.to(device)

        timings = []
        for _ in range(30):
            if use_cuda: torch.cuda.synchronize()
            start_iter = time.perf_counter()
            logits, aux = model(padded_seqs, return_aux=True)
            if use_cuda: torch.cuda.synchronize()
            timings.append(time.perf_counter() - start_iter)
        avg_time_ms = (sum(timings[5:]) / max(1, len(timings[5:]))) * 1000 / len(padded_seqs)

        preds = torch.argmax(logits, dim=1)
        correct = (preds == labels).sum().item()
        total = labels.size(0)
        total_correct += correct
        total_count += total

        if isinstance(model.encoder, DynamicEncoder):
            depth = aux.get('halting_ps').sum(dim=0).mean().item()
        else:
            depth = model.encoder.n_layers

        results[name] = {
            "accuracy": correct / total,
            "avg_time_ms": avg_time_ms,
            "avg_depth": depth
        }

    results["Общая точность"] = total_correct / total_count if total_count > 0 else 0
    # вернуть флаг как было (только для DAT)
    if isinstance(getattr(model, "encoder", None), DynamicEncoder):
        model.encoder.cfg.early_exit = was_early

    return results


# --- 4. Основной скрипт эксперимента ---

if __name__ == "__main__":
    NUM_SAMPLES = 25000
    MODEL_D = 256
    MODEL_LAYERS = 6
    MODEL_HEADS = 4

    TRAIN_CONFIG = TrainerConfig(max_steps=8000, batch_size=128, learning_rate=3e-4, log_every=100, eval_every=400)

    print("--- Фаза 1: Подготовка ---")
    if os.path.exists("train_data_brackets.pt"):
        print("Загрузка существующих данных...")
        train_data, val_data, test_data = torch.load("train_data_brackets.pt"), torch.load(
            "val_data_brackets.pt"), torch.load("test_data_brackets.pt")
    else:
        print("Генерация нового набора данных...")
        all_seqs, all_labels, all_comp = generate_dataset(NUM_SAMPLES)
        train_size, val_size = int(0.8 * NUM_SAMPLES), int(0.1 * NUM_SAMPLES)
        train_data = (all_seqs[:train_size], all_labels[:train_size], all_comp[:train_size])
        val_data = (all_seqs[train_size:train_size + val_size], all_labels[train_size:train_size + val_size],
                    all_comp[train_size:train_size + val_size])
        test_data = (all_seqs[train_size + val_size:], all_labels[train_size + val_size:],
                     all_comp[train_size + val_size:])
        torch.save(train_data, "train_data_brackets.pt");
        torch.save(val_data, "val_data_brackets.pt");
        torch.save(test_data, "test_data_brackets.pt")

    print("\n--- Обучение Baseline (Классический Трансформер) ---")
    baseline_encoder = BaselineEncoder(n_layers=MODEL_LAYERS, d_model=MODEL_D, n_heads=MODEL_HEADS, d_ff=4 * MODEL_D)
    baseline_model = SequenceClassifier(baseline_encoder, d_model=MODEL_D, n_classes=N_CLASSES)
    baseline_model = train(baseline_model, "Baseline", TRAIN_CONFIG, train_data, val_data)
    print("\n--- Оценка Baseline ---")
    baseline_results = evaluate(baseline_model, test_data)

    print("\n--- Обучение DAT ---")
    dat_layer_cfg = LayerConfig(
        d_model=MODEL_D, n_heads=MODEL_HEADS, d_head=MODEL_D // MODEL_HEADS, d_ff=4 * MODEL_D,
        halt_gate_hidden=64, head_gate_hidden=None  # Отключаем адаптивную ширину для чистоты эксперимента
    )
    dat_enc_cfg = EncoderConfig(n_layers=MODEL_LAYERS, layer=dat_layer_cfg, early_exit=True, exit_threshold=0.95)
    dat_encoder = DynamicEncoder(dat_enc_cfg)
    dat_model = SequenceClassifier(dat_encoder, d_model=MODEL_D, n_classes=N_CLASSES)
    dat_model = train(dat_model, "DAT", TRAIN_CONFIG, train_data, val_data)
    print("\n--- Оценка DAT ---")
    dat_results = evaluate(dat_model, test_data)

    print("\n\n" + "=" * 80)
    print("--- Итоговый сравнительный анализ: Проверка баланса скобок ---")
    print("=" * 80)
    print(f"{'Метрика':<25} | {'Baseline Трансформер':<25} | {'DAT':<25}")
    print("-" * 80)
    print(
        f"{'Общая точность':<25} | {baseline_results.get('Общая точность', 0):.4f}{'':<20} | {dat_results.get('Общая точность', 0):.4f}")
    print("-" * 80)
    print(
        f"{'Точность (легкие)':<25} | {baseline_results.get('Легкие', {}).get('accuracy', 0):.4f}{'':<20} | {dat_results.get('Легкие', {}).get('accuracy', 0):.4f}")
    print(
        f"{'Время/семпл (легкие), мс':<25} | {baseline_results.get('Легкие', {}).get('avg_time_ms', 0):.3f}{'':<21} | {dat_results.get('Легкие', {}).get('avg_time_ms', 0):.3f}")
    print(
        f"{'Глубина (легкие)':<25} | {baseline_results.get('Легкие', {}).get('avg_depth', 0):.2f} / {MODEL_LAYERS}{'':<17} | {dat_results.get('Легкие', {}).get('avg_depth', 0):.2f} / {MODEL_LAYERS}")
    print("-" * 80)
    print(
        f"{'Точность (сложные)':<25} | {baseline_results.get('Сложные', {}).get('accuracy', 0):.4f}{'':<20} | {dat_results.get('Сложные', {}).get('accuracy', 0):.4f}")
    print(
        f"{'Время/семпл (сложные), мс':<25} | {baseline_results.get('Сложные', {}).get('avg_time_ms', 0):.3f}{'':<21} | {dat_results.get('Сложные', {}).get('avg_time_ms', 0):.3f}")
    print(
        f"{'Глубина (сложные)':<25} | {baseline_results.get('Сложные', {}).get('avg_depth', 0):.2f} / {MODEL_LAYERS}{'':<17} | {dat_results.get('Сложные', {}).get('avg_depth', 0):.2f} / {MODEL_LAYERS}")
    print("-" * 80)