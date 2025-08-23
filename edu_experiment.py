"""
Эксперимент: Сортировка с отвлекающими факторами
================================================

Цель: Сравнить производительность и эффективность классического Трансформера (Baseline)
и Динамического Адаптивного Трансформера (DAT) на задаче классификации последовательностей,
сложность которой зависит от входных данных.

Задача: Определить, является ли последовательность чисел отсортированной по возрастанию.
Сложность варьируется за счет добавления "шума" - случайных буквенных токенов.

Гипотеза: DAT покажет схожее качество (точность), но будет значительно эффективнее
на "чистых" данных (без шума), используя меньше вычислительных ресурсов (слоёв),
в то время как Baseline будет тратить одинаковое количество ресурсов на все примеры.

Как запустить:
1. Убедитесь, что файл dat_transformer.py находится в той же директории.
2. Просто запустите этот скрипт: python experiment_sorting.py
3. Скрипт автоматически сгенерирует данные, обучит обе модели,
   проведет оценку и выведет итоговую сравнительную таблицу.

Ожидаемое время выполнения на RTX 3080 Ti Laptop: ~5-8 часов.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.data import DataLoader, TensorDataset

import random
import time
import numpy as np
from dataclasses import dataclass
from tqdm import tqdm
import os

# --- Импорт архитектуры DAT ---
# Убедитесь, что dat_core.py находится в той же папке
try:
    from dat_core import DynamicEncoder, EncoderConfig, LayerConfig
except ImportError:
    print("Ошибка: файл dat_core.py не найден. Пожалуйста, поместите его в ту же директорию.")
    exit()

# --- 1. Генерация данных ---

# Словарь: 0-999 для чисел, 1000-1025 для букв, 1026=[CLS], 1027=[SEP]
VOCAB_SIZE = 1028
CLS_TOKEN = 1026
SEP_TOKEN = 1027


def is_sorted(numbers):
    return all(numbers[i] <= numbers[i + 1] for i in range(len(numbers) - 1))


def generate_sequence(length, noise_ratio):
    numbers = sorted(list(np.random.randint(0, 1000, int(length * (1 - noise_ratio)))))
    if random.random() > 0.5:  # 50% отсортированных, 50% нет
        label = 1
    else:
        random.shuffle(numbers)
        label = 0 if not is_sorted(numbers) else 1  # Перепроверяем после shuffle

    sequence = [float(n) for n in numbers]
    num_noise = length - len(sequence)
    for _ in range(num_noise):
        noise_token = random.randint(1000, 1025)
        insert_pos = random.randint(0, len(sequence))
        sequence.insert(insert_pos, float(noise_token))

    final_sequence = [CLS_TOKEN] + sequence + [SEP_TOKEN]
    return torch.tensor(final_sequence, dtype=torch.long), torch.tensor(label, dtype=torch.long), noise_ratio


def generate_dataset(num_samples, min_len=10, max_len=40):
    sequences, labels, noise_ratios = [], [], []
    for _ in tqdm(range(num_samples), desc="Генерация данных"):
        length = random.randint(min_len, max_len)
        noise = random.random() * 0.5  # Шум от 0% до 50%
        seq, lab, noi = generate_sequence(length, noise)
        sequences.append(seq)
        labels.append(lab)
        noise_ratios.append(noise)
    return sequences, torch.stack(labels), torch.tensor(noise_ratios)


# --- 2. Определение Моделей ---

class SequenceClassifier(nn.Module):
    """ Обертка над энкодером для задачи классификации всей последовательности. """

    def __init__(self, encoder, d_model, n_classes=2):
        super().__init__()
        self.encoder = encoder
        self.head = nn.Linear(d_model, n_classes)
        self.embedding = nn.Embedding(VOCAB_SIZE, d_model)

    def forward(self, tokens, return_aux=False):
        # tokens: [B, T]
        x = self.embedding(tokens)  # -> [B, T, D]

        # Для DAT нам нужна возможность получать доп. информацию (aux)
        if isinstance(self.encoder, DynamicEncoder):
            feats, aux = self.encoder(x, return_aux=True)
        else:  # Для Baseline-трансформера
            feats = self.encoder(x)
            aux = {}  # Пустой словарь для совместимости

        # Используем выход [CLS] токена для классификации
        cls_token_feats = feats[:, 0, :]  # [B, D]
        logits = self.head(cls_token_feats)  # [B, n_classes]

        if return_aux:
            return logits, aux
        return logits


# --- 3. Цикл Обучения и Оценки ---

@dataclass
class TrainerConfig:
    max_steps: int = 10000
    batch_size: int = 64
    learning_rate: float = 1e-4
    warmup_steps: int = 500
    log_every: int = 100
    eval_every: int = 500


def pad_sequences(sequences, max_len=None):
    if max_len is None:
        max_len = max(len(s) for s in sequences)
    padded = torch.full((len(sequences), max_len), fill_value=0, dtype=torch.long)
    for i, seq in enumerate(sequences):
        padded[i, :len(seq)] = seq
    return padded


def train(model, config, train_data, val_data):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    optimizer = AdamW(model.parameters(), lr=config.learning_rate)
    loss_fn = nn.CrossEntropyLoss()

    # Создаем DataLoader с padding
    train_seqs, train_labels, _ = train_data
    val_seqs, val_labels, _ = val_data

    train_padded = pad_sequences(train_seqs)
    val_padded = pad_sequences(val_seqs)

    train_dataset = TensorDataset(train_padded, train_labels)
    val_dataset = TensorDataset(val_padded, val_labels)

    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size)

    best_val_acc = 0.0
    best_model_state = None

    pbar = tqdm(range(config.max_steps), desc="Обучение")
    step = 0
    while step < config.max_steps:
        for batch in train_loader:
            if step >= config.max_steps:
                break

            model.train()
            tokens, labels = [t.to(device) for t in batch]

            logits = model(tokens)
            loss = loss_fn(logits, labels)

            # Для DAT добавляем регуляризацию
            if isinstance(model.encoder, DynamicEncoder) and model.encoder.cfg.layer.halt_gate_hidden is not None:
                _, aux = model(tokens, return_aux=True)
                # Простой штраф за глубину
                depth_penalty = 0.01 * aux['halting_ps'].sum(dim=0).mean()
                loss += depth_penalty

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if step % config.log_every == 0:
                pbar.set_postfix({"loss": loss.item()})

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
                    best_model_state = model.state_dict()
                    pbar.set_postfix({"loss": loss.item(), "best_val_acc": best_val_acc})

            step += 1
            pbar.update(1)

    model.load_state_dict(best_model_state)
    return model


@torch.no_grad()
def evaluate(model, test_data):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    test_seqs, test_labels, test_noise = test_data

    # Разделяем на "чистые" и "шумные"
    clean_indices = torch.where(test_noise <= 0.1)[0]
    noisy_indices = torch.where(test_noise >= 0.4)[0]

    results = {}
    total_correct, total_count = 0, 0

    for name, indices in [("Чистые", clean_indices), ("Шумные", noisy_indices)]:
        if len(indices) == 0:
            continue

        subset_seqs = [test_seqs[i] for i in indices]
        subset_labels = test_labels[indices]

        padded_seqs = pad_sequences(subset_seqs).to(device)
        labels = subset_labels.to(device)

        # Измерение времени
        start_time = time.perf_counter()
        # Прогрев GPU
        for _ in range(5):
            _ = model(padded_seqs[:2], return_aux=True)

        timings = []
        for _ in range(20):
            start_iter = time.perf_counter()
            logits, aux = model(padded_seqs, return_aux=True)
            timings.append(time.perf_counter() - start_iter)

        avg_time_ms = (sum(timings) / len(timings)) * 1000 / len(padded_seqs)

        preds = torch.argmax(logits, dim=1)
        correct = (preds == labels).sum().item()
        total = labels.size(0)

        total_correct += correct
        total_count += total

        depth = aux.get('halting_ps', torch.zeros(model.encoder.cfg.n_layers)).sum(dim=0).mean().item()
        if depth == 0.0:  # Для Baseline
            depth = model.encoder.cfg.n_layers

        results[name] = {
            "accuracy": correct / total,
            "avg_time_ms": avg_time_ms,
            "avg_depth": depth
        }

    results["Общая точность"] = total_correct / total_count if total_count > 0 else 0
    return results


# --- 4. Основной скрипт эксперимента ---

if __name__ == "__main__":
    # --- Конфигурация ---
    NUM_SAMPLES = 20000
    MODEL_D = 256
    MODEL_LAYERS = 4
    MODEL_HEADS = 4

    TRAIN_CONFIG = TrainerConfig(
        max_steps=10000,
        batch_size=128,
        learning_rate=2e-4,
        warmup_steps=500,
        log_every=100,
        eval_every=500
    )

    # --- Генерация данных ---
    print("--- Фаза 1: Подготовка ---")
    if os.path.exists("train_data.pt"):
        print("Загрузка существующих данных...")
        train_data = torch.load("train_data.pt")
        val_data = torch.load("val_data.pt")
        test_data = torch.load("test_data.pt")
    else:
        print("Генерация нового набора данных...")
        all_seqs, all_labels, all_noise = generate_dataset(NUM_SAMPLES)
        # Делим на train/val/test
        train_size = int(0.8 * NUM_SAMPLES)
        val_size = int(0.1 * NUM_SAMPLES)

        train_data = (all_seqs[:train_size], all_labels[:train_size], all_noise[:train_size])
        val_data = (all_seqs[train_size:train_size + val_size], all_labels[train_size:train_size + val_size],
                    all_noise[train_size:train_size + val_size])
        test_data = (all_seqs[train_size + val_size:], all_labels[train_size + val_size:],
                     all_noise[train_size + val_size:])

        torch.save(train_data, "train_data.pt")
        torch.save(val_data, "val_data.pt")
        torch.save(test_data, "test_data.pt")

    # --- Обучение и оценка Baseline ---
    print("\n--- Фаза 2: Обучение Baseline (Классический Трансформер) ---")

    # Конфигурация без адаптивных элементов
    baseline_layer_cfg = LayerConfig(
        d_model=MODEL_D, n_heads=MODEL_HEADS, d_head=MODEL_D // MODEL_HEADS, d_ff=4 * MODEL_D,
        halt_gate_hidden=None, head_gate_hidden=None
    )
    baseline_enc_cfg = EncoderConfig(n_layers=MODEL_LAYERS, layer=baseline_layer_cfg)
    baseline_encoder = DynamicEncoder(baseline_enc_cfg)  # Используем тот же класс, но с выключенными опциями
    baseline_model = SequenceClassifier(baseline_encoder, d_model=MODEL_D)

    baseline_model = train(baseline_model, TRAIN_CONFIG, train_data, val_data)
    print("\n--- Фаза 3: Оценка Baseline ---")
    baseline_results = evaluate(baseline_model, test_data)

    # --- Обучение и оценка DAT ---
    print("\n--- Фаза 2: Обучение DAT ---")

    # Конфигурация с адаптивными элементами
    dat_layer_cfg = LayerConfig(
        d_model=MODEL_D, n_heads=MODEL_HEADS, d_head=MODEL_D // MODEL_HEADS, d_ff=4 * MODEL_D,
        halt_gate_hidden=64, head_gate_hidden=64
    )
    dat_enc_cfg = EncoderConfig(n_layers=MODEL_LAYERS, layer=dat_layer_cfg)
    dat_encoder = DynamicEncoder(dat_enc_cfg)
    dat_model = SequenceClassifier(dat_encoder, d_model=MODEL_D)

    # Включаем early_exit для инференса в DAT
    dat_model.encoder.cfg.early_exit = True
    dat_model.encoder.cfg.exit_threshold = 0.9

    dat_model = train(dat_model, TRAIN_CONFIG, train_data, val_data)
    print("\n--- Фаза 3: Оценка DAT ---")
    dat_results = evaluate(dat_model, test_data)

    # --- Итоговый отчёт ---
    print("\n\n--- Итоговый сравнительный анализ ---")
    print("-" * 80)
    print(f"{'Метрика':<25} | {'Baseline Трансформер':<25} | {'DAT':<25}")
    print("-" * 80)
    print(
        f"{'Общая точность':<25} | {baseline_results['Общая точность']:.4f}{'':<20} | {dat_results['Общая точность']:.4f}")
    print("-" * 80)
    print(
        f"{'Точность (чистые данные)':<25} | {baseline_results.get('Чистые', {}).get('accuracy', 0):.4f}{'':<20} | {dat_results.get('Чистые', {}).get('accuracy', 0):.4f}")
    print(
        f"{'Время/семпл (чистые), мс':<25} | {baseline_results.get('Чистые', {}).get('avg_time_ms', 0):.2f}{'':<22} | {dat_results.get('Чистые', {}).get('avg_time_ms', 0):.2f}")
    print(
        f"{'Глубина (чистые)':<25} | {baseline_results.get('Чистые', {}).get('avg_depth', 0):.2f} / {MODEL_LAYERS}{'':<17} | {dat_results.get('Чистые', {}).get('avg_depth', 0):.2f} / {MODEL_LAYERS}")
    print("-" * 80)
    print(
        f"{'Точность (шумные данные)':<25} | {baseline_results.get('Шумные', {}).get('accuracy', 0):.4f}{'':<20} | {dat_results.get('Шумные', {}).get('accuracy', 0):.4f}")
    print(
        f"{'Время/семпл (шумные), мс':<25} | {baseline_results.get('Шумные', {}).get('avg_time_ms', 0):.2f}{'':<22} | {dat_results.get('Шумные', {}).get('avg_time_ms', 0):.2f}")
    print(
        f"{'Глубина (шумные)':<25} | {baseline_results.get('Шумные', {}).get('avg_depth', 0):.2f} / {MODEL_LAYERS}{'':<17} | {dat_results.get('Шумные', {}).get('avg_depth', 0):.2f} / {MODEL_LAYERS}")
    print("-" * 80)
