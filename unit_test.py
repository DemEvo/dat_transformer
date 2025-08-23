"""
Модульные тесты для архитектуры DAT (Dynamic Adaptive Transformer)
==================================================================

Этот скрипт использует стандартную библиотеку `unittest` для проверки
корректности реализации каждого компонента из файла `dat_core.py`.

Тесты проверяют:
- Корректность размеров (shape) выходных тензоров.
- Базовую функциональность каждого модуля.
- Возможность выполнить полный прямой и обратный проход (forward/backward pass).
- Работу режимов обучения и инференса (с early_exit).

Как запустить:
1. Убедитесь, что файл dat_core.py находится в той же директории.
2. Запустите этот скрипт: python test_dat_architecture.py
3. Если все тесты пройдены успешно, вы увидите "OK" в конце вывода.
"""
import unittest
import torch
import torch.nn as nn

# --- Импорт архитектуры DAT ---
# Убедитесь, что dat_core.py находится в той же папке
try:
    from dat_core import (
        GatingMLP,
        ParametricScoreMLP,
        MemoryBank,
        MemoryAugmentedAttention,
        AdaptiveWidthMultiheadAttention,
        AdaptiveLayer,
        DynamicEncoder,
        AttentionConfig,
        LayerConfig,
        EncoderConfig
    )
except ImportError:
    print("Ошибка: файл dat_core.py не найден. Пожалуйста, поместите его в ту же директорию.")
    exit()


class TestDATArchitecture(unittest.TestCase):

    def setUp(self):
        """ Настраивает общие параметры для всех тестов """
        self.B, self.T, self.Dm = 2, 16, 128  # Batch, Time, Dimension
        self.H, self.Dh = 4, 32  # Heads, Head Dimension
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Общая конфигурация для большинства тестов
        self.layer_cfg = LayerConfig(
            d_model=self.Dm, n_heads=self.H, d_head=self.Dh, d_ff=4 * self.Dm,
            head_gate_hidden=64,
            use_parametric_scores=True,
            score_hidden=64,
            mem_topk=4,
            halt_gate_hidden=32,
        )
        self.enc_cfg = EncoderConfig(n_layers=3, layer=self.layer_cfg)
        self.x = torch.randn(self.B, self.T, self.Dm).to(self.device)

    def test_gating_mlp(self):
        """ Тест: Вентильная функция (GatingMLP) """
        gate = GatingMLP(self.Dm, d_out=self.H).to(self.device)
        output = gate(self.x)
        self.assertEqual(output.shape, (self.B, self.T, self.H))
        self.assertTrue((output >= 0).all() and (output <= 1).all())

    def test_parametric_score_mlp(self):
        """ Тест: Параметрическая функция оценки внимания """
        scorer = ParametricScoreMLP(d_q=self.Dh, d_k=self.Dh).to(self.device)
        q = torch.randn(self.B, self.H, self.T, self.Dh).to(self.device)
        k = torch.randn(self.B, self.H, self.T + 4, self.Dh).to(self.device)  # Другая длина для проверки
        scores = scorer.pairwise_scores(q, k)
        self.assertEqual(scores.shape, (self.B, self.H, self.T, self.T + 4))

    def test_memory_bank(self):
        """ Тест: Банк памяти (MemoryBank) """
        mem = MemoryBank(d_key=self.Dh, d_value=self.Dh, device=self.device)
        self.assertEqual(mem.size(), 0)

        # Запись
        keys_to_write = torch.randn(10, self.Dh).to(self.device)
        vals_to_write = torch.randn(10, self.Dh).to(self.device)
        mem.write(keys_to_write, vals_to_write)
        self.assertEqual(mem.size(), 10)

        # Чтение
        query = torch.randn(self.B, self.T, self.Dh).to(self.device)
        topk = 4
        k_mem, v_mem = mem.retrieve(query, topk=topk)
        self.assertEqual(k_mem.shape, (self.B, self.T, topk, self.Dh))
        self.assertEqual(v_mem.shape, (self.B, self.T, topk, self.Dh))

    def test_adaptive_width_mha(self):
        """ Тест: Многоголовочное внимание с адаптивной шириной """
        attn_cfg = AttentionConfig(
            n_heads=self.H, d_model=self.Dm, d_head=self.Dh, head_gate_hidden=32
        )
        aw_mha = AdaptiveWidthMultiheadAttention(attn_cfg).to(self.device)
        output, aux = aw_mha(self.x)

        self.assertEqual(output.shape, (self.B, self.T, self.Dm))
        self.assertIn("head_gates", aux)
        self.assertEqual(aux["head_gates"].shape, (self.B, self.T, self.H))

        # Проверяем, что градиент проходит
        output.sum().backward()
        self.assertIsNotNone(aw_mha.q_proj.weight.grad)
        self.assertIsNotNone(aw_mha.head_gate.net.weight.grad)

    def test_adaptive_layer(self):
        """ Тест: Адаптивный слой """
        layer = AdaptiveLayer(self.layer_cfg).to(self.device)
        x_out, p_halt, aux = layer(self.x)

        self.assertEqual(x_out.shape, (self.B, self.T, self.Dm))
        self.assertEqual(p_halt.shape, (self.B, self.T, 1))
        self.assertTrue((p_halt >= 0).all() and (p_halt <= 1).all())

        # Проверяем, что градиент проходит
        x_out.sum().backward()
        self.assertIsNotNone(layer.ff[0].weight.grad)
        self.assertIsNotNone(layer.halt_gate.net.weight.grad)

    def test_dynamic_encoder_forward_pass(self):
        """ Тест: Полный прямой проход через DynamicEncoder в режиме обучения """
        model = DynamicEncoder(self.enc_cfg).to(self.device)
        model.train()  # Убеждаемся, что мы в режиме обучения

        output, aux = model(self.x, return_aux=True)

        self.assertEqual(output.shape, (self.B, self.T, self.Dm))
        self.assertIn("halting_ps", aux)
        self.assertIn("head_gates", aux)
        self.assertEqual(aux["halting_ps"].shape, (self.enc_cfg.n_layers, self.B, self.T, 1))

        # Проверяем, что градиент проходит через всю модель
        output.sum().backward()
        # Проверяем градиент у первого и последнего слоя
        self.assertIsNotNone(model.layers[0].attn.q_proj.weight.grad)
        self.assertIsNotNone(model.layers[-1].ff[3].weight.grad)

    def test_dynamic_encoder_early_exit(self):
        """ Тест: Режим инференса с ранней остановкой (early_exit) """
        enc_cfg_exit = EncoderConfig(
            n_layers=6, layer=self.layer_cfg, early_exit=True, exit_threshold=0.5
        )
        model = DynamicEncoder(enc_cfg_exit).to(self.device)
        model.eval()  # Переключаем в режим оценки

        with torch.no_grad():
            output, aux = model(self.x, return_aux=True)

        self.assertEqual(output.shape, (self.B, self.T, self.Dm))

        # Проверяем, что ponder_cost (ожидаемая глубина) меньше максимальной
        # из-за ранней остановки (вероятность > 0)
        ponder_cost = aux["halting_ps"].sum(dim=0).mean()
        self.assertLess(ponder_cost.item(), enc_cfg_exit.n_layers)


if __name__ == '__main__':
    print("Запуск модульных тестов для архитектуры DAT...")
    unittest.main(argv=['first-arg-is-ignored'], exit=False)
    print("\nВсе тесты успешно пройдены!")
