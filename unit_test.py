"""
Модульные тесты для архитектуры DAT (Dynamic Adaptive Transformer)
==================================================================

Этот скрипт использует стандартную библиотеку `unittest` для проверки
корректности реализации каждого компонента из файла `dat_transformer.py`.

Тесты проверяют:
- Корректность размеров (shape) выходных тензоров.
- Базовую функциональность каждого модуля.
- Возможность выполнить полный прямой и обратный проход (forward/backward pass).
- Работу режимов обучения и инференса (с early_exit).

Как запустить:
1. Убедитесь, что файл dat_transformer.py находится в той же директории.
2. Запустите этот скрипт: python test_dat_architecture.py
3. Если все тесты пройдены успешно, вы увидите "OK" в конце вывода.
"""
import unittest
import torch
import torch.nn as nn

# --- Импорт архитектуры DAT ---
# Убедитесь, что dat_transformer.py находится в той же папке
try:
    from dat_transformer_v2 import (
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
        aw_mha = AdaptiveWidthMultiheadAttention(
            AttentionConfig(
                n_heads=self.H, d_model=self.Dm, d_head=self.Dh,
                use_parametric_scores=False
            )
        ).to(self.device)
        aw_mha.train()

        x = self.x.clone().requires_grad_(True)
        y, info = aw_mha(x)  # без памяти/маски
        loss = y.sum()
        loss.backward()

        # 1) есть градиент у Linear финального слоя гейта
        self.assertIsNotNone(aw_mha.head_gate.net.weight.grad)

        # 2) влияние гейтов на вывод: заглушаем гейт (sigmoid ≈ 0)
        with torch.no_grad():
            aw_mha.head_gate.net.weight.zero_()
            aw_mha.head_gate.net.bias.fill_(-20.0)
        y0, _ = aw_mha(self.x)
        self.assertLess(float(y0.abs().mean()), float(y.abs().mean()))

    def test_adaptive_layer(self):
        cfg = LayerConfig(
            d_model=self.Dm, n_heads=self.H, d_head=self.Dh, d_ff=4 * self.Dm,
            head_gate_hidden=32, halt_gate_hidden=32,
            use_parametric_scores=True, score_hidden=64, score_chunk_size=8,
            mem_topk=0,
        )
        layer = AdaptiveLayer(cfg).to(self.device)
        layer.train()
        x_out, p_halt, aux = layer(self.x)
        loss = x_out.sum() + 1e-3 * p_halt.mean()  # добавили мягкую зависимость
        loss.backward()
        self.assertIsNotNone(layer.ff[0].weight.grad)
        self.assertIsNotNone(layer.halt_gate.net.weight.grad)

        # 1) формы
        B, T, D = self.x.shape
        self.assertEqual(list(x_out.shape), [B, T, D])
        self.assertEqual(list(p_halt.shape), [B, T, 1])
        # (второй backward по тому же графу не делаем)

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

    def test_parametric_attention_forward(self):
        cfg = LayerConfig(
            d_model=self.Dm, n_heads=self.H, d_head=self.Dh,
            d_ff = 4 * self.Dm, use_parametric_scores = True,
            score_hidden = 64, score_chunk_size = 8
        )
        layer = AdaptiveLayer(cfg).to(self.device).train()
        x = self.x.clone().requires_grad_(True)
        out, p_halt, _ = layer(x)
        # простой loss даёт градиент в ParametricScoreMLP
        loss = out.sum()
        loss.backward()
        # найдём параметры MLP-скорера
        attn = layer.attn.attn
        params = list(attn.score.mlp.parameters())
        self.assertTrue(any(p.grad is not None for p in params))

    def test_memory_path(self):
        Dh = self.Dh
        mem = MemoryBank(d_key=Dh, d_value=self.Dm, device=self.device)
        # seed memory
        with torch.no_grad():
            mem.write(torch.randn(64, self.Dh, device=self.device),
                      torch.randn(64, self.Dh, device=self.device))

        cfg = LayerConfig(
            d_model=self.Dm, n_heads=self.H, d_head=Dh,
            d_ff=4 * self.Dm, mem_topk=4, use_parametric_scores=True,
            score_hidden=32, score_chunk_size=8
        )
        layer = AdaptiveLayer(cfg).to(self.device).eval() # форма инвариантна в eval/train
        out, p_halt, info = layer(self.x, memory_bank=mem)
        self.assertEqual(out.shape, self.x.shape)
        # длина по ключам = T + topk (память добавилась)
        self.assertEqual(info["attn_weights"].shape[-1], self.T + 4)

    def test_masking_blocks_pads(self):
        layer = AdaptiveLayer(self.layer_cfg).to(self.device).eval()
        B, T, D = self.x.shape
        # маска: половина тайм-степов паддинг
        mask = torch.ones(B, T, device=self.device);
        mask[:, T // 2:] = 0
        # additive attention mask [B,1,1,T]
        attn_mask = (1.0 - mask).unsqueeze(1).unsqueeze(2) * torch.finfo(self.x.dtype).min
        _, _, info = layer(self.x, attn_mask=attn_mask)  # уже на том же девайсе
        w = info["attn_weights"]  # [H или B? см. структуру] -> берем mean по H/Tq
        attn_mean_over_heads = w.mean(dim=(0, 1, 2))  # [Tk]
        # проверим, что среднее внимание на паддинги заметно меньше
        self.assertLess(float(attn_mean_over_heads[T // 2:].mean()),
                        float(attn_mean_over_heads[:T // 2].mean()))

    def test_dynamic_encoder_early_exit(self):
        cfg = EncoderConfig(n_layers=3, layer=self.layer_cfg, early_exit=True, exit_threshold=0.8)
        model = DynamicEncoder(cfg).to(self.device).eval()
        # «задираем» bias у всех halt_gate, чтобы p_halt≈1 на первом слое
        for lyr in model.layers:
            nn.init.constant_(lyr.halt_gate.net.bias, 10.0)
            if hasattr(lyr.halt_gate.net, "weight"):
                nn.init.zeros_(lyr.halt_gate.net.weight)
        _, aux = model(self.x, return_aux=True)
        # было выполнено только 1 слой
        self.assertEqual(aux["halting_ps"].shape[0], 1)

    def test_head_gate_disabled(self):
        cfg = LayerConfig(
            d_model=self.Dm, n_heads=self.H, d_head=self.Dh, d_ff=4 * self.Dm,
            head_gate_hidden=None,  # выключаем гейтинг
            halt_gate_hidden=32
        )
        aw = AdaptiveWidthMultiheadAttention(AttentionConfig(
            n_heads=self.H, d_model=self.Dm, d_head=self.Dh, head_gate_hidden=None
        )).to(self.device)
        x = torch.randn(self.B, self.T, self.Dm, device=self.device, requires_grad=True)
        out, info = aw(x)
        self.assertEqual(out.shape, (self.B, self.T, self.Dm))
        self.assertIsNone(info["head_gates"])
        out.sum().backward()
        # Проверяем, что градиенты вообще идут
        for p in aw.parameters():
            if p.requires_grad:
                self.assertIsNotNone(p.grad)


if __name__ == '__main__':
    print("Запуск модульных тестов для архитектуры DAT...")
    loader = unittest.TestLoader()
    suite = loader.loadTestsFromTestCase(TestDATArchitecture)
    result = unittest.TextTestRunner().run(suite)

    if result.wasSuccessful():
        print("\nВсе тесты успешно пройдены!")
    else:
        print("\nТесты не пройдены. Найдены ошибки.")
