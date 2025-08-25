"""
Модульные тесты для ort_transformer (Encoder-only, decoupled head dim, ortho-regularization)
===========================================================================================

Запуск:
1) Убедитесь, что ort_transformer.py находится в той же директории.
2) python test_ort_transformer_unittest.py
"""

import unittest
import torch
import torch.nn as nn

# --- Импорт тестируемой архитектуры ---
try:
    from ort_transformer import (
        TransformerEncoder,
        EncoderConfig,
        SequenceClassifier,
    )
except ImportError:
    print("Ошибка: файл ort_transformer.py не найден. Поместите его в ту же директорию.")
    raise


class TestOrtTransformer(unittest.TestCase):

    def setUp(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def test_shapes_decoupled_head_dim(self):
        """Проверка форм при развязке d_head != d_model / n_heads"""
        torch.manual_seed(0)
        cfg = EncoderConfig(
            n_layers=2, d_model=64,
            n_heads=2, d_head=64,   # decoupled: H*Dh = 128 != d_model
            d_ff=128,
            attn_dropout=0.0, resid_dropout=0.0,
            ortho_q=0.0, ortho_k=0.0, ortho_v=0.0,
        )
        enc = TransformerEncoder(cfg).to(self.device)
        x = torch.randn(3, 11, 64, device=self.device)  # [B,T,D]
        y, aux = enc(x, return_aux=True)
        self.assertEqual(tuple(y.shape), (3, 11, 64))
        self.assertIn('attn_probs', aux)
        self.assertIn('ortho_loss', aux)
        self.assertEqual(aux['attn_probs'].shape[:2], (3, cfg.n_heads))

    def test_additive_mask_zeroes_attention_on_pads(self):
        """Проверка, что additive mask зануляет вероятность внимания на паддинги"""
        torch.manual_seed(0)
        cfg = EncoderConfig(
            n_layers=1, d_model=32,
            n_heads=2, d_head=16,
            d_ff=64,
            attn_dropout=0.0, resid_dropout=0.0,
        )
        enc = TransformerEncoder(cfg).to(self.device)
        B, T, D = 1, 6, 32
        x = torch.randn(B, T, D, device=self.device)
        # маскируем последние 3 ключа
        padding = torch.tensor([[0, 0, 0, 1, 1, 1]], dtype=torch.bool, device=self.device)
        attn_mask = padding.unsqueeze(1).unsqueeze(2) * (-1e9)  # [B,1,1,T]
        y, aux = enc(x, attn_mask=attn_mask, return_aux=True)
        A = aux['attn_probs']  # [B,H,T,T]
        masked_probs = A[..., :, 3:].sum(dim=-1)  # [B,H,T]
        self.assertTrue(torch.all(masked_probs < 1e-8).item())

    def test_orthogonality_loss_zero_when_q_blocks_orthogonal(self):
        """Если блоки W_Q голов ортогональны, штраф ~0 (только Q включён)"""
        torch.manual_seed(0)
        cfg = EncoderConfig(
            n_layers=1, d_model=4,
            n_heads=2, d_head=2,  # W shape [4,4] split as [2,4] + [2,4]
            d_ff=16, attn_dropout=0.0, resid_dropout=0.0,
            ortho_q=1.0, ortho_k=0.0, ortho_v=0.0,
        )
        enc = TransformerEncoder(cfg).to(self.device)
        layer = enc.layers[0].mha
        W = torch.zeros(2*2, 4, device=self.device)  # [H*Dh, D] = [4,4]
        W[0,0] = 1.0; W[1,1] = 1.0   # head 0
        W[2,2] = 1.0; W[3,3] = 1.0   # head 1
        with torch.no_grad():
            layer.q_proj.weight.copy_(W)
        loss = layer.orthogonality_loss()
        self.assertLess(loss.item(), 1e-8)

    def test_orthogonality_loss_positive_on_random(self):
        """На случайных весах ортогональный штраф > 0"""
        torch.manual_seed(42)
        cfg = EncoderConfig(
            n_layers=1, d_model=32,
            n_heads=4, d_head=8,
            d_ff=64, attn_dropout=0.0, resid_dropout=0.0,
            ortho_q=1e-3, ortho_k=0.0, ortho_v=0.0,
        )
        enc = TransformerEncoder(cfg).to(self.device)
        loss = enc.layers[0].mha.orthogonality_loss()
        self.assertGreater(loss.item(), 0.0)

    def test_backward_flows_with_ortho_loss(self):
        """Градиент течёт через проекции при добавлении ortho_loss к целевой функции"""
        torch.manual_seed(0)
        cfg = EncoderConfig(
            n_layers=2, d_model=32,
            n_heads=2, d_head=16,
            d_ff=64, attn_dropout=0.0, resid_dropout=0.0,
            ortho_q=1e-3, ortho_k=1e-3, ortho_v=0.0,
        )
        enc = TransformerEncoder(cfg).to(self.device)
        x = torch.randn(2, 7, 32, device=self.device, requires_grad=True)
        y, aux = enc(x, return_aux=True)
        loss = y.mean() + aux['ortho_loss']
        loss.backward()
        for layer in enc.layers:
            self.assertIsNotNone(layer.mha.q_proj.weight.grad)
            self.assertIsNotNone(layer.mha.k_proj.weight.grad)

    def test_sequence_classifier_shapes_and_padding_mask(self):
        """Проверка SequenceClassifier: формы логитов и маскирование паддингов"""
        torch.manual_seed(0)
        cfg = EncoderConfig(
            n_layers=1, d_model=16,
            n_heads=2, d_head=8,
            d_ff=64, attn_dropout=0.0, resid_dropout=0.0,
            ortho_q=0.0, ortho_k=0.0, ortho_v=0.0,
        )
        from ort_transformer import SequenceClassifier
        enc = TransformerEncoder(cfg).to(self.device)
        clf = SequenceClassifier(enc, vocab_size=10, d_model=16, n_classes=3, pad_token=0).to(self.device)
        tokens = torch.tensor([[1,2,3,0,0]], device=self.device)  # [B=1,T=5]
        logits, aux = clf(tokens, return_aux=True)
        self.assertEqual(tuple(logits.shape), (1,3))
        A = aux['attn_probs']  # [B,H,T,T]
        masked_cols = A[..., :, 3:].sum(dim=-1)
        self.assertTrue(torch.all(masked_cols < 1e-8).item())

    def test_various_head_dims_subtests(self):
        """Несколько конфигураций голов через subTest"""
        torch.manual_seed(0)
        for n_heads, d_head in [(1,32), (2,16), (4,8)]:
            with self.subTest(n_heads=n_heads, d_head=d_head):
                cfg = EncoderConfig(
                    n_layers=1, d_model=32,
                    n_heads=n_heads, d_head=d_head, d_ff=64,
                    attn_dropout=0.0, resid_dropout=0.0
                )
                enc = TransformerEncoder(cfg).to(self.device)
                x = torch.randn(2, 5, 32, device=self.device)
                y = enc(x)
                self.assertEqual(tuple(y.shape), (2,5,32))


if __name__ == '__main__':
    print("Запуск модульных тестов для ort_transformer...")
    suite = unittest.TestLoader().loadTestsFromTestCase(TestOrtTransformer)
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    if result.wasSuccessful():
        print("\nВсе тесты успешно пройдены!")
    else:
        print("\nТесты не пройдены. Найдены ошибки.")
