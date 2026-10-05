"""Gradient check for the mini GPT backward pass.

Compares every analytic gradient from loss_and_grads against a central
finite-difference estimate on a tiny model. If a parameter is missing from the
backward pass, its analytic gradient is zero and this test fails.
"""

from __future__ import annotations

import unittest

import numpy as np

from main import MiniGPT, cross_entropy_loss, loss_and_grads, parameters, train_mini_gpt


def tiny_model():
    np.random.seed(0)
    model = MiniGPT(vocab_size=11, embed_dim=8, num_heads=2,
                    num_layers=2, max_seq_len=6, ff_dim=16)
    for p in parameters(model):
        p += np.random.randn(*p.shape) * 0.1
    return model


class GradientCheck(unittest.TestCase):
    def test_every_parameter_matches_finite_differences(self):
        model = tiny_model()
        rng = np.random.default_rng(1)
        input_ids = rng.integers(0, 11, size=(2, 5))
        target_ids = rng.integers(0, 11, size=(2, 5))

        _, grads = loss_and_grads(model, input_ids, target_ids)
        params = parameters(model)
        self.assertEqual(len(grads), len(params))

        eps = 1e-5
        for i, (p, g) in enumerate(zip(params, grads)):
            numeric = np.zeros_like(p)
            for idx in np.ndindex(p.shape):
                old = p[idx]
                p[idx] = old + eps
                plus = cross_entropy_loss(model.forward(input_ids), target_ids)
                p[idx] = old - eps
                minus = cross_entropy_loss(model.forward(input_ids), target_ids)
                p[idx] = old
                numeric[idx] = (plus - minus) / (2 * eps)
            denom = np.abs(numeric).max() + np.abs(g).max() + 1e-12
            rel_err = np.abs(numeric - g).max() / denom
            self.assertLess(rel_err, 1e-5, f"parameter {i} shape {p.shape}: rel err {rel_err:.2e}")


class TrainingLearns(unittest.TestCase):
    def test_loss_drops_well_below_random_guessing(self):
        np.random.seed(42)
        text = "the cat sat on the mat. " * 40
        model = train_mini_gpt(text, embed_dim=32, num_heads=2, num_layers=2,
                               seq_len=16, num_steps=150)
        tokens = np.array(list(text.encode("utf-8")))
        x, y = tokens[:16].reshape(1, -1), tokens[1:17].reshape(1, -1)
        loss = cross_entropy_loss(model.forward(x), y)
        self.assertLess(loss, 0.5 * np.log(256))


if __name__ == "__main__":
    unittest.main()
