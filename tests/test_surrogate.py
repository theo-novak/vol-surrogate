import numpy as np
import pytest
import torch

from vol_surrogate.surrogate.losses import arbitrage_penalty, relative_mse, torch_bs_call
from vol_surrogate.surrogate.mlp import (
    MLPSurrogate,
    Normalizer,
    load_checkpoint,
    predict_iv,
    save_checkpoint,
)
from vol_surrogate.surrogate.train import TrainConfig, train_surrogate
from vol_surrogate.surrogate.transformer import GridTransformer


class TestModel:
    def test_mlp_forward_shape(self):
        model = MLPSurrogate(in_dim=5, hidden=32, depth=2, out_dim=48)
        out = model(torch.randn(7, 5))
        assert out.shape == (7, 48)

    def test_transformer_forward_shape(self):
        model = GridTransformer(in_dim=5, n_tokens=48, d_model=32, nhead=2, num_layers=1)
        out = model(torch.randn(7, 5))
        assert out.shape == (7, 48)

    def test_normalizer_round_trip(self, rng):
        X = rng.normal(2.0, 3.0, (100, 5))
        Y = rng.normal(0.2, 0.05, (100, 48))
        norm = Normalizer.fit(X, Y)
        assert np.allclose(norm.denorm_x(norm.norm_x(X)), X)
        assert np.allclose(norm.denorm_y(norm.norm_y(Y)), Y)
        # normalised stats are standard
        assert np.allclose(norm.norm_x(X).mean(axis=0), 0.0, atol=1e-10)
        assert np.allclose(norm.norm_x(X).std(axis=0), 1.0, atol=1e-6)

    def test_checkpoint_round_trip(self, tmp_path, rng, grid):
        model = MLPSurrogate(hidden=16, depth=2)
        X = rng.uniform(0.5, 2.0, (50, 5))
        Y = rng.uniform(0.1, 0.4, (50, 48))
        norm = Normalizer.fit(X, Y)
        path = save_checkpoint(tmp_path / "ck.pt", model, norm,
                               grid["moneyness"], grid["maturities"])
        model2, norm2, meta = load_checkpoint(path)
        x = np.array([2.0, 0.04, 0.5, -0.7, 0.04])
        a = predict_iv(model, norm, x)
        b = predict_iv(model2, norm2, x)
        # fp16 storage round-trip: predictions agree to ~1e-3
        assert np.allclose(a, b, atol=2e-3)
        assert np.allclose(meta["moneyness"], grid["moneyness"])


class TestLosses:
    def _flat_iv(self, grid, sigma=0.2, batch=3):
        n_mat, n_mon = len(grid["maturities"]), len(grid["moneyness"])
        return torch.full((batch, n_mat, n_mon), sigma)

    def test_torch_bs_call_matches_scipy(self, grid):
        from vol_surrogate.pricing.black_scholes import bs_price

        m = torch.as_tensor(grid["moneyness"], dtype=torch.float64)
        T = torch.as_tensor(grid["maturities"], dtype=torch.float64)
        iv = self._flat_iv(grid, 0.25, batch=1).double()
        C = torch_bs_call(iv, m, T)[0]
        for i, Tm in enumerate(grid["maturities"]):
            for j, mj in enumerate(grid["moneyness"]):
                ref = bs_price(1.0, float(mj), 0.0, float(Tm), 0.25)
                assert float(C[i, j]) == pytest.approx(ref, abs=1e-10)

    def test_penalty_zero_on_clean_bs_surface(self, grid):
        m = torch.as_tensor(grid["moneyness"], dtype=torch.float32)
        T = torch.as_tensor(grid["maturities"], dtype=torch.float32)
        pen = arbitrage_penalty(self._flat_iv(grid), m, T)
        assert float(pen["total"]) == pytest.approx(0.0, abs=1e-10)

    def test_penalty_positive_on_corrupted_surface(self, grid):
        m = torch.as_tensor(grid["moneyness"], dtype=torch.float32)
        T = torch.as_tensor(grid["maturities"], dtype=torch.float32)
        iv = self._flat_iv(grid)
        iv[:, 2, 4] = 1.4          # butterfly/call-spread violation in strike
        iv[:, -1, :] = 0.05        # calendar violation: long-end total var collapses
        pen = arbitrage_penalty(iv, m, T)
        assert float(pen["total"]) > 1e-8
        assert float(pen["calendar"]) > 0.0
        assert float(pen["butterfly"]) > 0.0

    def test_relative_mse_scale_invariance(self):
        true = torch.tensor([[0.1, 0.4]], dtype=torch.float64)
        pred = true * 1.01  # 1% relative error everywhere
        assert float(relative_mse(pred, true)) == pytest.approx(1e-4, rel=1e-6)


class TestTraining:
    def test_short_training_reduces_loss(self, tiny_dataset):
        cfg = TrainConfig(hidden=32, depth=2, max_epochs=15, patience=15,
                          batch_size=64, lr=3e-3, seed=42)
        model, norm, history = train_surrogate(
            tiny_dataset["X"], tiny_dataset["Y"],
            tiny_dataset["moneyness"], tiny_dataset["maturities"],
            cfg, verbose=False,
        )
        assert history["val_loss"][-1] < history["val_loss"][0]
        assert history["best"]["val_rel_mse"] < 0.04
        # prediction has the right shape and sane values
        iv = predict_iv(model, norm, tiny_dataset["X"][0])
        assert iv.shape == (6, 8)
        assert 0.0 < iv.mean() < 1.0


class TestOverfit:
    def test_mlp_overfits_50_samples(self, tiny_dataset):
        """A 2x32 MLP must drive 50 training samples to near-zero relative MSE
        — the standard sanity check that the architecture and loss can learn."""
        X, Y = tiny_dataset["X"][:50], tiny_dataset["Y"][:50]
        torch.manual_seed(42)
        model = MLPSurrogate(in_dim=5, hidden=32, depth=2, out_dim=Y.shape[1])
        norm = Normalizer.fit(X, Y)
        Xn = torch.as_tensor(norm.norm_x(X), dtype=torch.float32)
        iv_true = torch.as_tensor(Y, dtype=torch.float32)
        y_mean, y_std = norm.torch_y_stats()
        opt = torch.optim.Adam(model.parameters(), lr=5e-3)
        for _ in range(400):
            opt.zero_grad()
            loss = relative_mse(model(Xn) * y_std + y_mean, iv_true)
            loss.backward()
            opt.step()
        assert float(loss.detach()) < 1e-4


class TestExport:
    def test_numpy_forward_matches_torch(self, tmp_path, rng, grid):
        """The dashboard's pure-numpy forward pass must reproduce the torch
        model through the 5-significant-digit JSON round trip."""
        import json

        from vol_surrogate.surrogate.mlp import export_weights_json, numpy_forward

        torch.manual_seed(42)
        model = MLPSurrogate(in_dim=5, hidden=16, depth=2, out_dim=48)
        X = rng.uniform(0.5, 2.0, (60, 5))
        Y = rng.uniform(0.1, 0.4, (60, 48))
        norm = Normalizer.fit(X, Y)
        path = export_weights_json(tmp_path / "w.json", model, norm,
                                   grid["moneyness"], grid["maturities"])
        weights = json.loads(path.read_text())

        x = np.array([2.0, 0.04, 0.5, -0.7, 0.04])
        iv_torch = predict_iv(model, norm, x).ravel()
        iv_numpy = numpy_forward(weights, x)
        assert np.allclose(iv_torch, iv_numpy, atol=5e-4)
