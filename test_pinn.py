"""
Test Suite - PINN Degradation Model (pinn_model.py)
"""

import numpy as np
import torch
import pytest

from pinn_model import (
    PINNDegradation,
    PINNHeatingElements,
    PINNTrainConfig,
    compare_models,
    generate_training_data,
    train_pinn,
)
from model import HeatingElements, Inputs


# --- Data Generation ---

class TestDataGeneration:
    def test_returns_expected_keys(self):
        data = generate_training_data(n_trajectories=5, steps_per_trajectory=10, seed=42)
        assert set(data.keys()) == {"cumulative_load", "load", "health"}

    def test_correct_length(self):
        data = generate_training_data(n_trajectories=5, steps_per_trajectory=10, seed=42)
        assert len(data["health"]) == 50  # 5 * 10

    def test_health_in_range(self):
        data = generate_training_data(n_trajectories=10, steps_per_trajectory=50, seed=42)
        assert np.all(data["health"] >= 0.0)
        assert np.all(data["health"] <= 1.0)

    def test_cumulative_load_positive(self):
        data = generate_training_data(n_trajectories=5, steps_per_trajectory=10, seed=42)
        assert np.all(data["cumulative_load"] > 0.0)

    def test_deterministic(self):
        d1 = generate_training_data(n_trajectories=5, steps_per_trajectory=10, seed=99)
        d2 = generate_training_data(n_trajectories=5, steps_per_trajectory=10, seed=99)
        np.testing.assert_array_equal(d1["health"], d2["health"])


# --- PINN Network ---

class TestPINNNetwork:
    def test_forward_shape(self):
        model = PINNDegradation()
        x = torch.randn(16, 1)
        out = model(x)
        assert out.shape == (16, 1)

    def test_output_in_01(self):
        model = PINNDegradation()
        x = torch.randn(100, 1)
        out = model(x)
        assert (out >= 0.0).all()
        assert (out <= 1.0).all()

    def test_gradients_flow(self):
        model = PINNDegradation()
        x = torch.randn(1, 1, requires_grad=True)
        out = model(x)
        out.backward()
        assert x.grad is not None
        for p in model.parameters():
            assert p.grad is not None

    def test_autograd_for_physics(self):
        """Test that we can compute dH/dx via autograd (needed for physics loss)."""
        model = PINNDegradation()
        x = torch.tensor([[5.0]], requires_grad=True)
        h = model(x)
        dh_dx = torch.autograd.grad(h, x, create_graph=True)[0]
        assert dh_dx.shape == (1, 1)
        assert torch.isfinite(dh_dx).all()


# --- Training ---

class TestTraining:
    def test_short_training_completes(self):
        data = generate_training_data(n_trajectories=10, steps_per_trajectory=20, seed=42)
        model = train_pinn(data, PINNTrainConfig(epochs=50, log_interval=50, batch_size=64))
        assert isinstance(model, PINNDegradation)

    def test_trained_model_predicts_correctly(self):
        """After training, PINN should predict health close to analytical."""
        data = generate_training_data(n_trajectories=50, steps_per_trajectory=100, seed=42)
        model = train_pinn(data, PINNTrainConfig(
            epochs=500, log_interval=500, batch_size=512, seed=42
        ))
        # Test: at cumulative_load=0, health should be ~1.0
        with torch.no_grad():
            h_0 = model(torch.tensor([[0.1]])).item()
            assert h_0 > 0.9, f"Health at load~0 should be ~1.0, got {h_0}"

        # At high cumulative load, health should be low
        with torch.no_grad():
            h_high = model(torch.tensor([[500.0]])).item()
            assert h_high < 0.1, f"Health at high load should be low, got {h_high}"


# --- Drop-in Component ---

class TestPINNHeatingElements:
    def test_interface_matches(self):
        model = PINNDegradation()
        comp = PINNHeatingElements(model)
        assert comp.name == "HeatingElements"
        assert comp.health == 1.0
        assert comp.resistance_ohms == 10.0

    def test_update_returns_report(self):
        model = PINNDegradation()
        comp = PINNHeatingElements(model)
        inp = Inputs(temperature_stress=25.0, contamination=0.1,
                     operational_load=5.0, maintenance_level=0.5)
        report = comp.update(inp)
        assert 0.0 <= report.health_index <= 1.0
        assert "resistance_ohms" in report.metrics

    def test_reset(self):
        model = PINNDegradation()
        comp = PINNHeatingElements(model)
        inp = Inputs(temperature_stress=25.0, contamination=0.1,
                     operational_load=10.0, maintenance_level=0.5)
        comp.update(inp)
        comp.reset()
        assert comp.health == 1.0
        assert comp.resistance_ohms == 10.0
        assert comp._cumulative_load == 0.0


# --- Comparison ---

class TestComparison:
    def test_comparison_returns_stats(self):
        model = PINNDegradation()  # untrained, just test the function runs
        stats = compare_models(model, n_steps=10, seed=42)
        assert "mae" in stats
        assert "rmse" in stats
        assert "max_error" in stats
