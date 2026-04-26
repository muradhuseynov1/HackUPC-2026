"""
Physics-Informed Neural Network (PINN) — Degradation Model
============================================================
Replaces the HeatingElements' hand-tuned exponential decay formula
with a neural network trained on synthetic data, regularised by the
known ODE: dH/dt = -lambda * load * H(t).

Architecture:
  MLP: (cumulative_load, current_load) -> predicted_health
  Loss = MSE_data + lambda_phys * MSE_physics

The physics loss enforces the ODE via automatic differentiation:
  dH/d(cumulative_load) should equal -lambda * H

Usage:
  python pinn_model.py              # generate data, train, compare
  python pinn_model.py --epochs 5000
"""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from model import (
    Component,
    HeatingElements,
    Inputs,
    OperationalStatus,
    StateReport,
)


# =====================================================================
# 1. SYNTHETIC DATA GENERATION
# =====================================================================

def generate_training_data(
    n_trajectories: int = 500,
    steps_per_trajectory: int = 200,
    seed: int = 42,
) -> dict[str, np.ndarray]:
    """
    Run the analytical HeatingElements model many times with random loads
    to generate (cumulative_load, load, health) training triplets.
    """
    rng = np.random.default_rng(seed)
    cum_loads, loads, healths = [], [], []

    for _ in range(n_trajectories):
        heater = HeatingElements()
        for step in range(steps_per_trajectory):
            load = float(rng.uniform(0.5, 25.0))
            inp = Inputs(
                temperature_stress=25.0,
                contamination=0.1,
                operational_load=load,
                maintenance_level=0.5,
            )
            heater.update(inp)

            cum_loads.append(heater._cumulative_load)
            loads.append(load)
            healths.append(heater.health)

    return {
        "cumulative_load": np.array(cum_loads, dtype=np.float32),
        "load": np.array(loads, dtype=np.float32),
        "health": np.array(healths, dtype=np.float32),
    }


# =====================================================================
# 2. PINN NETWORK
# =====================================================================

class PINNDegradation(nn.Module):
    """
    Physics-Informed Neural Network for degradation prediction.

    Input:  cumulative_load (scalar, requires_grad for ODE loss)
    Output: predicted health H in [0, 1]

    Architecture:
        cumulative_load -> FC(64, Tanh) -> FC(64, Tanh) -> FC(32, Tanh) -> FC(1, Sigmoid)

    Uses Tanh activations (standard for PINNs — smooth, differentiable everywhere)
    and Sigmoid output to enforce H in [0, 1].
    """

    def __init__(self) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(1, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, 32),
            nn.Tanh(),
            nn.Linear(32, 1),
            nn.Sigmoid(),
        )

    def forward(self, cumulative_load: torch.Tensor) -> torch.Tensor:
        """Predict health from cumulative_load. Input shape: (N, 1)."""
        return self.net(cumulative_load)


# =====================================================================
# 3. TRAINING
# =====================================================================

@dataclass
class PINNTrainConfig:
    epochs: int = 3000
    lr: float = 1e-3
    lambda_physics: float = 1.0     # weight of physics loss vs data loss
    batch_size: int = 2048
    true_lambda: float = 0.008      # known decay constant from analytical model
    log_interval: int = 500
    seed: int = 42


def train_pinn(
    data: dict[str, np.ndarray],
    config: PINNTrainConfig | None = None,
) -> PINNDegradation:
    """Train the PINN on synthetic data with physics-informed loss."""
    cfg = config or PINNTrainConfig()
    torch.manual_seed(cfg.seed)

    model = PINNDegradation()
    optimizer = optim.Adam(model.parameters(), lr=cfg.lr)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, cfg.epochs)

    # Prepare data tensors
    cum_load = torch.tensor(data["cumulative_load"]).unsqueeze(1)  # (N, 1)
    health_true = torch.tensor(data["health"]).unsqueeze(1)        # (N, 1)
    n_samples = cum_load.shape[0]

    print("=" * 60)
    print("  PINN Training - HeatingElements Degradation")
    print(f"  Samples: {n_samples}  |  Epochs: {cfg.epochs}"
          f"  |  lambda_phys: {cfg.lambda_physics}")
    print("=" * 60)

    for epoch in range(1, cfg.epochs + 1):
        # --- Mini-batch sampling ---
        idx = torch.randperm(n_samples)[:cfg.batch_size]
        x_batch = cum_load[idx].requires_grad_(True)
        y_batch = health_true[idx]

        # --- Forward pass ---
        h_pred = model(x_batch)

        # --- Data loss: MSE ---
        data_loss = nn.functional.mse_loss(h_pred, y_batch)

        # --- Physics loss: enforce dH/d(cum_load) = -lambda * H ---
        # Use autograd to compute dH/d(cum_load)
        dh_dx = torch.autograd.grad(
            outputs=h_pred,
            inputs=x_batch,
            grad_outputs=torch.ones_like(h_pred),
            create_graph=True,
        )[0]

        # The ODE: dH/dx = -lambda * H  =>  residual = dH/dx + lambda * H = 0
        physics_residual = dh_dx + cfg.true_lambda * h_pred
        physics_loss = torch.mean(physics_residual ** 2)

        # --- Total loss ---
        loss = data_loss + cfg.lambda_physics * physics_loss

        # --- Backprop ---
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()

        if epoch % cfg.log_interval == 0:
            print(f"  Epoch {epoch:>5}/{cfg.epochs}"
                  f"  data_loss={data_loss.item():.6f}"
                  f"  physics_loss={physics_loss.item():.6f}"
                  f"  total={loss.item():.6f}")

    print("\n  Training complete.")
    return model


# =====================================================================
# 4. DROP-IN REPLACEMENT COMPONENT
# =====================================================================

class PINNHeatingElements(Component):
    """
    Drop-in replacement for HeatingElements that uses a trained PINN
    instead of the analytical formula H(t) = exp(-lambda * t).

    Interface is identical to the original HeatingElements class.
    """

    INITIAL_RESISTANCE: float = 10.0
    FAIL_RESISTANCE: float = 15.0

    def __init__(self, model: PINNDegradation) -> None:
        super().__init__(name="HeatingElements")
        self._pinn = model
        self._pinn.eval()
        self.resistance_ohms: float = self.INITIAL_RESISTANCE
        self._cumulative_load: float = 0.0

    def update(self, inputs: Inputs) -> StateReport:
        self._cumulative_load += inputs.operational_load

        # Use PINN to predict health
        with torch.no_grad():
            x = torch.tensor([[self._cumulative_load]], dtype=torch.float32)
            predicted_health = self._pinn(x).item()

        self.health = max(0.0, min(1.0, predicted_health))

        # Resistance mapping (same as original)
        resistance_range = self.FAIL_RESISTANCE - self.INITIAL_RESISTANCE
        self.resistance_ohms = (
            self.INITIAL_RESISTANCE + (1.0 - self._health) * resistance_range
        )

        return self._make_report(
            {"resistance_ohms": round(self.resistance_ohms, 4)}
        )

    def reset(self) -> None:
        super().reset()
        self.resistance_ohms = self.INITIAL_RESISTANCE
        self._cumulative_load = 0.0


# =====================================================================
# 5. COMPARISON: PINN vs ANALYTICAL
# =====================================================================

def compare_models(pinn_model: PINNDegradation, n_steps: int = 300,
                   seed: int = 42) -> dict[str, float]:
    """Run both models side-by-side and report error statistics."""
    rng = np.random.default_rng(seed)

    analytical = HeatingElements()
    pinn_comp = PINNHeatingElements(pinn_model)

    errors = []
    analytical_healths = []
    pinn_healths = []

    for step in range(n_steps):
        load = float(rng.uniform(1.0, 20.0))
        inp = Inputs(
            temperature_stress=25.0,
            contamination=0.1,
            operational_load=load,
            maintenance_level=0.5,
        )

        r_analytical = analytical.update(inp)
        r_pinn = pinn_comp.update(inp)

        analytical_healths.append(r_analytical.health_index)
        pinn_healths.append(r_pinn.health_index)
        errors.append(abs(r_analytical.health_index - r_pinn.health_index))

    stats = {
        "mae": float(np.mean(errors)),
        "max_error": float(np.max(errors)),
        "rmse": float(np.sqrt(np.mean(np.array(errors) ** 2))),
        "final_analytical": analytical_healths[-1],
        "final_pinn": pinn_healths[-1],
    }

    print("\n" + "=" * 60)
    print("  PINN vs Analytical Model Comparison")
    print("=" * 60)
    print(f"  Mean Absolute Error:   {stats['mae']:.6f}")
    print(f"  RMSE:                  {stats['rmse']:.6f}")
    print(f"  Max Error:             {stats['max_error']:.6f}")
    print(f"  Final Health (Analyt): {stats['final_analytical']:.6f}")
    print(f"  Final Health (PINN):   {stats['final_pinn']:.6f}")
    print()

    if stats["mae"] < 0.01:
        print("  [PASS] PINN matches analytical model within 1% MAE")
    elif stats["mae"] < 0.05:
        print("  [OK]   PINN approximates analytical model within 5% MAE")
    else:
        print("  [WARN] PINN has >5% MAE - needs more training")

    return stats


# =====================================================================
# 6. SAVE / LOAD
# =====================================================================

def save_pinn(model: PINNDegradation,
              path: str = "pinn_heater_model.pt") -> None:
    torch.save(model.state_dict(), path)
    print(f"  Model saved to {path}")


def load_pinn(path: str = "pinn_heater_model.pt") -> PINNDegradation:
    model = PINNDegradation()
    model.load_state_dict(torch.load(path, weights_only=True))
    model.eval()
    print(f"  Model loaded from {path}")
    return model


# =====================================================================
# CLI ENTRY POINT
# =====================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="PINN Degradation Model for HeatingElements"
    )
    parser.add_argument("--epochs", type=int, default=3000,
                        help="Training epochs (default: 3000)")
    parser.add_argument("--trajectories", type=int, default=500,
                        help="Number of synthetic trajectories (default: 500)")
    parser.add_argument("--lambda-phys", type=float, default=1.0,
                        help="Physics loss weight (default: 1.0)")
    parser.add_argument("--lr", type=float, default=1e-3,
                        help="Learning rate (default: 1e-3)")
    parser.add_argument("--eval-only", type=str, default=None,
                        help="Skip training; load model and evaluate")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if args.eval_only:
        pinn = load_pinn(args.eval_only)
    else:
        print("  Generating synthetic training data...")
        data = generate_training_data(
            n_trajectories=args.trajectories, seed=args.seed
        )
        print(f"  Generated {len(data['health'])} samples.\n")

        pinn = train_pinn(
            data,
            PINNTrainConfig(
                epochs=args.epochs,
                lr=args.lr,
                lambda_physics=args.lambda_phys,
                seed=args.seed,
            ),
        )
        save_pinn(pinn)

    compare_models(pinn, seed=args.seed)
