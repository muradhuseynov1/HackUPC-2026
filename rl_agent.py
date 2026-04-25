"""
RL Maintenance Agent — Advantage Actor-Critic (A2C)
=====================================================
Trains a reinforcement learning agent to discover the optimal
maintenance policy for the HP Metal Jet S100 Digital Twin.

The agent observes component health and environmental conditions,
then decides which components to maintain at each simulation step.

Architecture:
  - Environment: wraps Phase 1 engine; state = healths + drivers
  - Actor-Critic: shared MLP backbone, policy (actor) + value (critic) heads
  - Training: A2C with advantage estimation and entropy bonus

Run:
  python rl_agent.py                 # train + compare vs rule-based
  python rl_agent.py --episodes 5000 # custom episode count
"""

from __future__ import annotations

import argparse
import math
import random
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from model import Engine as PhysicsEngine, Inputs, OperationalStatus


# ═══════════════════════════════════════════════════════════════════════════
# 1. ENVIRONMENT
# ═══════════════════════════════════════════════════════════════════════════

# Environmental scenario profiles for training diversity
_SCENARIOS = [
    {"temp_mean": 25.0,  "temp_std": 5.0,  "load_mean": 5.0,  "load_std": 1.0, "cont_mean": 0.15, "cont_std": 0.05},
    {"temp_mean": 60.0,  "temp_std": 15.0, "load_mean": 8.0,  "load_std": 2.0, "cont_mean": 0.30, "cont_std": 0.08},
    {"temp_mean": 120.0, "temp_std": 25.0, "load_mean": 12.0, "load_std": 3.0, "cont_mean": 0.50, "cont_std": 0.10},
    {"temp_mean": 180.0, "temp_std": 10.0, "load_mean": 10.0, "load_std": 2.0, "cont_mean": 0.20, "cont_std": 0.05},
]


@dataclass
class EnvConfig:
    """Knobs for the RL environment."""
    max_steps: int = 200
    maintenance_cost: float = 0.5       # small penalty per maintenance action
    failure_penalty: float = 10.0       # one-time penalty when a component first fails
    seed: int | None = None


class PrinterEnv:
    """
    Gym-like environment wrapping the Phase 1 physics engine.

    State (9-dim continuous):
        [blade_health, nozzle_health, heater_health,
         blade_velocity, nozzle_velocity, heater_velocity,
         norm_temperature, norm_load, norm_contamination]
        velocity = health change from previous step (negative = degrading)

    Action (8 discrete — 3-bit encoding):
        bit 0 = maintain blade, bit 1 = maintain nozzle, bit 2 = maintain heater
        e.g. action=5 → binary 101 → maintain blade + heater, skip nozzle

    Reward (per step):
        + sum(health_i)          continuous uptime signal (0-3)
        - maintenance_cost       per maintenance action taken
        - failure_penalty        ONE-TIME when component first fails
        Episode ends if ANY component fails (teaches prevention).
    """

    STATE_DIM = 9
    ACTION_DIM = 8  # 2^3 combinations

    def __init__(self, config: EnvConfig | None = None) -> None:
        self.config = config or EnvConfig()
        self.rng = np.random.default_rng(self.config.seed)
        self._engine = PhysicsEngine()
        self._step_count = 0
        self._scenario: dict[str, float] = {}
        self._done = False
        self._prev_health = np.array([1.0, 1.0, 1.0], dtype=np.float32)
        self._failed_set: set[str] = set()  # track first-time failures

    def reset(self) -> np.ndarray:
        """Reset environment for a new episode. Returns initial state."""
        self._engine.reset()
        self._step_count = 0
        self._done = False
        self._prev_health = np.array([1.0, 1.0, 1.0], dtype=np.float32)
        self._failed_set = set()
        # Pick a random scenario for training diversity
        self._scenario = self.rng.choice(_SCENARIOS)  # type: ignore[arg-type]
        return self._get_state()

    def step(self, action: int) -> tuple[np.ndarray, float, bool, dict[str, Any]]:
        """
        Execute one time step.

        Parameters
        ----------
        action : int  (0–7)
            3-bit encoded maintenance decision.

        Returns
        -------
        (next_state, reward, done, info)
        """
        assert 0 <= action < self.ACTION_DIM, f"Invalid action: {action}"
        assert not self._done, "Episode already done — call reset()"

        # --- Decode action into maintenance decisions ---
        maintain_blade  = bool(action & 1)
        maintain_nozzle = bool(action & 2)
        maintain_heater = bool(action & 4)

        # Apply maintenance BEFORE physics step (reset = full repair)
        maintenance_count = 0
        for comp in self._engine.components:
            if comp.name == "RecoaterBlade" and maintain_blade:
                comp.reset(); maintenance_count += 1
            elif comp.name == "NozzlePlate" and maintain_nozzle:
                comp.reset(); maintenance_count += 1
            elif comp.name == "HeatingElements" and maintain_heater:
                comp.reset(); maintenance_count += 1

        # --- Generate environmental drivers ---
        s = self._scenario
        temperature = max(0.0, self.rng.normal(s["temp_mean"], s["temp_std"]))
        load = max(0.0, self.rng.normal(s["load_mean"], s["load_std"]))
        contamination = float(np.clip(
            self.rng.normal(s["cont_mean"], s["cont_std"]), 0.0, 1.0
        ))

        inputs = Inputs(
            temperature_stress=temperature,
            contamination=contamination,
            operational_load=load,
            maintenance_level=0.6,
        )

        # --- Run physics ---
        reports = self._engine.update_state(inputs)
        self._step_count += 1

        # --- Calculate reward ---
        # Continuous: reward proportional to health (smooth gradient signal)
        reward = sum(r.health_index for r in reports.values())

        # One-time failure penalty (only on first transition to FAILED)
        new_failures = 0
        for name, report in reports.items():
            if report.operational_status == OperationalStatus.FAILED:
                if name not in self._failed_set:
                    reward -= self.config.failure_penalty
                    self._failed_set.add(name)
                    new_failures += 1

        # Maintenance cost
        reward -= maintenance_count * self.config.maintenance_cost

        failures = len(self._failed_set)

        # --- Check termination ---
        # End on ANY failure — teaches the agent to prevent failures
        self._done = (
            self._step_count >= self.config.max_steps
            or new_failures > 0
        )

        info = {
            "step": self._step_count,
            "maintenance_count": maintenance_count,
            "failures": failures,
            "reports": reports,
        }

        return self._get_state(), reward, self._done, info

    def _get_state(self) -> np.ndarray:
        """Build the 9-dim normalized state vector."""
        healths = np.array([comp.health for comp in self._engine.components],
                          dtype=np.float32)
        velocities = healths - self._prev_health  # negative = degrading
        self._prev_health = healths.copy()
        # Use scenario means as normalized driver proxies (0-1 range)
        s = self._scenario if self._scenario else _SCENARIOS[0]
        norm_temp = min(s.get("temp_mean", 25.0) / 200.0, 1.0)
        norm_load = min(s.get("load_mean", 5.0) / 20.0, 1.0)
        norm_cont = s.get("cont_mean", 0.15)
        return np.concatenate([
            healths,
            velocities,
            np.array([norm_temp, norm_load, norm_cont], dtype=np.float32),
        ])


# ═══════════════════════════════════════════════════════════════════════════
# 2. ACTOR-CRITIC NETWORK
# ═══════════════════════════════════════════════════════════════════════════

class ActorCritic(nn.Module):
    """
    Shared-backbone Actor-Critic network.

    Architecture:
        Input (9) -> FC(128, ReLU) -> FC(64, ReLU) --+-- Actor  -> FC(8, Softmax)
                                                     +-- Critic -> FC(1)
    """

    def __init__(self, state_dim: int = 9, action_dim: int = 8) -> None:
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
        )
        self.actor = nn.Linear(64, action_dim)   # policy logits
        self.critic = nn.Linear(64, 1)            # state value

    def forward(self, state: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Returns
        -------
        (action_probs, state_value)
            action_probs : (batch, action_dim) probability distribution
            state_value  : (batch, 1) estimated V(s)
        """
        features = self.backbone(state)
        action_probs = F.softmax(self.actor(features), dim=-1)
        state_value = self.critic(features)
        return action_probs, state_value

    def select_action(self, state: np.ndarray) -> tuple[int, torch.Tensor, torch.Tensor]:
        """Sample an action from the policy, returning (action, log_prob, value)."""
        state_t = torch.FloatTensor(state).unsqueeze(0)
        probs, value = self(state_t)
        dist = torch.distributions.Categorical(probs)
        action = dist.sample()
        return action.item(), dist.log_prob(action), value.squeeze()


# ═══════════════════════════════════════════════════════════════════════════
# 3. A2C TRAINER
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class TrainConfig:
    """Hyperparameters for A2C training."""
    episodes: int = 3000
    gamma: float = 0.99           # discount factor
    lr: float = 1e-3              # learning rate (higher for faster learning)
    entropy_coeff: float = 0.02   # entropy bonus for exploration
    value_coeff: float = 0.5      # critic loss weight
    max_grad_norm: float = 0.5    # gradient clipping
    log_interval: int = 100       # print every N episodes
    seed: int = 42


class A2CTrainer:
    """Advantage Actor-Critic trainer."""

    def __init__(self, train_config: TrainConfig | None = None,
                 env_config: EnvConfig | None = None) -> None:
        self.tc = train_config or TrainConfig()
        self.ec = env_config or EnvConfig()

        torch.manual_seed(self.tc.seed)
        np.random.seed(self.tc.seed)

        self.env = PrinterEnv(self.ec)
        self.model = ActorCritic(PrinterEnv.STATE_DIM, PrinterEnv.ACTION_DIM)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.tc.lr)

        # Tracking
        self.episode_rewards: list[float] = []
        self.episode_lengths: list[int] = []

    def train(self) -> ActorCritic:
        """Run the full training loop. Returns the trained model."""
        print("=" * 60)
        print("  A2C Training - RL Maintenance Agent")
        print(f"  Episodes: {self.tc.episodes}  |  gamma: {self.tc.gamma}"
              f"  |  lr: {self.tc.lr}")
        print("=" * 60)

        for episode in range(1, self.tc.episodes + 1):
            ep_reward, ep_length = self._run_episode()
            self.episode_rewards.append(ep_reward)
            self.episode_lengths.append(ep_length)

            if episode % self.tc.log_interval == 0:
                avg_r = np.mean(self.episode_rewards[-self.tc.log_interval:])
                avg_l = np.mean(self.episode_lengths[-self.tc.log_interval:])
                print(f"  Episode {episode:>5}/{self.tc.episodes}"
                      f"  avg_reward={avg_r:8.1f}  avg_length={avg_l:5.1f}")

        print("\n  Training complete.")
        return self.model

    def _run_episode(self) -> tuple[float, int]:
        """Run one episode, collect transitions, update model."""
        state = self.env.reset()
        log_probs: list[torch.Tensor] = []
        values: list[torch.Tensor] = []
        rewards: list[float] = []
        entropies: list[torch.Tensor] = []

        done = False
        while not done:
            action, log_prob, value = self.model.select_action(state)
            next_state, reward, done, info = self.env.step(action)

            # Compute entropy for this step
            state_t = torch.FloatTensor(state).unsqueeze(0)
            probs, _ = self.model(state_t)
            dist = torch.distributions.Categorical(probs)
            entropies.append(dist.entropy())

            log_probs.append(log_prob)
            values.append(value)
            rewards.append(reward)
            state = next_state

        # --- Compute returns and advantages ---
        returns = self._compute_returns(rewards)
        returns_t = torch.FloatTensor(returns)
        values_t = torch.stack(values)
        log_probs_t = torch.stack(log_probs)
        entropies_t = torch.stack(entropies)

        advantage = returns_t - values_t.detach()

        # --- Losses ---
        actor_loss = -(log_probs_t * advantage).mean()
        critic_loss = F.mse_loss(values_t, returns_t)
        entropy_loss = -entropies_t.mean()

        loss = (
            actor_loss
            + self.tc.value_coeff * critic_loss
            + self.tc.entropy_coeff * entropy_loss
        )

        # --- Backprop ---
        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.model.parameters(), self.tc.max_grad_norm)
        self.optimizer.step()

        return sum(rewards), len(rewards)

    def _compute_returns(self, rewards: list[float]) -> list[float]:
        """Discounted returns (Monte Carlo)."""
        returns = []
        R = 0.0
        for r in reversed(rewards):
            R = r + self.tc.gamma * R
            returns.insert(0, R)
        return returns


# ═══════════════════════════════════════════════════════════════════════════
# 4. EVALUATION — RL vs RULE-BASED
# ═══════════════════════════════════════════════════════════════════════════

def evaluate_rl_agent(model: ActorCritic, env: PrinterEnv,
                      num_episodes: int = 50) -> dict[str, float]:
    """Run the trained RL agent and collect stats."""
    model.eval()
    total_rewards, total_lengths, total_maintenance, total_failures = [], [], [], []

    for _ in range(num_episodes):
        state = env.reset()
        ep_reward, ep_maint, ep_fail = 0.0, 0, 0
        done = False
        steps = 0
        while not done:
            with torch.no_grad():
                state_t = torch.FloatTensor(state).unsqueeze(0)
                probs, _ = model(state_t)
                action = probs.argmax(dim=-1).item()  # greedy
            state, reward, done, info = env.step(action)
            ep_reward += reward
            ep_maint += info["maintenance_count"]
            ep_fail += info["failures"]
            steps += 1

        total_rewards.append(ep_reward)
        total_lengths.append(steps)
        total_maintenance.append(ep_maint)
        total_failures.append(ep_fail)

    return {
        "avg_reward": float(np.mean(total_rewards)),
        "avg_length": float(np.mean(total_lengths)),
        "avg_maintenance": float(np.mean(total_maintenance)),
        "avg_failures": float(np.mean(total_failures)),
    }


def evaluate_rule_based(env: PrinterEnv, threshold: float = 0.15,
                        num_episodes: int = 50) -> dict[str, float]:
    """Run the ProactiveAgent-style rule-based policy and collect stats."""
    total_rewards, total_lengths, total_maintenance, total_failures = [], [], [], []

    for _ in range(num_episodes):
        state = env.reset()
        ep_reward, ep_maint, ep_fail = 0.0, 0, 0
        done = False
        steps = 0
        prev_health = [1.0, 1.0, 1.0]

        while not done:
            # Rule-based: maintain if predicted next-step health < threshold
            action = 0
            for i in range(3):
                decay_rate = prev_health[i] - state[i]
                predicted = state[i] - decay_rate
                if predicted <= threshold:
                    action |= (1 << i)  # set bit i
                prev_health[i] = state[i] if not (action & (1 << i)) else 1.0

            state, reward, done, info = env.step(action)
            ep_reward += reward
            ep_maint += info["maintenance_count"]
            ep_fail += info["failures"]
            steps += 1

        total_rewards.append(ep_reward)
        total_lengths.append(steps)
        total_maintenance.append(ep_maint)
        total_failures.append(ep_fail)

    return {
        "avg_reward": float(np.mean(total_rewards)),
        "avg_length": float(np.mean(total_lengths)),
        "avg_maintenance": float(np.mean(total_maintenance)),
        "avg_failures": float(np.mean(total_failures)),
    }


def compare(model: ActorCritic, num_episodes: int = 50,
            seed: int = 123) -> None:
    """Print a side-by-side comparison of RL vs rule-based agents."""
    env_cfg = EnvConfig(seed=seed)
    env = PrinterEnv(env_cfg)

    rl_stats = evaluate_rl_agent(model, env, num_episodes)

    # Reset env with same seed for fair comparison
    env2 = PrinterEnv(EnvConfig(seed=seed))
    rb_stats = evaluate_rule_based(env2, num_episodes=num_episodes)

    print("\n" + "=" * 60)
    print("  RL Agent  vs  Rule-Based Agent  (head-to-head)")
    print("=" * 60)
    print(f"  {'Metric':<25s} {'RL Agent':>12s} {'Rule-Based':>12s} {'Winner':>10s}")
    print(f"  {'-'*25} {'-'*12} {'-'*12} {'-'*10}")

    for key in ["avg_reward", "avg_length", "avg_maintenance", "avg_failures"]:
        rl_val = rl_stats[key]
        rb_val = rb_stats[key]
        # Higher reward/length is better; lower maintenance/failures is better
        if key in ("avg_reward", "avg_length"):
            winner = "RL *" if rl_val > rb_val else ("Rule *" if rb_val > rl_val else "Tie")
        else:
            winner = "RL *" if rl_val < rb_val else ("Rule *" if rb_val < rl_val else "Tie")
        print(f"  {key:<25s} {rl_val:>12.1f} {rb_val:>12.1f} {winner:>10s}")

    print()
    improvement = rl_stats["avg_reward"] - rb_stats["avg_reward"]
    pct = (improvement / abs(rb_stats["avg_reward"]) * 100) if rb_stats["avg_reward"] != 0 else 0
    if improvement > 0:
        print(f"  [WIN] RL agent outperforms rule-based by {improvement:.1f} reward ({pct:+.1f}%)")
    elif improvement < 0:
        print(f"  [LOSS] Rule-based outperforms RL agent by {-improvement:.1f} reward ({-pct:+.1f}%)")
    else:
        print("  [TIE] Both agents perform equally.")
    print()


# ═══════════════════════════════════════════════════════════════════════════
# 5. SAVE / LOAD
# ═══════════════════════════════════════════════════════════════════════════

def save_model(model: ActorCritic, path: str = "rl_maintenance_agent.pt") -> None:
    torch.save(model.state_dict(), path)
    print(f"  Model saved to {path}")


def load_model(path: str = "rl_maintenance_agent.pt") -> ActorCritic:
    model = ActorCritic()
    model.load_state_dict(torch.load(path, weights_only=True))
    model.eval()
    print(f"  Model loaded from {path}")
    return model


# ═══════════════════════════════════════════════════════════════════════════
# CLI ENTRY POINT
# ═══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="RL Maintenance Agent (A2C)")
    parser.add_argument("--episodes", type=int, default=3000,
                        help="Number of training episodes (default: 3000)")
    parser.add_argument("--max-steps", type=int, default=200,
                        help="Max steps per episode (default: 200)")
    parser.add_argument("--lr", type=float, default=1e-3,
                        help="Learning rate (default: 1e-3)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed (default: 42)")
    parser.add_argument("--eval-only", type=str, default=None,
                        help="Skip training; load model from path and evaluate")
    args = parser.parse_args()

    if args.eval_only:
        model = load_model(args.eval_only)
    else:
        trainer = A2CTrainer(
            train_config=TrainConfig(
                episodes=args.episodes,
                lr=args.lr,
                seed=args.seed,
            ),
            env_config=EnvConfig(
                max_steps=args.max_steps,
                seed=args.seed,
            ),
        )
        model = trainer.train()
        save_model(model)

    compare(model, num_episodes=50, seed=args.seed)
