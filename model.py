"""
Phase 1 – Mathematical Logic Engine
=====================================
Deterministic, rule-based degradation model for three HP Metal Jet S100
subsystem components: Recoater Blade, Nozzle Plate, and Heating Elements.

No AI/ML, PINNs, or stochastic models. Classical math only.
"""

from __future__ import annotations

import math
import os
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


# ──────────────────────────────────────────────
# 2.1  Global Data Contracts – Inputs
# ──────────────────────────────────────────────

@dataclass(frozen=True)
class Inputs:
    """Standardized environmental & operational vector for a single time step."""

    temperature_stress: float
    """Ambient / internal temperature variance (°C)."""

    contamination: float
    """Powder purity / air moisture – 0.0 (clean) … 1.0 (highly contaminated)."""

    operational_load: float
    """Hours of active printing or cycles completed in the current step."""

    maintenance_level: float
    """Care coefficient – 0.0 (no maintenance) … 1.0 (perfect maintenance)."""

    def __post_init__(self) -> None:
        if not 0.0 <= self.contamination <= 1.0:
            raise ValueError(
                f"contamination must be in [0, 1], got {self.contamination}"
            )
        if not 0.0 <= self.maintenance_level <= 1.0:
            raise ValueError(
                f"maintenance_level must be in [0, 1], got {self.maintenance_level}"
            )
        if self.operational_load < 0.0:
            raise ValueError(
                f"operational_load must be >= 0, got {self.operational_load}"
            )


# ──────────────────────────────────────────────
# 2.2  Global Data Contracts – Outputs
# ──────────────────────────────────────────────

class OperationalStatus(Enum):
    """Discrete health bucket derived from the continuous health_index."""

    FUNCTIONAL = "FUNCTIONAL"
    DEGRADED = "DEGRADED"
    CRITICAL = "CRITICAL"
    FAILED = "FAILED"


def _health_to_status(health: float) -> OperationalStatus:
    """Map a health_index ∈ [0, 1] to an OperationalStatus."""
    if health > 0.7:
        return OperationalStatus.FUNCTIONAL
    if health > 0.3:
        return OperationalStatus.DEGRADED
    if health > 0.0:
        return OperationalStatus.CRITICAL
    return OperationalStatus.FAILED


@dataclass
class StateReport:
    """Output state for a single component after one time step."""

    health_index: float
    """Normalized value: 1.0 (perfect) … 0.0 (completely broken)."""

    operational_status: OperationalStatus
    """Discrete status derived from health_index."""

    metrics: dict[str, Any]
    """Component-specific custom physical variables."""


# ──────────────────────────────────────────────
# 3.  Component Base Class
# ──────────────────────────────────────────────

class Component(ABC):
    """Abstract base for every degradable subsystem component."""

    name: str

    def __init__(self, name: str) -> None:
        self.name = name
        self._health: float = 1.0  # start at perfect health

    @property
    def health(self) -> float:
        return self._health

    @health.setter
    def health(self, value: float) -> None:
        self._health = max(0.0, min(1.0, value))

    @abstractmethod
    def update(self, inputs: Inputs) -> StateReport:
        """Apply one time-step of degradation and return the new state."""
        ...

    def _make_report(self, metrics: dict[str, Any]) -> StateReport:
        return StateReport(
            health_index=round(self._health, 6),
            operational_status=_health_to_status(self._health),
            metrics=metrics,
        )

    def reset(self) -> None:
        """Restore component to factory-new condition."""
        self._health = 1.0


# ──────────────────────────────────────────────
# 3.1  Recoater Blade  (Linear Decay)
# ──────────────────────────────────────────────

class RecoaterBlade(Component):
    """
    Subsystem: Recoating
    Failure mode: Abrasive Wear
    Primary input driver: **contamination**

    Degradation math — Linear Decay driven by Contamination
    --------------------------------------------------------
    Higher contamination (powder impurity / humidity) causes
    exponentially faster abrasive wear on the blade.

    Custom metric: blade_thickness_mm
        starts at 2.0 mm, fails at 1.5 mm
    """

    INITIAL_THICKNESS_MM: float = 2.0
    FAIL_THICKNESS_MM: float = 1.5

    # Tuning constants
    BASE_WEAR_RATE: float = 0.01    # baseline health loss per step
    CONTAMINATION_EXP: float = 2.0  # exponent for contamination driver

    def __init__(self) -> None:
        super().__init__(name="RecoaterBlade")
        self.blade_thickness_mm: float = self.INITIAL_THICKNESS_MM

    def update(self, inputs: Inputs) -> StateReport:
        # Primary driver: contamination drives wear exponentially
        wear = self.BASE_WEAR_RATE * (1.0 + inputs.contamination ** self.CONTAMINATION_EXP * 5.0)

        self.health -= wear

        # Map health linearly to blade thickness
        thickness_range = self.INITIAL_THICKNESS_MM - self.FAIL_THICKNESS_MM
        self.blade_thickness_mm = (
            self.FAIL_THICKNESS_MM + self._health * thickness_range
        )

        return self._make_report(
            {"blade_thickness_mm": round(self.blade_thickness_mm, 4)}
        )

    def reset(self) -> None:
        super().reset()
        self.blade_thickness_mm = self.INITIAL_THICKNESS_MM


# ──────────────────────────────────────────────
# 3.2  Nozzle Plate  (Weibull-inspired / Threshold)
# ──────────────────────────────────────────────

class NozzlePlate(Component):
    """
    Subsystem: Printhead Array
    Failure mode: Clogging & Thermal Fatigue
    Primary input driver: **temperature_stress**

    Degradation math — Weibull-inspired / Threshold-based Decay
    ------------------------------------------------------------
    If temperature stress exceeds an optimal bound the health
    drops sharply via a Weibull hazard function.

    Custom metric: clogging_percentage
        starts at 0.0 %, fails at 100.0 %
    """

    OPTIMAL_TEMP: float = 25.0     # deg-C optimal operating variance
    TEMP_SHAPE: float = 2.5        # Weibull shape parameter (beta)
    TEMP_SCALE: float = 50.0       # Weibull scale parameter (eta, deg-C)
    BASE_WEAR_RATE: float = 0.005  # small baseline health loss per step

    def __init__(self) -> None:
        super().__init__(name="NozzlePlate")
        self.clogging_percentage: float = 0.0

    def update(self, inputs: Inputs) -> StateReport:
        # Primary driver: temperature_stress
        temp_excess = max(0.0, abs(inputs.temperature_stress) - self.OPTIMAL_TEMP)

        # Weibull hazard contribution: (beta/eta) * (x/eta)^(beta-1)
        if temp_excess > 0.0:
            thermal_damage = (
                (self.TEMP_SHAPE / self.TEMP_SCALE)
                * (temp_excess / self.TEMP_SCALE) ** (self.TEMP_SHAPE - 1)
            )
        else:
            thermal_damage = 0.0

        total_damage = self.BASE_WEAR_RATE + thermal_damage
        self.health -= total_damage

        # Clogging is the inverse of health
        self.clogging_percentage = (1.0 - self._health) * 100.0

        return self._make_report(
            {"clogging_percentage": round(self.clogging_percentage, 4)}
        )

    def reset(self) -> None:
        super().reset()
        self.clogging_percentage = 0.0


# ──────────────────────────────────────────────
# 3.3  Heating Elements  (Exponential Decay)
# ──────────────────────────────────────────────

class HeatingElements(Component):
    """
    Subsystem: Thermal Control
    Failure mode: Electrical Degradation
    Primary input driver: **operational_load**

    Degradation math — Standard Exponential Decay
    -----------------------------------------------
    H(t) = e^(-lambda * t)  where t is cumulative operational_load.

    Custom metric: resistance_ohms
        starts at 10.0 Ohms, fails at > 15.0 Ohms
    """

    INITIAL_RESISTANCE: float = 10.0
    FAIL_RESISTANCE: float = 15.0
    BASE_LAMBDA: float = 0.008  # decay constant per unit operational_load

    def __init__(self) -> None:
        super().__init__(name="HeatingElements")
        self.resistance_ohms: float = self.INITIAL_RESISTANCE
        self._cumulative_load: float = 0.0

    def update(self, inputs: Inputs) -> StateReport:
        # Primary driver: operational_load accumulates over time
        self._cumulative_load += inputs.operational_load

        # Standard exponential decay: H(t) = e^(-lambda * t)
        self.health = math.exp(-self.BASE_LAMBDA * self._cumulative_load)

        # Resistance rises as health falls
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


# ──────────────────────────────────────────────
# 4.  Engine – top-level orchestrator
# ──────────────────────────────────────────────

class Engine:
    """
    Main Logic Engine.

    Manages a set of components and advances them through time
    steps deterministically.
    """

    def __init__(self, use_pinn: bool = False) -> None:
        heater: Component
        if use_pinn:
            heater = self._load_pinn_heater()
        else:
            heater = HeatingElements()

        self.components: list[Component] = [
            RecoaterBlade(),
            NozzlePlate(),
            heater,
        ]
        self._step: int = 0
        self._use_pinn = use_pinn

    @staticmethod
    def _load_pinn_heater() -> Component:
        """Load the trained PINN model and return a PINNHeatingElements component."""
        from pinn_model import PINNHeatingElements, load_pinn
        model_path = os.path.join(os.path.dirname(__file__), "pinn_heater_model.pt")
        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"PINN model not found at {model_path}. "
                f"Train it first: python pinn_model.py"
            )
        pinn = load_pinn(model_path)
        return PINNHeatingElements(pinn)

    def update_state(self, inputs: Inputs) -> dict[str, StateReport]:
        """
        Apply one time-step of degradation to every component.

        Parameters
        ----------
        inputs : Inputs
            Environmental & operational vector for this step.

        Returns
        -------
        dict[str, StateReport]
            Mapping of component name → its updated state report.
        """
        self._step += 1
        reports: dict[str, StateReport] = {}
        for component in self.components:
            report = component.update(inputs)
            reports[component.name] = report
        return reports

    def get_all_states(self) -> dict[str, StateReport]:
        """Return the current state of every component without advancing."""
        return {
            comp.name: comp._make_report(self._current_metrics(comp))
            for comp in self.components
        }

    def reset(self) -> None:
        """Reset every component to factory-new."""
        self._step = 0
        for comp in self.components:
            comp.reset()

    @property
    def step(self) -> int:
        return self._step

    # ----- helpers -----

    @staticmethod
    def _current_metrics(comp: Component) -> dict[str, Any]:
        if isinstance(comp, RecoaterBlade):
            return {"blade_thickness_mm": round(comp.blade_thickness_mm, 4)}
        if isinstance(comp, NozzlePlate):
            return {"clogging_percentage": round(comp.clogging_percentage, 4)}
        # HeatingElements or PINNHeatingElements — both have resistance_ohms
        if hasattr(comp, "resistance_ohms"):
            return {"resistance_ohms": round(comp.resistance_ohms, 4)}
        return {}


# ──────────────────────────────────────────────
# Quick smoke test (runs only when executed directly)
# ──────────────────────────────────────────────

if __name__ == "__main__":
    engine = Engine()

    sample_inputs = Inputs(
        temperature_stress=30.0,
        contamination=0.3,
        operational_load=10.0,
        maintenance_level=0.7,
    )

    print("=" * 60)
    print("  HP Metal Jet S100 - Logic Engine  (Phase 1)")
    print("=" * 60)

    for step in range(1, 11):
        reports = engine.update_state(sample_inputs)
        print(f"\n-- Step {step} --")
        for name, report in reports.items():
            print(
                f"  {name:20s}  "
                f"health={report.health_index:.4f}  "
                f"status={report.operational_status.value:12s}  "
                f"metrics={report.metrics}"
            )
