"""
Phase 2 - Simulation Engine (Script A)
========================================
Background process that:
  1. Runs a printer State Machine (IDLE -> HEATING -> PRINTING -> COOLDOWN)
  2. Generates realistic synthetic telemetry with Gaussian noise
  3. Feeds each tick into the Phase 1 mathematical model
  4. Persists every row to a shared SQLite database

Run:  python engine.py
"""

from __future__ import annotations

import asyncio
import json
import math
import sqlite3
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any

import numpy as np

# Phase 1 model lives in model.py
from model import Engine as PhysicsEngine
from model import Inputs

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
DB_PATH = Path(__file__).parent / "telemetry.db"
CONFIG_PATH = Path(__file__).parent / "sim_config.json"

TICK_INTERVAL_S = 1.0  # real-world seconds between ticks


# ---------------------------------------------------------------------------
# Printer State Machine
# ---------------------------------------------------------------------------

class PrinterState(Enum):
    """
    The printer cycles through four phases in order:
        IDLE  ->  HEATING  ->  PRINTING  ->  COOLDOWN  -> (back to IDLE)

    Each state defines baseline temperature, load, and contamination,
    plus the duration (in ticks) before transitioning to the next state.
    """
    IDLE = "IDLE"
    HEATING = "HEATING"
    PRINTING = "PRINTING"
    COOLDOWN = "COOLDOWN"


# base profiles: (temperature_C, load, contamination, duration_ticks)
_STATE_PROFILES: dict[PrinterState, dict[str, Any]] = {
    PrinterState.IDLE: {
        "temp": 25.0,
        "load": 0.0,
        "contamination": 0.05,
        "duration": 10,
    },
    PrinterState.HEATING: {
        "temp": 180.0,   # temperature rises sharply toward target
        "load": 5.0,     # load spikes during ramp-up
        "contamination": 0.1,
        "duration": 15,
    },
    PrinterState.PRINTING: {
        "temp": 180.0,   # plateaus at target
        "load": 8.0,     # steady high load
        "contamination": 0.25,
        "duration": 60,
    },
    PrinterState.COOLDOWN: {
        "temp": 25.0,    # drops back exponentially
        "load": 0.5,
        "contamination": 0.08,
        "duration": 20,
    },
}

_STATE_ORDER = [
    PrinterState.IDLE,
    PrinterState.HEATING,
    PrinterState.PRINTING,
    PrinterState.COOLDOWN,
]


@dataclass
class StateMachine:
    """
    Tracks which phase the printer is in and how many ticks remain before
    the next transition.  Provides smoothed driver values (e.g. exponential
    ramp for HEATING / COOLDOWN instead of instant jumps).
    """
    current: PrinterState = PrinterState.IDLE
    ticks_in_state: int = 0
    _prev_temp: float = 25.0

    def tick(self) -> dict[str, float]:
        """Advance one tick and return the base (noise-free) driver values."""
        profile = _STATE_PROFILES[self.current]
        duration = profile["duration"]
        self.ticks_in_state += 1

        # --- Temperature smoothing ---
        target_temp = profile["temp"]
        if self.current == PrinterState.HEATING:
            # exponential approach toward target
            alpha = 1 - math.exp(-3.0 * self.ticks_in_state / duration)
            temp = 25.0 + (target_temp - 25.0) * alpha
        elif self.current == PrinterState.COOLDOWN:
            # exponential decay from previous temp back to 25 C
            alpha = math.exp(-3.0 * self.ticks_in_state / duration)
            temp = 25.0 + (self._prev_temp - 25.0) * alpha
        else:
            temp = target_temp

        load = profile["load"]
        contamination = profile["contamination"]

        # --- Transition logic ---
        if self.ticks_in_state >= duration:
            idx = _STATE_ORDER.index(self.current)
            self._prev_temp = temp
            self.current = _STATE_ORDER[(idx + 1) % len(_STATE_ORDER)]
            self.ticks_in_state = 0

        return {
            "temperature": temp,
            "load": load,
            "contamination": contamination,
        }


# ---------------------------------------------------------------------------
# Database helpers
# ---------------------------------------------------------------------------

def _init_db(conn: sqlite3.Connection) -> None:
    """Create the telemetry_log table if it does not exist."""
    conn.execute("""
        CREATE TABLE IF NOT EXISTS telemetry_log (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp       TEXT    NOT NULL,
            run_id          TEXT    NOT NULL,
            printer_state   TEXT    NOT NULL,
            temperature     REAL    NOT NULL,
            load            REAL    NOT NULL,
            contamination   REAL    NOT NULL,
            blade_health    REAL    NOT NULL,
            nozzle_health   REAL    NOT NULL,
            heater_health   REAL    NOT NULL,
            failure_log     TEXT
        )
    """)
    conn.commit()


def _insert_row(
    conn: sqlite3.Connection,
    *,
    run_id: str,
    printer_state: str,
    timestamp: str,
    temperature: float,
    load: float,
    contamination: float,
    blade_health: float,
    nozzle_health: float,
    heater_health: float,
    failure_log: str | None,
) -> None:
    conn.execute(
        """
        INSERT INTO telemetry_log
            (timestamp, run_id, printer_state,
             temperature, load, contamination,
             blade_health, nozzle_health, heater_health,
             failure_log)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            timestamp,
            run_id,
            printer_state,
            temperature,
            load,
            contamination,
            blade_health,
            nozzle_health,
            heater_health,
            failure_log,
        ),
    )
    conn.commit()


# ---------------------------------------------------------------------------
# Config reader (sidebar writes, engine reads)
# ---------------------------------------------------------------------------

def _read_config() -> dict[str, Any]:
    """
    Read user-adjustable parameters written by the Streamlit sidebar.
    Returns sensible defaults if the file does not exist yet.
    """
    defaults = {
        "base_temperature_offset": 0.0,
        "production_volume": 1.0,       # multiplier on load
        "inject_thermal_anomaly": False,
    }
    if CONFIG_PATH.exists():
        try:
            with open(CONFIG_PATH, "r") as f:
                data = json.load(f)
            defaults.update(data)
            # Reset one-shot flags after reading
            if data.get("inject_thermal_anomaly"):
                data["inject_thermal_anomaly"] = False
                with open(CONFIG_PATH, "w") as f:
                    json.dump(data, f)
        except (json.JSONDecodeError, OSError):
            pass
    return defaults


# ---------------------------------------------------------------------------
# Proactive Maintenance Agent (Phase 2 Go-Further)
# ---------------------------------------------------------------------------

class ProactiveAgent:
    """
    Velocity-based lookahead maintenance agent.

    Monitors component health decay rates and autonomously triggers a
    maintenance reset (health -> 1.0) *before* a component fails.

    Algorithm:
        1. Observe:  decay_rate = previous_health - current_health
        2. Predict:  predicted_health = current_health - decay_rate
        3. Evaluate: if predicted_health <= critical_threshold -> intervene
        4. Act:      reset health to 1.0 and flag proactive maintenance
    """

    def __init__(self, critical_threshold: float = 0.15) -> None:
        self.critical_threshold = critical_threshold
        self.state_memory: dict[str, float] = {}  # {component_name: prev_health}

    def evaluate_and_act(
        self,
        component_name: str,
        current_health: float,
    ) -> tuple[float, bool, str]:
        """
        Evaluate a component's health and decide whether to intervene.

        Returns
        -------
        (final_health, action_taken, log_message)
            final_health  : 1.0 if maintained, else current_health
            action_taken  : True if maintenance was triggered
            log_message   : descriptive string for the historian
        """
        # First observation — store and pass through
        if component_name not in self.state_memory:
            self.state_memory[component_name] = current_health
            return (current_health, False, "OK")

        # Calculate local derivative (decay velocity per tick)
        decay_rate = self.state_memory[component_name] - current_health

        # Project health one tick into the future
        predicted_health = current_health - decay_rate

        if predicted_health <= self.critical_threshold:
            # Intervene: proactive maintenance reset
            self.state_memory[component_name] = 1.0
            msg = (
                f"[PROACTIVE] Agent reset {component_name} "
                f"at {current_health:.2f} health "
                f"(predicted next-tick: {predicted_health:.2f})."
            )
            return (1.0, True, msg)

        # No intervention needed — update memory
        self.state_memory[component_name] = current_health
        return (current_health, False, "OK")


# ---------------------------------------------------------------------------
# Main simulation loop
# ---------------------------------------------------------------------------

async def run_simulation() -> None:
    """Async loop: tick state machine, add noise, run physics, persist."""

    run_id = uuid.uuid4().hex[:12]
    print(f"[engine] Starting run {run_id}  (DB: {DB_PATH})")

    conn = sqlite3.connect(str(DB_PATH))
    _init_db(conn)

    physics = PhysicsEngine()
    sm = StateMachine()
    agent = ProactiveAgent(critical_threshold=0.15)
    rng = np.random.default_rng(seed=None)  # real randomness for sensor noise

    tick = 0
    while True:
        tick += 1
        cfg = _read_config()

        # 1. State machine produces base drivers
        base = sm.tick()

        # 2. Apply user config offsets
        temperature = base["temperature"] + cfg["base_temperature_offset"]
        load = base["load"] * cfg["production_volume"]
        contamination = base["contamination"]

        # Thermal anomaly: spike temp for this tick
        if cfg.get("inject_thermal_anomaly"):
            temperature += 120.0  # massive spike

        # 3. Gaussian noise for realistic sensor variance
        temperature += rng.normal(0, 1.5)
        load = max(0.0, load + rng.normal(0, 0.3))
        contamination = float(np.clip(contamination + rng.normal(0, 0.02), 0.0, 1.0))

        # 4. Build Phase 1 Inputs and run physics
        inputs = Inputs(
            temperature_stress=temperature,
            contamination=contamination,
            operational_load=load,
            maintenance_level=0.6,  # fixed maintenance schedule
        )
        reports = physics.update_state(inputs)

        blade_h = reports["RecoaterBlade"].health_index
        nozzle_h = reports["NozzlePlate"].health_index
        heater_h = reports["HeatingElements"].health_index

        # 5. Proactive Maintenance Agent — evaluate each component
        #    Runs AFTER physics but BEFORE DB write.
        #    If the agent intervenes, it resets health to 1.0 and
        #    also resets the corresponding physics component.
        now_str = datetime.now(timezone.utc).isoformat()
        agent_logs: list[str] = []

        blade_h, blade_maint, blade_log = agent.evaluate_and_act("RecoaterBlade", blade_h)
        if blade_maint:
            for comp in physics.components:
                if comp.name == "RecoaterBlade":
                    comp.reset()
            agent_logs.append(blade_log)

        nozzle_h, nozzle_maint, nozzle_log = agent.evaluate_and_act("NozzlePlate", nozzle_h)
        if nozzle_maint:
            for comp in physics.components:
                if comp.name == "NozzlePlate":
                    comp.reset()
            agent_logs.append(nozzle_log)

        heater_h, heater_maint, heater_log = agent.evaluate_and_act("HeatingElements", heater_h)
        if heater_maint:
            for comp in physics.components:
                if comp.name == "HeatingElements":
                    comp.reset()
            agent_logs.append(heater_log)

        # 6. Build failure / maintenance log string
        failures: list[str] = []
        for name, rpt in reports.items():
            if rpt.operational_status.value == "FAILED":
                failures.append(f"[{now_str}] {name} FAILED.")
        all_events = agent_logs + failures
        failure_log = " | ".join(all_events) if all_events else None

        # Print agent interventions immediately
        for log_msg in agent_logs:
            print(f"  ** {log_msg}")

        # 7. Persist to SQLite
        _insert_row(
            conn,
            run_id=run_id,
            printer_state=sm.current.value,
            timestamp=now_str,
            temperature=round(temperature, 3),
            load=round(load, 3),
            contamination=round(contamination, 4),
            blade_health=blade_h,
            nozzle_health=nozzle_h,
            heater_health=heater_h,
            failure_log=failure_log,
        )

        # 8. Console heartbeat (every 10 ticks)
        if tick % 10 == 0:
            print(
                f"  tick {tick:>5}  state={sm.current.value:10s}  "
                f"T={temperature:6.1f}  L={load:5.2f}  C={contamination:.3f}  "
                f"blade={blade_h:.3f}  nozzle={nozzle_h:.3f}  heater={heater_h:.3f}"
            )

        await asyncio.sleep(TICK_INTERVAL_S)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=" * 60)
    print("  HP Metal Jet S100 - Simulation Engine  (Phase 2)")
    print("=" * 60)
    try:
        asyncio.run(run_simulation())
    except KeyboardInterrupt:
        print("\n[engine] Stopped by user.")
