# Phase 1: Mathematical Logic Engine (Copilot Prompt Specification)

## 1. Project Overview & Objective
This document serves as the architectural and mathematical specification for **Phase 1: Model** of the HP Metal Jet S100 Digital Twin hackathon. 

**Goal:** Generate Python code to build a deterministic, rule-based **Logic Engine** that calculates component degradation over time. 
**Constraint:** Do NOT use AI/ML, PINNs, or stochastic models for this phase. Use classical mathematical formulas (e.g., Exponential Decay, Linear Wear, Weibull).

---

## 2. Global Data Contracts

### 2.1 Inputs (Environmental & Operational Vectors)
The Engine must accept a standardized input object (e.g., a Python `dataclass` or `dict`) containing the following drivers at each time step:
* **`temperature_stress`** (float): Ambient or internal temperature variance (Celsius).
* **`contamination`** (float): Powder purity / air moisture (normalized 0.0 to 1.0, where 1.0 is highly contaminated).
* **`operational_load`** (float): Hours of active printing or cycles completed in the current step.
* **`maintenance_level`** (float): Coefficient representing care (0.0 to 1.0, where 1.0 is perfect maintenance).

### 2.2 Outputs (State Report)
For every component, the Engine must return a structured state containing:
* **`health_index`** (float): Normalized value from `1.0` (Perfect) to `0.0` (Completely broken).
* **`operational_status`** (string/Enum): Must be one of `FUNCTIONAL`, `DEGRADED`, `CRITICAL`, or `FAILED`.
    * *Logic:* * `1.0 >= health > 0.7` -> `FUNCTIONAL`
        * `0.7 >= health > 0.3` -> `DEGRADED`
        * `0.3 >= health > 0.0` -> `CRITICAL`
        * `health <= 0.0` -> `FAILED`
* **`metrics`** (dict): Component-specific custom physical variables.

---

## 3. Component Specifications (The 3 Targets)

Please implement an Object-Oriented design (e.g., a base `Component` class and three subclasses) for the following components.

### 3.1 Recoater Blade (Subsystem: Recoating)
* **Primary Function:** Spreads thin layers of metal powder.
* **Failure Mode:** Abrasive Wear.
* **Degradation Math (Linear Decay with Multipliers):** * The wear rate is directly proportional to `operational_load`.
    * `contamination` acts as an aggressive multiplier (higher contamination = exponentially faster abrasive wear).
    * `maintenance_level` slows down the baseline wear rate.
* **Custom Metric:** `blade_thickness_mm` (Starts at 2.0 mm, fails at 1.5 mm).

### 3.2 Nozzle Plate (Subsystem: Printhead Array)
* **Primary Function:** Jeta binder liquid.
* **Failure Mode:** Clogging & Thermal Fatigue.
* **Degradation Math (Weibull-inspired / Threshold-based Decay):**
    * Highly sensitive to `temperature_stress`. If temperature stress exceeds an optimal bound, health drops sharply.
    * Wear accumulates via `operational_load`.
* **Custom Metric:** `clogging_percentage` (Starts at 0.0%, fails at 100.0%).

### 3.3 Heating Elements (Subsystem: Thermal Control)
* **Primary Function:** Maintains optimal build temperature.
* **Failure Mode:** Electrical Degradation.
* **Degradation Math (Standard Exponential Decay):**
    * Follows standard electrical degradation curve: $H(t) = H_0 * e^{-\lambda t}$, where $t$ is `operational_load`.
    * `maintenance_level` can periodically reset or flatten the $\lambda$ (decay constant).
* **Custom Metric:** `resistance_ohms` (Starts at 10.0 Ohms, increases as it degrades, fails at > 15.0 Ohms).

---

## 4. Implementation Instructions for Copilot
1. **Language:** Strict Python 3.10+.
2. **Typing:** Use the `typing` module extensively (`TypedDict`, `dataclass`, `Enum` for operational states).
3. **Modularity:** Ensure the main `Engine` class has an `update_state(inputs: Inputs)` method that iterates through all components, applies their specific math functions, and returns a compiled dictionary of all state reports.
4. **Determinism:** Two runs with the exact same sequence of inputs must produce the exact same sequence of outputs. Do not use `random`.
