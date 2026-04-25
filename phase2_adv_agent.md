# Phase 2: Proactive Maintenance Agent (Copilot Prompt Spec)

## 1. Project Overview & Objective
This document outlines the specification for the **Proactive Maintenance Agent**, a "Go-Further" feature for Phase 2 of the HP Metal Jet S100 Digital Twin.
**Goal:** Generate a Python class that acts as an inline watcher within the `asyncio` simulation loop (`engine.py`). The agent must monitor the health of components, calculate their degradation velocity, and autonomously trigger a maintenance reset (health = 1.0) *before* the component fails (health <= 0.0).

## 2. Core Logic: Velocity-Based Lookahead
Because the Phase 1 Engine currently uses deterministic mathematical equations, the Agent will use a localized derivative (velocity) approach to predict failure.

**The Algorithm:**
1. **Observe:** Compare the component's `current_health` to its `previous_health` to find the `decay_rate` per tick.
2. **Predict:** Project the health for the *next* tick: `predicted_health = current_health - decay_rate`.
3. **Evaluate Thresholds:** If `predicted_health <= CRITICAL_THRESHOLD`, trigger an intervention.
4. **Act:** Override the health to `1.0` and flag a proactive maintenance event for the UI/Historian.

## 3. Class Specification: `ProactiveAgent`

### 3.1 Initialization & State Management
* Create a class `ProactiveAgent`.
* `__init__(self, critical_threshold: float = 0.15)`
* Use a dictionary to store the previous health state of each component: `self.state_memory = {}` (e.g., `{"Recoater Blade": 0.85}`).

### 3.2 Main Method: `evaluate_and_act`
* **Inputs:** * `component_name` (str)
  * `current_health` (float)
* **Outputs:** * `final_health` (float): Returns `1.0` if maintained, otherwise returns `current_health`.
  * `action_taken` (bool): `True` if maintenance was triggered.
  * `log_message` (str): E.g., `"OK"` or `"[PROACTIVE] Agent intervened. Nozzle Plate projected to fail next tick."`
* **Logic Flow:**
  1. If `component_name` is not in `self.state_memory`, store `current_health` and return `(current_health, False, "OK")`.
  2. Calculate `decay_rate = self.state_memory[component_name] - current_health`.
  3. Calculate `predicted_health = current_health - decay_rate`.
  4. If `predicted_health <= self.critical_threshold`:
     * Set `self.state_memory[component_name] = 1.0`.
     * Return `(1.0, True, f"[PROACTIVE] Agent reset {component_name} at {current_health:.2f} health.")`.
  5. Else:
     * Update `self.state_memory[component_name] = current_health`.
     * Return `(current_health, False, "OK")`.

## 4. Integration Instructions for `engine.py`
Instruct Copilot to insert this Agent directly into the existing `asyncio` simulation loop:
1. Instantiate `agent = ProactiveAgent(critical_threshold=0.15)` *outside* the `while True` loop.
2. Inside the loop, *after* calling the Phase 1 model but *before* writing to the SQLite database, pass the raw health values through the agent.
3. Example flow:
   ```python
   raw_blade_health = phase1_engine.calculate_blade(...)
   final_blade_health, blade_maintained, blade_log = agent.evaluate_and_act("Recoater Blade", raw_blade_health)