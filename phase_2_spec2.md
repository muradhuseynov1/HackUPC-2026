# Phase 2: Simulation Engine & Proactive Agent (Copilot Prompt Spec)

## 1. Project Overview & Objective
This document outlines the specification for **Phase 2: Simulate**, integrating both the core Simulation Engine and the "Go-Further" **Proactive Maintenance Agent**.
**Goal:** Generate a decoupled, real-time Python simulation architecture using `asyncio`. The engine must strictly adhere to a `SimulationConfig` object to bound execution time, advance by specific time steps, calculate component health via the Phase 1 model, run the data through a predictive agent, and persist everything to SQLite.

---

## 2. Global Data Contracts

### 2.1 Simulation Configuration (`SimulationConfig`)
To ensure the simulation does not run indefinitely, the engine must be initialized with a `SimulationConfig` object. 
**Instruction for Copilot:** Implement this explicitly as a Python `@dataclass`.
* **`total_duration`** (float or int): The maximum total simulated time to be modeled (e.g., total hours). 
* **`time_step`** (float or int): The time interval advanced per simulation tick.
* **`environmental_profile`** (Callable or Dict): A function or sequence defining the base environment (Temperature, Contamination, Load) for a given state or time.

---

## 3. The Proactive Agent Class
The Agent acts as an inline watcher that predicts failures using a localized derivative (velocity) approach.

### 3.1 Class Specification: `ProactiveAgent`
* **`__init__(self, critical_threshold: float = 0.15)`**: Initialize `self.state_memory = {}`.
* **`evaluate_and_act(self, component_name: str, current_health: float)`**:
  1. Compare `current_health` against the stored value in `self.state_memory` to calculate the `decay_rate` per tick.
  2. Project the next tick: `predicted_health = current_health - decay_rate`.
  3. **Act:** If `predicted_health <= self.critical_threshold`, override health to `1.0`, reset memory, and return `(1.0, True, "[PROACTIVE] Agent reset component.")`.
  4. Otherwise, update memory and return `(current_health, False, "OK")`.

---

## 4. Script Architecture: `engine.py` (The Clock)
This script runs the actual simulation loop in the background.

### 4.1 Database Setup (The Historian)
* Initialize `sqlite3` with a table `telemetry_log`: `timestamp` (datetime), `current_time` (float), `temperature`, `load`, `contamination`, `blade_health`, `nozzle_health`, `heater_health`, `event_log` (string).

### 4.2 The Execution Loop (`asyncio`)
**Instruction for Copilot:** Do NOT use an infinite `while True` loop. The loop must be driven by `SimulationConfig`.

```python
# Example logic flow to generate:
async def run_simulation(config: SimulationConfig):
    current_time = 0.0
    agent = ProactiveAgent(critical_threshold=0.15)
    
    while current_time < config.total_duration:
        # 1. Fetch environmental drivers from config.environmental_profile based on state
        # 2. Add Numpy Gaussian noise for realism
        # 3. Call Phase 1 Engine to get raw health values
        # 4. Pass raw health values through the ProactiveAgent
        final_blade, blade_maintained, blade_log = agent.evaluate_and_act("Blade", raw_blade)
        
        # 5. Insert final values and logs into SQLite
        # 6. Advance Time
        current_time += config.time_step
        await asyncio.sleep(1) # Real-world delay for streaming effect