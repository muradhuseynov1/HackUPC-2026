# Phase 2: Simulate

## 2.1 Mission & Goal

Transform your Phase 1 **Mathematical Brain** into a dynamic, time-advancing **Digital Twin**. You are building the **Simulation Engine (The Clock)** and the **Historical Record (The Historian)**.

The goal is to move from a static calculation to a continuous simulation that evolves over time, creating a rich, timestamped history of the printer's health and operational state.

---

## 2.2 The Integration Bridge (Connecting Phase 1 to Phase 2)

**Crucial Instruction:** Your Phase 2 engine is not a separate project; it is the "driver" for your Phase 1 engine. You must bridge the two phases through a clear interface.

* **The Relationship:** Your Phase 2 Simulation Loop must call your Phase 1 Logic Engine at every time step.
* **The Data Flow:**
    1. **Phase 2** generates or receives the environmental and operational input drivers for the current time step — the ones you defined in Phase 1.
    2. **Phase 2** passes these inputs to the **Phase 1** engine.
    3. **Phase 1** calculates the new state and returns it.
    4. **Phase 2** captures that state, attaches a timestamp, and saves it to a database.

**Driver Definition:** The drivers are the input values that change over time, such as temperature, humidity, contamination, load, and maintenance level. Time itself is owned by the Phase 2 loop; Phase 1 receives the current driver values, not the clock.

**If your Phase 1 engine is not modular and callable by an external loop, your Phase 2 will fail.**

---

## 2.3 Implementation Modes

Your simulation can be built in two modes — both are valid and **neither affects your evaluation score**. Choose the one that best fits your architecture.

### Mode 1: Batch Simulation

The engine runs a predefined scenario from start to finish. Once the simulation is complete, it exports a single, complete historical file (e.g., a CSV or JSON file). This is the simpler approach and a solid foundation.

### Mode 2: Real-Time Streaming

The engine mimics a live machine. It "emits" data points one by one in real-time, simulating a continuous telemetry stream. This is more complex but creates a compelling demo.

---

## 2.4 Simulation Design Patterns (Architectural Choices)

When designing your simulation, you must choose an architectural pattern. This choice determines how your "Clock" (the simulation loop) interacts with your "Brain" (the Phase 1 engine).

**Note:** You are only required to implement **one** of the following patterns. Pattern A is the minimum baseline for passing this phase.

| Pattern | Requirement Level | Core Logic | Architectural Impact |
| :--- | :--- | :--- | :--- |
| **A: Deterministic Replay** | **Minimum Baseline** | The simulation follows a fixed, pre-defined path of environmental inputs. | **Simple Loop:** The engine reads sequential inputs from a file and applies them linearly. |
| **B: Informed Extrapolation** | **Advanced** | The engine ingests historical data to "sync" the current state, then predicts the future. | **Two-Phase Execution:** Requires a "Sync Phase" to initialize the model, followed by a "Prediction Phase." |
| **C: Stochastic Simulation** | **Advanced** | The simulation introduces non-linear "noise" or random variables into the environment. | **Noise Engine:** Requires implementing a randomness generator within the input loop. |

---

## 2.5 The Data Contract (Inputs & Outputs)

### 2.5.1 Expected Inputs (The Configuration)

To start the simulation, your engine must accept a `SimulationConfig` object:

* **Total Duration:** The total simulated time to be modeled.
* **Time Step (Resolution):** The interval between updates (e.g., 1 minute, 1 hour).
* **Environmental Profile:** A sequence, function, or scenario definition that describes how the full set of input drivers expected by Phase 1 changes over the simulation.

### 2.5.2 Expected Outputs (The Digital History)

Your engine must produce a **Historical Log** (The Historian). This log must be a structured collection of every state change, containing:

* **Timestamp:** The exact simulated time of the event.
* **Component States:** The full output from your Phase 1 Engine (Health, Status, and Metrics).
* **Contextual Metadata:** The input vectors that caused that specific state.

If you run multiple scenarios, persist each one with a unique run or scenario identifier so the historical records remain queryable and comparable.

---

## 2.6 Minimum Requirements

To successfully complete this phase, your engine must meet these requirements:

1. **The Simulation Loop:** A functional "Clock" that advances time according to your defined time step.
2. **Persistence (The Basic Historian):** Every state must be saved to a structured format (e.g., **CSV, JSON, or SQLite**).
3. **Integration:** Successful, automated execution of the Phase 1 engine within the Phase 2 loop.

---

## 2.7 Go Further

The minimum gives you a working twin. These ideas can make it remarkable. They are starting points — feel free to push them further or invent your own.

**1. What-if Scenarios**
Run the same machine under radically different conditions: high-humidity factory floor vs. dry climate-controlled lab, aggressive 24/7 production vs. light usage. How does the degradation story change? Let the data make the argument.

**2. Chaos Engineering**
Inject random shocks into the simulation — a sudden temperature spike, a contamination burst, an unexpected maintenance skip. Can your twin survive the unexpected gracefully, or does it spiral into cascading failures? Real machines aren't predictable; neither should your simulation be.

**3. AI Maintenance Agent**
Build an agent that watches the simulation state in real time and autonomously decides *when* to trigger a maintenance event (using the maintenance model you built in Phase 1) to maximise printer uptime. How much longer can it keep the machine healthy compared to a fixed maintenance schedule?

**4. Reinforcement Learning Policy**
Take the Maintenance Agent further: instead of hand-coded rules, train an RL agent across thousands of simulation episodes to discover the *optimal* maintenance policy. No rules — just reward. Can it beat a human-designed schedule?

**5. Digital Twin Synchronisation**
Feed your twin a stream of "real" sensor readings (even if synthetic) and watch it self-correct its predicted state against actual observations. How closely can a simulation track reality when the world doesn't behave as expected?

---

## 2.8 Deliverables & Evaluation

### What to deliver

* **Time Series Visualization** — a running plot or dashboard showing component health evolving over simulated time.
* **Degradation Model explanation** — a clear description of how each component's decay logic was designed and implemented.
* **Failure Analysis** — identify when each component fails during the simulation and explain why.

### How it will be evaluated

| Criterion | Description |
| :--- | :--- |
| **Time moves** | The simulation loop advances time correctly and consistently. |
| **Systemic Integration** | The Phase 1 engine is called at every time step and results are persisted to a time-series record. |
| **Complexity & Innovation** *(bonus)* | Advanced behaviours such as scenario analysis, chaos injection, or an AI Maintenance Agent. |
| **Realism & Fidelity** *(bonus)* | Degradation curves reflect physically plausible behaviour; the simulation handles non-linear events gracefully. |

> **Tip:** A convincing time-series plot of component health declining over time — with at least one component reaching `FAILED` — is one of the most impactful things you can show in the demo.

---
[Back to Main Brief](./hackathon.md)
