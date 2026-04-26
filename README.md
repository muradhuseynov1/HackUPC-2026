# HP Metal Jet S100 — Digital Twin & AI Co-Pilot

> HackUPC 2026 · HP Challenge: *When AI Meets Reality*

A full-stack Digital Twin for the HP Metal Jet S100 3D metal printer, featuring physics-based degradation models, a real-time simulation engine, an AI diagnostic co-pilot, a Physics-Informed Neural Network (PINN), and a Reinforcement Learning maintenance agent.

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        Phase 3: Interact                        │
│  phase3_chat.py — Agentic AI Co-Pilot (ReAct + Tool Use)       │
│  Queries historian via get_telemetry() tool                     │
└──────────────────────────┬──────────────────────────────────────┘
                           │ reads from
┌──────────────────────────▼──────────────────────────────────────┐
│                        Phase 2: Simulate                        │
│  engine.py — Simulation Loop + State Machine + Historian        │
│  app.py — Streamlit Dashboard (health charts, failure log)      │
│  telemetry.db — SQLite historian (timestamped telemetry)        │
│  ProactiveAgent — Velocity-based predictive maintenance         │
└──────────────────────────┬──────────────────────────────────────┘
                           │ calls at every tick
┌──────────────────────────▼──────────────────────────────────────┐
│                        Phase 1: Model                           │
│  model.py — Physics Engine (3 degradation models)               │
│  ├─ RecoaterBlade   (Archard linear wear)                       │
│  ├─ NozzlePlate     (Arrhenius + clogging)                      │
│  └─ HeatingElements (Exponential decay / PINN)                  │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│                     Go-Further Extensions                       │
│  pinn_model.py — Physics-Informed Neural Network (Phase 1 #4)   │
│  rl_agent.py   — A2C Actor-Critic RL Agent (Phase 2 #4)         │
└─────────────────────────────────────────────────────────────────┘
```

---

## Quick Start

### Prerequisites

- Python 3.10+
- An OpenAI API key (for Phase 3 AI co-pilot)

### Installation

```bash
git clone https://github.com/muradhuseynov1/HackUPC-2026.git
cd HackUPC-2026

pip install numpy torch streamlit openai pandas streamlit-autorefresh
```

### Environment Variables

```bash
# Required for Phase 3 only
export OPENAI_API_KEY="sk-..."
```

---

## Running the Project

### Phase 1 — Physics Engine (standalone test)

```bash
python -c "from model import Engine, Inputs; e = Engine(); r = e.update_state(Inputs(25.0, 0.1, 5.0, 0.5)); print({k: round(v.health_index,4) for k,v in r.items()})"
```

### Phase 2 — Simulation + Dashboard

Run in two terminals:

```bash
# Terminal 1: Start the simulation engine
python engine.py

# Terminal 2: Start the Streamlit dashboard
streamlit run app.py
```

The dashboard at `http://localhost:8501` shows:
- Real-time health metrics (3 components)
- Time-series health decay charts
- Failure analysis log
- Simulation controls (temperature offset, production volume, anomaly injection)

### Phase 3 — AI Co-Pilot

```bash
streamlit run phase3_chat.py
```

Ask questions like:
- *"What is the current health of all components?"*
- *"When did the recoater blade first reach CRITICAL status?"*
- *"Compare the last two simulation runs."*

### PINN — Physics-Informed Neural Network

```bash
# Train the PINN (generates synthetic data, trains, compares)
python pinn_model.py

# Evaluate a saved model
python pinn_model.py --eval-only pinn_heater_model.pt
```

To run the simulation with the PINN-powered HeatingElements:
```python
from model import Engine
engine = Engine(use_pinn=True)  # swaps HeatingElements for PINN
```

### RL Agent — Reinforcement Learning Maintenance Policy

```bash
# Train the A2C agent (default: 3000 episodes)
python rl_agent.py

# Custom training
python rl_agent.py --episodes 5000 --max-steps 200

# Evaluate a saved model
python rl_agent.py --eval-only rl_maintenance_agent.pt
```

---

## Running Tests

```bash
# All tests
python -m pytest

# Individual suites
python -m pytest test_model.py     # Phase 1 engine (42 tests)
python -m pytest test_engine.py    # Phase 2 simulation (25 tests)
python -m pytest test_phase3.py    # Phase 3 AI interface (25 tests)
python -m pytest test_pinn.py      # PINN model (15 tests)
python -m pytest test_rl_agent.py  # RL agent (25 tests)
```

---

## Component Degradation Models

### RecoaterBlade — Archard Linear Wear
- **Subsystem:** Recoating System
- **Primary driver:** `contamination`
- **Physics:** Blade thickness decreases linearly with contamination exposure. Higher contamination accelerates abrasive wear.
- **Failure:** Health reaches 0 when blade is worn down to minimum thickness.

### NozzlePlate — Arrhenius Thermal + Clogging
- **Subsystem:** Printhead Array
- **Primary driver:** `temperature_stress`
- **Physics:** Nozzle clogging follows an Arrhenius rate equation — clogging rate increases exponentially with temperature. Above optimal temperature bounds, thermal fatigue accelerates.
- **Failure:** Health drops below threshold when clogging exceeds critical percentage.

### HeatingElements — Exponential Decay (or PINN)
- **Subsystem:** Thermal Control
- **Primary driver:** `operational_load`
- **Physics:** H(t) = exp(-λ · cumulative_load) where λ = 0.008. Resistance rises as health falls.
- **PINN mode:** Neural network trained with physics-informed loss enforcing the ODE dH/dt = -λ·load·H(t). Achieves 0.36% MAE vs analytical formula.
- **Failure:** Health reaches 0 when resistance exceeds 15 Ω.

---

## Project Structure

```
HackUPC-2026/
├── model.py              # Phase 1 — Physics Engine (3 components)
├── engine.py             # Phase 2 — Simulation Loop + Historian
├── app.py                # Phase 2 — Streamlit Dashboard
├── phase3_chat.py        # Phase 3 — AI Co-Pilot (Agentic ReAct)
├── pinn_model.py         # Go-Further — PINN Degradation Model
├── rl_agent.py           # Go-Further — A2C RL Maintenance Agent
├── sim_config.json       # Runtime simulation configuration
├── telemetry.db          # SQLite historian (generated at runtime)
├── pinn_heater_model.pt  # Trained PINN weights
├── rl_maintenance_agent.pt # Trained RL agent weights
├── test_model.py         # Tests: Phase 1
├── test_engine.py        # Tests: Phase 2
├── test_phase3.py        # Tests: Phase 3
├── test_pinn.py          # Tests: PINN
├── test_rl_agent.py      # Tests: RL Agent
└── README.md             # This file
```

---

## Technical Highlights

| Feature | Details |
|:--------|:--------|
| **Degradation Models** | 3 models: linear wear, Arrhenius + clogging, exponential decay |
| **State Machine** | 4-phase printer cycle: IDLE → HEATING → PRINTING → COOLDOWN |
| **Historian** | SQLite with component health, physical metrics, timestamps, run IDs |
| **AI Co-Pilot** | Agentic ReAct pattern (Pattern C) with tool-based historian queries |
| **Grounding** | All AI responses cite specific timestamps and data points |
| **PINN** | Physics-Informed Neural Network with ODE loss via autograd |
| **RL Agent** | A2C Actor-Critic, 9-dim state, 8-action space, 0% failure rate |
| **Proactive Agent** | Rule-based velocity-decay predictive maintenance |
| **Test Coverage** | 132+ tests across 5 test suites |

---

## Team

Built at HackUPC 2026 for the HP "When AI Meets Reality" challenge.

- **Giorgia Barboni**
- **Murad Hüseynov**
- **Riccardo Bastiani**
