# Phase 3: Intelligent Agentic Interface (Copilot Prompt Spec)

## 1. Project Overview & Objective
This document outlines the architectural specification for **Phase 3: Interact** of the HP Metal Jet S100 Digital Twin.
**Goal:** Generate a Python-based intelligent interface (accessible via Streamlit text chat). This interface must move beyond simple tool-calling to implement a full **Agentic ReAct Loop**, feature **Collaborative Persistent Memory**, actively monitor for **Proactive Alerts**, and safely manipulate the Phase 2 `SimulationConfig` via a **Neurosymbolic Firewall**.

## 2. Global AI Directives & Setup
* **Modality:** Text-only interface (no voice).
* **Grounding Protocol (Zero Hallucination):** The LLM must strictly source answers from the SQLite Historian.
* **Output Format:** Every final response must include:
  1. A clear textual explanation.
  2. A severity indicator (`INFO`, `WARNING`, `CRITICAL`).
  3. Explicit timestamp/run ID citations.
* **Collaborative Memory (Persistent):** Chat history must survive session reloads. Do not use Streamlit's ephemeral `st.session_state` alone. Store all chat messages in a new SQLite table called `chat_history` (`id`, `timestamp`, `role`, `content`) and load them on startup.

## 3. Pattern C: The Explicit ReAct Loop (Agentic Diagnosis)
The LLM must not just call tools; it must reason step-by-step to diagnose root causes.

### 3.1 ReAct Loop Architecture
Instruct Copilot to implement a strict ReAct (Reasoning and Acting) parser or use a framework like LangChain's `create_react_agent`. The loop must follow this exact trace format until it reaches a conclusion:
1. **Thought:** The LLM explains what it needs to figure out.
2. **Action:** The LLM selects a tool (e.g., `get_telemetry`).
3. **Action Input:** The parameters for the tool.
4. **Observation:** The Python backend executes the tool and returns the raw data.
5. *(Loop repeats until the LLM has enough data)*
6. **Final Answer:** The LLM generates the grounded response.

### 3.2 The Diagnostic Tools (Available to the Agent)
Provide these explicit tools for the Agent to use:
* `get_telemetry(start_time, end_time, component_name)`: Queries the Phase 2 SQLite `telemetry_log`.
* `get_failure_logs()`: Returns only rows where health is `0.0` or a component state is `FAILED`.
* `get_simulation_config()`: Returns the current `SimulationConfig` (Total Duration, Time Step, Environmental Profile variables).

## 4. Proactive Intelligence (Background Monitoring)
The interface must not wait for the user to ask questions if the machine is failing.

### 4.1 The Background Alerter
* Implement a background polling mechanism in Streamlit (e.g., using `st_autorefresh` or an asynchronous thread).
* **Logic:** Every 5 seconds, query the `telemetry_log` for the latest timestep. If any component's health drops below a `WARNING` (e.g., 0.3) or `CRITICAL` (0.0) threshold, automatically push a message into the chat interface:
  `"🚨 **PROACTIVE ALERT:** Heating Elements health has dropped to 0.28 at simulated hour 42. Would you like me to diagnose the root cause?"`
* Ensure alerts are deduplicated (do not spam the same alert every 5 seconds).

## 5. Neurosymbolic AI & SimulationConfig (The Action Layer)
When the Agent diagnoses a problem, it can adjust the running simulation parameters.

### 5.1 The Safety Envelope Dictionary
Define absolute physical limits to constrain the LLM.
```python
SAFETY_ENVELOPE = {
    "temperature_stress": {"max": 220.0, "min": 20.0},
    "contamination": {"max": 1.0, "min": 0.0},
    "operational_load": {"max": 100.0, "min": 0.0}
}