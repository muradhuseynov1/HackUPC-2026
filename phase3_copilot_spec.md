# Phase 3: Intelligent Interface (Copilot Prompt Spec)

## 1. Project Overview & Objective
This document outlines the architectural specification for **Phase 3: Interact** of the HP Metal Jet S100 Digital Twin.
**Goal:** Generate a Python-based intelligent interface (accessible via Streamlit text chat) that implements **Contextual RAG (Structured Text-to-SQL)** and a **Neurosymbolic AI Firewall**. The LLM must query the Phase 2 SQLite database to answer historical questions and must pass any parameter change requests through hard-coded safety bounds.

## 2. Global AI Directives & Setup
* **Modality:** Text-only interface (no voice).
* **Grounding Protocol (Zero Hallucination):** The LLM must not guess or answer from its training data. Every answer must be sourced from the `sqlite` database.
* **Output Format:** Every response must include:
  1. A clear textual explanation.
  2. A severity indicator (`INFO`, `WARNING`, `CRITICAL`).
  3. Explicit timestamp citations based on the retrieved rows.

## 3. The Contextual RAG Pipeline (Structured Retrieval)
Instead of semantic vector search, use Tool Calling / Function Calling.

### 3.1 The Retrieval Tool Schema
Define a JSON schema/function tool for the LLM called `get_telemetry`.
* **Tool Name:** `get_telemetry`
* **Parameters:**
  * `start_time` (datetime or int)
  * `end_time` (datetime or int)
  * `component_name` (string)
* **Description:** "Fetches the historical health index, operational state, and environmental drivers (temp, load, contamination) for a specific component over a requested time window from the Phase 2 Historian."

### 3.2 The Python Execution Loop
When the LLM outputs a request to use `get_telemetry`:
1. Intercept the tool call.
2. Formulate a SQL Query against the `telemetry_log` table:
   `SELECT * FROM telemetry_log WHERE timestamp BETWEEN ? AND ?`
3. Execute the query via `sqlite3`.
4. Inject the raw rows back into the LLM context for it to generate the final human-readable response.

## 4. Neurosymbolic AI (The Deterministic Firewall)
Implement a formal logic gate to protect the simulation from dangerous LLM commands.

### 4.1 The Safety Envelope Dictionary
Define the absolute physical limits of the machine.
```python
SAFETY_ENVELOPE = {
    "Heating Elements": {"max_temp": 220, "min_temp": 150},
    "Print Speed": {"max_speed_multiplier": 1.2, "min_speed_multiplier": 0.5},
    "Recoater Blade": {"max_load": 5.0} # example threshold
}
```

### 4.2 The Action Tool Schema
Give the LLM a tool to alter the simulation state.
* **Tool Name:** `adjust_machine_parameter`
* **Parameters:** `component` (string), `parameter` (string), `new_value` (float).
* **Description:** "Use this tool to adjust the machine's running parameters if the user asks for a fix."

### 4.3 The Firewall Middleware Function
Create a function `deterministic_firewall(component, parameter, requested_value)` that intercepts the LLM's action call.
1. Lookup the `component` and `parameter` in `SAFETY_ENVELOPE`.
2. Compare `requested_value` against the min/max bounds.
3. **If OUT OF BOUNDS:** Return a string to the LLM: `"SAFETY OVERRIDE: Requested {parameter} of {requested_value} exceeds hardware limits. Min: {min}, Max: {max}."`
4. **If SAFE:** Update the parameter in the Phase 2 `SimulationConfig` and return `"SUCCESS"`.

## 5. Instructions for Copilot
1. Create a `phase3_chat.py` file or integrate into the Phase 2 `app.py` Streamlit script.
2. Create the Streamlit chat UI using `st.chat_message()` and `st.chat_input()`.
3. Implement the `get_telemetry` and `adjust_machine_parameter` python functions.
4. Set up the LLM integration (use an agnostic setup or the official library for OpenAI/Anthropic/Gemini depending on environment, assuming OpenAI-style function calling syntax).
5. Ensure the LLM system prompt strictly enforces the Grounding Protocol and citation requirements.
