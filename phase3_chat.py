"""
Phase 3 - Intelligent Agentic Interface
==========================================
ReAct Loop (Agentic Diagnosis) + Persistent Memory +
Proactive Alerts + Neurosymbolic AI Firewall.

Run:  streamlit run phase3_chat.py --server.port 8502
"""

from __future__ import annotations

import json
import sqlite3
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import streamlit as st

# ---------------------------------------------------------------------------
# Paths (must match engine.py)
# ---------------------------------------------------------------------------
DB_PATH = Path(__file__).parent / "telemetry.db"
CONFIG_PATH = Path(__file__).parent / "sim_config.json"

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="HP Metal Jet S100 - AI Co-Pilot",
    page_icon="🤖",
    layout="wide",
)

# ---------------------------------------------------------------------------
# Custom CSS
# ---------------------------------------------------------------------------
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #12121e 0%, #1a1a2e 100%);
    }
    .severity-info {
        background: #1e3a5f; color: #7ec8f2; padding: 3px 10px;
        border-radius: 6px; font-size: 0.75rem; font-weight: 600;
        display: inline-block; margin-bottom: 6px;
    }
    .severity-warning {
        background: #5f4b1e; color: #f2d07e; padding: 3px 10px;
        border-radius: 6px; font-size: 0.75rem; font-weight: 600;
        display: inline-block; margin-bottom: 6px;
    }
    .severity-critical {
        background: #5f1e1e; color: #f27e7e; padding: 3px 10px;
        border-radius: 6px; font-size: 0.75rem; font-weight: 600;
        display: inline-block; margin-bottom: 6px;
    }
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Auto-refresh for proactive alerts (every 5 seconds)
# ---------------------------------------------------------------------------
try:
    from streamlit_autorefresh import st_autorefresh
    st_autorefresh(interval=5000, limit=None, key="proactive_refresh")
except ImportError:
    pass  # graceful fallback if not installed

# ═══════════════════════════════════════════════════════════════════════════
# PERSISTENT CHAT HISTORY (SQLite)
# ═══════════════════════════════════════════════════════════════════════════

def _init_chat_db(conn: sqlite3.Connection) -> None:
    conn.execute("""
        CREATE TABLE IF NOT EXISTS chat_history (
            id        INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT    NOT NULL,
            role      TEXT    NOT NULL,
            content   TEXT    NOT NULL
        )
    """)
    conn.commit()


def _load_chat_history() -> list[dict]:
    if not DB_PATH.exists():
        return []
    conn = sqlite3.connect(str(DB_PATH))
    _init_chat_db(conn)
    rows = conn.execute(
        "SELECT role, content FROM chat_history ORDER BY id ASC"
    ).fetchall()
    conn.close()
    return [{"role": r[0], "content": r[1]} for r in rows]


def _save_chat_message(role: str, content: str) -> None:
    conn = sqlite3.connect(str(DB_PATH))
    _init_chat_db(conn)
    conn.execute(
        "INSERT INTO chat_history (timestamp, role, content) VALUES (?, ?, ?)",
        (datetime.now(timezone.utc).isoformat(), role, content),
    )
    conn.commit()
    conn.close()


def _clear_chat_history_db() -> None:
    if not DB_PATH.exists():
        return
    conn = sqlite3.connect(str(DB_PATH))
    conn.execute("DELETE FROM chat_history")
    conn.commit()
    conn.close()


# ═══════════════════════════════════════════════════════════════════════════
# NEUROSYMBOLIC AI FIREWALL — Safety Envelope (§5.1)
# ═══════════════════════════════════════════════════════════════════════════

SAFETY_ENVELOPE: dict[str, dict[str, float]] = {
    "temperature_stress": {"max": 220.0, "min": 20.0},
    "contamination":      {"max": 1.0,   "min": 0.0},
    "operational_load":   {"max": 100.0, "min": 0.0},
}


def deterministic_firewall(
    component: str, parameter: str, requested_value: float,
) -> str:
    envelope = SAFETY_ENVELOPE.get(parameter)
    if envelope is None:
        return (
            f"SAFETY OVERRIDE: Unknown parameter '{parameter}'. "
            f"Known: {list(SAFETY_ENVELOPE.keys())}."
        )
    for key, bound in envelope.items():
        if key == "max" and requested_value > bound:
            return (
                f"SAFETY OVERRIDE: Requested {parameter} of {requested_value} "
                f"exceeds maximum limit ({bound}). NOT applied."
            )
        if key == "min" and requested_value < bound:
            return (
                f"SAFETY OVERRIDE: Requested {parameter} of {requested_value} "
                f"is below minimum limit ({bound}). NOT applied."
            )
    _apply_parameter(component, parameter, requested_value)
    return f"SUCCESS: {component}.{parameter} set to {requested_value}."


def _apply_parameter(component: str, parameter: str, value: float) -> None:
    cfg: dict[str, Any] = {}
    if CONFIG_PATH.exists():
        try:
            with open(CONFIG_PATH, "r") as f:
                cfg = json.load(f)
        except (json.JSONDecodeError, OSError):
            pass
    if parameter == "temperature_stress":
        cfg["base_temperature_offset"] = value
    elif parameter == "operational_load":
        cfg["production_volume"] = value
    else:
        cfg[f"{component}.{parameter}"] = value
    with open(CONFIG_PATH, "w") as f:
        json.dump(cfg, f, indent=2)


# ═══════════════════════════════════════════════════════════════════════════
# DIAGNOSTIC TOOLS (§3.2)
# ═══════════════════════════════════════════════════════════════════════════

def get_telemetry(
    start_time: str | None = None,
    end_time: str | None = None,
    component_name: str | None = None,
    run_id: str | None = None,
    limit: int = 30,
) -> str:
    if not DB_PATH.exists():
        return json.dumps({"error": "No telemetry database found."})
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    conditions, params = [], []
    if start_time:
        conditions.append("timestamp >= ?"); params.append(start_time)
    if end_time:
        conditions.append("timestamp <= ?"); params.append(end_time)
    if run_id:
        conditions.append("run_id = ?"); params.append(run_id)
    where = f"WHERE {' AND '.join(conditions)}" if conditions else ""
    query = f"SELECT * FROM telemetry_log {where} ORDER BY id DESC LIMIT {limit}"
    try:
        rows = conn.execute(query, params).fetchall()
        conn.close()
    except Exception as e:
        conn.close()
        return json.dumps({"error": str(e)})
    if not rows:
        return json.dumps({"result": "No data found."})
    data = [dict(row) for row in rows]
    data.reverse()
    col_map = {
        "Recoater Blade": "blade_health", "RecoaterBlade": "blade_health",
        "Nozzle Plate": "nozzle_health", "NozzlePlate": "nozzle_health",
        "Heating Elements": "heater_health", "HeatingElements": "heater_health",
    }
    if component_name and component_name in col_map:
        col = col_map[component_name]
        return json.dumps({
            "component": component_name, "health_column": col,
            "rows_returned": len(data),
            "latest_health": data[-1][col] if data else None,
            "earliest_health": data[0][col] if data else None,
            "data": data,
        }, default=str)
    return json.dumps({"rows_returned": len(data), "data": data}, default=str)


def get_failure_logs() -> str:
    """Return only rows where a component FAILED or agent intervened."""
    if not DB_PATH.exists():
        return json.dumps({"error": "No telemetry database found."})
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    rows = conn.execute(
        "SELECT * FROM telemetry_log WHERE failure_log IS NOT NULL "
        "ORDER BY id DESC LIMIT 50"
    ).fetchall()
    conn.close()
    if not rows:
        return json.dumps({"result": "No failure or maintenance events found."})
    data = [dict(r) for r in rows]
    data.reverse()
    return json.dumps({"events": len(data), "data": data}, default=str)


def get_simulation_config() -> str:
    """Return the current SimulationConfig and sim_config.json overrides."""
    result: dict[str, Any] = {
        "default_total_duration": 500.0,
        "default_time_step": 1.0,
    }
    if CONFIG_PATH.exists():
        try:
            with open(CONFIG_PATH, "r") as f:
                result["overrides"] = json.load(f)
        except (json.JSONDecodeError, OSError):
            result["overrides"] = {}
    else:
        result["overrides"] = {}
    result["safety_envelope"] = SAFETY_ENVELOPE
    return json.dumps(result, default=str)


def get_latest_snapshot() -> str:
    if not DB_PATH.exists():
        return json.dumps({"error": "No telemetry database found."})
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    row = conn.execute(
        "SELECT * FROM telemetry_log ORDER BY id DESC LIMIT 1"
    ).fetchone()
    conn.close()
    if row is None:
        return json.dumps({"error": "No data available."})
    return json.dumps(dict(row), default=str)


# ═══════════════════════════════════════════════════════════════════════════
# OPENAI TOOL DEFINITIONS
# ═══════════════════════════════════════════════════════════════════════════

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_telemetry",
            "description": (
                "Query the Phase 2 SQLite Historian for component health, "
                "environmental drivers, and operational metrics."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "start_time": {"type": "string", "description": "ISO-8601 start time."},
                    "end_time": {"type": "string", "description": "ISO-8601 end time."},
                    "component_name": {
                        "type": "string",
                        "description": "'Recoater Blade', 'Nozzle Plate', or 'Heating Elements'.",
                    },
                    "run_id": {"type": "string", "description": "Filter by simulation run ID for multi-scenario comparison."},
                    "limit": {"type": "integer", "description": "Max rows (default 30)."},
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_failure_logs",
            "description": "Returns rows where a component FAILED or the proactive agent intervened.",
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_simulation_config",
            "description": "Returns the current SimulationConfig (duration, step, overrides, safety envelope).",
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "adjust_machine_parameter",
            "description": (
                "Adjust a simulation parameter. Validated against the safety envelope. "
                "Parameters: temperature_stress, contamination, operational_load."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "component": {"type": "string", "description": "Component name."},
                    "parameter": {
                        "type": "string",
                        "description": "One of: temperature_stress, contamination, operational_load.",
                    },
                    "new_value": {"type": "number", "description": "Requested new value."},
                },
                "required": ["component", "parameter", "new_value"],
            },
        },
    },
]

# ═══════════════════════════════════════════════════════════════════════════
# SYSTEM PROMPT — ReAct Loop (§3.1)
# ═══════════════════════════════════════════════════════════════════════════

SYSTEM_PROMPT = """You are the HP Metal Jet S100 Digital Co-Pilot, an AI diagnostic assistant.

## GROUNDING PROTOCOL (MANDATORY)
- NEVER answer from training knowledge. ALWAYS use tools first.
- Every response MUST cite timestamps or run IDs from retrieved data.

## REACT REASONING (MANDATORY)
For every user question, follow this exact reasoning pattern:
1. **Thought:** Explain what you need to figure out and which tool to call.
2. **Action:** Call the appropriate tool (get_telemetry, get_failure_logs, get_simulation_config).
3. **Observation:** Analyze the returned data.
4. Repeat steps 1-3 if more data is needed.
5. **Final Answer:** Provide a grounded response with severity and citations.

## RESPONSE FORMAT
Every final response must include:
1. Severity: [INFO], [WARNING], or [CRITICAL]
2. Clear explanation grounded in data
3. Explicit timestamp/sim_time citations

## TOOLS
- **get_telemetry**: Query historian for health & drivers over a time window.
- **get_failure_logs**: Get rows where components FAILED or agent intervened.
- **get_simulation_config**: Get current SimulationConfig and overrides.
- **adjust_machine_parameter**: Change a parameter (goes through safety firewall).

## COMPONENTS
- Recoater Blade (blade_health) — Recoating subsystem
- Nozzle Plate (nozzle_health) — Printhead Array
- Heating Elements (heater_health) — Thermal Control

## SAFETY ENVELOPE
- temperature_stress: 20.0–220.0
- contamination: 0.0–1.0
- operational_load: 0.0–100.0

Be concise, professional, and always cite data sources."""

# ═══════════════════════════════════════════════════════════════════════════
# TOOL DISPATCHER
# ═══════════════════════════════════════════════════════════════════════════

def dispatch_tool_call(name: str, arguments: dict[str, Any]) -> str:
    if name == "get_telemetry":
        return get_telemetry(
            start_time=arguments.get("start_time"),
            end_time=arguments.get("end_time"),
            component_name=arguments.get("component_name"),
            run_id=arguments.get("run_id"),
            limit=arguments.get("limit", 30),
        )
    elif name == "get_failure_logs":
        return get_failure_logs()
    elif name == "get_simulation_config":
        return get_simulation_config()
    elif name == "adjust_machine_parameter":
        return deterministic_firewall(
            component=arguments["component"],
            parameter=arguments["parameter"],
            requested_value=arguments["new_value"],
        )
    return json.dumps({"error": f"Unknown tool: {name}"})


# ═══════════════════════════════════════════════════════════════════════════
# REACT CHAT LOOP
# ═══════════════════════════════════════════════════════════════════════════

def run_chat(user_message: str, chat_history: list[dict]) -> tuple[str, list[dict]]:
    """Run the ReAct loop. Returns (final_answer, reasoning_trace)."""
    trace: list[dict] = []  # collects every intermediate step

    try:
        from openai import OpenAI
    except ImportError:
        return ("[CRITICAL] OpenAI package not installed. Run: `pip install openai`", trace)

    api_key = st.session_state.get("openai_api_key", "")
    if not api_key:
        return ("[INFO] Please enter your OpenAI API key in the sidebar.", trace)

    client = OpenAI(api_key=api_key)
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]

    # Inject latest snapshot for baseline awareness
    snapshot = get_latest_snapshot()
    messages.append({"role": "system", "content": f"[CURRENT STATE]\n{snapshot}"})

    # Add persistent chat history (last 20 turns)
    for msg in chat_history[-20:]:
        messages.append({"role": msg["role"], "content": msg["content"]})
    messages.append({"role": "user", "content": user_message})

    # ReAct loop — up to 5 rounds of tool calls
    for iteration in range(5):
        try:
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=messages,
                tools=TOOLS,
                tool_choice="auto",
                temperature=0.1,
            )
        except Exception as e:
            return (f"[CRITICAL] OpenAI API error: {e}", trace)

        choice = response.choices[0]

        if choice.finish_reason == "tool_calls" or choice.message.tool_calls:
            # Save the AI's intermediate thought (if any)
            if choice.message.content:
                trace.append({"step": "thought", "content": choice.message.content})

            messages.append(choice.message)
            for tool_call in choice.message.tool_calls:
                fn_name = tool_call.function.name
                try:
                    fn_args = json.loads(tool_call.function.arguments)
                except json.JSONDecodeError:
                    fn_args = {}

                # Save the tool call
                trace.append({
                    "step": "tool_call",
                    "tool": fn_name,
                    "arguments": fn_args,
                })

                result = dispatch_tool_call(fn_name, fn_args)

                # Save the tool result (truncated for readability)
                trace.append({
                    "step": "tool_result",
                    "tool": fn_name,
                    "result_preview": result[:500] + ("..." if len(result) > 500 else ""),
                })

                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": result,
                })
            continue

        return (choice.message.content or "[INFO] No response generated.", trace)

    return ("[WARNING] Maximum ReAct iterations reached.", trace)


def _format_trace(trace: list[dict]) -> str:
    """Format the reasoning trace as readable markdown."""
    if not trace:
        return "_No intermediate reasoning steps._"
    lines = []
    for i, step in enumerate(trace, 1):
        if step["step"] == "thought":
            lines.append(f"**💭 Thought:** {step['content']}")
        elif step["step"] == "tool_call":
            args_str = json.dumps(step["arguments"], indent=2)
            lines.append(f"**🔧 Tool Call:** `{step['tool']}({args_str})`")
        elif step["step"] == "tool_result":
            lines.append(f"**📋 Result from `{step['tool']}`:**\n```json\n{step['result_preview']}\n```")
    return "\n\n".join(lines)


# ═══════════════════════════════════════════════════════════════════════════
# PROACTIVE ALERTING (§4.1)
# ═══════════════════════════════════════════════════════════════════════════

WARNING_THRESHOLD = 0.3
CRITICAL_THRESHOLD = 0.0


def _check_proactive_alerts() -> str | None:
    """Check latest telemetry for alerts. Returns alert message or None."""
    if not DB_PATH.exists():
        return None
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    row = conn.execute(
        "SELECT * FROM telemetry_log ORDER BY id DESC LIMIT 1"
    ).fetchone()
    conn.close()
    if row is None:
        return None

    data = dict(row)
    alerts = []
    components = [
        ("Recoater Blade", "blade_health"),
        ("Nozzle Plate", "nozzle_health"),
        ("Heating Elements", "heater_health"),
    ]
    sim_time = data.get("sim_time", "?")

    for name, col in components:
        health = data.get(col, 1.0)
        if health <= CRITICAL_THRESHOLD:
            alerts.append(
                f"🚨 **PROACTIVE ALERT:** {name} health is {health:.2f} "
                f"(FAILED) at sim_time {sim_time}. Diagnose root cause?"
            )
        elif health <= WARNING_THRESHOLD:
            alerts.append(
                f"⚠️ **PROACTIVE ALERT:** {name} health dropped to {health:.2f} "
                f"at sim_time {sim_time}. Want me to investigate?"
            )

    return "\n\n".join(alerts) if alerts else None


# ═══════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ═══════════════════════════════════════════════════════════════════════════

with st.sidebar:
    st.markdown("## 🤖 AI Co-Pilot Settings")
    st.markdown("---")

    api_key = st.text_input(
        "🔑 OpenAI API Key", type="password",
        value=st.session_state.get("openai_api_key", ""),
    )
    if api_key:
        st.session_state["openai_api_key"] = api_key

    st.markdown("---")
    st.markdown("### 🛡️ Safety Envelope")
    for param, bounds in SAFETY_ENVELOPE.items():
        bounds_str = ", ".join(f"{k}: {v}" for k, v in bounds.items())
        st.caption(f"**{param}**: {bounds_str}")

    st.markdown("---")
    if st.button("🗑️ Clear Chat History", use_container_width=True):
        _clear_chat_history_db()
        st.session_state["chat_history"] = []
        st.session_state.pop("_alerted", None)
        st.rerun()

    st.markdown("---")
    st.caption("Phase 3: Agentic ReAct Interface")
    st.caption("Persistent Memory + Proactive Alerts + Firewall")

# ═══════════════════════════════════════════════════════════════════════════
# MAIN CHAT UI
# ═══════════════════════════════════════════════════════════════════════════

st.markdown("# 🤖 HP Metal Jet S100 — AI Co-Pilot")
st.caption(
    "Agentic ReAct diagnosis grounded in Phase 2 telemetry. "
    "Persistent chat history survives reloads."
)

# Load persistent chat history on first run
if "chat_history" not in st.session_state:
    st.session_state["chat_history"] = _load_chat_history()

if "_alerted" not in st.session_state:
    st.session_state["_alerted"] = set()


def _render_message(content: str) -> None:
    """Render a message with severity badge styling."""
    if content.startswith("[INFO]"):
        st.markdown('<span class="severity-info">INFO</span>', unsafe_allow_html=True)
        st.markdown(content[6:].strip())
    elif content.startswith("[WARNING]"):
        st.markdown('<span class="severity-warning">WARNING</span>', unsafe_allow_html=True)
        st.markdown(content[9:].strip())
    elif content.startswith("[CRITICAL]"):
        st.markdown('<span class="severity-critical">CRITICAL</span>', unsafe_allow_html=True)
        st.markdown(content[10:].strip())
    else:
        st.markdown(content)


# Display chat history
for msg in st.session_state["chat_history"]:
    with st.chat_message(msg["role"]):
        _render_message(msg["content"])

# --- Proactive alert check (runs on every refresh) ---
alert_msg = _check_proactive_alerts()
if alert_msg:
    alert_key = alert_msg[:80]  # dedup key
    if alert_key not in st.session_state["_alerted"]:
        st.session_state["_alerted"].add(alert_key)
        st.session_state["chat_history"].append(
            {"role": "assistant", "content": alert_msg}
        )
        _save_chat_message("assistant", alert_msg)
        st.rerun()

# Chat input
user_input = st.chat_input("Ask the Co-Pilot about the printer...")

if user_input:
    with st.chat_message("user"):
        st.markdown(user_input)
    st.session_state["chat_history"].append({"role": "user", "content": user_input})
    _save_chat_message("user", user_input)

    with st.chat_message("assistant"):
        with st.spinner("ReAct reasoning..."):
            response, trace = run_chat(user_input, st.session_state["chat_history"])
        _render_message(response)
        if trace:
            with st.expander("🧠 Reasoning Trace", expanded=False):
                st.markdown(_format_trace(trace))

    st.session_state["chat_history"].append({"role": "assistant", "content": response})
    _save_chat_message("assistant", response)
    if trace:
        trace_text = _format_trace(trace)
        st.session_state["chat_history"].append({"role": "assistant", "content": f"[TRACE]\n{trace_text}"})
        _save_chat_message("assistant", f"[TRACE]\n{trace_text}")

# --- Quick-action buttons ---
st.markdown("---")
st.markdown("### 💡 Quick Actions")
qcol1, qcol2, qcol3, qcol4 = st.columns(4)

with qcol1:
    if st.button("📊 System Status", use_container_width=True):
        st.session_state["_quick"] = "What is the current health status of all components?"
        st.rerun()
with qcol2:
    if st.button("📉 Degradation Trends", use_container_width=True):
        st.session_state["_quick"] = "Show degradation trends over the last 50 data points."
        st.rerun()
with qcol3:
    if st.button("🚨 Failure Analysis", use_container_width=True):
        st.session_state["_quick"] = "Show all failure and maintenance events from the log."
        st.rerun()
with qcol4:
    if st.button("🔮 Predict Failures", use_container_width=True):
        st.session_state["_quick"] = "Based on current rates, when will each component reach CRITICAL?"
        st.rerun()

if "_quick" in st.session_state:
    quick_msg = st.session_state.pop("_quick")
    st.session_state["chat_history"].append({"role": "user", "content": quick_msg})
    _save_chat_message("user", quick_msg)
    response, trace = run_chat(quick_msg, st.session_state["chat_history"])
    st.session_state["chat_history"].append({"role": "assistant", "content": response})
    _save_chat_message("assistant", response)
    if trace:
        trace_text = _format_trace(trace)
        st.session_state["chat_history"].append({"role": "assistant", "content": f"[TRACE]\n{trace_text}"})
        _save_chat_message("assistant", f"[TRACE]\n{trace_text}")
    st.rerun()
