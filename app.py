"""
Phase 2 - Streamlit Dashboard (Script B)
==========================================
Read-only viewer for the SQLite telemetry database.
Provides real-time health metrics, time-series charts,
simulation controls, and a failure analysis log.

Run:  streamlit run app.py
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import pandas as pd
import streamlit as st

# ---------------------------------------------------------------------------
# Paths (must match engine.py)
# ---------------------------------------------------------------------------
DB_PATH = Path(__file__).parent / "telemetry.db"
CONFIG_PATH = Path(__file__).parent / "sim_config.json"

# ---------------------------------------------------------------------------
# Page configuration
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="HP Metal Jet S100 - Digital Twin",
    page_icon="🏭",
    layout="wide",
)

# ---------------------------------------------------------------------------
# Custom CSS for a premium dark theme
# ---------------------------------------------------------------------------
st.markdown("""
<style>
    /* Global font */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

    /* Metric cards */
    div[data-testid="stMetric"] {
        background: linear-gradient(135deg, #1e1e2e 0%, #2a2a3e 100%);
        border: 1px solid rgba(255,255,255,0.08);
        border-radius: 12px;
        padding: 18px 20px;
        box-shadow: 0 4px 20px rgba(0,0,0,0.3);
    }
    div[data-testid="stMetric"] label {
        color: #a0a0c0 !important;
        font-weight: 500;
    }
    div[data-testid="stMetric"] [data-testid="stMetricValue"] {
        font-size: 2rem !important;
        font-weight: 700 !important;
    }

    /* Sidebar */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #12121e 0%, #1a1a2e 100%);
    }

    /* Header styling */
    h1 { letter-spacing: -0.5px; }

    /* Chart containers */
    div[data-testid="stVegaLiteChart"] {
        border-radius: 12px;
        overflow: hidden;
    }
</style>
""", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Helper: read config / write config
# ---------------------------------------------------------------------------

def _read_config() -> dict:
    defaults = {
        "base_temperature_offset": 0.0,
        "production_volume": 1.0,
        "inject_thermal_anomaly": False,
    }
    if CONFIG_PATH.exists():
        try:
            with open(CONFIG_PATH, "r") as f:
                data = json.load(f)
            defaults.update(data)
        except (json.JSONDecodeError, OSError):
            pass
    return defaults


def _write_config(cfg: dict) -> None:
    with open(CONFIG_PATH, "w") as f:
        json.dump(cfg, f, indent=2)


# ---------------------------------------------------------------------------
# Helper: load data from SQLite
# ---------------------------------------------------------------------------

@st.cache_data(ttl=1)
def load_latest_rows(n: int = 100) -> pd.DataFrame:
    """Fetch the last *n* rows from telemetry_log."""
    if not DB_PATH.exists():
        return pd.DataFrame()
    try:
        import sqlite3
        conn = sqlite3.connect(str(DB_PATH))
        df = pd.read_sql_query(
            f"SELECT * FROM telemetry_log ORDER BY id DESC LIMIT {n}",
            conn,
        )
        conn.close()
        if df.empty:
            return df
        df = df.sort_values("id").reset_index(drop=True)
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        return df
    except Exception:
        return pd.DataFrame()


@st.cache_data(ttl=1)
def load_failure_log() -> pd.DataFrame:
    """Fetch rows where failure_log is not null."""
    if not DB_PATH.exists():
        return pd.DataFrame()
    try:
        import sqlite3
        conn = sqlite3.connect(str(DB_PATH))
        df = pd.read_sql_query(
            "SELECT timestamp, failure_log FROM telemetry_log "
            "WHERE failure_log IS NOT NULL ORDER BY id DESC LIMIT 50",
            conn,
        )
        conn.close()
        return df
    except Exception:
        return pd.DataFrame()


# ---------------------------------------------------------------------------
# Sidebar - Simulation Controls
# ---------------------------------------------------------------------------

with st.sidebar:
    st.markdown("## ⚙️ Simulation Controls")
    st.markdown("---")

    cfg = _read_config()

    base_temp = st.slider(
        "🌡️ Base Temperature Offset (°C)",
        min_value=-20.0,
        max_value=80.0,
        value=float(cfg.get("base_temperature_offset", 0.0)),
        step=1.0,
        help="Shift the baseline temperature up or down.",
    )

    prod_vol = st.slider(
        "📦 Production Volume (multiplier)",
        min_value=0.1,
        max_value=3.0,
        value=float(cfg.get("production_volume", 1.0)),
        step=0.1,
        help="Scale the operational load.",
    )

    st.markdown("---")
    st.markdown("### 🔥 Chaos Injection")
    chaos_btn = st.button(
        "⚡ Inject Thermal Anomaly",
        use_container_width=True,
        type="primary",
    )

    new_cfg = {
        "base_temperature_offset": base_temp,
        "production_volume": prod_vol,
        "inject_thermal_anomaly": chaos_btn,
    }
    _write_config(new_cfg)

    st.markdown("---")
    st.caption("Dashboard auto-refreshes every second.")
    st.caption(f"DB: `{DB_PATH.name}`")


# ---------------------------------------------------------------------------
# Main content
# ---------------------------------------------------------------------------

st.markdown("# 🏭 HP Metal Jet S100 — Digital Twin Dashboard")
st.caption("Real-time telemetry from the Phase 2 Simulation Engine")

df = load_latest_rows(100)

if df.empty:
    st.warning(
        "No telemetry data yet. Start the simulation engine first:\n\n"
        "```\npython engine.py\n```"
    )
    time.sleep(2)
    st.rerun()

# ---------------------------------------------------------------------------
# Top Row: System Health Metrics
# ---------------------------------------------------------------------------

st.markdown("## System Health")

latest = df.iloc[-1]
prev = df.iloc[-2] if len(df) >= 2 else latest

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric(
        label="🔪 Recoater Blade",
        value=f"{latest['blade_health']:.3f}",
        delta=f"{latest['blade_health'] - prev['blade_health']:.4f}",
    )
with col2:
    st.metric(
        label="🖨️ Nozzle Plate",
        value=f"{latest['nozzle_health']:.3f}",
        delta=f"{latest['nozzle_health'] - prev['nozzle_health']:.4f}",
    )
with col3:
    st.metric(
        label="🔥 Heating Elements",
        value=f"{latest['heater_health']:.3f}",
        delta=f"{latest['heater_health'] - prev['heater_health']:.4f}",
    )
with col4:
    st.metric(
        label="⏱️ Printer State",
        value=latest.get("printer_state", "N/A"),
        delta=f"tick #{int(latest['id'])}",
        delta_color="off",
    )

# ---------------------------------------------------------------------------
# Middle Row: Time-Series Charts
# ---------------------------------------------------------------------------

st.markdown("---")
st.markdown("## Time-Series Telemetry")

chart_col1, chart_col2 = st.columns(2)

with chart_col1:
    st.markdown("#### 🌡️ Environmental Drivers")
    env_df = df[["timestamp", "temperature", "contamination"]].set_index("timestamp")
    st.line_chart(env_df, use_container_width=True)

with chart_col2:
    st.markdown("#### ❤️ Component Health")
    health_df = df[["timestamp", "blade_health", "nozzle_health", "heater_health"]].set_index("timestamp")
    st.line_chart(health_df, use_container_width=True)

# ---------------------------------------------------------------------------
# Load chart
# ---------------------------------------------------------------------------

st.markdown("#### ⚙️ Operational Load")
load_df = df[["timestamp", "load"]].set_index("timestamp")
st.line_chart(load_df, use_container_width=True)

# ---------------------------------------------------------------------------
# Bottom Row: Failure Analysis Log
# ---------------------------------------------------------------------------

st.markdown("---")
st.markdown("## 🚨 Failure Analysis Log")

failures = load_failure_log()
if failures.empty:
    st.success("No component failures recorded.")
else:
    st.dataframe(failures, use_container_width=True, hide_index=True)

# ---------------------------------------------------------------------------
# Raw telemetry (expandable)
# ---------------------------------------------------------------------------

with st.expander("📊 Raw Telemetry (last 100 rows)"):
    st.dataframe(df, use_container_width=True, hide_index=True)

# ---------------------------------------------------------------------------
# Auto-refresh heartbeat
# ---------------------------------------------------------------------------
time.sleep(1)
st.rerun()
