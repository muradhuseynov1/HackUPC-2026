"""
Test Suite — Phase 2: Simulation Engine (engine.py)
"""
import json
import sqlite3
from pathlib import Path
from unittest.mock import patch
import pytest
from engine import (
    PrinterState, ProactiveAgent, SimulationConfig, StateMachine,
    _STATE_ORDER, _STATE_PROFILES, _init_db, _insert_row, _read_config,
    run_simulation,
)

@pytest.fixture
def tmp_db(tmp_path):
    db_file = tmp_path / "test.db"
    conn = sqlite3.connect(str(db_file))
    _init_db(conn)
    yield conn, db_file
    conn.close()

@pytest.fixture
def tmp_config(tmp_path):
    return tmp_path / "cfg.json"

# --- SimulationConfig ---
class TestSimulationConfig:
    def test_basic(self):
        cfg = SimulationConfig(total_duration=100.0, time_step=1.0)
        assert cfg.total_duration == 100.0
        assert cfg.time_step == 1.0
        assert cfg.environmental_profile is None

# --- StateMachine ---
class TestStateMachine:
    def test_initial_state(self):
        sm = StateMachine()
        assert sm.current == PrinterState.IDLE

    def test_tick_returns_keys(self):
        d = StateMachine().tick()
        assert set(d.keys()) == {"temperature", "load", "contamination"}

    def test_full_cycle(self):
        sm = StateMachine()
        visited = [sm.current]
        for _ in range(200):
            sm.tick()
            if sm.current != visited[-1]:
                visited.append(sm.current)
            if len(visited) >= 5:
                break
        assert visited == [
            PrinterState.IDLE, PrinterState.HEATING,
            PrinterState.PRINTING, PrinterState.COOLDOWN, PrinterState.IDLE,
        ]

    def test_idle_duration(self):
        sm = StateMachine()
        dur = _STATE_PROFILES[PrinterState.IDLE]["duration"]
        for _ in range(dur):
            sm.tick()
        assert sm.current == PrinterState.HEATING

    def test_heating_ramps(self):
        sm = StateMachine()
        for _ in range(_STATE_PROFILES[PrinterState.IDLE]["duration"]):
            sm.tick()
        temps = [sm.tick()["temperature"] for _ in range(
            _STATE_PROFILES[PrinterState.HEATING]["duration"])]
        for i in range(1, len(temps)):
            assert temps[i] >= temps[i-1]

    def test_cooldown_drops(self):
        sm = StateMachine()
        skip = sum(_STATE_PROFILES[s]["duration"]
                   for s in [PrinterState.IDLE, PrinterState.HEATING, PrinterState.PRINTING])
        for _ in range(skip):
            sm.tick()
        assert sm.current == PrinterState.COOLDOWN
        temps = [sm.tick()["temperature"] for _ in range(
            _STATE_PROFILES[PrinterState.COOLDOWN]["duration"])]
        for i in range(1, len(temps)):
            assert temps[i] <= temps[i-1]

# --- ProactiveAgent ---
class TestProactiveAgent:
    def test_first_obs_no_action(self):
        h, acted, _ = ProactiveAgent(0.15).evaluate_and_act("C", 0.9)
        assert acted is False and h == 0.9

    def test_no_intervention_gradual(self):
        a = ProactiveAgent(0.15)
        a.evaluate_and_act("C", 0.9)
        _, acted, _ = a.evaluate_and_act("C", 0.88)
        assert acted is False

    def test_intervention_sharp_drop(self):
        a = ProactiveAgent(0.15)
        a.evaluate_and_act("C", 0.5)
        h, acted, msg = a.evaluate_and_act("C", 0.2)
        assert acted is True and h == 1.0 and "[PROACTIVE]" in msg

    def test_independent_components(self):
        a = ProactiveAgent(0.15)
        a.evaluate_and_act("A", 0.9); a.evaluate_and_act("B", 0.9)
        _, a1, _ = a.evaluate_and_act("A", 0.88)
        _, a2, _ = a.evaluate_and_act("B", 0.2)
        assert a1 is False and a2 is True

    def test_memory_resets_after_intervention(self):
        a = ProactiveAgent(0.15)
        a.evaluate_and_act("C", 0.5)
        a.evaluate_and_act("C", 0.2)
        _, acted, _ = a.evaluate_and_act("C", 0.95)
        assert acted is False

# --- DB Helpers ---
class TestDatabase:
    def test_creates_table(self, tmp_db):
        conn, _ = tmp_db
        cur = conn.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='telemetry_log'")
        assert cur.fetchone() is not None

    def test_metric_columns_exist(self, tmp_db):
        conn, _ = tmp_db
        cols = {r[1] for r in conn.execute("PRAGMA table_info(telemetry_log)").fetchall()}
        assert {"blade_thickness", "nozzle_clogging", "heater_resistance"} <= cols

    def test_insert_row(self, tmp_db):
        conn, _ = tmp_db
        _insert_row(conn, run_id="r1", printer_state="IDLE",
                     timestamp="2026-01-01T00:00:00Z", sim_time=0.0,
                     temperature=25.0, load=0.0, contamination=0.05,
                     blade_health=1.0, nozzle_health=1.0, heater_health=1.0,
                     blade_thickness=2.0, nozzle_clogging=0.0, heater_resistance=10.0)
        assert conn.execute("SELECT COUNT(*) FROM telemetry_log").fetchone()[0] == 1

    def test_insert_row_with_metrics(self, tmp_db):
        conn, _ = tmp_db
        _insert_row(conn, run_id="r2", printer_state="PRINTING",
                     timestamp="2026-01-01T01:00:00Z", sim_time=60.0,
                     temperature=180.0, load=8.0, contamination=0.25,
                     blade_health=0.8, nozzle_health=0.75, heater_health=0.9,
                     blade_thickness=1.9, nozzle_clogging=25.0, heater_resistance=10.5)
        conn.row_factory = sqlite3.Row
        row = conn.execute("SELECT * FROM telemetry_log WHERE run_id='r2'").fetchone()
        assert row["blade_thickness"] == pytest.approx(1.9)
        assert row["nozzle_clogging"] == pytest.approx(25.0)
        assert row["heater_resistance"] == pytest.approx(10.5)

    def test_migration_adds_columns(self, tmp_path):
        db_file = tmp_path / "legacy.db"
        conn = sqlite3.connect(str(db_file))
        conn.execute("""CREATE TABLE telemetry_log (
            id INTEGER PRIMARY KEY AUTOINCREMENT, timestamp TEXT NOT NULL,
            sim_time REAL NOT NULL, run_id TEXT NOT NULL, printer_state TEXT NOT NULL,
            temperature REAL NOT NULL, load REAL NOT NULL, contamination REAL NOT NULL,
            blade_health REAL NOT NULL, nozzle_health REAL NOT NULL,
            heater_health REAL NOT NULL, failure_log TEXT)""")
        conn.commit()
        _init_db(conn)
        cols = {r[1] for r in conn.execute("PRAGMA table_info(telemetry_log)").fetchall()}
        assert {"blade_thickness", "nozzle_clogging", "heater_resistance"} <= cols
        conn.close()

# --- Config Reader ---
class TestConfigReader:
    def test_defaults_no_file(self, tmp_path):
        with patch("engine.CONFIG_PATH", tmp_path / "x.json"):
            cfg = _read_config()
            assert cfg["base_temperature_offset"] == 0.0

    def test_reads_file(self, tmp_config):
        with open(tmp_config, "w") as f:
            json.dump({"base_temperature_offset": 15.0, "production_volume": 2.0}, f)
        with patch("engine.CONFIG_PATH", tmp_config):
            cfg = _read_config()
            assert cfg["base_temperature_offset"] == 15.0

    def test_corrupt_json(self, tmp_config):
        with open(tmp_config, "w") as f:
            f.write("{bad!")
        with patch("engine.CONFIG_PATH", tmp_config):
            cfg = _read_config()
            assert cfg["base_temperature_offset"] == 0.0

    def test_anomaly_auto_resets(self, tmp_config):
        with open(tmp_config, "w") as f:
            json.dump({"inject_thermal_anomaly": True}, f)
        with patch("engine.CONFIG_PATH", tmp_config):
            _read_config()
            with open(tmp_config) as f:
                assert json.load(f)["inject_thermal_anomaly"] is False

# --- Simulation Loop (Integration) ---
class TestSimulationLoop:
    @pytest.mark.asyncio
    async def test_short_run(self, tmp_path):
        db = tmp_path / "sim.db"
        cfg_f = tmp_path / "cfg.json"
        with open(cfg_f, "w") as f:
            json.dump({}, f)
        with patch("engine.DB_PATH", db), patch("engine.CONFIG_PATH", cfg_f), \
             patch("engine.TICK_INTERVAL_S", 0.0):
            await run_simulation(SimulationConfig(5.0, 1.0))
        conn = sqlite3.connect(str(db))
        assert conn.execute("SELECT COUNT(*) FROM telemetry_log").fetchone()[0] == 5
        conn.close()

    @pytest.mark.asyncio
    async def test_unique_run_id(self, tmp_path):
        db = tmp_path / "rid.db"
        cfg_f = tmp_path / "cfg.json"
        with open(cfg_f, "w") as f:
            json.dump({}, f)
        with patch("engine.DB_PATH", db), patch("engine.CONFIG_PATH", cfg_f), \
             patch("engine.TICK_INTERVAL_S", 0.0):
            await run_simulation(SimulationConfig(3.0, 1.0))
        conn = sqlite3.connect(str(db))
        ids = conn.execute("SELECT DISTINCT run_id FROM telemetry_log").fetchall()
        conn.close()
        assert len(ids) == 1 and len(ids[0][0]) == 12

    @pytest.mark.asyncio
    async def test_stores_metrics(self, tmp_path):
        db = tmp_path / "met.db"
        cfg_f = tmp_path / "cfg.json"
        with open(cfg_f, "w") as f:
            json.dump({}, f)
        with patch("engine.DB_PATH", db), patch("engine.CONFIG_PATH", cfg_f), \
             patch("engine.TICK_INTERVAL_S", 0.0):
            await run_simulation(SimulationConfig(3.0, 1.0))
        conn = sqlite3.connect(str(db))
        conn.row_factory = sqlite3.Row
        row = conn.execute("SELECT * FROM telemetry_log LIMIT 1").fetchone()
        conn.close()
        assert row["blade_thickness"] is not None
        assert row["nozzle_clogging"] is not None
        assert row["heater_resistance"] is not None

    @pytest.mark.asyncio
    async def test_sim_time_advances(self, tmp_path):
        db = tmp_path / "time.db"
        cfg_f = tmp_path / "cfg.json"
        with open(cfg_f, "w") as f:
            json.dump({}, f)
        with patch("engine.DB_PATH", db), patch("engine.CONFIG_PATH", cfg_f), \
             patch("engine.TICK_INTERVAL_S", 0.0):
            await run_simulation(SimulationConfig(5.0, 1.0))
        conn = sqlite3.connect(str(db))
        times = [r[0] for r in conn.execute(
            "SELECT sim_time FROM telemetry_log ORDER BY id ASC").fetchall()]
        conn.close()
        for i in range(1, len(times)):
            assert times[i] > times[i-1]
