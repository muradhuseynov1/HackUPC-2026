"""
Test Suite — Phase 1: Mathematical Logic Engine  (model.py)
============================================================
Covers:
  • Input validation
  • Component construction & reset
  • Degradation math for each component
  • Health → OperationalStatus mapping
  • Engine orchestration & determinism
  • Edge cases (boundary values, zero-health clamping)
"""

import math
import pytest

from model import (
    Component,
    Engine,
    HeatingElements,
    Inputs,
    NozzlePlate,
    OperationalStatus,
    RecoaterBlade,
    StateReport,
    _health_to_status,
)


# ═══════════════════════════════════════════════════════════════════════════
# Fixtures
# ═══════════════════════════════════════════════════════════════════════════

@pytest.fixture
def default_inputs() -> Inputs:
    return Inputs(
        temperature_stress=30.0,
        contamination=0.3,
        operational_load=10.0,
        maintenance_level=0.7,
    )


@pytest.fixture
def harsh_inputs() -> Inputs:
    """Extreme but valid operating conditions."""
    return Inputs(
        temperature_stress=200.0,
        contamination=0.95,
        operational_load=50.0,
        maintenance_level=0.1,
    )


@pytest.fixture
def mild_inputs() -> Inputs:
    """Gentle operating conditions."""
    return Inputs(
        temperature_stress=25.0,
        contamination=0.0,
        operational_load=0.0,
        maintenance_level=1.0,
    )


@pytest.fixture
def engine() -> Engine:
    return Engine()


# ═══════════════════════════════════════════════════════════════════════════
# 1. Input Validation
# ═══════════════════════════════════════════════════════════════════════════

class TestInputs:
    """Validate the Inputs dataclass guards."""

    def test_valid_inputs(self, default_inputs: Inputs):
        assert default_inputs.temperature_stress == 30.0
        assert default_inputs.contamination == 0.3
        assert default_inputs.operational_load == 10.0
        assert default_inputs.maintenance_level == 0.7

    def test_contamination_too_high(self):
        with pytest.raises(ValueError, match="contamination"):
            Inputs(temperature_stress=25.0, contamination=1.5,
                   operational_load=1.0, maintenance_level=0.5)

    def test_contamination_negative(self):
        with pytest.raises(ValueError, match="contamination"):
            Inputs(temperature_stress=25.0, contamination=-0.1,
                   operational_load=1.0, maintenance_level=0.5)

    def test_maintenance_level_too_high(self):
        with pytest.raises(ValueError, match="maintenance_level"):
            Inputs(temperature_stress=25.0, contamination=0.5,
                   operational_load=1.0, maintenance_level=1.1)

    def test_maintenance_level_negative(self):
        with pytest.raises(ValueError, match="maintenance_level"):
            Inputs(temperature_stress=25.0, contamination=0.5,
                   operational_load=1.0, maintenance_level=-0.5)

    def test_operational_load_negative(self):
        with pytest.raises(ValueError, match="operational_load"):
            Inputs(temperature_stress=25.0, contamination=0.5,
                   operational_load=-1.0, maintenance_level=0.5)

    def test_boundary_values(self):
        # Exact boundary values should be valid
        inp = Inputs(temperature_stress=0.0, contamination=0.0,
                     operational_load=0.0, maintenance_level=0.0)
        assert inp.contamination == 0.0

        inp = Inputs(temperature_stress=0.0, contamination=1.0,
                     operational_load=0.0, maintenance_level=1.0)
        assert inp.contamination == 1.0
        assert inp.maintenance_level == 1.0

    def test_inputs_are_frozen(self, default_inputs: Inputs):
        with pytest.raises(AttributeError):
            default_inputs.temperature_stress = 99.0  # type: ignore[misc]


# ═══════════════════════════════════════════════════════════════════════════
# 2. Health → Status Mapping
# ═══════════════════════════════════════════════════════════════════════════

class TestHealthToStatus:
    def test_functional(self):
        assert _health_to_status(1.0) == OperationalStatus.FUNCTIONAL
        assert _health_to_status(0.71) == OperationalStatus.FUNCTIONAL

    def test_degraded(self):
        assert _health_to_status(0.7) == OperationalStatus.DEGRADED
        assert _health_to_status(0.5) == OperationalStatus.DEGRADED
        assert _health_to_status(0.31) == OperationalStatus.DEGRADED

    def test_critical(self):
        assert _health_to_status(0.3) == OperationalStatus.CRITICAL
        assert _health_to_status(0.01) == OperationalStatus.CRITICAL

    def test_failed(self):
        assert _health_to_status(0.0) == OperationalStatus.FAILED


# ═══════════════════════════════════════════════════════════════════════════
# 3. Recoater Blade (Linear Decay / Contamination)
# ═══════════════════════════════════════════════════════════════════════════

class TestRecoaterBlade:
    def test_initial_state(self):
        blade = RecoaterBlade()
        assert blade.health == 1.0
        assert blade.blade_thickness_mm == 2.0
        assert blade.name == "RecoaterBlade"

    def test_degrades_on_update(self, default_inputs: Inputs):
        blade = RecoaterBlade()
        report = blade.update(default_inputs)
        assert report.health_index < 1.0
        assert report.operational_status == OperationalStatus.FUNCTIONAL
        assert "blade_thickness_mm" in report.metrics
        assert report.metrics["blade_thickness_mm"] < 2.0

    def test_higher_contamination_faster_wear(self):
        blade_clean = RecoaterBlade()
        blade_dirty = RecoaterBlade()

        clean = Inputs(temperature_stress=25.0, contamination=0.0,
                       operational_load=5.0, maintenance_level=0.5)
        dirty = Inputs(temperature_stress=25.0, contamination=0.9,
                       operational_load=5.0, maintenance_level=0.5)

        r_clean = blade_clean.update(clean)
        r_dirty = blade_dirty.update(dirty)

        assert r_dirty.health_index < r_clean.health_index, \
            "Higher contamination should cause more wear"

    def test_health_clamps_at_zero(self, harsh_inputs: Inputs):
        blade = RecoaterBlade()
        for _ in range(500):
            blade.update(harsh_inputs)
        assert blade.health == 0.0

    def test_thickness_at_failure(self, harsh_inputs: Inputs):
        blade = RecoaterBlade()
        for _ in range(500):
            blade.update(harsh_inputs)
        assert blade.blade_thickness_mm == pytest.approx(
            RecoaterBlade.FAIL_THICKNESS_MM, abs=0.01
        )

    def test_reset(self, default_inputs: Inputs):
        blade = RecoaterBlade()
        blade.update(default_inputs)
        blade.reset()
        assert blade.health == 1.0
        assert blade.blade_thickness_mm == RecoaterBlade.INITIAL_THICKNESS_MM


# ═══════════════════════════════════════════════════════════════════════════
# 4. Nozzle Plate (Weibull / Temperature)
# ═══════════════════════════════════════════════════════════════════════════

class TestNozzlePlate:
    def test_initial_state(self):
        nozzle = NozzlePlate()
        assert nozzle.health == 1.0
        assert nozzle.clogging_percentage == 0.0
        assert nozzle.name == "NozzlePlate"

    def test_degrades_on_update(self, default_inputs: Inputs):
        nozzle = NozzlePlate()
        report = nozzle.update(default_inputs)
        assert report.health_index < 1.0
        assert "clogging_percentage" in report.metrics
        assert report.metrics["clogging_percentage"] > 0.0

    def test_high_temp_stress_causes_more_damage(self):
        nozzle_cool = NozzlePlate()
        nozzle_hot = NozzlePlate()

        cool = Inputs(temperature_stress=25.0, contamination=0.1,
                      operational_load=5.0, maintenance_level=0.5)
        hot = Inputs(temperature_stress=150.0, contamination=0.1,
                     operational_load=5.0, maintenance_level=0.5)

        r_cool = nozzle_cool.update(cool)
        r_hot = nozzle_hot.update(hot)

        assert r_hot.health_index < r_cool.health_index, \
            "Higher temperature stress should cause more thermal fatigue"

    def test_no_thermal_damage_below_optimal(self):
        nozzle = NozzlePlate()
        # Temperature within optimal range (<=25°C)
        mild = Inputs(temperature_stress=20.0, contamination=0.0,
                      operational_load=0.0, maintenance_level=1.0)
        report = nozzle.update(mild)
        # Only BASE_WEAR_RATE should apply (no thermal damage)
        expected = 1.0 - NozzlePlate.BASE_WEAR_RATE
        assert report.health_index == pytest.approx(expected, abs=1e-5)

    def test_clogging_inverse_of_health(self, default_inputs: Inputs):
        nozzle = NozzlePlate()
        report = nozzle.update(default_inputs)
        expected_clogging = (1.0 - report.health_index) * 100.0
        assert report.metrics["clogging_percentage"] == pytest.approx(
            expected_clogging, abs=0.01
        )

    def test_health_clamps_at_zero(self, harsh_inputs: Inputs):
        nozzle = NozzlePlate()
        for _ in range(500):
            nozzle.update(harsh_inputs)
        assert nozzle.health == 0.0
        assert nozzle.clogging_percentage == pytest.approx(100.0, abs=0.01)

    def test_reset(self, default_inputs: Inputs):
        nozzle = NozzlePlate()
        nozzle.update(default_inputs)
        nozzle.reset()
        assert nozzle.health == 1.0
        assert nozzle.clogging_percentage == 0.0


# ═══════════════════════════════════════════════════════════════════════════
# 5. Heating Elements (Exponential Decay / Operational Load)
# ═══════════════════════════════════════════════════════════════════════════

class TestHeatingElements:
    def test_initial_state(self):
        heater = HeatingElements()
        assert heater.health == 1.0
        assert heater.resistance_ohms == HeatingElements.INITIAL_RESISTANCE
        assert heater.name == "HeatingElements"

    def test_degrades_on_update(self, default_inputs: Inputs):
        heater = HeatingElements()
        report = heater.update(default_inputs)
        assert report.health_index < 1.0
        assert "resistance_ohms" in report.metrics
        assert report.metrics["resistance_ohms"] > HeatingElements.INITIAL_RESISTANCE

    def test_exponential_decay_formula(self, default_inputs: Inputs):
        heater = HeatingElements()
        report = heater.update(default_inputs)
        expected_health = math.exp(
            -HeatingElements.BASE_LAMBDA * default_inputs.operational_load
        )
        assert report.health_index == pytest.approx(expected_health, abs=1e-5)

    def test_cumulative_load(self, default_inputs: Inputs):
        heater = HeatingElements()
        heater.update(default_inputs)
        heater.update(default_inputs)
        expected_health = math.exp(
            -HeatingElements.BASE_LAMBDA * default_inputs.operational_load * 2
        )
        assert heater.health == pytest.approx(expected_health, abs=1e-5)

    def test_higher_load_faster_decay(self):
        h_low = HeatingElements()
        h_high = HeatingElements()

        low_load = Inputs(temperature_stress=25.0, contamination=0.1,
                          operational_load=1.0, maintenance_level=0.5)
        high_load = Inputs(temperature_stress=25.0, contamination=0.1,
                           operational_load=50.0, maintenance_level=0.5)

        r_low = h_low.update(low_load)
        r_high = h_high.update(high_load)

        assert r_high.health_index < r_low.health_index

    def test_resistance_rises_with_degradation(self, default_inputs: Inputs):
        heater = HeatingElements()
        prev_resistance = heater.resistance_ohms
        for _ in range(5):
            report = heater.update(default_inputs)
            assert report.metrics["resistance_ohms"] >= prev_resistance
            prev_resistance = report.metrics["resistance_ohms"]

    def test_zero_load_no_degradation(self):
        heater = HeatingElements()
        zero_load = Inputs(temperature_stress=25.0, contamination=0.0,
                           operational_load=0.0, maintenance_level=1.0)
        report = heater.update(zero_load)
        assert report.health_index == pytest.approx(1.0, abs=1e-5)

    def test_reset(self, default_inputs: Inputs):
        heater = HeatingElements()
        heater.update(default_inputs)
        heater.reset()
        assert heater.health == 1.0
        assert heater.resistance_ohms == HeatingElements.INITIAL_RESISTANCE


# ═══════════════════════════════════════════════════════════════════════════
# 6. Engine Orchestration
# ═══════════════════════════════════════════════════════════════════════════

class TestEngine:
    def test_has_three_components(self, engine: Engine):
        assert len(engine.components) == 3
        names = {c.name for c in engine.components}
        assert names == {"RecoaterBlade", "NozzlePlate", "HeatingElements"}

    def test_update_state_returns_all_components(
        self, engine: Engine, default_inputs: Inputs
    ):
        reports = engine.update_state(default_inputs)
        assert set(reports.keys()) == {
            "RecoaterBlade", "NozzlePlate", "HeatingElements"
        }
        for name, report in reports.items():
            assert isinstance(report, StateReport)
            assert 0.0 <= report.health_index <= 1.0

    def test_step_increments(self, engine: Engine, default_inputs: Inputs):
        assert engine.step == 0
        engine.update_state(default_inputs)
        assert engine.step == 1
        engine.update_state(default_inputs)
        assert engine.step == 2

    def test_determinism(self, default_inputs: Inputs):
        """Two runs with identical inputs must produce identical outputs."""
        engine_a = Engine()
        engine_b = Engine()
        for _ in range(20):
            ra = engine_a.update_state(default_inputs)
            rb = engine_b.update_state(default_inputs)
            for name in ra:
                assert ra[name].health_index == rb[name].health_index
                assert ra[name].operational_status == rb[name].operational_status
                assert ra[name].metrics == rb[name].metrics

    def test_get_all_states_without_advancing(
        self, engine: Engine, default_inputs: Inputs
    ):
        engine.update_state(default_inputs)
        step_before = engine.step
        states = engine.get_all_states()
        assert engine.step == step_before
        assert set(states.keys()) == {
            "RecoaterBlade", "NozzlePlate", "HeatingElements"
        }

    def test_reset(self, engine: Engine, default_inputs: Inputs):
        for _ in range(10):
            engine.update_state(default_inputs)
        engine.reset()
        assert engine.step == 0
        for comp in engine.components:
            assert comp.health == 1.0

    def test_components_eventually_fail(self, engine: Engine, harsh_inputs: Inputs):
        """Under harsh conditions, at least one component should reach FAILED."""
        for _ in range(500):
            reports = engine.update_state(harsh_inputs)
        any_failed = any(
            r.operational_status == OperationalStatus.FAILED
            for r in reports.values()
        )
        assert any_failed, "At least one component should fail under harsh conditions"


# ═══════════════════════════════════════════════════════════════════════════
# 7. StateReport / OperationalStatus
# ═══════════════════════════════════════════════════════════════════════════

class TestStateReport:
    def test_status_enum_values(self):
        assert OperationalStatus.FUNCTIONAL.value == "FUNCTIONAL"
        assert OperationalStatus.DEGRADED.value == "DEGRADED"
        assert OperationalStatus.CRITICAL.value == "CRITICAL"
        assert OperationalStatus.FAILED.value == "FAILED"

    def test_state_report_fields(self):
        report = StateReport(
            health_index=0.85,
            operational_status=OperationalStatus.FUNCTIONAL,
            metrics={"some_metric": 42.0},
        )
        assert report.health_index == 0.85
        assert report.operational_status == OperationalStatus.FUNCTIONAL
        assert report.metrics["some_metric"] == 42.0
