"""Domain definitions shared by the simulator, the documentation generator and the
rule-based baseline: sensors, failure modes, their signatures and error codes."""
from __future__ import annotations

from dataclasses import dataclass, field

SENSORS = ["volt", "rotate", "pressure", "vibration", "temperature", "current"]
UNITS = {"volt": "V", "rotate": "rpm", "pressure": "psi", "vibration": "mm/s", "temperature": "°C", "current": "A"}

# Baseline mean / std of each sensor per machine model.
MODEL_BASELINES = {
    "model1": {"volt": (170, 6), "rotate": (450, 18), "pressure": (100, 4), "vibration": (40, 2.5), "temperature": (65, 2), "current": (12, 0.6)},
    "model2": {"volt": (230, 7), "rotate": (520, 20), "pressure": (110, 5), "vibration": (35, 2.0), "temperature": (60, 2), "current": (15, 0.7)},
    "model3": {"volt": (170, 6), "rotate": (380, 15), "pressure": (140, 6), "vibration": (45, 3.0), "temperature": (70, 2.5), "current": (11, 0.5)},
    "model4": {"volt": (400, 10), "rotate": (600, 25), "pressure": (95, 4), "vibration": (30, 2.0), "temperature": (68, 2), "current": (20, 1.0)},
}

COMPONENTS = ["bearing", "cooling_fan", "psu", "hydraulic_seal", "motor", "sensor_array", "controller"]


@dataclass(frozen=True)
class FailureMode:
    name: str
    component: str
    title: str
    precursor_hours: int
    # sensor -> (kind, magnitude). kind: drift (mean shift in std units), noise (std multiplier)
    signature: dict = field(default_factory=dict)
    error_codes: tuple = ()
    base_rate: float = 1.0            # relative frequency
    wear: bool = False                # hazard grows with time since component maintenance
    model_bias: dict = field(default_factory=dict)  # machine model -> hazard multiplier
    operator_hints: tuple = ()


FAILURE_MODES: dict[str, FailureMode] = {
    m.name: m
    for m in [
        FailureMode(
            "bearing_wear", "bearing", "Bearing wear / spindle bearing degradation", 96,
            {"vibration": ("drift", 4.0), "temperature": ("drift", 1.2)},
            ("E101", "E105"), 1.2, True, {"model2": 1.6},
            ("grinding noise from the drive end", "operator felt heavy vibration on the housing"),
        ),
        FailureMode(
            "cooling_failure", "cooling_fan", "Cooling system failure / overheating", 48,
            {"temperature": ("drift", 4.5), "current": ("drift", 0.8)},
            ("E201", "E203"), 1.0, True, {"model4": 1.8},
            ("enclosure was hot to the touch", "fan not audible during walkdown"),
        ),
        FailureMode(
            "power_supply_fault", "psu", "Power supply instability", 36,
            {"volt": ("noise", 3.5), "current": ("noise", 2.5)},
            ("E301", "E302"), 0.9, False, {"model1": 1.5},
            ("lights in the cabinet flickered", "breaker tripped once before shutdown"),
        ),
        FailureMode(
            "hydraulic_leak", "hydraulic_seal", "Hydraulic seal leak / pressure loss", 120,
            {"pressure": ("drift", -4.0), "vibration": ("drift", 0.6)},
            ("E401", "E402"), 1.0, True, {"model3": 2.2},
            ("oil found under the unit", "clamping felt weak on the last batch"),
        ),
        FailureMode(
            "motor_overload", "motor", "Drive motor overload", 24,
            {"current": ("drift", 3.5), "rotate": ("drift", -2.5), "temperature": ("drift", 2.0)},
            ("E501", "E201"), 0.9, False, {},
            ("motor labored under load", "heavy batch running when it stopped"),
        ),
        FailureMode(
            "sensor_malfunction", "sensor_array", "Sensor malfunction / bad instrumentation", 48,
            {"__one_random__": ("noise", 6.0)},
            ("E601",), 0.7, False, {},
            ("HMI showed readings jumping around", "readings looked wrong to the operator"),
        ),
        FailureMode(
            "controller_fault", "controller", "Controller / firmware fault", 4,
            {},
            ("E701", "E702"), 0.6, False, {},
            ("screen froze before the stop", "unit rebooted by itself"),
        ),
    ]
}
MODE_NAMES = list(FAILURE_MODES)

ERROR_CODES = {
    "E000": ("Fault trip - machine stopped", "critical"),
    "E101": ("High vibration amplitude", "warning"),
    "E105": ("Spindle temperature above soft limit", "warning"),
    "E201": ("Over-temperature threshold exceeded", "error"),
    "E203": ("Cooling fan tachometer fault", "error"),
    "E301": ("Supply undervoltage / brown-out detected", "error"),
    "E302": ("DC bus ripple out of tolerance", "warning"),
    "E401": ("Hydraulic pressure low", "error"),
    "E402": ("Hydraulic reservoir level low", "warning"),
    "E501": ("Drive overcurrent", "error"),
    "E601": ("Sensor value out of plausible range", "warning"),
    "E701": ("Controller watchdog reset", "critical"),
    "E702": ("Firmware exception in PLC task", "critical"),
    "E900": ("Shift change / operator login", "info"),
    "E901": ("Recipe changed", "info"),
}

# Which modes each code is commonly associated with (codes overlap on purpose).
CODE_TO_MODES: dict[str, list[str]] = {}
for _m in FAILURE_MODES.values():
    for _c in _m.error_codes:
        CODE_TO_MODES.setdefault(_c, []).append(_m.name)

GENERIC_REPORTS = (
    "unit tripped offline during production",
    "machine stopped unexpectedly, alarm on HMI",
    "line halted, operator could not restart",
    "unplanned stop reported by second shift",
    "no obvious cause noticed by operator",
)
