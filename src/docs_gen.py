"""Generate the maintenance documentation corpus (markdown) that the agent retrieves from.

Runbooks deliberately include differential-diagnosis notes, because several failure
modes share error codes and sensors; plus distractor documents that should not be retrieved.
"""
from __future__ import annotations

import re

from src import config
from src.knowledge import CODE_TO_MODES, COMPONENTS, ERROR_CODES, FAILURE_MODES, UNITS

DIFFERENTIALS = {
    "bearing_wear": "Distinguish from hydraulic_leak (small vibration rise but pressure falls) and from transient vibration "
                    "spikes lasting only a few hours. Bearing wear builds over several days and vibration keeps climbing. "
                    "Long time since the last bearing service makes this more likely.",
    "cooling_failure": "Distinguish from motor_overload, which also raises temperature and logs E201 but shows a large "
                       "current increase together with a drop in rotation speed. Cooling failure raises temperature "
                       "much more than current, and E203 (fan tachometer fault) is specific to it.",
    "power_supply_fault": "The signature is instability (variance), not a level shift: voltage and current become noisy. "
                          "A single noisy sensor with other sensors normal points to sensor_malfunction instead.",
    "hydraulic_leak": "Pressure decays slowly over up to five days. Model3 units are especially prone. "
                      "A sudden pressure jump or a stuck pressure value is more likely sensor_malfunction.",
    "motor_overload": "Short precursor (about a day): current rises sharply, rotation speed drops, temperature rises. "
                      "E501 is the key code; E201 alone is not enough to tell it from cooling_failure.",
    "sensor_malfunction": "Exactly one sensor is erratic or stuck at an implausible value while the physically related "
                          "sensors stay normal. E601 is common. Physical failure modes move several related sensors together.",
    "controller_fault": "No telemetry precursor: sensors look normal until the trip. The evidence is in the controller "
                        "log (E701 watchdog reset, E702 firmware exception) in the last hours before the stop. "
                        "If logs are missing and telemetry is normal, controller_fault is the most likely cause.",
}

CHECKS = {
    "bearing_wear": ["Measure vibration spectrum at drive-end and non-drive-end bearings", "Check bearing temperature with IR thermometer", "Inspect lubrication condition and grease interval"],
    "cooling_failure": ["Verify cooling fan spins and tachometer reads correctly", "Check filters and heat-exchanger fins for blockage", "Confirm coolant level and pump operation"],
    "power_supply_fault": ["Measure incoming supply voltage under load", "Check DC bus ripple with an oscilloscope", "Inspect PSU capacitors and terminal tightness"],
    "hydraulic_leak": ["Inspect seals, fittings and hoses for oil", "Check reservoir level", "Pressure-test the circuit with the pump isolated"],
    "motor_overload": ["Check load on the drive and for mechanical binding", "Measure phase currents and insulation resistance", "Review recipe / batch size at time of trip"],
    "sensor_malfunction": ["Compare the suspect reading with a handheld reference instrument", "Check sensor cable and connector", "Swap the sensor channel to confirm"],
    "controller_fault": ["Download the PLC diagnostic buffer", "Check firmware version against the latest approved release", "Inspect controller power and backplane"],
}

FIX = {
    "bearing_wear": "Replace the bearing set, re-lubricate and re-align the spindle. Return to service after a 2-hour run-in with vibration below baseline + 1 std.",
    "cooling_failure": "Replace the cooling fan or pump, clean filters, and verify temperature returns to baseline within 1 hour.",
    "power_supply_fault": "Replace the power supply unit or its filter capacitors; tighten terminals; verify voltage stability.",
    "hydraulic_leak": "Replace the failed seal, refill the reservoir and bleed the circuit. Recheck pressure after 24 hours.",
    "motor_overload": "Remove the mechanical cause of overload, inspect the motor windings, and reset the drive.",
    "sensor_malfunction": "Replace or recalibrate the faulty sensor; no mechanical repair is required.",
    "controller_fault": "Reflash approved firmware and restore the PLC program; replace the controller if resets recur.",
}

INTERVALS = {
    "bearing": "Service every 90-150 days; failure risk grows sharply beyond 120 days since last service.",
    "cooling_fan": "Inspect every 90 days, replace every 150 days or on tachometer fault.",
    "psu": "Inspect yearly; replace on recurring E301/E302.",
    "hydraulic_seal": "Replace seals every 90-150 days; model3 units every 90 days.",
    "motor": "Inspect windings and insulation every 180-300 days.",
    "sensor_array": "Calibrate every 180 days.",
    "controller": "Apply firmware updates as released; inspect yearly.",
}

DISTRACTORS = {
    "safety_lockout": "# Lockout / tagout procedure\n\n## Scope\nApplies to all personnel performing service on energized equipment.\n\n"
                      "## Procedure\nNotify affected staff, shut down the machine, isolate all energy sources, apply locks and tags, "
                      "verify zero energy before starting work.\n\n## Release\nRemove tools, reinstall guards, remove locks in reverse order.",
    "spare_parts_ordering": "# Spare parts ordering\n\n## Process\nRaise a purchase requisition in the ERP with the part number and machine ID. "
                            "Critical spares are stocked at the central warehouse.\n\n## Lead times\nStandard parts ship in 3-5 business days; "
                            "expedited shipping is available for line-down situations.",
    "shift_handover": "# Shift handover guideline\n\n## Content\nRecord open work orders, machines running in degraded mode and pending "
                      "quality holds.\n\n## Format\nUse the shift log template; handover must be signed by both supervisors.",
    "it_access": "# IT access for maintenance laptops\n\n## VPN\nConnect to the plant VPN before downloading PLC diagnostics.\n\n"
                 "## Passwords\nRotate service passwords every 90 days.",
}


def slug(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", s.lower()).strip("-")


def describe_signature(mode) -> str:
    if not mode.signature:
        return "No telemetry precursor; sensors stay within normal range until the trip."
    if "__one_random__" in mode.signature:
        return "A single sensor (any channel) becomes erratic or reads a stuck, implausible value; other sensors stay normal."
    parts = []
    for s, (kind, mag) in mode.signature.items():
        if kind == "drift":
            parts.append(f"{s} ({UNITS[s]}) {'rises' if mag > 0 else 'falls'} gradually, up to ~{abs(mag):.0f} std from baseline")
        else:
            parts.append(f"{s} ({UNITS[s]}) becomes unstable/noisy (variance up to ~{mag:.0f}x normal)")
    return "; ".join(parts) + f". Precursor typically starts {mode.precursor_hours} hours before failure."


def runbook(mode) -> str:
    codes = ", ".join(f"{c} ({ERROR_CODES[c][0]})" for c in mode.error_codes)
    prone = ", ".join(f"{k} ({v:.1f}x fleet rate)" for k, v in mode.model_bias.items()) or "no model-specific bias"
    return (
        f"# Runbook: {mode.title}\n\nFailure mode ID: {mode.name}. Component: {mode.component}.\n\n"
        f"## Symptoms\n{describe_signature(mode)}\n\n"
        f"## Associated error codes\n{codes}.\n\n"
        f"## Differential diagnosis\n{DIFFERENTIALS[mode.name]}\n\n"
        f"## Diagnostic checks\n" + "\n".join(f"- {c}" for c in CHECKS[mode.name]) + "\n\n"
        f"## Corrective action\n{FIX[mode.name]}\n\n"
        f"## Fleet notes\nMore frequent on: {prone}."
        + (" Wear-related: risk increases with time since the component was last serviced." if mode.wear else "")
        + "\n"
    )


def error_code_reference() -> str:
    out = ["# Error code reference\n"]
    for code, (desc, sev) in ERROR_CODES.items():
        modes = CODE_TO_MODES.get(code, [])
        causes = ", ".join(modes) if modes else "informational; not a fault indicator"
        out.append(f"## {code}\n{code} - {desc}. Severity: {sev}. Commonly associated failure modes: {causes}. "
                   + ("Isolated occurrences also appear as background noise; look for repeated events in the hours "
                      "before a stop." if modes else "") + "\n")
    return "\n".join(out)


def maintenance_manual() -> str:
    out = ["# Maintenance manual: service intervals\n"]
    for c in COMPONENTS:
        out.append(f"## {c}\nComponent {c.replace('_', ' ')}. {INTERVALS[c]}\n")
    return "\n".join(out)


def write_docs() -> int:
    d = config.DOCS_DIR
    d.mkdir(parents=True, exist_ok=True)
    files = {f"runbook_{m.name}": runbook(m) for m in FAILURE_MODES.values()}
    files["error_codes"] = error_code_reference()
    files["maintenance_manual"] = maintenance_manual()
    files.update(DISTRACTORS)
    for name, text in files.items():
        (d / f"{name}.md").write_text(text)
    return len(files)


def chunk_docs() -> list[dict]:
    """Split every markdown doc on '## ' headings. chunk_id = '<doc_id>#<section-slug>'."""
    chunks = []
    for path in sorted(config.DOCS_DIR.glob("*.md")):
        doc_id = path.stem
        text = path.read_text()
        title = text.splitlines()[0].lstrip("# ").strip()
        preamble, *sections = re.split(r"\n## ", text)
        intro = preamble.split("\n", 1)[1].strip() if "\n" in preamble else ""
        for sec in sections:
            heading, _, body = sec.partition("\n")
            chunks.append({
                "chunk_id": f"{doc_id}#{slug(heading)}",
                "doc_id": doc_id,
                "title": title,
                "section": heading.strip(),
                "text": f"{title} — {heading.strip()}\n{intro}\n{body.strip()}".strip(),
            })
    return chunks
