"""Simulate a fleet of industrial machines with three data sources plus documentation.

Outputs (data/):
    machines.csv          machine_id, model, install_year
    telemetry.parquet     hourly volt / rotate / pressure / vibration / temperature / current
    logs.csv              controller event log (error codes, with background noise)
    maintenance.csv       scheduled + corrective maintenance records
    incidents.csv         every failure with its ground-truth root cause
    docs/*.md             runbooks, error-code reference, maintenance manual, fleet notes, distractors
    eval_questions.jsonl  held-out questions for the agent (see src/eval_set.py)

Each failure mode leaves a precursor signature in telemetry (drift / noise on specific
sensors) and in the logs (characteristic error codes) before the machine trips. Difficulty
comes from: random signature strength, error codes shared between modes, background
warning noise, transient disturbances that never become failures, missing log ingestion
for some incidents, and operator reports that are usually generic and sometimes misleading.

Usage:  python -m src.simulate                # 100 machines x 365 days
        python -m src.simulate --machines 20 --days 120   # small/fast
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from src import config
from src.knowledge import (
    COMPONENTS, ERROR_CODES, FAILURE_MODES, GENERIC_REPORTS, MODEL_BASELINES, SENSORS,
)

DAILY_FAILURE_RATE = 1 / 30          # fleet-average failures per machine-day
LOG_MISSING_P = 0.12                 # incident whose logs were never ingested
WARNING_NOISE_P = 0.006              # random warning per machine-hour
TRANSIENT_P = 1 / 20                 # harmless disturbance per machine-day
HINT_P, MISLEADING_P = 0.35, 0.08    # operator report informativeness


def simulate(n_machines: int, n_days: int, seed: int = config.SEED) -> dict[str, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    H = n_days * 24
    times = pd.date_range(config.START_DATE, periods=H, freq="h")
    models = rng.choice(list(MODEL_BASELINES), size=n_machines, p=[0.3, 0.25, 0.25, 0.2])
    machines = pd.DataFrame({
        "machine_id": [f"M{i + 1:03d}" for i in range(n_machines)],
        "model": models,
        "install_year": rng.integers(2012, 2024, n_machines),
    })

    modes = list(FAILURE_MODES.values())
    base = np.array([m.base_rate for m in modes])
    base = base / base.sum() * DAILY_FAILURE_RATE

    incidents, maint, logs = [], [], []
    tel = np.zeros((n_machines, H, len(SENSORS)), dtype=np.float32)
    keep = np.ones((n_machines, H), dtype=bool)

    for mi, row in machines.iterrows():
        mid, model = row.machine_id, row.model
        mu = np.array([MODEL_BASELINES[model][s][0] for s in SENSORS]) * rng.uniform(0.97, 1.03, len(SENSORS))
        sd = np.array([MODEL_BASELINES[model][s][1] for s in SENSORS])

        # --- normal operation: AR(1) noise + shift cycle on temperature/current
        z = np.zeros((H, len(SENSORS)))
        eps = rng.normal(size=(H, len(SENSORS)))
        for t in range(1, H):
            z[t] = 0.7 * z[t - 1] + 0.714 * eps[t]
        hours = np.arange(H) % 24
        cycle = np.where((hours >= 6) & (hours < 22), 0.5, -0.5)
        z[:, SENSORS.index("temperature")] += cycle
        z[:, SENSORS.index("current")] += 0.6 * cycle
        x = mu + sd * z

        # --- maintenance + failure process (daily)
        last = {c: -rng.uniform(0, 90) for c in COMPONENTS}            # day of last service
        interval = {c: rng.uniform(75, 150) if c in ("bearing", "cooling_fan", "hydraulic_seal") else rng.uniform(180, 320)
                    for c in COMPONENTS}
        next_due = {c: last[c] + interval[c] for c in COMPONENTS}
        busy_until = -1  # hour index until which no new failure window may start

        for d in range(n_days):
            for c in COMPONENTS:
                if d >= next_due[c]:
                    t = d * 24 + int(rng.integers(6, 18))
                    maint.append((mid, times[min(t, H - 1)], c, "scheduled", f"Scheduled service of {c.replace('_', ' ')}"))
                    last[c], next_due[c] = d, d + interval[c]
            hazard = []
            for m, b in zip(modes, base):
                h = b * m.model_bias.get(model, 1.0)
                if m.wear:
                    h *= 0.4 + 1.2 * min((d - last[m.component]) / 120, 2.0)
                hazard.append(h)
            hazard = np.array(hazard)
            if rng.random() > 1 - np.exp(-hazard.sum()):
                continue
            mode = modes[rng.choice(len(modes), p=hazard / hazard.sum())]
            tf = d * 24 + int(rng.integers(0, 24))
            if tf - mode.precursor_hours <= busy_until + 12 or tf >= H - 1:
                continue
            down = int(rng.integers(6, 17))
            strength = float(rng.uniform(0.35, 1.0))
            logs_missing = bool(rng.random() < LOG_MISSING_P)

            # telemetry signature
            W = mode.precursor_hours
            idx = np.arange(tf - W, tf)
            r = ((idx - (tf - W)) / W) ** 1.5 * strength
            detail = ""
            for sensor, (kind, mag) in mode.signature.items():
                if sensor == "__one_random__":
                    k = int(rng.integers(len(SENSORS)))
                    detail = SENSORS[k]
                    if rng.random() < 0.6:
                        x[idx, k] += sd[k] * mag * r * rng.normal(size=W)
                    else:  # stuck reading with an offset
                        x[idx, k] = np.where(r > 0.15 * strength, mu[k] + sd[k] * 5 * strength * np.sign(rng.normal()), x[idx, k])
                    continue
                k = SENSORS.index(sensor)
                if kind == "drift":
                    x[idx, k] += sd[k] * mag * r
                else:
                    x[idx, k] += sd[k] * mag * r * rng.normal(size=W)

            # precursor log events
            if not logs_missing:
                for j, code in enumerate(mode.error_codes):
                    n = rng.poisson((1 + 5 * strength) * (1.0 if j == 0 else 0.6))
                    lo = tf - max(int(W * 0.6), 2)
                    for t in np.sort(rng.integers(lo, tf, n)):
                        logs.append((mid, times[t], code))
                logs.append((mid, times[tf], "E000"))

            keep[mi, tf: min(tf + down, H)] = False
            maint.append((mid, times[min(tf + down, H - 1)], mode.component, "corrective",
                          f"Replaced/repaired {mode.component.replace('_', ' ')} after fault trip"))
            last[mode.component] = (tf + down) / 24
            next_due[mode.component] = last[mode.component] + interval[mode.component]
            busy_until = tf + down

            u = rng.random()
            if u < MISLEADING_P:
                other = rng.choice([m for m in modes if m.name != mode.name and m.operator_hints])
                report = rng.choice(other.operator_hints)
            elif u < MISLEADING_P + HINT_P and mode.operator_hints:
                report = rng.choice(mode.operator_hints)
            else:
                report = rng.choice(GENERIC_REPORTS)
            incidents.append({
                "machine_id": mid, "model": model, "failure_time": times[tf], "root_cause": mode.name,
                "component": mode.component, "operator_report": str(report), "signature_strength": round(strength, 3),
                "logs_missing": logs_missing, "affected_sensor": detail,
            })

        # transient disturbances that never become failures (false-alarm material)
        for d in np.nonzero(rng.random(n_days) < TRANSIENT_P)[0]:
            t0 = d * 24 + int(rng.integers(0, 18))
            k = int(rng.integers(len(SENSORS)))
            dur = int(rng.integers(3, 8))
            x[t0: t0 + dur, k] += sd[k] * rng.uniform(1.5, 3.0) * np.sign(rng.normal())
            if rng.random() < 0.3:
                logs.append((mid, times[min(t0 + 1, H - 1)], rng.choice(["E101", "E105", "E201", "E302", "E601"])))

        # background warning noise + shift-change info events
        noisy = np.nonzero(rng.random(H) < WARNING_NOISE_P)[0]
        for t in noisy:
            logs.append((mid, times[t], rng.choice(["E105", "E101", "E201", "E302", "E601", "E402", "E501"])))
        for t in range(6, H, 8):
            logs.append((mid, times[t], "E900"))
        for t in np.nonzero(rng.random(H) < 0.01)[0]:
            logs.append((mid, times[t], "E901"))

        tel[mi] = x

    # ---- assemble tables
    mi_idx, t_idx = np.nonzero(keep)
    telemetry = pd.DataFrame({"machine_id": machines.machine_id.values[mi_idx], "datetime": times[t_idx]})
    for k, s in enumerate(SENSORS):
        telemetry[s] = tel[mi_idx, t_idx, k].round(3)

    logs_df = pd.DataFrame(logs, columns=["machine_id", "datetime", "code"])
    logs_df["severity"] = logs_df.code.map(lambda c: ERROR_CODES[c][1])
    logs_df["message"] = logs_df.code.map(lambda c: ERROR_CODES[c][0])
    logs_df = logs_df.sort_values(["machine_id", "datetime"]).reset_index(drop=True)
    # drop log lines that fall inside downtime (machine was off)
    down = telemetry.set_index(["machine_id", "datetime"]).index
    logs_df = logs_df[logs_df.set_index(["machine_id", "datetime"]).index.isin(down) | (logs_df.code == "E000")]

    maint_df = pd.DataFrame(maint, columns=["machine_id", "datetime", "component", "type", "note"]).sort_values(
        ["machine_id", "datetime"]).reset_index(drop=True)

    inc = pd.DataFrame(incidents).sort_values("failure_time").reset_index(drop=True)
    inc.insert(0, "incident_id", [f"INC-{i + 1:05d}" for i in range(len(inc))])
    split_time = times[0] + pd.Timedelta(days=int(n_days * config.TRAIN_FRAC))
    inc["split"] = np.where(inc.failure_time < split_time, "train", "test")

    return {"machines": machines, "telemetry": telemetry, "logs": logs_df.reset_index(drop=True),
            "maintenance": maint_df, "incidents": inc}


def save(tables: dict[str, pd.DataFrame]) -> None:
    tables["telemetry"].to_parquet(config.DATA_DIR / "telemetry.parquet", index=False)
    for name in ("machines", "logs", "maintenance", "incidents"):
        tables[name].to_csv(config.DATA_DIR / f"{name}.csv", index=False)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--machines", type=int, default=config.N_MACHINES)
    ap.add_argument("--days", type=int, default=config.N_DAYS)
    args = ap.parse_args()

    from src import docs_gen, eval_set

    tables = simulate(args.machines, args.days)
    save(tables)
    n_docs = docs_gen.write_docs()
    qs = eval_set.build_and_save(tables["incidents"], tables["telemetry"], args.days)

    inc = tables["incidents"]
    print(f"Machines        : {len(tables['machines'])}")
    print(f"Telemetry rows  : {len(tables['telemetry']):,}")
    print(f"Log events      : {len(tables['logs']):,}")
    print(f"Maintenance     : {len(tables['maintenance']):,}")
    print(f"Incidents       : {len(inc):,}  (train {sum(inc.split == 'train')}, test {sum(inc.split == 'test')})")
    print(f"Docs written    : {n_docs}")
    print(f"Eval questions  : {len(qs)}  {pd.Series([q['type'] for q in qs]).value_counts().to_dict()}")
    print("Root causes     :", inc.root_cause.value_counts().to_dict())


if __name__ == "__main__":
    main()
