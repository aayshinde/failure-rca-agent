"""Data shaping for the dashboard: fleet risk snapshot, per-machine drill-down, incident list.

Everything here is a thin layer over Toolbox, so the dashboard shows exactly what the agent's tools see.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import xgboost as xgb

from src.anomaly import windows_ending_at
from src.features import failure_times, machine_features
from src.knowledge import SENSORS, UNITS
from src.tools import Toolbox, parse_time


def time_range(tb: Toolbox) -> dict:
    inc = tb.fleet.incidents
    test = inc[inc.failure_time >= tb.fleet.split_time]
    return {"start": str(tb.fleet.split_time + pd.Timedelta(hours=72)), "end": str(test.failure_time.max()),
            "train_end": str(tb.fleet.split_time)}


def incident_list(tb: Toolbox, split: str = "test") -> list[dict]:
    inc = tb.fleet.incidents
    inc = inc[inc.split == split].sort_values("failure_time")
    return [{"id": r.incident_id, "machine_id": r.machine_id, "time": str(r.failure_time),
             "root_cause": r.root_cause, "component": r.component} for r in inc.itertuples()]


def _features(tb: Toolbox, m: str) -> pd.DataFrame:
    if m not in tb._feat_cache:
        tb._feat_cache[m] = machine_features(tb.fleet, m, tb.norm)
    return tb._feat_cache[m]


def _fails_within(tb: Toolbox, m: str, t: pd.Timestamp, hours: int = 24) -> bool:
    ft = failure_times(tb.fleet, m)
    return bool(len(ft) and ((ft > np.datetime64(t)) & (ft <= np.datetime64(t + pd.Timedelta(hours=hours)))).any())


def fleet_snapshot(tb: Toolbox, at: str) -> dict:
    t = parse_time(at)
    thr = tb.risk_meta["threshold"]
    rows = []
    for m, info in tb.fleet.machines.iterrows():
        f = _features(tb, m).loc[:t]
        if f.empty:
            continue
        cols = tb.risk_meta["features"]
        p = float(tb.risk_model.predict(xgb.DMatrix(f[cols].iloc[[-1]]))[0])
        rows.append({"machine_id": m, "model": info["model"], "p": round(p, 3), "flagged": p >= thr,
                     "fails_24h": _fails_within(tb, m, t)})
    flagged = [r for r in rows if r["flagged"]]
    return {"at": str(t), "threshold": round(thr, 2), "machines": rows,
            "summary": {"n": len(rows), "flagged": len(flagged),
                        "true_positive": sum(r["fails_24h"] for r in flagged),
                        "missed": sum(r["fails_24h"] and not r["flagged"] for r in rows),
                        "failing_soon": sum(r["fails_24h"] for r in rows)}}


def machine_detail(tb: Toolbox, machine_id: str, at: str, hours: int = 168) -> dict:
    m, t = tb._check_machine(machine_id), parse_time(at)
    lo = t - pd.Timedelta(hours=hours - 1)
    z, raw = tb._z(m).loc[lo:t], tb.fleet.tel(m).loc[lo:t]
    idx = z.index

    # anomaly score for every hour in the window (same windows the detect_anomalies tool uses)
    w, ok = windows_ending_at(tb._z(m), idx)
    anomaly = [None] * len(idx)
    if ok.any():
        sc = tb.detector.score_windows(w[ok])["score"]
        for i, v in zip(np.flatnonzero(ok), sc):
            anomaly[i] = round(float(v), 2)

    # risk history across the window
    f = _features(tb, m).loc[lo:t]
    cols = tb.risk_meta["features"]
    risk = dict(zip(f.index, tb.risk_model.predict(xgb.DMatrix(f[cols])))) if len(f) else {}

    factors = []
    if len(f):
        dm = xgb.DMatrix(f[cols].iloc[[-1]])
        contrib = tb.risk_model.predict(dm, pred_contribs=True)[0][:-1]
        factors = [{"feature": cols[i], "value": round(float(f[cols].iloc[-1, i]), 2), "shap": round(float(contrib[i]), 3)}
                   for i in np.argsort(-np.abs(contrib))[:6]]

    ft = failure_times(tb.fleet, m)
    fails = [str(pd.Timestamp(x)) for x in ft if np.datetime64(lo) <= x <= np.datetime64(t + pd.Timedelta(hours=24))]
    lg = tb.fleet.machine_logs(m).loc[lo:t]
    lg = lg[lg.severity != "info"]
    ts = [str(x) for x in idx]
    return {
        "machine_id": m, "model": tb.fleet.machines.loc[m, "model"], "at": str(t),
        "times": ts,
        "sensors": {s: {"z": [None if np.isnan(v) else round(float(v), 2) for v in z[s]],
                        "raw": [None if np.isnan(v) else round(float(v), 1) for v in raw[s].reindex(idx)],
                        "unit": UNITS[s]} for s in SENSORS},
        "anomaly": {"score": anomaly, "threshold": round(tb.detector.meta["threshold"], 2)},
        "risk": {"p": [round(float(risk[i]), 3) if i in risk else None for i in idx],
                 "threshold": round(tb.risk_meta["threshold"], 2)},
        "failures": fails,
        "log_events": [{"time": str(r.Index), "code": r.code, "severity": r.severity} for r in lg.itertuples()][-60:],
        "factors": factors,
        "risk_now": tb.predict_failure_risk(m, str(t)),
        "anomaly_now": tb.detect_anomalies(m, str(t)),
    }
