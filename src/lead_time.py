"""Operational view of the risk model: how much warning do operators get, and how many false alarms do they eat?

AUROC says little about whether a model is usable. This turns hourly risk scores into *alert episodes* (consecutive
flagged hours, merged across short gaps) on the held-out test period and reports, per decision threshold:

  caught            share of real failures preceded by an alert in the previous 24h
  lead time         hours between the start of that alert episode and the failure (median / p25)
  false alerts      alert episodes with no failure within 24h of their end, per machine-month
  precision         share of alert episodes that were real

    python -m src.lead_time
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
import xgboost as xgb

from src import config
from src.features import FleetData, load_norm, machine_features
from src.tools import Toolbox

MERGE_GAP_H = 12          # flagged hours closer than this belong to one alert episode
LOOKAHEAD_H = config.RISK_HORIZON_H


def hourly_risk(tb: Toolbox, test_start: pd.Timestamp) -> dict[str, pd.Series]:
    cols = tb.risk_meta["features"]
    out = {}
    for m in tb.fleet.machines.index:
        f = machine_features(tb.fleet, m, tb.norm)
        f = f.loc[test_start:]
        if len(f):
            out[m] = pd.Series(tb.risk_model.predict(xgb.DMatrix(f[cols])), index=f.index)
    return out


def episodes(flag: pd.Series) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
    hours = flag.index[flag.values]
    if not len(hours):
        return []
    eps, start, prev = [], hours[0], hours[0]
    for h in hours[1:]:
        if h - prev > pd.Timedelta(hours=MERGE_GAP_H):
            eps.append((start, prev))
            start = h
        prev = h
    eps.append((start, prev))
    return eps


def evaluate(risk: dict[str, pd.Series], fleet: FleetData, thr: float) -> dict:
    inc = fleet.incidents
    n_fail = caught = n_ep = true_ep = 0
    leads, machine_hours = [], 0
    for m, p in risk.items():
        machine_hours += len(p)
        eps = episodes(p >= thr)
        fails = inc[(inc.machine_id == m) & (inc.failure_time >= p.index[0])].failure_time
        fails = [pd.Timestamp(x) for x in fails if pd.Timestamp(x) <= p.index[-1]]
        n_ep += len(eps)
        for s, e in eps:
            true_ep += any(s < f <= e + pd.Timedelta(hours=LOOKAHEAD_H) for f in fails)
        for f in fails:
            n_fail += 1
            hit = [(s, e) for s, e in eps if e >= f - pd.Timedelta(hours=LOOKAHEAD_H) and s < f]
            if hit:
                caught += 1
                leads.append((f - hit[0][0]).total_seconds() / 3600)
    machine_months = machine_hours / 24 / 30
    return {"threshold": round(thr, 2), "failures": n_fail, "caught_pct": round(100 * caught / max(n_fail, 1), 1),
            "lead_median_h": round(float(np.median(leads)), 1) if leads else None,
            "lead_p25_h": round(float(np.percentile(leads, 25)), 1) if leads else None,
            "alert_episodes": n_ep, "precision_pct": round(100 * true_ep / max(n_ep, 1), 1),
            "false_alerts_per_machine_month": round((n_ep - true_ep) / machine_months, 2)}


def main() -> dict:
    tb = Toolbox()
    risk = hourly_risk(tb, tb.fleet.split_time)
    thr0 = tb.risk_meta["threshold"]
    rows = [evaluate(risk, tb.fleet, t) for t in sorted({0.3, 0.5, 0.7, round(thr0, 2), 0.95})]
    res = {"merge_gap_h": MERGE_GAP_H, "lookahead_h": LOOKAHEAD_H, "tuned_threshold": round(thr0, 2),
           "machines": len(risk), "operating_points": rows}
    (config.RESULTS_DIR / "lead_time.json").write_text(json.dumps(res, indent=2))
    print(pd.DataFrame(rows).to_string(index=False))
    return res


if __name__ == "__main__":
    main()
