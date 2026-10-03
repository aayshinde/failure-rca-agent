"""Export a static, serverless copy of the dashboard for GitHub Pages (no Python needed to view it).

Everything the live console computes on request is pre-computed here from the real models, on a 3-hour grid over the
held-out test period: fleet risk, per-machine sensor / anomaly / risk series, SHAP factors, warning/error log events, and
the `rules` agent's investigation for every test incident. The page in docs/ is the same dashboard/index.html with a flag
that swaps its data layer from the API to these files.

    python -m src.export_static            # writes docs/index.html and docs/data/*.json (~4 min)
    python -m src.export_static --html-only   # just refresh the page from dashboard/index.html
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb

from src import config, viz
from src.anomaly import windows_ending_at
from src.features import failure_times
from src.knowledge import ERROR_CODES, SENSORS
from src.tools import Toolbox

STEP_H = 3
HIST_H = 168                       # drill-down window length
TOP_FACTORS = 4
OUT = config.ROOT / "docs"


def _ints(a, scale, none=-9999):
    return [none if (v is None or (isinstance(v, float) and np.isnan(v))) else int(round(float(v) * scale)) for v in a]


def write_html() -> None:
    html = (config.ROOT / "dashboard" / "index.html").read_text(encoding="utf-8")
    html = html.replace("<script>\nconst $ =", "<script>window.STATIC_DEMO = true;</script>\n<script>\nconst $ =", 1)
    (OUT / "index.html").write_text(html, encoding="utf-8")
    (OUT / ".nojekyll").write_text("")


def main() -> None:
    if "--html-only" in sys.argv:          # refresh the page after editing dashboard/index.html, keep the data
        return write_html()
    tb = Toolbox()
    rng = viz.time_range(tb)
    t0, t1 = pd.Timestamp(rng["start"]), pd.Timestamp(rng["end"]).ceil("h")
    grid = pd.date_range(t0 - pd.Timedelta(hours=HIST_H), t1, freq=f"{STEP_H}h")
    first = int(np.searchsorted(grid, t0))                 # first grid index at which the demo can "stand"
    cols = tb.risk_meta["features"]
    thr_r, thr_a = tb.risk_meta["threshold"], tb.detector.meta["threshold"]
    (OUT / "data").mkdir(parents=True, exist_ok=True)

    machines, fleet_p = [], []
    codes = sorted(ERROR_CODES)
    for n, (m, info) in enumerate(tb.fleet.machines.iterrows(), 1):
        z = tb._z(m).reindex(grid)
        raw = tb.fleet.tel(m).reindex(grid)
        w, ok = windows_ending_at(tb._z(m), grid)
        anomaly = np.full(len(grid), np.nan)
        if ok.any():
            anomaly[ok] = tb.detector.score_windows(w[ok])["score"]

        feats = viz._features(tb, m).reindex(grid)
        risk = np.full(len(grid), np.nan)
        has = feats[cols].notna().any(axis=1).to_numpy()
        factors = [[] for _ in grid]
        if has.any():
            dm = xgb.DMatrix(feats.loc[has, cols])
            risk[has] = tb.risk_model.predict(dm)
            contrib = tb.risk_model.predict(dm, pred_contribs=True)[:, :-1]
            for gi, c in zip(np.flatnonzero(has), contrib):
                top = np.argsort(-np.abs(c))[:TOP_FACTORS]
                factors[gi] = [[int(i), int(round(c[i] * 100))] for i in top]

        lg = tb.fleet.machine_logs(m).loc[grid[0]: t1]
        lg = lg[lg.severity != "info"]
        logs = [[str(ts)[:16], codes.index(r.code), 1 if r.severity == "error" else 0] for ts, r in lg.iterrows()]
        fails = [str(pd.Timestamp(x))[:16] for x in failure_times(tb.fleet, m) if grid[0] <= pd.Timestamp(x) <= t1]
        doc = {"id": m, "model": info["model"], "t0": str(grid[0])[:16], "step_h": STEP_H,
               "z": {s: _ints(z[s], 10) for s in SENSORS}, "raw": {s: _ints(raw[s], 10) for s in SENSORS},
               "anomaly": _ints(anomaly, 100), "risk": _ints(risk, 1000), "factors": factors[first:], "factors_from": first,
               "logs": logs, "failures": fails}
        (OUT / "data" / f"m_{m}.json").write_text(json.dumps(doc, separators=(",", ":")))
        machines.append({"id": m, "model": info["model"]})
        fleet_p.append(_ints(risk[first:], 1000))
        print(f"  [{n:3d}/{len(tb.fleet.machines)}] {m}", end="\r", flush=True)

    # investigations: the offline `rules` agent on every test incident, exactly as the live API would answer
    from src.agent import RCAAgent

    agent = RCAAgent("rules", tb)
    inv = {}
    for i in viz.incident_list(tb):
        q = f"Machine {i['machine_id']} tripped offline at {i['time'][:16]}. What is the root cause?"
        r = agent.run(q)
        inv[i["id"]] = {"question": q, "root_cause": r["root_cause"], "answer": r["answer"], "tools_called": r["tools_called"],
                        "retrieved_chunks": r["retrieved_chunks"], "evidence": r["evidence_text"][:2500],
                        "latency_s": r["latency_s"]}
    (OUT / "data" / "investigations.json").write_text(json.dumps(inv, separators=(",", ":")))

    ops = json.loads((config.RESULTS_DIR / "lead_time.json").read_text())
    mm = json.loads((config.RESULTS_DIR / "model_metrics.json").read_text())
    ag = json.loads((config.RESULTS_DIR / "agent_eval_summary.json").read_text())["variants"]
    meta = {"t0": str(grid[first])[:16], "step_h": STEP_H, "n": len(grid) - first, "hist_points": HIST_H // STEP_H,
            "risk_threshold": round(thr_r, 3), "anomaly_threshold": round(thr_a, 3), "machines": machines,
            "incidents": viz.incident_list(tb), "features": cols, "codes": codes,
            "ops": ops, "risk_metrics": mm["risk_xgb"], "rules_agent": ag.get("rules") or next(iter(ag.values()))}
    (OUT / "data" / "meta.json").write_text(json.dumps(meta, separators=(",", ":")))
    (OUT / "data" / "fleet.json").write_text(json.dumps({"p": fleet_p}, separators=(",", ":")))

    write_html()
    mb = sum(f.stat().st_size for f in (OUT / "data").glob("*.json")) / 1e6
    print(f"\nwrote docs/index.html and {len(list((OUT / 'data').glob('*.json')))} data files ({mb:.1f} MB)")


if __name__ == "__main__":
    main()
