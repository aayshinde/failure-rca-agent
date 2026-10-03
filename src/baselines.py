"""Is XGBoost earning its keep, and which data source matters? Same split and features as train_risk.py.

  * logistic regression on all features   -> does non-linearity help?
  * XGBoost / logistic on feature subsets  -> telemetry-only, logs-only, maintenance-only ablations
  * "age only" (hours since service)       -> the naive scheduled-maintenance heuristic

    python -m src.baselines
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score
from sklearn.preprocessing import StandardScaler

from src import config
from src.features import FleetData, build_feature_table, load_norm

GROUPS = {
    "telemetry": lambda c: c.split("_")[0] in {"volt", "rotate", "pressure", "vibration", "temperature", "current"},
    "logs": lambda c: c.startswith("E") and c[1:4].isdigit() or c == "warnings_6h",
    "maintenance": lambda c: c.startswith("hrs_since"),
}


def best_threshold(y, p) -> float:
    grid = np.linspace(0.05, 0.95, 91)
    return float(grid[int(np.argmax([f1_score(y, p >= g) for g in grid]))])


def main() -> dict:
    fleet = FleetData()
    table = build_feature_table(fleet, load_norm())
    allc = [c for c in table.columns if c not in ("datetime", "machine_id", "label")]
    tr = table[table.datetime < fleet.val_start]
    va = table[(table.datetime >= fleet.val_start) & (table.datetime < fleet.split_time)]
    te = table[table.datetime >= fleet.split_time]
    spw = float((tr.label == 0).sum() / max((tr.label == 1).sum(), 1))
    meta = json.loads((config.MODEL_DIR / "risk_meta.json").read_text())

    def run(name: str, cols: list[str], kind: str) -> dict:
        if kind == "xgb":
            m = xgb.XGBClassifier(n_estimators=400, max_depth=5, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8,
                                  scale_pos_weight=spw, eval_metric="aucpr", random_state=config.SEED, n_jobs=1)
            m.fit(tr[cols], tr.label, verbose=False)
            pv, pt = m.predict_proba(va[cols])[:, 1], m.predict_proba(te[cols])[:, 1]
        else:
            med = tr[cols].median()                       # XGBoost handles NaN natively; logistic regression needs imputing
            prep = lambda d: sc.transform(d[cols].fillna(med))
            sc = StandardScaler().fit(tr[cols].fillna(med))
            m = LogisticRegression(max_iter=3000, class_weight="balanced").fit(prep(tr), tr.label)
            pv, pt = m.predict_proba(prep(va))[:, 1], m.predict_proba(prep(te))[:, 1]
        thr = best_threshold(va.label, pv)
        return {"model": name, "features": len(cols), "test_auroc": round(roc_auc_score(te.label, pt), 3),
                "test_auprc": round(average_precision_score(te.label, pt), 3), "test_f1": round(f1_score(te.label, pt >= thr), 3)}

    rows = [run("XGBoost, all features", allc, "xgb"), run("Logistic regression, all features", allc, "lr")]
    for g, fn in GROUPS.items():
        cols = [c for c in allc if fn(c)]
        rows.append(run(f"XGBoost, {g} only", cols, "xgb"))
    rows.append(run("XGBoost, telemetry + logs (no maintenance)", [c for c in allc if GROUPS["telemetry"](c) or GROUPS["logs"](c)], "xgb"))
    age = te[[c for c in allc if c.startswith("hrs_since")]].min(axis=1).fillna(0)
    rows.append({"model": "Heuristic: hours since last service", "features": 1,
                 "test_auroc": round(roc_auc_score(te.label, -age), 3), "test_auprc": round(average_precision_score(te.label, -age), 3),
                 "test_f1": None})
    res = {"split": "time-based, same as train_risk.py", "positive_rate_test": round(float(te.label.mean()), 3), "rows": rows}
    (config.RESULTS_DIR / "baselines.json").write_text(json.dumps(res, indent=2))
    print(pd.DataFrame(rows).to_string(index=False))
    return res


if __name__ == "__main__":
    main()
