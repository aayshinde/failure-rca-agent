"""Train the XGBoost failure-risk model: P(failure within the next 24h | machine-hour features).

Time-based split: train < val_start <= val < split_time <= test. The decision threshold is
chosen on validation to maximise F1, then reported on test. Everything is logged to MLflow.

Usage:  python -m src.train_risk
"""
from __future__ import annotations

import json
import time

import mlflow
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import average_precision_score, f1_score, precision_score, recall_score, roc_auc_score

from src import config
from src.features import FleetData, build_feature_table, fit_norm

PARAMS = dict(n_estimators=600, max_depth=6, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8,
              min_child_weight=5, tree_method="hist", eval_metric="aucpr", early_stopping_rounds=50, n_jobs=-1)


def event_recall(df: pd.DataFrame, pred: np.ndarray, fleet: FleetData) -> float:
    """Share of test failures that got at least one alert in the 24h before they happened."""
    df = df.assign(pred=pred)
    inc = fleet.incidents[fleet.incidents.split == "test"]
    hit = 0
    for _, r in inc.iterrows():
        w = df[(df.machine_id == r.machine_id) & (df.datetime < r.failure_time)
               & (df.datetime >= r.failure_time - pd.Timedelta(hours=config.RISK_HORIZON_H))]
        hit += int(w.pred.any())
    return hit / max(len(inc), 1)


def main() -> None:
    t0 = time.time()
    fleet = FleetData()
    norm = fit_norm(fleet)
    table = build_feature_table(fleet, norm)
    feat_cols = [c for c in table.columns if c not in ("datetime", "machine_id", "label")]
    print(f"Feature table: {table.shape}, positives {table.label.mean():.2%}  ({time.time() - t0:.0f}s)")

    tr = table[table.datetime < fleet.val_start]
    va = table[(table.datetime >= fleet.val_start) & (table.datetime < fleet.split_time)]
    te = table[table.datetime >= fleet.split_time]

    spw = float((tr.label == 0).sum() / max((tr.label == 1).sum(), 1))
    model = xgb.XGBClassifier(**PARAMS, scale_pos_weight=spw, random_state=config.SEED)
    model.fit(tr[feat_cols], tr.label, eval_set=[(va[feat_cols], va.label)], verbose=False)

    p_va = model.predict_proba(va[feat_cols])[:, 1]
    grid = np.linspace(0.05, 0.95, 91)
    f1s = [f1_score(va.label, p_va >= g) for g in grid]
    thr = float(grid[int(np.argmax(f1s))])

    p_te = model.predict_proba(te[feat_cols])[:, 1]
    yhat = p_te >= thr
    metrics = {
        "test_auroc": roc_auc_score(te.label, p_te),
        "test_auprc": average_precision_score(te.label, p_te),
        "test_f1": f1_score(te.label, yhat),
        "test_precision": precision_score(te.label, yhat),
        "test_recall": recall_score(te.label, yhat),
        "test_event_recall": event_recall(te[["machine_id", "datetime"]], yhat, fleet),
        "val_f1": float(max(f1s)),
        "threshold": thr,
        "best_iteration": int(model.best_iteration),
    }
    metrics = {k: round(float(v), 4) for k, v in metrics.items()}

    model.save_model(config.MODEL_DIR / "risk_xgb.json")
    (config.MODEL_DIR / "risk_meta.json").write_text(json.dumps({"features": feat_cols, "threshold": thr}, indent=2))
    imp = pd.Series(model.feature_importances_, index=feat_cols).sort_values(ascending=False)

    mlflow.set_tracking_uri(config.MLFLOW_TRACKING_URI)
    mlflow.set_experiment("failure-risk")
    with mlflow.start_run(run_name="xgb_risk_24h"):
        mlflow.log_params({k: v for k, v in PARAMS.items() if k != "n_jobs"} | {"scale_pos_weight": round(spw, 2),
                          "horizon_h": config.RISK_HORIZON_H, "n_features": len(feat_cols), "n_train": len(tr)})
        mlflow.log_metrics(metrics)
        mlflow.log_artifact(str(config.MODEL_DIR / "risk_xgb.json"))
        mlflow.log_text(imp.head(30).to_string(), "top_features.txt")

    _merge_model_metrics({"risk_xgb": metrics})
    print(json.dumps(metrics, indent=2))
    print("Top features:\n" + imp.head(10).to_string())
    print(f"done in {time.time() - t0:.0f}s")


def _merge_model_metrics(new: dict) -> None:
    p = config.RESULTS_DIR / "model_metrics.json"
    cur = json.loads(p.read_text()) if p.exists() else {}
    cur.update(new)
    p.write_text(json.dumps(cur, indent=2))


if __name__ == "__main__":
    main()
