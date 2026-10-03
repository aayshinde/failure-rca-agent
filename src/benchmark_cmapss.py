"""Sanity-check the modelling approach on a public benchmark: NASA C-MAPSS turbofan degradation (FD001).

The fleet data in this repo is simulated, so this answers "does the same recipe work on data I did not write?"
Task: P(engine fails within the next H cycles) from rolling sensor features, same as the fleet risk model.
Evaluation is by *engine* (held-out engines never appear in training) and on NASA's official test set.

    python -m src.benchmark_cmapss            # downloads ~12 MB once into data/external/
"""
from __future__ import annotations

import io
import json
import subprocess
import urllib.request
import zipfile

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score
from sklearn.preprocessing import StandardScaler

from src import config

URL = "https://data.nasa.gov/docs/legacy/CMAPSSData.zip"
HORIZON = 30                                                   # "fails within 30 cycles"
COLS = ["unit", "cycle", "op1", "op2", "op3"] + [f"s{i}" for i in range(1, 22)]
# Sensors with (near-)zero variance in FD001 carry no signal.
SENSORS = [f"s{i}" for i in (2, 3, 4, 7, 8, 9, 11, 12, 13, 14, 15, 17, 20, 21)]
EXT = config.DATA_DIR / "external" / "cmapss"


def download() -> None:
    if (EXT / "train_FD001.txt").exists():
        return
    EXT.mkdir(parents=True, exist_ok=True)
    print(f"downloading {URL} ...")
    try:
        with urllib.request.urlopen(URL, timeout=60) as r:
            blob = r.read()
    except OSError:                       # some networks block Python's resolver but not curl
        blob = subprocess.run(["curl", "-fsSL", "-m", "180", URL], capture_output=True, check=True).stdout
    zipfile.ZipFile(io.BytesIO(blob)).extractall(EXT)


def read(name: str) -> pd.DataFrame:
    return pd.read_csv(EXT / name, sep=r"\s+", header=None, names=COLS)


def features(df: pd.DataFrame) -> pd.DataFrame:
    """Per-engine rolling level, volatility and slope: what a maintenance engineer would eyeball."""
    out = []
    for _, g in df.groupby("unit"):
        g = g.sort_values("cycle")
        f = pd.DataFrame({"unit": g.unit.values, "cycle": g.cycle.values})
        for s in SENSORS:
            x = g[s]
            f[f"{s}_last"] = x.values
            f[f"{s}_mean10"] = x.rolling(10, min_periods=1).mean().values
            f[f"{s}_std10"] = x.rolling(10, min_periods=2).std().fillna(0).values
            f[f"{s}_slope10"] = (x - x.shift(10)).fillna(0).values
            f[f"{s}_dev"] = (x - x.iloc[:10].mean()).values       # drift from the engine's own healthy start
        out.append(f)
    return pd.concat(out, ignore_index=True)


def label(df: pd.DataFrame) -> np.ndarray:
    rul = df.groupby("unit").cycle.transform("max") - df.cycle
    return (rul <= HORIZON).astype(int).values


def best_f1(y, p) -> tuple[float, float]:
    ts = np.linspace(0.05, 0.95, 91)
    f = [f1_score(y, p >= t) for t in ts]
    return float(max(f)), float(ts[int(np.argmax(f))])


def main() -> dict:
    download()
    train, test, rul = read("train_FD001.txt"), read("test_FD001.txt"), pd.read_csv(EXT / "RUL_FD001.txt", header=None)[0].values
    units = train.unit.unique()
    rng = np.random.default_rng(config.SEED)
    val_units = set(rng.choice(units, size=20, replace=False))     # 80 engines to train, 20 held-out engines
    tr, va = train[~train.unit.isin(val_units)], train[train.unit.isin(val_units)]

    Xtr, Xva, Xte = features(tr), features(va), features(test)
    ytr, yva = label(tr), label(va)
    drop = ["unit", "cycle"]
    cols = [c for c in Xtr.columns if c not in drop]

    # official test set: only the last observed cycle of each engine has a known remaining useful life
    last = Xte.groupby("unit").tail(1)
    yte = (rul <= HORIZON).astype(int)

    scaler = StandardScaler().fit(Xtr[cols])
    models = {}
    pos = ytr.mean()
    booster = xgb.XGBClassifier(n_estimators=300, max_depth=4, learning_rate=0.05, subsample=0.8, colsample_bytree=0.7,
                                scale_pos_weight=(1 - pos) / pos, eval_metric="aucpr", random_state=config.SEED, n_jobs=1)
    booster.fit(Xtr[cols], ytr)
    models["xgboost (rolling features)"] = (lambda X: booster.predict_proba(X[cols])[:, 1])
    lr = LogisticRegression(max_iter=2000, class_weight="balanced").fit(scaler.transform(Xtr[cols]), ytr)
    models["logistic regression (same features)"] = (lambda X: lr.predict_proba(scaler.transform(X[cols]))[:, 1])
    lr_raw_cols = [f"{s}_last" for s in SENSORS]
    sc2 = StandardScaler().fit(Xtr[lr_raw_cols])
    lr2 = LogisticRegression(max_iter=2000, class_weight="balanced").fit(sc2.transform(Xtr[lr_raw_cols]), ytr)
    models["logistic regression (raw sensors only)"] = (lambda X: lr2.predict_proba(sc2.transform(X[lr_raw_cols]))[:, 1])
    age_max = Xtr.groupby("unit").cycle.max().median()
    models["age only (cycle count)"] = (lambda X: np.clip(X["cycle"].values / age_max, 0, 1))

    rows = []
    for name, fn in models.items():
        pv, pt = fn(Xva), fn(last)
        f1v, thr = best_f1(yva, pv)
        rows.append({"model": name, "heldout_engines_auroc": round(roc_auc_score(yva, pv), 4),
                     "heldout_engines_auprc": round(average_precision_score(yva, pv), 4), "heldout_engines_best_f1": round(f1v, 3),
                     "official_test_auroc": round(roc_auc_score(yte, pt), 4),
                     "official_test_f1_at_val_thr": round(float(f1_score(yte, pt >= thr)), 3)})
    res = {"dataset": "NASA C-MAPSS FD001", "task": f"fails within {HORIZON} cycles", "train_engines": int(len(units) - 20),
           "heldout_engines": 20, "official_test_engines": int(len(rul)), "positive_rate_heldout": round(float(yva.mean()), 3),
           "results": rows}
    (config.RESULTS_DIR / "cmapss_benchmark.json").write_text(json.dumps(res, indent=2))
    print(pd.DataFrame(rows).to_string(index=False))
    return res


if __name__ == "__main__":
    main()
