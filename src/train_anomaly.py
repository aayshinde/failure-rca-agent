"""Train the PyTorch LSTM autoencoder on NORMAL 24h telemetry windows and calibrate the
combined anomaly detector (see src/anomaly.py).

Evaluated on the test period: AUROC of the window score for "a failure happens within the
next 24h" vs normal operation, overall and per failure mode, plus an ablation showing what
the autoencoder alone and the level score alone achieve.

Usage:  python -m src.train_anomaly [--epochs 12]
"""
from __future__ import annotations

import argparse
import json
import time

import mlflow
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score
from torch import nn

from src import config
from src.anomaly import AnomalyDetector, LSTMAutoencoder, raw_scores, windows_ending_at
from src.features import FleetData, failure_times, hours_since_prev, hours_to_next, load_norm, zscores
from src.knowledge import SENSORS

HORIZON = 24


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--n-train", type=int, default=40000)
    args = ap.parse_args()
    torch.manual_seed(config.SEED)
    rng = np.random.default_rng(config.SEED)
    t0 = time.time()

    fleet, norm = FleetData(), load_norm()
    inc = fleet.incidents
    train_w, val_w, test = [], [], []
    for m in fleet.machines.index:
        z = zscores(fleet.tel(m), norm[m])
        idx = z.index
        ft = failure_times(fleet, m)
        to_fail, since = hours_to_next(ft, idx), hours_since_prev(ft, idx)
        normal = (np.isnan(to_fail) | (to_fail > 168)) & (np.isnan(since) | (since > 24))
        w, ok = windows_ending_at(z, idx[(idx < fleet.val_start) & normal][::3])
        train_w.append(w[ok])
        w, ok = windows_ending_at(z, idx[(idx >= fleet.val_start) & (idx < fleet.split_time) & normal][::6])
        val_w.append(w[ok])
        ends = idx[idx >= fleet.split_time][::3]
        w, ok = windows_ending_at(z, ends)
        tf = hours_to_next(ft, ends)[ok]
        nxt = (ends[ok] + pd.to_timedelta(np.nan_to_num(tf), unit="h")).round("h")
        test.append((m, w[ok], tf, nxt))

    Xtr = np.concatenate(train_w)
    Xtr = Xtr[rng.permutation(len(Xtr))[: args.n_train]]
    Xva = np.concatenate(val_w)
    print(f"train windows {len(Xtr):,}  val windows {len(Xva):,}  ({time.time() - t0:.0f}s)")

    model = LSTMAutoencoder()
    opt = torch.optim.Adam(model.parameters(), lr=2e-3)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    loss_fn, Xt = nn.MSELoss(), torch.from_numpy(Xtr)
    for ep in range(args.epochs):
        model.train()
        perm, tot = torch.randperm(len(Xt)), 0.0
        for i in range(0, len(Xt), 256):
            xb = Xt[perm[i: i + 256]]
            opt.zero_grad()
            loss = loss_fn(model(xb), xb)
            loss.backward()
            opt.step()
            tot += loss.item() * len(xb)
        sched.step()
        rec_va, _ = raw_scores(model, Xva)
        print(f"epoch {ep + 1:2d}: train {tot / len(Xt):.4f}  val(recent) {rec_va.mean():.4f}")

    # ---- calibrate on normal validation windows
    rec_va, lvl_va = raw_scores(model, Xva)
    meta = {"window": 24, "recent": 6, "sensors": SENSORS,
            "recon_q99": np.quantile(rec_va, 0.99, axis=0).tolist(),
            "level_q99": np.quantile(lvl_va, 0.99, axis=0).tolist()}
    det = AnomalyDetector(model, meta)
    meta["threshold"] = float(np.quantile(det.score_windows(Xva)["score"], 0.99))

    # ---- test evaluation
    lookup = inc.set_index(["machine_id", "failure_time"]).root_cause
    y, comb, rec_only, lvl_only, modes = [], [], [], [], []
    for m, w, tf, nxt in test:
        pre = (tf > 0) & (tf <= HORIZON)
        sel = pre | np.isnan(tf) | (tf > 168)
        s = det.score_windows(w[sel])
        y.append(pre[sel]); comb.append(s["score"])
        rec_only.append(s["recon"].max(1)); lvl_only.append(s["level"].max(1))
        modes += [lookup.get((m, t), "none") if p else "none" for t, p in zip(nxt[sel], pre[sel])]
    y, comb, rec_only, lvl_only, modes = map(np.concatenate, (y, comb, rec_only, lvl_only, [modes]))
    thr = meta["threshold"]
    metrics = {
        f"test_auroc_{HORIZON}h": roc_auc_score(y, comb),
        "test_auroc_autoencoder_only": roc_auc_score(y, rec_only),
        "test_auroc_level_only": roc_auc_score(y, lvl_only),
        "test_false_alarm_rate": float((comb[~y] > thr).mean()),
        "test_detection_rate": float((comb[y] > thr).mean()),
        "threshold": thr,
    }
    metrics = {k: round(float(v), 4) for k, v in metrics.items()}
    per_mode = {mo: round(float(roc_auc_score(y[(modes == mo) | ~y], comb[(modes == mo) | ~y])), 4)
                for mo in sorted(set(modes) - {"none"})}

    torch.save(model.state_dict(), config.MODEL_DIR / "anomaly_lstm_ae.pt")
    (config.MODEL_DIR / "anomaly_meta.json").write_text(json.dumps(meta, indent=2))

    mlflow.set_tracking_uri(config.MLFLOW_TRACKING_URI)
    mlflow.set_experiment("anomaly-detection")
    with mlflow.start_run(run_name="lstm_autoencoder"):
        mlflow.log_params({"window": 24, "epochs": args.epochs, "n_train": len(Xtr), "hidden": 48, "latent": 16})
        mlflow.log_metrics(metrics | {f"auroc_{k}": v for k, v in per_mode.items()})
        mlflow.log_artifact(str(config.MODEL_DIR / "anomaly_lstm_ae.pt"))

    from src.train_risk import _merge_model_metrics
    _merge_model_metrics({"anomaly_detector": metrics | {"auroc_by_mode": per_mode}})
    print(json.dumps(metrics | {"auroc_by_mode": per_mode}, indent=2))
    print(f"done in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
