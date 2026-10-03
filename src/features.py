"""Shared data access + feature engineering (used by training AND by the agent's tools,
so the model sees identical features offline and online)."""
from __future__ import annotations

import json
from functools import cached_property

import numpy as np
import pandas as pd

from src import config
from src.knowledge import COMPONENTS, ERROR_CODES, MODEL_BASELINES, SENSORS

FAULT_CODES = [c for c, (_, sev) in ERROR_CODES.items() if sev != "info" and c != "E000"]
MODELS = list(MODEL_BASELINES)


class FleetData:
    """Lazy loader for the simulated (or your own) fleet tables."""

    def __init__(self, data_dir=config.DATA_DIR):
        self.dir = data_dir

    @cached_property
    def machines(self) -> pd.DataFrame:
        return pd.read_csv(self.dir / "machines.csv").set_index("machine_id")

    @cached_property
    def telemetry(self) -> pd.DataFrame:
        return pd.read_parquet(self.dir / "telemetry.parquet")

    @cached_property
    def logs(self) -> pd.DataFrame:
        return pd.read_csv(self.dir / "logs.csv", parse_dates=["datetime"])

    @cached_property
    def maintenance(self) -> pd.DataFrame:
        return pd.read_csv(self.dir / "maintenance.csv", parse_dates=["datetime"])

    @cached_property
    def incidents(self) -> pd.DataFrame:
        return pd.read_csv(self.dir / "incidents.csv", parse_dates=["failure_time"])

    @cached_property
    def split_time(self) -> pd.Timestamp:
        t0 = self.telemetry.datetime.min().normalize()
        days = (self.telemetry.datetime.max() - t0).days + 1
        return t0 + pd.Timedelta(days=int(days * config.TRAIN_FRAC))

    @cached_property
    def val_start(self) -> pd.Timestamp:
        t0 = self.telemetry.datetime.min().normalize()
        return t0 + (self.split_time - t0) * (1 - config.VAL_FRAC)

    @cached_property
    def _tel_by_machine(self) -> dict[str, pd.DataFrame]:
        return {m: g.set_index("datetime")[SENSORS] for m, g in self.telemetry.groupby("machine_id")}

    def tel(self, machine_id: str) -> pd.DataFrame:
        return self._tel_by_machine[machine_id]

    @cached_property
    def _logs_by_machine(self):
        return {m: g.set_index("datetime").sort_index() for m, g in self.logs.groupby("machine_id")}

    def machine_logs(self, machine_id: str) -> pd.DataFrame:
        return self._logs_by_machine.get(machine_id, self.logs.iloc[:0].set_index("datetime"))

    def machine_maint(self, machine_id: str) -> pd.DataFrame:
        m = self.maintenance
        return m[m.machine_id == machine_id].sort_values("datetime")


# ------------------------------------------------------------------ normalisation


def fit_norm(fleet: FleetData) -> dict:
    """Per-machine sensor mean/std from the TRAIN period only (no test leakage)."""
    tr = fleet.telemetry[fleet.telemetry.datetime < fleet.split_time]
    g = tr.groupby("machine_id")[SENSORS]
    mu, sd = g.median(), g.std()
    norm = {m: {"mu": mu.loc[m].tolist(), "sd": sd.loc[m].tolist()} for m in mu.index}
    (config.MODEL_DIR / "norm.json").write_text(json.dumps(norm))
    return norm


def load_norm() -> dict:
    return json.loads((config.MODEL_DIR / "norm.json").read_text())


def zscores(tel_m: pd.DataFrame, norm_m: dict) -> pd.DataFrame:
    """Reindex to a full hourly grid (downtime -> NaN) and z-score each sensor."""
    full = tel_m.reindex(pd.date_range(tel_m.index.min(), tel_m.index.max(), freq="h"))
    return (full - np.array(norm_m["mu"])) / np.array(norm_m["sd"])


# ------------------------------------------------------------------ features


def machine_features(fleet: FleetData, machine_id: str, norm: dict) -> pd.DataFrame:
    """One row per machine-hour with telemetry, log and maintenance features."""
    z = zscores(fleet.tel(machine_id), norm[machine_id])
    feats = {}
    for s in SENSORS:
        col = z[s]
        m6, m24, m72 = (col.rolling(w, min_periods=max(w // 2, 2)).mean() for w in (6, 24, 72))
        sd6, sd72 = (col.rolling(w, min_periods=max(w // 2, 2)).std() for w in (6, 72))
        feats[f"{s}_mean6"], feats[f"{s}_mean24"] = m6, m24
        feats[f"{s}_delta"] = m6 - m72
        feats[f"{s}_std6"], feats[f"{s}_std72"] = sd6, sd72
        feats[f"{s}_volratio"] = sd6 / (sd72 + 1e-3)
    f = pd.DataFrame(feats, index=z.index)

    # error-code counts in the last 24h / 6h
    lg = fleet.machine_logs(machine_id)
    lg = lg[lg.code.isin(FAULT_CODES)]
    counts = pd.crosstab(lg.index.floor("h"), lg.code).reindex(columns=FAULT_CODES, fill_value=0)
    counts = counts.reindex(f.index, fill_value=0)
    for c in FAULT_CODES:
        f[f"{c}_24h"] = counts[c].rolling(24, min_periods=1).sum()
    f["warnings_6h"] = counts.sum(axis=1).rolling(6, min_periods=1).sum()

    # hours since last service per component
    mt = fleet.machine_maint(machine_id)
    base = pd.DataFrame({"datetime": f.index})
    for c in COMPONENTS:
        mc = mt[mt.component == c][["datetime"]].assign(last=lambda d: d.datetime)
        if mc.empty:
            f[f"hrs_since_{c}"] = np.nan
            continue
        merged = pd.merge_asof(base, mc, on="datetime", direction="backward")
        f[f"hrs_since_{c}"] = ((merged["datetime"] - merged["last"]).dt.total_seconds() / 3600).values

    model = fleet.machines.loc[machine_id, "model"]
    for mm in MODELS:
        f[f"is_{mm}"] = float(model == mm)

    f = f[z.notna().all(axis=1)]  # drop downtime hours
    f.index.name = "datetime"
    return f.astype(np.float32)


def hours_to_next(ft: np.ndarray, t) -> np.ndarray:
    """Hours from each time in t to the next failure strictly after it (NaN if none)."""
    tv = pd.DatetimeIndex(t).values
    if len(ft) == 0:
        return np.full(len(tv), np.nan)
    pos = np.searchsorted(ft, tv, side="right")
    hrs = (ft[np.minimum(pos, len(ft) - 1)] - tv) / np.timedelta64(1, "h")
    return np.where(pos < len(ft), hrs, np.nan)


def hours_since_prev(ft: np.ndarray, t) -> np.ndarray:
    """Hours since the most recent failure at or before each time in t (NaN if none)."""
    tv = pd.DatetimeIndex(t).values
    if len(ft) == 0:
        return np.full(len(tv), np.nan)
    pos = np.searchsorted(ft, tv, side="right") - 1
    hrs = (tv - ft[np.maximum(pos, 0)]) / np.timedelta64(1, "h")
    return np.where(pos >= 0, hrs, np.nan)


def failure_times(fleet: FleetData, machine_id: str) -> np.ndarray:
    return np.sort(fleet.incidents.loc[fleet.incidents.machine_id == machine_id, "failure_time"].values)


def add_labels(f: pd.DataFrame, fleet: FleetData, machine_id: str, horizon_h: int = config.RISK_HORIZON_H) -> pd.Series:
    hrs = hours_to_next(failure_times(fleet, machine_id), f.index)
    return pd.Series((hrs > 0) & (hrs <= horizon_h), index=f.index, name="label").astype(np.int8)


def build_feature_table(fleet: FleetData, norm: dict) -> pd.DataFrame:
    parts = []
    for m in fleet.machines.index:
        f = machine_features(fleet, m, norm)
        f["label"] = add_labels(f, fleet, m)
        f["machine_id"] = m
        parts.append(f.reset_index())
    return pd.concat(parts, ignore_index=True)
