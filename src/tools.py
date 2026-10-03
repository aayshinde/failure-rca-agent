"""The agent's tools. Each returns a compact JSON-serialisable dict.

query_telemetry          sensor levels/trends vs the machine's baseline
detect_anomalies         LSTM-AE + level detector over a recent window
predict_failure_risk     XGBoost P(failure in 24h) + top SHAP contributions
search_logs              controller error codes in a window
get_maintenance_history  last service per component, recent repairs
search_docs              vector search over runbooks / manuals
"""
from __future__ import annotations

import json
import re
from functools import cached_property

import numpy as np
import pandas as pd
import xgboost as xgb

from src import config
from src.anomaly import AnomalyDetector, windows_ending_at
from src.features import FleetData, load_norm, machine_features, zscores
from src.knowledge import COMPONENTS, ERROR_CODES, SENSORS, UNITS
from src.retrieval import DocIndex

TOOL_DESCRIPTIONS = {
    "query_telemetry": "Summarise sensor readings (volt, rotate, pressure, vibration, temperature, current) for a machine "
                       "over the hours before a given time: recent level vs baseline, trend and volatility.",
    "detect_anomalies": "Run the anomaly-detection model on a machine's recent telemetry; returns anomaly score vs threshold, "
                        "when anomalies started and which sensors are abnormal (pattern vs level, direction).",
    "predict_failure_risk": "Forward-looking: predicted probability that the machine fails within the next 24 hours "
                            "at a given time, with the top contributing features.",
    "search_logs": "Controller error/event log for a machine in a time window: error codes, counts, first/last seen.",
    "get_maintenance_history": "Maintenance records for a machine before a given time: days since each component was "
                               "serviced and recent corrective repairs.",
    "search_docs": "Search maintenance documentation (runbooks, error-code reference, service manual) by text query.",
}


def parse_time(t) -> pd.Timestamp:
    return pd.Timestamp(t).floor("h")


class Toolbox:
    def __init__(self, fleet: FleetData | None = None, docs: DocIndex | None = None):
        self.fleet = fleet or FleetData()
        self._docs = docs
        self._feat_cache: dict[str, pd.DataFrame] = {}
        self._z_cache: dict[str, pd.DataFrame] = {}

    # ------------------------------------------------------------- lazy resources
    @cached_property
    def norm(self):
        return load_norm()

    @cached_property
    def detector(self):
        return AnomalyDetector.load()

    @cached_property
    def risk_model(self):
        b = xgb.Booster()
        b.load_model(config.MODEL_DIR / "risk_xgb.json")
        return b

    @cached_property
    def risk_meta(self):
        return json.loads((config.MODEL_DIR / "risk_meta.json").read_text())

    @property
    def docs(self) -> DocIndex:
        if self._docs is None:
            self._docs = DocIndex.load()
        return self._docs

    def _z(self, m: str) -> pd.DataFrame:
        if m not in self._z_cache:
            self._z_cache[m] = zscores(self.fleet.tel(m), self.norm[m])
        return self._z_cache[m]

    def _check_machine(self, m: str) -> str:
        m = str(m).strip().upper()
        if m not in self.fleet.machines.index:
            raise ValueError(f"Unknown machine_id {m!r}")
        return m

    # ------------------------------------------------------------- tools
    def query_telemetry(self, machine_id: str, end_time: str, hours: int = 72) -> dict:
        m, t = self._check_machine(machine_id), parse_time(end_time)
        z = self._z(m).loc[t - pd.Timedelta(hours=hours - 1): t]
        raw = self.fleet.tel(m).loc[t - pd.Timedelta(hours=hours - 1): t]
        mu = dict(zip(SENSORS, self.norm[m]["mu"]))
        out, notable = {}, []
        for s in SENSORS:
            col = z[s].dropna()
            if len(col) < 12:
                continue
            recent, early = col.iloc[-6:].mean(), col.iloc[:12].mean()
            vol = col.iloc[-12:].std()
            out[s] = {"baseline": round(mu[s], 1), "recent_6h_mean": round(float(raw[s].iloc[-6:].mean()), 1), "unit": UNITS[s],
                      "recent_z": round(float(recent), 2), "trend_z_over_window": round(float(recent - early), 2),
                      "volatility_x_normal": round(float(vol), 2)}
            if abs(recent) > 2:
                notable.append(f"{s} {'above' if recent > 0 else 'below'} baseline by {abs(recent):.1f} std")
            if vol > 2:
                notable.append(f"{s} unstable ({vol:.1f}x normal variability)")
        return {"machine_id": m, "window": f"{t - pd.Timedelta(hours=hours - 1)} to {t}", "sensors": out,
                "notable": notable or ["all sensors within normal range"]}

    def detect_anomalies(self, machine_id: str, end_time: str, hours: int = 72) -> dict:
        m, t = self._check_machine(machine_id), parse_time(end_time)
        z = self._z(m)
        ends = z.index[(z.index > t - pd.Timedelta(hours=hours)) & (z.index <= t)]
        w, ok = windows_ending_at(z, ends)
        if not ok.any():
            return {"machine_id": m, "error": "not enough telemetry in window"}
        ends, s = ends[ok], self.detector.score_windows(w[ok])
        thr = self.detector.meta["threshold"]
        above = s["score"] > thr
        recent = slice(-6, None)
        per = s["per_sensor"][recent].max(axis=0)
        rec, lvl = s["recon"][recent].max(axis=0), s["level"][recent].max(axis=0)
        direction = np.sign(w[ok][-1, -6:, :].mean(axis=0))
        top = []
        for k in np.argsort(-per):
            if per[k] <= thr and top:
                break
            top.append({"sensor": SENSORS[k], "score": round(float(per[k]), 2),
                        "kind": "pattern/volatility" if rec[k] >= lvl[k] else ("level shift up" if direction[k] > 0 else "level shift down")})
            if len(top) == 3:
                break
        return {
            "machine_id": m, "window_hours": hours, "threshold": round(thr, 2),
            "max_score": round(float(s["score"].max()), 2), "latest_score": round(float(s["score"][-1]), 2),
            "anomalous": bool(above[-6:].any()), "hours_above_threshold": int(above.sum()),
            "first_anomalous_at": str(ends[above][0]) if above.any() else None,
            "top_sensors": top if above[-6:].any() else [],
        }

    def predict_failure_risk(self, machine_id: str, at_time: str) -> dict:
        m, t = self._check_machine(machine_id), parse_time(at_time)
        if m not in self._feat_cache:
            self._feat_cache[m] = machine_features(self.fleet, m, self.norm)
        f = self._feat_cache[m].loc[:t]
        if f.empty:
            return {"machine_id": m, "error": "no data before this time"}
        cols = self.risk_meta["features"]
        dm = xgb.DMatrix(f[cols].iloc[[-1]])
        p = float(self.risk_model.predict(dm)[0])
        contrib = self.risk_model.predict(dm, pred_contribs=True)[0][:-1]
        top = np.argsort(-np.abs(contrib))[:5]
        thr = self.risk_meta["threshold"]
        return {"machine_id": m, "as_of": str(f.index[-1]), "p_fail_24h": round(p, 3), "decision_threshold": round(thr, 2),
                "risk_level": "high" if p >= thr else "medium" if p >= thr / 2 else "low",
                "top_factors": [{"feature": cols[i], "value": round(float(f[cols].iloc[-1, i]), 2),
                                 "effect": "raises risk" if contrib[i] > 0 else "lowers risk"} for i in top]}

    def search_logs(self, machine_id: str, end_time: str, hours: int = 72, include_info: bool = False) -> dict:
        m, t = self._check_machine(machine_id), parse_time(end_time)
        lg = self.fleet.machine_logs(m).loc[t - pd.Timedelta(hours=hours): t]
        if not include_info:
            lg = lg[lg.severity != "info"]
        events = []
        for code, g in lg.groupby("code"):
            events.append({"code": code, "message": ERROR_CODES[code][0], "severity": ERROR_CODES[code][1],
                           "count": int(len(g)), "first_seen": str(g.index.min()), "last_seen": str(g.index.max())})
        events.sort(key=lambda e: -e["count"])
        return {"machine_id": m, "window": f"{t - pd.Timedelta(hours=hours)} to {t}", "n_events": int(len(lg)),
                "events": events or "no warning/error events in this window"}

    def get_maintenance_history(self, machine_id: str, before_time: str) -> dict:
        m, t = self._check_machine(machine_id), parse_time(before_time)
        mt = self.fleet.machine_maint(m)
        mt = mt[mt.datetime < t]
        last = {}
        for c in COMPONENTS:
            g = mt[mt.component == c]
            last[c] = {"days_since_service": round((t - g.datetime.max()).total_seconds() / 86400, 1),
                       "last_type": g.type.iloc[-1]} if len(g) else {"days_since_service": None, "last_type": None}
        recent = mt[(mt.type == "corrective") & (mt.datetime >= t - pd.Timedelta(days=180))]
        return {"machine_id": m, "model": self.fleet.machines.loc[m, "model"], "as_of": str(t),
                "components": last,
                "recent_repairs": [{"date": str(r.datetime.date()), "component": r.component} for r in recent.itertuples()]}

    def search_docs(self, query: str, k: int = config.DOC_TOP_K) -> dict:
        hits = self.docs.search(query, k)
        return {"query": query, "results": [{"chunk_id": h["chunk_id"], "score": h["score"], "text": h["text"]} for h in hits]}

    # ------------------------------------------------------------- plumbing
    def call(self, name: str, **kwargs) -> dict:
        if name not in TOOL_DESCRIPTIONS:
            raise ValueError(f"unknown tool {name}")
        return getattr(self, name)(**kwargs)

    def as_langchain_tools(self):
        """Wrap as LangChain tools for the ReAct baseline (LLM chooses tools and arguments freely)."""
        from langchain_core.tools import StructuredTool

        tools = []
        for name, desc in TOOL_DESCRIPTIONS.items():
            fn = getattr(self, name)

            def _run(_fn=fn, **kw):
                try:
                    return json.dumps(_fn(**kw), default=str)
                except Exception as e:  # surface errors to the model instead of crashing the loop
                    return json.dumps({"error": f"{type(e).__name__}: {e}"})

            tools.append(StructuredTool.from_function(func=_run, name=name, description=desc,
                                                      args_schema=_schema_for(name)))
        return tools


def _schema_for(name: str):
    from pydantic import BaseModel, Field

    class MachineWindow(BaseModel):
        machine_id: str = Field(description="Machine ID, e.g. M017")
        end_time: str = Field(description="End of window, 'YYYY-MM-DD HH:00'")
        hours: int = Field(72, description="Look-back window in hours")

    class Risk(BaseModel):
        machine_id: str
        at_time: str = Field(description="'YYYY-MM-DD HH:00'")

    class Maint(BaseModel):
        machine_id: str
        before_time: str = Field(description="'YYYY-MM-DD HH:00'")

    class Docs(BaseModel):
        query: str
        k: int = 4

    return {"query_telemetry": MachineWindow, "detect_anomalies": MachineWindow, "search_logs": MachineWindow,
            "predict_failure_risk": Risk, "get_maintenance_history": Maint, "search_docs": Docs}[name]


MACHINE_RE = re.compile(r"\bM\d{3}\b", re.I)
TIME_RE = re.compile(r"\d{4}-\d{2}-\d{2} \d{2}:\d{2}")


def parse_question(q: str) -> tuple[str | None, str | None]:
    m, t = MACHINE_RE.search(q), TIME_RE.search(q)
    return (m.group(0).upper() if m else None, t.group(0) if t else None)
