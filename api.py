"""FastAPI service for the reliability system.

Run:   uvicorn api:app --reload --port 8000      (docs at http://localhost:8000/docs)
"""
from __future__ import annotations

import json
from functools import lru_cache
from typing import Literal

from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field

from src import config

app = FastAPI(title="Predictive Failure & Root Cause Agent", version="1.1.0")
DASHBOARD = Path(__file__).parent / "dashboard" / "index.html"


class MachineAt(BaseModel):
    machine_id: str = Field(examples=["M017"])
    time: str = Field(examples=["2025-09-01 10:00"], description="'YYYY-MM-DD HH:00'")


class Investigate(BaseModel):
    question: str = Field(examples=["Machine M017 tripped offline at 2025-09-01 10:00. What is the root cause?"])
    variant: Literal["graph", "react", "rules"] = "graph"


@lru_cache(maxsize=1)
def toolbox():
    from src.tools import Toolbox

    if not (config.MODEL_DIR / "risk_xgb.json").exists():
        raise HTTPException(503, "Models not trained yet. Run `python run_pipeline.py`.")
    return Toolbox()


@lru_cache(maxsize=3)
def agent(variant: str):
    from src.agent import RCAAgent

    return RCAAgent(variant, toolbox())


def _call(fn, *a, **kw):
    try:
        return fn(*a, **kw)
    except ValueError as e:
        raise HTTPException(400, str(e))


@app.get("/health")
def health():
    return {"status": "ok", "models_ready": (config.MODEL_DIR / "risk_xgb.json").exists(),
            "docs_index_ready": any(config.INDEX_DIR.glob("*/index_meta.json")),
            "chat_model": config.CHAT_MODEL, "vector_backend": config.VECTOR_BACKEND}


@app.get("/machines")
def machines():
    return toolbox().fleet.machines.reset_index().to_dict(orient="records")


@app.post("/risk")
def risk(q: MachineAt):
    """XGBoost probability of failure within 24h, with top contributing features."""
    return _call(toolbox().predict_failure_risk, q.machine_id, q.time)


@app.post("/anomalies")
def anomalies(q: MachineAt):
    """LSTM-autoencoder + level anomaly detection over the previous 72h."""
    return _call(toolbox().detect_anomalies, q.machine_id, q.time)


@app.post("/investigate")
def investigate(q: Investigate):
    """Run the root-cause agent. Returns the diagnosis, the tools it used and the evidence it saw."""
    res = _call(agent(q.variant).run, q.question)
    res["evidence_text"] = res["evidence_text"][:20000]
    return res


@app.get("/metrics")
def metrics():
    p = config.RESULTS_DIR / "agent_eval_summary.json"
    m = config.RESULTS_DIR / "model_metrics.json"
    if not (p.exists() or m.exists()):
        raise HTTPException(404, "No evaluation yet. Run `python -m src.evaluate`.")
    return {"agent": json.loads(p.read_text()) if p.exists() else None,
            "models": json.loads(m.read_text()) if m.exists() else None}


# ----------------------------------------------------------------------------- dashboard
@app.get("/", response_class=HTMLResponse, include_in_schema=False)
def dashboard():
    return DASHBOARD.read_text(encoding="utf-8")


@app.get("/api/range")
def api_range():
    from src import viz

    return viz.time_range(toolbox())


@app.get("/api/incidents")
def api_incidents():
    from src import viz

    return viz.incident_list(toolbox())


@app.get("/api/fleet")
def api_fleet(at: str):
    """Risk snapshot of every machine at a point in time, with ground truth for comparison."""
    from src import viz

    return _call(viz.fleet_snapshot, toolbox(), at)


@app.get("/api/machine/{machine_id}")
def api_machine(machine_id: str, at: str, hours: int = 168):
    from src import viz

    return _call(viz.machine_detail, toolbox(), machine_id, at, hours)


@app.get("/api/llm")
def api_llm():
    """Which agent variants can run here: 'rules' always, the LLM agents only with an OpenAI key."""
    return {"rules": True, "llm": config.llm_available(), "provider": config.LLM_PROVIDER, "model": config.CHAT_MODEL}


@app.get("/api/operating-points")
def api_operating_points():
    """Warning lead time and false-alert burden per decision threshold (computed once, cached in results/)."""
    path = config.RESULTS_DIR / "lead_time.json"
    if not path.exists():
        from src import lead_time

        toolbox()
        lead_time.main()
    return json.loads(path.read_text())
