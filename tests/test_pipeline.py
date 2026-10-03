"""Offline end-to-end tests: simulate -> train both models -> index docs -> tools -> agents -> eval.
No API key or network needed (LLMs are replaced with scripted fakes).  Run:  pytest -q
"""
from __future__ import annotations

import json
import sys

import pandas as pd
import pytest
from langchain_core.messages import AIMessage, ToolMessage
from langchain_core.runnables import RunnableLambda

from src import config


@pytest.fixture(scope="session")
def built():
    from src import docs_gen, eval_set, simulate, train_anomaly, train_risk
    from src.retrieval import DocIndex

    tables = simulate.simulate(n_machines=16, n_days=90)
    simulate.save(tables)
    docs_gen.write_docs()
    eval_set.build_and_save(tables["incidents"], tables["telemetry"], 90)
    train_risk.main()
    sys.argv = ["x", "--epochs", "1", "--n-train", "2000"]
    train_anomaly.main()
    DocIndex.build("faiss")
    return tables


@pytest.fixture(scope="session")
def toolbox(built):
    from src.tools import Toolbox

    return Toolbox()


def a_test_incident(toolbox):
    inc = toolbox.fleet.incidents
    return inc[inc.split == "test"].iloc[0]


# ------------------------------------------------------------------ data + models
def test_simulation_shapes(built):
    inc = built["incidents"]
    assert len(inc) > 20 and set(inc.split) == {"train", "test"}
    assert {"volt", "vibration", "current"} <= set(built["telemetry"].columns)
    assert (built["logs"].code == "E000").sum() > 0


def test_eval_questions_are_test_period_only(built):
    qs = [json.loads(l) for l in (config.DATA_DIR / "eval_questions.jsonl").read_text().splitlines()]
    inc = built["incidents"].set_index("incident_id")
    pm = [q for q in qs if q["type"] == "postmortem"]
    assert pm and all(inc.loc[q["incident_id"], "split"] == "test" for q in pm)
    assert {q["type"] for q in qs} == {"postmortem", "risk_assessment", "doc_lookup"}


def test_model_metrics_written(built):
    m = json.loads((config.RESULTS_DIR / "model_metrics.json").read_text())
    assert 0 <= m["risk_xgb"]["test_auroc"] <= 1
    assert "test_auroc_24h" in m["anomaly_detector"]


# ------------------------------------------------------------------ tools
def test_tools(toolbox):
    r = a_test_incident(toolbox)
    t = str(r.failure_time)
    assert "sensors" in toolbox.query_telemetry(r.machine_id, t)
    a = toolbox.detect_anomalies(r.machine_id, t)
    assert "max_score" in a and "top_sensors" in a
    p = toolbox.predict_failure_risk(r.machine_id, t)
    assert 0 <= p["p_fail_24h"] <= 1 and len(p["top_factors"]) == 5
    logs = toolbox.search_logs(r.machine_id, t)
    assert "events" in logs
    docs = toolbox.search_docs("E401 hydraulic pressure low")
    assert docs["results"][0]["chunk_id"].startswith(("error_codes", "runbook_hydraulic"))


def test_maintenance_history_has_no_future_leakage(toolbox):
    r = a_test_incident(toolbox)
    h = toolbox.get_maintenance_history(r.machine_id, str(r.failure_time))
    for rep in h["recent_repairs"]:
        assert pd.Timestamp(rep["date"]) <= r.failure_time.normalize()
    for c in h["components"].values():
        assert c["days_since_service"] is None or c["days_since_service"] > 0


def test_qdrant_backend(built):
    from src.retrieval import DocIndex

    idx = DocIndex.build("qdrant")
    assert idx.search("cooling fan tachometer fault", k=3)[0]["chunk_id"]


# ------------------------------------------------------------------ agents
class FakeLLM:
    """Scripted stand-in for ChatOpenAI supporting with_structured_output and bind_tools."""

    def __init__(self, root="hydraulic_leak", unsupported_first=True):
        self.root, self.unsupported_first, self.verify_calls = root, unsupported_first, 0

    def with_structured_output(self, schema, method=None):
        from src.agent import Diagnosis, Extraction, Plan, Verification

        def respond(pv):
            text = pv.to_string()
            if schema is Plan:
                doc = "M0" not in text.split("Human:")[-1]
                return Plan(question_type="doc_lookup" if doc else "postmortem",
                            tools=["search_docs"] if doc else ["detect_anomalies", "search_logs", "get_maintenance_history", "search_docs"],
                            rationale="test")
            if schema is Diagnosis:
                return Diagnosis(root_cause=self.root, confidence=0.7, answer="Pressure fell before the trip [E1].", evidence_ids=["E1"])
            if schema is Verification:
                self.verify_calls += 1
                bad = self.unsupported_first and self.verify_calls == 1
                return Verification(claims=[{"claim": "pressure fell", "supported": not bad}])
            if schema is Extraction:
                return Extraction(root_cause=self.root)
            raise TypeError(schema)

        return RunnableLambda(respond)

    def bind_tools(self, tools):
        def respond(msgs):
            if not any(isinstance(m, ToolMessage) for m in msgs):
                q = msgs[-1].content
                from src.tools import parse_question
                mid, t = parse_question(q)
                return AIMessage("", tool_calls=[
                    {"name": "search_logs", "args": {"machine_id": mid, "end_time": t}, "id": "c1"},
                    {"name": "search_docs", "args": {"query": "pressure low"}, "id": "c2"}])
            return AIMessage("The root cause is a hydraulic seal leak.")

        return RunnableLambda(respond)


def test_graph_agent_runs_plan_and_revision_loop(toolbox):
    from src.agent import RCAAgent

    r = a_test_incident(toolbox)
    q = f"Machine {r.machine_id} tripped offline at {r.failure_time:%Y-%m-%d %H:00}. What is the root cause?"
    llm = FakeLLM()
    res = RCAAgent("graph", toolbox, llm=llm).run(q)
    assert res["question_type"] == "postmortem" and res["root_cause"] == "hydraulic_leak"
    assert res["tools_called"] == ["detect_anomalies", "search_logs", "get_maintenance_history", "search_docs"]
    assert res["retrieved_chunks"] and "[E1] detect_anomalies" in res["evidence_text"]
    assert res["revisions"] == 1 and llm.verify_calls == 2  # unsupported claim -> one revision -> re-verified


def test_graph_doc_question_only_searches_docs(toolbox):
    from src.agent import RCAAgent

    res = RCAAgent("graph", toolbox, llm=FakeLLM(unsupported_first=False)).run("What does error code E203 mean?")
    assert res["tools_called"] == ["search_docs"] and res["revisions"] == 0


def test_react_agent_records_tool_calls(toolbox):
    from src.agent import RCAAgent

    r = a_test_incident(toolbox)
    res = RCAAgent("react", toolbox, llm=FakeLLM()).run(f"{r.machine_id} went down at {r.failure_time:%Y-%m-%d %H:00}. Root cause?")
    assert res["tools_called"] == ["search_logs", "search_docs"]
    assert res["retrieved_chunks"] and res["root_cause"] == "hydraulic_leak"


# ------------------------------------------------------------------ evaluation
def test_rules_eval_end_to_end(toolbox):
    from src import eval_set
    from src.evaluate import run_variant, summarise

    qs = eval_set.load()
    df = run_variant("rules", qs, toolbox, judge=None, workers=2)
    s = summarise(df)
    assert s["n_questions"] == len(qs)
    assert s["tool_selection_acc"] == 1.0
    assert 0 <= s["root_cause_acc"] <= 1 and s["retrieval_hit_at_k"] is not None


def test_tool_selection_scoring():
    from src.evaluate import score_question

    q = {"qid": "x", "type": "postmortem", "question": "", "gold_root_cause": "bearing_wear", "gold_chunks": ["runbook_bearing_wear#symptoms"],
         "required_tools": ["detect_anomalies", "search_logs", "get_maintenance_history", "search_docs"], "optional_tools": ["query_telemetry"]}
    base = {"variant": "t", "root_cause": "bearing_wear", "will_fail_24h": None, "answer": "", "latency_s": 1,
            "usage": {"llm_calls": 0}, "retrieved_chunks": ["error_codes#e101", "runbook_bearing_wear#symptoms"]}
    ok = score_question(q, base | {"tools_called": q["required_tools"] + ["query_telemetry"]})
    assert ok["tool_selection_ok"] and ok["root_cause_correct"] and ok["retrieval_rr"] == 0.5
    extra = score_question(q, base | {"tools_called": q["required_tools"] + ["predict_failure_risk"]})
    assert not extra["tool_selection_ok"] and extra["tool_precision"] == 0.8
    missing = score_question(q, base | {"tools_called": ["search_logs"]})
    assert not missing["tool_selection_ok"] and missing["tool_recall"] == 0.25


# ------------------------------------------------------------------ dashboard
def test_dashboard_data_layer(toolbox):
    from src import viz

    inc = a_test_incident(toolbox)
    t = pd.Timestamp(inc.failure_time) - pd.Timedelta(hours=3)
    snap = viz.fleet_snapshot(toolbox, str(t))
    assert snap["summary"]["n"] == len(toolbox.fleet.machines)
    assert all(0 <= m["p"] <= 1 for m in snap["machines"])
    row = next(m for m in snap["machines"] if m["machine_id"] == inc.machine_id)
    assert row["fails_24h"]                       # ground truth marks the machine that is about to fail

    d = viz.machine_detail(toolbox, inc.machine_id, str(t), hours=96)
    n = len(d["times"])
    assert n == 96 and len(d["risk"]["p"]) == n and len(d["anomaly"]["score"]) == n
    assert set(d["sensors"]) == set(viz.SENSORS) and d["factors"]
    assert viz.incident_list(toolbox) and viz.time_range(toolbox)["start"] < viz.time_range(toolbox)["end"]


def test_dashboard_served_by_api(built):
    from fastapi.testclient import TestClient

    import api

    c = TestClient(api.app)
    assert "Fleet Reliability Console" in c.get("/").text
    assert c.get("/api/range").status_code == 200
    assert c.get("/api/llm").json()["llm"] is False      # no key in tests
    assert c.get("/api/machine/M999?at=2025-02-01 10:00").status_code == 400
