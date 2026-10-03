"""Build the held-out evaluation questions (all from the TEST period, never seen in training).

Three question types, each with the tools a competent investigator must use (required)
and may use (optional). Tool selection is correct when required <= selected <= required | optional.

postmortem       "Machine M017 tripped at <time>. Find the root cause."     gold: root cause
risk_assessment  "Is M042 likely to fail in the next 24h as of <time>?"      gold: will_fail (+ mode)
doc_lookup       "What does E203 mean / service interval / procedure"       gold: doc chunk(s)
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from src import config
from src.docs_gen import chunk_docs, slug
from src.knowledge import COMPONENTS, ERROR_CODES, FAILURE_MODES

TOOLS = ["query_telemetry", "detect_anomalies", "predict_failure_risk", "search_logs", "get_maintenance_history", "search_docs"]

TOOL_POLICY = {
    "postmortem": {"required": ["detect_anomalies", "search_logs", "get_maintenance_history", "search_docs"],
                   "optional": ["query_telemetry"]},
    "risk_assessment": {"required": ["predict_failure_risk", "detect_anomalies"],
                        "optional": ["query_telemetry", "search_logs", "get_maintenance_history", "search_docs"]},
    "doc_lookup": {"required": ["search_docs"], "optional": []},
}

PM_TEMPLATES = [
    "Machine {m} tripped offline at {t}. Operator note: \"{r}\". What is the root cause?",
    "Investigate the unplanned stop of {m} at {t}. Shift report says: \"{r}\". Identify the failure mode and supporting evidence.",
    "{m} went down at {t} (\"{r}\"). Run a root-cause analysis and recommend the fix.",
]
RISK_TEMPLATES = [
    "As of {t}, is machine {m} likely to fail within the next 24 hours? If so, what is the most likely failure mode?",
    "Assess the failure risk of {m} at {t} for the next day and say which component is at risk.",
]


def fmt(ts) -> str:
    return pd.Timestamp(ts).strftime("%Y-%m-%d %H:00")


def build(incidents: pd.DataFrame, telemetry: pd.DataFrame, n_days: int, seed: int = config.SEED) -> list[dict]:
    rng = np.random.default_rng(seed + 1)
    test = incidents[incidents.split == "test"].reset_index(drop=True)
    qs: list[dict] = []

    # ---- postmortems: one per test incident (shuffled; evaluation takes the first N)
    for _, inc in test.sample(frac=1, random_state=seed).iterrows():
        qs.append({
            "qid": f"PM-{inc.incident_id}", "type": "postmortem",
            "question": rng.choice(PM_TEMPLATES).format(m=inc.machine_id, t=fmt(inc.failure_time), r=inc.operator_report),
            "machine_id": inc.machine_id, "time": fmt(inc.failure_time),
            "gold_root_cause": inc.root_cause, "incident_id": inc.incident_id,
            "gold_chunks": [c["chunk_id"] for c in CHUNKS() if c["doc_id"] == f"runbook_{inc.root_cause}"],
        })

    # ---- risk assessments: positives shortly before a test failure, negatives far from any failure
    n_pos = max(config.N_RISK // 2, 1)
    for _, inc in test.sample(n=min(n_pos, len(test)), random_state=seed + 2).iterrows():
        t = pd.Timestamp(inc.failure_time) - pd.Timedelta(hours=int(rng.integers(4, 20)))
        qs.append({"qid": f"RK-{inc.incident_id}", "type": "risk_assessment",
                   "question": rng.choice(RISK_TEMPLATES).format(m=inc.machine_id, t=fmt(t)),
                   "machine_id": inc.machine_id, "time": fmt(t), "gold_will_fail": True,
                   "gold_root_cause": inc.root_cause, "gold_chunks": []})
    split_time = pd.Timestamp(config.START_DATE) + pd.Timedelta(days=int(n_days * config.TRAIN_FRAC))
    tel_test = telemetry[telemetry.datetime >= split_time + pd.Timedelta(days=2)]
    fails = incidents.groupby("machine_id").failure_time.apply(lambda s: pd.to_datetime(s).values)
    k = 0
    while k < config.N_RISK - n_pos:
        row = tel_test.iloc[int(rng.integers(len(tel_test)))]
        ft = fails.get(row.machine_id, np.array([], dtype="datetime64[ns]"))
        dt = (ft - np.datetime64(row.datetime)) / np.timedelta64(1, "h")
        if np.any((dt > -48) & (dt < 96)):
            continue
        qs.append({"qid": f"RK-NEG-{k:03d}", "type": "risk_assessment",
                   "question": rng.choice(RISK_TEMPLATES).format(m=row.machine_id, t=fmt(row.datetime)),
                   "machine_id": row.machine_id, "time": fmt(row.datetime), "gold_will_fail": False,
                   "gold_root_cause": "none", "gold_chunks": []})
        k += 1

    # interleave positives and negatives so any --n cap stays balanced
    risk = [q for q in qs if q["type"] == "risk_assessment"]
    qs = [q for q in qs if q["type"] != "risk_assessment"] + [risk[i] for i in rng.permutation(len(risk))]

    # ---- documentation lookups
    codes = [c for c in ERROR_CODES if c not in ("E000", "E900", "E901")]
    doc_qs = []
    for c in codes:
        doc_qs.append((f"What does error code {c} mean and which failure modes is it associated with?", [f"error_codes#{slug(c)}"]))
    for comp in COMPONENTS:
        doc_qs.append((f"What is the recommended service interval for the {comp.replace('_', ' ')}?", [f"maintenance_manual#{slug(comp)}"]))
    for m in FAILURE_MODES.values():
        doc_qs.append((f"What is the corrective action for {m.title.lower()}?", [f"runbook_{m.name}#corrective-action"]))
        doc_qs.append((f"Which diagnostic checks should a technician run for suspected {m.name.replace('_', ' ')}?", [f"runbook_{m.name}#diagnostic-checks"]))
    order = rng.permutation(len(doc_qs))
    for i in order[: config.N_DOC]:
        q, gold = doc_qs[i]
        qs.append({"qid": f"DOC-{i:03d}", "type": "doc_lookup", "question": q, "machine_id": None, "time": None,
                   "gold_chunks": gold})

    for q in qs:
        q["required_tools"] = TOOL_POLICY[q["type"]]["required"]
        q["optional_tools"] = TOOL_POLICY[q["type"]]["optional"]
    return qs


_CHUNKS: list[dict] | None = None


def CHUNKS() -> list[dict]:
    global _CHUNKS
    if _CHUNKS is None:
        _CHUNKS = chunk_docs()
    return _CHUNKS


def build_and_save(incidents, telemetry, n_days) -> list[dict]:
    qs = build(incidents, telemetry, n_days)
    with (config.DATA_DIR / "eval_questions.jsonl").open("w") as f:
        for q in qs:
            f.write(json.dumps(q, default=str) + "\n")
    return qs


def load(n_postmortem=config.N_POSTMORTEM, n_risk=config.N_RISK, n_doc=config.N_DOC) -> list[dict]:
    qs = [json.loads(l) for l in (config.DATA_DIR / "eval_questions.jsonl").read_text().splitlines()]
    caps = {"postmortem": n_postmortem, "risk_assessment": n_risk, "doc_lookup": n_doc}
    out, seen = [], {k: 0 for k in caps}
    for q in qs:
        if seen[q["type"]] < caps[q["type"]]:
            out.append(q)
            seen[q["type"]] += 1
    return out
