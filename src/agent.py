"""Root-cause investigation agents built with LangGraph.

Three variants share the same tools and output fields so they can be compared directly:

react     (v1 baseline)  Generic ReAct loop: the LLM picks tools and arguments freely, answers in
                         free text; root cause is then extracted from that text.
graph     (v2, deployed) Explicit LangGraph workflow:
                           plan  -> structured question typing + tool routing policy
                           gather-> run the planned data tools deterministically
                           retrieve -> doc search with a query built from the evidence
                                       (error codes, abnormal sensors), not just the question
                           diagnose -> structured diagnosis constrained to known failure modes,
                                       every claim cites [E#]/[D#] evidence
                           verify -> groundedness self-check; unsupported claims are sent back
                                     to diagnose once for revision
rules     (no LLM)       Deterministic planner + evidence-scoring diagnoser. Offline reference
                         baseline; also lets the whole eval harness run without an API key.
"""
from __future__ import annotations

import json
import time
from typing import Literal, Optional, TypedDict

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.prebuilt import ToolNode, tools_condition
from pydantic import BaseModel, Field, field_validator

from src import config
from src.eval_set import TOOLS
from src.knowledge import CODE_TO_MODES, FAILURE_MODES, MODE_NAMES
from src.llm import UsageTracker, get_chat, structured
from src.tools import TOOL_DESCRIPTIONS, Toolbox, parse_question

ROOT_CAUSES = tuple(MODE_NAMES + ["none", "unknown"])
QTYPES = ("postmortem", "risk_assessment", "doc_lookup")
MAX_REVISIONS = 1


# ============================================================ schemas
class Plan(BaseModel):
    question_type: Literal[QTYPES]
    machine_id: Optional[str] = Field(None, description="e.g. M017, if the question is about a machine")
    time: Optional[str] = Field(None, description="'YYYY-MM-DD HH:00' from the question, if any")
    tools: list[Literal[tuple(TOOLS)]] = Field(description="tools to run")
    rationale: str


class Diagnosis(BaseModel):
    root_cause: Literal[ROOT_CAUSES] = Field(description="failure mode; 'none' if no failure is expected; 'unknown' for doc questions or if undeterminable")
    will_fail_24h: Optional[bool] = Field(None, description="risk questions only")
    confidence: float = Field(ge=0, le=1)

    @field_validator("confidence", mode="before")
    @classmethod
    def _percent_to_fraction(cls, v):
        """Small local models sometimes answer 85 or 100 (percent) instead of 0.85; accept it rather than crash."""
        return v / 100 if isinstance(v, (int, float)) and 1 < v <= 100 else v
    answer: str = Field(description="concise answer; cite evidence like [E2] or [D1] after every factual statement")
    evidence_ids: list[str] = Field(default_factory=list)


class ClaimCheck(BaseModel):
    claim: str
    supported: bool


class Verification(BaseModel):
    claims: list[ClaimCheck]


class Extraction(BaseModel):
    root_cause: Literal[ROOT_CAUSES]
    will_fail_24h: Optional[bool] = None


# ============================================================ prompts
TOOL_LIST = "\n".join(f"- {k}: {v}" for k, v in TOOL_DESCRIPTIONS.items())

PLANNER_PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "You route maintenance questions to investigation tools.\nTools:\n{tools}\n\n"
     "Tool policy:\n"
     "- postmortem (a machine already failed/tripped; find the root cause): detect_anomalies, search_logs, "
     "get_maintenance_history, search_docs. query_telemetry is optional. Do NOT use predict_failure_risk — it is "
     "forward-looking and says nothing about why a past failure happened.\n"
     "- risk_assessment (will a machine fail soon?): predict_failure_risk and detect_anomalies; optionally "
     "query_telemetry, search_logs, get_maintenance_history, search_docs.\n"
     "- doc_lookup (general question about codes, procedures, intervals; no specific machine event): search_docs only."),
    ("human", "{question}"),
])

DIAGNOSE_PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "You are a senior reliability engineer. Answer using ONLY the evidence below.\n"
     "Failure modes: {modes}\n\n"
     "Method for machine questions:\n"
     "1. Abnormal telemetry: which sensors, direction, level shift vs volatility/pattern. Note if telemetry is normal.\n"
     "2. Error codes: repeated codes in the hours before the event are strong evidence; a single occurrence may be "
     "background noise; E000 only marks the trip itself. Several modes share codes (see the error-code reference).\n"
     "3. Compare with runbook symptoms and the differential-diagnosis notes; physical modes move related sensors "
     "together, a single erratic sensor suggests instrumentation; no telemetry precursor suggests the controller.\n"
     "4. Maintenance: wear modes are more likely when the component is overdue for service. Machine model may matter.\n"
     "5. Operator notes are often generic or wrong — weak tie-breaker only.\n"
     "postmortem: pick the single most likely root_cause. risk_assessment: set will_fail_24h from the risk model and "
     "anomalies; root_cause = most likely mode if a failure is likely, else 'none'. doc_lookup: root_cause='unknown', "
     "answer from the docs.\n"
     "Cite [E#]/[D#] after every factual statement and never state numbers that are not in the evidence."),
    ("human", "Question type: {qtype}\nQuestion: {question}\n\nEVIDENCE:\n{evidence}\n\nDOCUMENTATION:\n{docs}{feedback}"),
])

VERIFY_PROMPT = ChatPromptTemplate.from_messages([
    ("system", "Split the ANSWER into atomic factual claims. Mark each supported=true only if the EVIDENCE states or "
               "directly implies it (a cited [E#]/[D#] must actually say it). Ignore the final root-cause label itself."),
    ("human", "EVIDENCE:\n{evidence}\n\nANSWER:\n{answer}"),
])

REACT_SYSTEM = ("You are a maintenance assistant for an industrial fleet. Use the tools as needed to answer the "
                "question, then give a final answer. For failures, name the root cause. For risk questions, say "
                "whether the machine will fail in the next 24 hours.")

EXTRACT_PROMPT = ChatPromptTemplate.from_messages([
    ("system", "Map the assistant's final answer to labels. root_cause must be one of: {labels}. Use 'none' if the "
               "answer says no failure is expected and 'unknown' if no cause is stated. Do not add your own judgement."),
    ("human", "Question: {question}\n\nAnswer: {answer}"),
])


# ============================================================ helpers
def fmt_evidence(ev: list[dict]) -> str:
    return "\n\n".join(f"[{e['id']}] {e['tool']}: {json.dumps(e['output'], default=str)}" for e in ev) or "(none)"


def fmt_docs(docs: list[dict]) -> str:
    return "\n\n".join(f"[{d['id']}] ({d['chunk_id']}) {d['text']}" for d in docs) or "(none)"


def evidence_query(question: str, evidence: list[dict]) -> str:
    """Doc-search query from what the tools found, so retrieval targets the matching runbooks."""
    terms = []
    for e in evidence:
        out = e["output"]
        if e["tool"] == "search_logs" and isinstance(out.get("events"), list):
            terms += [f"{x['code']} {x['message']}" for x in out["events"] if x["code"] != "E000"][:4]
        if e["tool"] == "detect_anomalies":
            terms += [f"{s['sensor']} {s['kind']}" for s in out.get("top_sensors", [])]
        if e["tool"] == "query_telemetry":
            terms += [n for n in out.get("notable", []) if "normal range" not in n]
    if not terms:
        return question + " no telemetry precursor"
    return "symptoms: " + "; ".join(terms)


def result(variant: str, **kw) -> dict:
    base = {"variant": variant, "question_type": None, "root_cause": "unknown", "will_fail_24h": None, "answer": "",
            "confidence": None, "tools_called": [], "retrieved_chunks": [], "evidence_text": "", "revisions": 0}
    return base | kw


# ============================================================ v2: LangGraph workflow
class RCAState(TypedDict, total=False):
    question: str
    plan: dict
    evidence: list
    docs: list
    diagnosis: dict
    feedback: str
    revisions: int


def build_graph(toolbox: Toolbox, llm=None, verify: bool = True):
    llm = llm or get_chat()
    planner = PLANNER_PROMPT | structured(llm, Plan)
    diagnoser = DIAGNOSE_PROMPT | structured(llm, Diagnosis)
    verifier = VERIFY_PROMPT | structured(llm, Verification)

    def plan(state: RCAState, config=None):
        p: Plan = planner.invoke({"tools": TOOL_LIST, "question": state["question"]}, config)
        mid, t = parse_question(state["question"])
        p.machine_id, p.time = (p.machine_id or mid), (p.time or t)
        tools = list(dict.fromkeys(p.tools))
        if not (p.machine_id and p.time):
            tools = [x for x in tools if x == "search_docs"] or ["search_docs"]
        return {"plan": p.model_dump() | {"tools": tools}, "evidence": [], "docs": [], "revisions": 0}

    def gather(state: RCAState):
        p, ev = state["plan"], []
        args = {"query_telemetry": {"end_time": p.get("time")}, "detect_anomalies": {"end_time": p.get("time")},
                "search_logs": {"end_time": p.get("time")}, "predict_failure_risk": {"at_time": p.get("time")},
                "get_maintenance_history": {"before_time": p.get("time")}}
        for name in p["tools"]:
            if name == "search_docs":
                continue
            try:
                out = toolbox.call(name, machine_id=p["machine_id"], **args[name])
            except Exception as e:
                out = {"error": f"{type(e).__name__}: {e}"}
            ev.append({"id": f"E{len(ev) + 1}", "tool": name, "output": out})
        return {"evidence": ev}

    def retrieve(state: RCAState):
        if "search_docs" not in state["plan"]["tools"]:
            return {"docs": []}
        q = state["question"] if state["plan"]["question_type"] == "doc_lookup" else evidence_query(state["question"], state["evidence"])
        k = config.DOC_TOP_K if state["plan"]["question_type"] == "doc_lookup" else config.DOC_TOP_K + 2
        hits = toolbox.search_docs(q, k=k)["results"]
        return {"docs": [{"id": f"D{i + 1}", **h} for i, h in enumerate(hits)]}

    def diagnose(state: RCAState, config=None):
        fb = f"\n\nREVISION REQUEST: {state['feedback']}" if state.get("feedback") else ""
        d: Diagnosis = diagnoser.invoke({
            "modes": ", ".join(MODE_NAMES), "qtype": state["plan"]["question_type"], "question": state["question"],
            "evidence": fmt_evidence(state["evidence"]), "docs": fmt_docs(state["docs"]), "feedback": fb}, config)
        return {"diagnosis": d.model_dump()}

    def check(state: RCAState, config=None):
        ctx = fmt_evidence(state["evidence"]) + "\n\n" + fmt_docs(state["docs"])
        v: Verification = verifier.invoke({"evidence": ctx, "answer": state["diagnosis"]["answer"]}, config)
        bad = [c.claim for c in v.claims if not c.supported]
        if bad and state.get("revisions", 0) < MAX_REVISIONS:
            return {"feedback": "These claims are not supported by the evidence; remove or correct them and re-check "
                                "the diagnosis: " + " | ".join(bad), "revisions": state.get("revisions", 0) + 1}
        return {"feedback": ""}

    g = StateGraph(RCAState)
    g.add_node("plan", plan)
    g.add_node("gather", gather)
    g.add_node("retrieve", retrieve)
    g.add_node("diagnose", diagnose)
    g.add_edge(START, "plan")
    g.add_edge("plan", "gather")
    g.add_edge("gather", "retrieve")
    g.add_edge("retrieve", "diagnose")
    if verify:
        g.add_node("verify", check)
        g.add_edge("diagnose", "verify")
        g.add_conditional_edges("verify", lambda s: "diagnose" if s.get("feedback") else END, ["diagnose", END])
    else:
        g.add_edge("diagnose", END)
    return g.compile()


# ============================================================ v1: ReAct baseline
def build_react(toolbox: Toolbox, llm=None):
    llm = llm or get_chat()
    tools = toolbox.as_langchain_tools()
    bound = llm.bind_tools(tools)

    def agent(state: MessagesState, config=None):
        return {"messages": [bound.invoke([SystemMessage(REACT_SYSTEM)] + state["messages"], config)]}

    g = StateGraph(MessagesState)
    g.add_node("agent", agent)
    g.add_node("tools", ToolNode(tools))
    g.add_edge(START, "agent")
    g.add_conditional_edges("agent", tools_condition)
    g.add_edge("tools", "agent")
    return g.compile()


# ============================================================ rules baseline (no LLM)
def rules_answer(toolbox: Toolbox, question: str) -> dict:
    q = question.lower()
    mid, t = parse_question(question)
    if mid and t and any(w in q for w in ("likely to fail", "risk")):
        qtype, tools = "risk_assessment", ["predict_failure_risk", "detect_anomalies", "search_logs"]
    elif mid and t:
        qtype, tools = "postmortem", ["detect_anomalies", "search_logs", "get_maintenance_history", "search_docs"]
    else:
        qtype, tools = "doc_lookup", ["search_docs"]

    ev, out = [], {}
    for name in tools:
        if name == "search_docs":
            continue
        kw = {"at_time": t} if name == "predict_failure_risk" else {"before_time": t} if name == "get_maintenance_history" else {"end_time": t}
        out[name] = toolbox.call(name, machine_id=mid, **kw)
        ev.append({"id": f"E{len(ev) + 1}", "tool": name, "output": out[name]})
    docs = []
    if "search_docs" in tools:
        query = question if qtype == "doc_lookup" else evidence_query(question, ev)
        docs = [{"id": f"D{i + 1}", **h} for i, h in enumerate(toolbox.search_docs(query)["results"])]

    if qtype == "doc_lookup":
        root, will, ans = "unknown", None, f"{docs[0]['text']} [D1]" if docs else "No documentation found."
    else:
        scores = score_modes(out)
        root = max(scores, key=scores.get) if max(scores.values()) > 0 else "unknown"
        will = None
        if qtype == "risk_assessment":
            r = out["predict_failure_risk"]
            will = r.get("risk_level") == "high" or (r.get("risk_level") == "medium" and out["detect_anomalies"].get("anomalous"))
            root = root if will else "none"
        ans = f"Most likely root cause: {root} (evidence [E1]-[E{len(ev)}])."
    return result("rules", question_type=qtype, root_cause=root, will_fail_24h=will, answer=ans, tools_called=tools,
                  retrieved_chunks=[d["chunk_id"] for d in docs], evidence_text=fmt_evidence(ev) + "\n\n" + fmt_docs(docs))


def score_modes(out: dict) -> dict:
    s = {m: 0.0 for m in MODE_NAMES}
    logs = out.get("search_logs", {}).get("events")
    for e in logs if isinstance(logs, list) else []:
        for m in CODE_TO_MODES.get(e["code"], []):
            primary = FAILURE_MODES[m].error_codes[0] == e["code"]
            s[m] += min(e["count"], 6) * (1.0 if primary else 0.5) / len(CODE_TO_MODES[e["code"]])
    an = out.get("detect_anomalies", {})
    top = an.get("top_sensors", [])
    for t in top:
        for m, fm in FAILURE_MODES.items():
            sig = fm.signature.get(t["sensor"])
            if sig and ((sig[0] == "noise" and "pattern" in t["kind"]) or
                        (sig[0] == "drift" and ("up" in t["kind"]) == (sig[1] > 0) and "level" in t["kind"])):
                s[m] += 2.0 * min(t["score"], 3)
    if len(top) == 1 and "pattern" in top[0]["kind"]:
        s["sensor_malfunction"] += 2.5
    if not an.get("anomalous"):
        s["controller_fault"] += 0.5
    maint = out.get("get_maintenance_history", {}).get("components", {})
    for m, fm in FAILURE_MODES.items():
        d = (maint.get(fm.component) or {}).get("days_since_service")
        if fm.wear and d and d > 120:
            s[m] += 0.5
    return s


# ============================================================ unified runner
class RCAAgent:
    def __init__(self, variant: Literal["graph", "react", "rules"] = "graph", toolbox: Toolbox | None = None, llm=None):
        self.variant, self.toolbox, self.llm = variant, toolbox or Toolbox(), llm
        self._app = None

    @property
    def app(self):
        if self._app is None and self.variant != "rules":
            self._app = build_graph(self.toolbox, self.llm) if self.variant == "graph" else build_react(self.toolbox, self.llm)
        return self._app

    def run(self, question: str) -> dict:
        tracker, t0 = UsageTracker(), time.time()
        cfg = {"callbacks": [tracker], "recursion_limit": 25}
        if self.variant == "rules":
            res = rules_answer(self.toolbox, question)
        elif self.variant == "graph":
            s = self.app.invoke({"question": question}, cfg)
            d = s["diagnosis"]
            res = result("graph", question_type=s["plan"]["question_type"], root_cause=d["root_cause"],
                         will_fail_24h=d["will_fail_24h"], answer=d["answer"], confidence=d["confidence"],
                         tools_called=s["plan"]["tools"], retrieved_chunks=[x["chunk_id"] for x in s["docs"]],
                         evidence_text=fmt_evidence(s["evidence"]) + "\n\n" + fmt_docs(s["docs"]),
                         revisions=s.get("revisions", 0), plan_rationale=s["plan"]["rationale"])
        else:
            res = self._run_react(question, cfg)
        res["latency_s"] = round(time.time() - t0, 2)
        res["usage"] = tracker.summary()
        return res

    def _run_react(self, question: str, cfg: dict) -> dict:
        s = self.app.invoke({"messages": [HumanMessage(question)]}, cfg)
        msgs = s["messages"]
        tools, chunks, ev = [], [], []
        for m in msgs:
            if isinstance(m, AIMessage):
                tools += [tc["name"] for tc in m.tool_calls]
            if isinstance(m, ToolMessage):
                ev.append(f"[{m.name}] {m.content}")
                if m.name == "search_docs":
                    try:
                        chunks += [r["chunk_id"] for r in json.loads(m.content).get("results", [])]
                    except (json.JSONDecodeError, AttributeError):
                        pass
        answer = msgs[-1].content if isinstance(msgs[-1].content, str) else str(msgs[-1].content)
        llm = self.llm or get_chat()
        ex: Extraction = (EXTRACT_PROMPT | structured(llm, Extraction)).invoke(
            {"labels": ", ".join(ROOT_CAUSES), "question": question, "answer": answer}, cfg)
        return result("react", root_cause=ex.root_cause, will_fail_24h=ex.will_fail_24h, answer=answer,
                      tools_called=list(dict.fromkeys(tools)), retrieved_chunks=list(dict.fromkeys(chunks)),
                      evidence_text="\n\n".join(ev))


if __name__ == "__main__":
    import sys

    v = sys.argv[1] if len(sys.argv) > 1 and sys.argv[1] in ("graph", "react", "rules") else "graph"
    q = " ".join(a for a in sys.argv[1:] if a != v) or "Machine M017 tripped offline at 2025-09-01 10:00. What is the root cause?"
    r = RCAAgent(v).run(q)
    r.pop("evidence_text")
    print(json.dumps(r, indent=2, default=str))
