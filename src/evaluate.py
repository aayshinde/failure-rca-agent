"""Evaluation harness for the root-cause agents (held-out test-period questions).

Per question
  tool_selection_ok   required <= tools_called <= required | optional
  tool precision / recall
  retrieval hit@k, reciprocal rank   (gold = the runbook of the true failure mode, or the doc chunk asked about)
  root_cause_correct  postmortems: predicted failure mode == ground truth
  risk_correct        risk questions: will_fail_24h == ground truth
  groundedness        LLM judge splits the answer into claims and checks each against the evidence the agent
                      actually saw (tool outputs + retrieved docs); "unsupported" = groundedness < 0.8
  latency, LLM tokens, inference cost (USD)

Also pulls in the model metrics (XGBoost F1/AUROC, anomaly AUROC) from results/model_metrics.json.

Usage
  python -m src.evaluate --variants rules                      # offline, no API key
  python -m src.evaluate --variants react graph                # the real comparison (needs OPENAI_API_KEY)
  python -m src.evaluate --variants react graph --n-pm 30 --n-risk 10 --n-doc 10   # cheap smoke run
"""
from __future__ import annotations

import argparse
import json
import os
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import pandas as pd
from tqdm import tqdm

from src import config, eval_set
from src.agent import RCAAgent, Verification, VERIFY_PROMPT
from src.knowledge import MODE_NAMES
from src.llm import UsageTracker, get_chat, structured
from src.tools import Toolbox


class GroundednessJudge:
    def __init__(self, llm=None):
        llm = llm or get_chat(config.JUDGE_MODEL)
        self.chain = VERIFY_PROMPT | structured(llm, Verification)

    def __call__(self, answer: str, evidence: str) -> tuple[float, list[dict]]:
        v: Verification = self.chain.invoke({"evidence": evidence[:60000], "answer": answer})
        claims = [c.model_dump() for c in v.claims]
        return (sum(c["supported"] for c in claims) / len(claims) if claims else 1.0), claims


def score_question(q: dict, r: dict) -> dict:
    called, req, opt = set(r["tools_called"]), set(q["required_tools"]), set(q["optional_tools"])
    rec = {
        "qid": q["qid"], "type": q["type"], "variant": r["variant"], "question": q["question"],
        "tools_called": r["tools_called"], "tool_selection_ok": req <= called <= (req | opt),
        "tool_recall": len(called & req) / len(req), "tool_precision": len(called & (req | opt)) / len(called) if called else 0.0,
        "pred_root_cause": r["root_cause"], "gold_root_cause": q.get("gold_root_cause"),
        "pred_will_fail": r["will_fail_24h"], "gold_will_fail": q.get("gold_will_fail"),
        "answer": r["answer"], "latency_s": r["latency_s"], **{f"usage_{k}": v for k, v in r["usage"].items()},
        "revisions": r.get("revisions", 0),
    }
    if q["type"] == "postmortem":
        rec["root_cause_correct"] = r["root_cause"] == q["gold_root_cause"]
    if q["type"] == "risk_assessment":
        rec["risk_correct"] = bool(r["will_fail_24h"]) == q["gold_will_fail"]
    if q.get("gold_chunks"):
        gold, ret = set(q["gold_chunks"]), r["retrieved_chunks"]
        ranks = [i for i, c in enumerate(ret, 1) if c in gold]
        rec["retrieval_hit"] = bool(ranks and ranks[0] <= config.DOC_TOP_K + 2)
        rec["retrieval_rr"] = 1 / ranks[0] if ranks else 0.0
    return rec


def run_variant(variant: str, qs: list[dict], toolbox: Toolbox, judge, workers: int) -> pd.DataFrame:
    path = config.RESULTS_DIR / f"agent_eval_{variant}.jsonl"
    done = {}
    if path.exists():
        for line in path.read_text().splitlines():
            rec = json.loads(line)
            done[rec["qid"]] = rec
    todo = [q for q in qs if q["qid"] not in done]
    agent = RCAAgent(variant, toolbox)
    if variant != "rules":
        _ = agent.app  # build once before threading

    def work(q):
        r = agent.run(q["question"])
        rec = score_question(q, r)
        if judge is not None:
            g, claims = judge(r["answer"], r["evidence_text"])
            rec |= {"groundedness": g, "unsupported": g < config.GROUNDED_THRESHOLD, "claims": claims}
        return rec

    with path.open("a") as f, ThreadPoolExecutor(max_workers=workers) as pool:
        futs = {pool.submit(work, q): q["qid"] for q in todo}
        for fut in tqdm(as_completed(futs), total=len(futs), desc=f"eval[{variant}]"):
            try:
                rec = fut.result()
            except Exception as e:  # failed rows are retried on the next run
                tqdm.write(f"  ! {futs[fut]}: {type(e).__name__}: {e}")
                continue
            done[rec["qid"]] = rec
            f.write(json.dumps(rec, default=str) + "\n")
            f.flush()
    return pd.DataFrame([done[q["qid"]] for q in qs if q["qid"] in done])


def bootstrap_ci(x: np.ndarray, n: int = 2000, seed: int = 0) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    if len(x) == 0:
        return (float("nan"), float("nan"))
    means = rng.choice(x, size=(n, len(x)), replace=True).mean(1)
    return float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


def summarise(df: pd.DataFrame) -> dict:
    pm, rk = df[df.type == "postmortem"], df[df.type == "risk_assessment"]
    has_ret = df[df.retrieval_hit.notna()] if "retrieval_hit" in df else df.iloc[:0]
    acc = pm.root_cause_correct.astype(float).to_numpy() if len(pm) else np.array([])
    lo, hi = bootstrap_ci(acc)
    s = {
        "n_questions": int(len(df)), "n_postmortem": int(len(pm)),
        "root_cause_acc": round(float(acc.mean()), 4) if len(acc) else None,
        "root_cause_acc_95ci": [round(lo, 4), round(hi, 4)],
        "root_cause_macro_recall": round(float(pm.groupby("gold_root_cause").root_cause_correct.mean().mean()), 4) if len(pm) else None,
        "risk_acc": round(float(rk.risk_correct.mean()), 4) if len(rk) else None,
        "tool_selection_acc": round(float(df.tool_selection_ok.mean()), 4),
        "tool_precision": round(float(df.tool_precision.mean()), 4),
        "tool_recall": round(float(df.tool_recall.mean()), 4),
        "retrieval_hit_at_k": round(float(has_ret.retrieval_hit.astype(float).mean()), 4) if len(has_ret) else None,
        "retrieval_mrr": round(float(has_ret.retrieval_rr.mean()), 4) if len(has_ret) else None,
        "latency_p50_s": round(float(df.latency_s.median()), 2),
        "latency_p95_s": round(float(df.latency_s.quantile(0.95)), 2),
        "llm_calls_per_q": round(float(df.usage_llm_calls.mean()), 2),
        "tokens_per_q": round(float((df.usage_input_tokens + df.usage_output_tokens).mean()), 0),
        "cost_per_q_usd": round(float(df.usage_cost_usd.mean()), 6),
        "cost_total_usd": round(float(df.usage_cost_usd.sum()), 4),
    }
    if "groundedness" in df and df.groundedness.notna().any():
        g = df[df.groundedness.notna()]
        s["mean_groundedness"] = round(float(g.groundedness.mean()), 4)
        s["unsupported_rate"] = round(float(g.unsupported.astype(float).mean()), 4)
    if len(pm):
        s["root_cause_acc_by_mode"] = {m: round(float(v), 3) for m, v in pm.groupby("gold_root_cause").root_cause_correct.mean().items()}
    return s


def write_report(summary: dict) -> str:
    v = summary["variants"]
    names = list(v)
    rows = [("Root-cause accuracy (postmortems)", "root_cause_acc", "pct"), ("  95% bootstrap CI", "root_cause_acc_95ci", "ci"),
            ("Root-cause macro recall", "root_cause_macro_recall", "pct"), ("Risk-question accuracy", "risk_acc", "pct"),
            ("Tool-selection accuracy", "tool_selection_acc", "pct"), ("Tool precision", "tool_precision", "pct"),
            ("Tool recall", "tool_recall", "pct"), ("Retrieval hit@k", "retrieval_hit_at_k", "pct"),
            ("Retrieval MRR", "retrieval_mrr", "num"), ("Mean groundedness", "mean_groundedness", "pct"),
            ("Unsupported-response rate", "unsupported_rate", "pct"), ("Latency p50 (s)", "latency_p50_s", "num"),
            ("Latency p95 (s)", "latency_p95_s", "num"), ("LLM calls / question", "llm_calls_per_q", "num"),
            ("Tokens / question", "tokens_per_q", "num"), ("Cost / question (USD)", "cost_per_q_usd", "num")]

    def cell(x, kind):
        if x is None:
            return "–"
        if kind == "pct":
            return f"{x:.1%}"
        if kind == "ci":
            return f"{x[0]:.1%} – {x[1]:.1%}"
        return f"{x}"

    out = ["# Evaluation report", "",
           f"Questions: {summary['n_questions']} held-out ({summary['counts']}), chat model `{config.CHAT_MODEL}`, "
           f"judge `{config.JUDGE_MODEL}`, embeddings `{config.EMBED_MODEL}`, vector store `{config.VECTOR_BACKEND}`.", "",
           "## Agent", "", "| Metric | " + " | ".join(names) + " |", "|---|" + "---|" * len(names)]
    out += [f"| {label} | " + " | ".join(cell(v[n].get(k), kind) for n in names) + " |" for label, k, kind in rows]
    for a, b, k, label in summary.get("comparisons", []):
        out.append(f"\n**{label}**: {a} → {b}: {k}")
    out += ["", "## Root-cause accuracy by failure mode", "", "| Mode | " + " | ".join(names) + " |", "|---|" + "---|" * len(names)]
    for m in MODE_NAMES:
        out.append(f"| {m} | " + " | ".join(cell(v[n].get("root_cause_acc_by_mode", {}).get(m), "pct") for n in names) + " |")
    mm = summary.get("model_metrics", {})
    if mm:
        out += ["", "## Predictive models (test period)", ""]
        for name, metrics in mm.items():
            flat = {k: x for k, x in metrics.items() if not isinstance(x, dict)}
            out.append(f"- **{name}**: " + ", ".join(f"{k}={x}" for k, x in flat.items()))
            for k, x in metrics.items():
                if isinstance(x, dict):
                    out.append(f"  - {k}: " + ", ".join(f"{a}={b}" for a, b in x.items()))
    text = "\n".join(out) + "\n"
    (config.RESULTS_DIR / "report.md").write_text(text)
    return text


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--variants", nargs="+", default=["react", "graph"], choices=["react", "graph", "rules"])
    ap.add_argument("--n-pm", type=int, default=config.N_POSTMORTEM)
    ap.add_argument("--n-risk", type=int, default=config.N_RISK)
    ap.add_argument("--n-doc", type=int, default=config.N_DOC)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--no-judge", action="store_true")
    ap.add_argument("--fresh", action="store_true")
    args = ap.parse_args()

    qs = eval_set.load(args.n_pm, args.n_risk, args.n_doc)
    if args.fresh:
        for v in args.variants:
            (config.RESULTS_DIR / f"agent_eval_{v}.jsonl").unlink(missing_ok=True)
    judge = None if args.no_judge or not config.llm_available() else GroundednessJudge()
    toolbox = Toolbox()

    summary = {"n_questions": len(qs), "counts": pd.Series([q["type"] for q in qs]).value_counts().to_dict(), "variants": {}}
    for v in args.variants:
        df = run_variant(v, qs, toolbox, judge, args.workers)
        summary["variants"][v] = summarise(df)
        cm = pd.crosstab(df[df.type == "postmortem"].gold_root_cause, df[df.type == "postmortem"].pred_root_cause)
        cm.to_csv(config.RESULTS_DIR / f"confusion_{v}.csv")

    comps = []
    if {"react", "graph"} <= summary["variants"].keys():
        a, b = summary["variants"]["react"], summary["variants"]["graph"]
        comps.append((f"{a['root_cause_acc']:.1%}", f"{b['root_cause_acc']:.1%}", "", "Root-cause accuracy, react → graph"))
        if a.get("unsupported_rate"):
            red = (a["unsupported_rate"] - b["unsupported_rate"]) / a["unsupported_rate"]
            summary["unsupported_reduction"] = round(red, 4)
            comps.append((f"{a['unsupported_rate']:.1%}", f"{b['unsupported_rate']:.1%}", f"relative reduction {red:.1%}",
                          "Unsupported responses"))
    summary["comparisons"] = comps
    mp = config.RESULTS_DIR / "model_metrics.json"
    summary["model_metrics"] = json.loads(mp.read_text()) if mp.exists() else {}
    (config.RESULTS_DIR / "agent_eval_summary.json").write_text(json.dumps(summary, indent=2))
    print(write_report(summary))

    try:
        import mlflow

        mlflow.set_tracking_uri(config.MLFLOW_TRACKING_URI)
        mlflow.set_experiment("rca-agent-eval")
        for v, s in summary["variants"].items():
            with mlflow.start_run(run_name=f"agent_{v}"):
                mlflow.log_params({"variant": v, "chat_model": config.CHAT_MODEL, "judge_model": config.JUDGE_MODEL,
                                   "embed_model": config.EMBED_MODEL, "vector_backend": config.VECTOR_BACKEND,
                                   "n_questions": s["n_questions"]})
                mlflow.log_metrics({k: float(x) for k, x in s.items() if isinstance(x, (int, float)) and x is not None})
                mlflow.log_artifact(str(config.RESULTS_DIR / "report.md"))
                mlflow.log_artifact(str(config.RESULTS_DIR / f"agent_eval_{v}.jsonl"))
    except Exception as e:
        print(f"(MLflow logging skipped: {e})")


if __name__ == "__main__":
    main()
