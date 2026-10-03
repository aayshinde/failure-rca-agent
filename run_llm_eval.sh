#!/usr/bin/env bash
# Free LLM evaluation on a local Ollama model: compares the react baseline with the LangGraph workflow.
# Usage: ./run_llm_eval.sh [n-pm n-risk n-doc]     (defaults to a 32-question sample; ~1 min per question on a laptop)
set -e
# Ollama defaults to a 4k context, which silently truncates the ReAct loop's accumulated tool output and would
# handicap the baseline. Create a 12k-context copy of the model so both agents get a fair setup.
if ! ollama list | grep -q "^qwen2.5-12k"; then
  printf 'FROM qwen2.5:7b-instruct\nPARAMETER num_ctx 12288\nPARAMETER temperature 0\n' > /tmp/Modelfile.rca
  ollama create qwen2.5-12k -f /tmp/Modelfile.rca
fi
cd "$(dirname "$0")"
[ -f .venv/bin/activate ] && source .venv/bin/activate
export LLM_PROVIDER=ollama CHAT_MODEL=${CHAT_MODEL:-qwen2.5-12k} JUDGE_MODEL=${JUDGE_MODEL:-qwen2.5-12k}
export EMBED_MODEL=${EMBED_MODEL:-ollama:nomic-embed-text}
python -m src.retrieval build
python -m src.evaluate --variants react graph --n-pm "${1:-20}" --n-risk "${2:-6}" --n-doc "${3:-6}" --workers 1 --fresh
