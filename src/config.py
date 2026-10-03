"""Central configuration — every value can be overridden with an environment variable / .env."""
from __future__ import annotations

import os
import sys
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

# torch, xgboost and faiss each ship their own OpenMP runtime; on macOS loading them together can segfault.
if sys.platform == "darwin":
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = Path(os.getenv("DATA_DIR", ROOT / "data"))
DOCS_DIR = DATA_DIR / "docs"
MODEL_DIR = Path(os.getenv("MODEL_DIR", ROOT / "models"))
INDEX_DIR = Path(os.getenv("INDEX_DIR", ROOT / "index"))
RESULTS_DIR = Path(os.getenv("RESULTS_DIR", ROOT / "results"))

# --- simulation ---------------------------------------------------------------
SEED = int(os.getenv("SEED", "7"))
N_MACHINES = int(os.getenv("N_MACHINES", "100"))
N_DAYS = int(os.getenv("N_DAYS", "365"))
START_DATE = os.getenv("START_DATE", "2025-01-01")
TRAIN_FRAC = float(os.getenv("TRAIN_FRAC", "0.6"))   # time-based split: first 60% of days train, rest test
VAL_FRAC = float(os.getenv("VAL_FRAC", "0.15"))      # tail of the train period used to pick thresholds
RISK_HORIZON_H = int(os.getenv("RISK_HORIZON_H", "24"))

# --- LLMs -----------------------------------------------------------------------
# LLM_PROVIDER=ollama runs the agents and the judge on a local model (free, no API key); openai is the default.
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "openai").lower()
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434/v1")
CHAT_MODEL = os.getenv("CHAT_MODEL", "gpt-4o-mini")
JUDGE_MODEL = os.getenv("JUDGE_MODEL", "gpt-4o-mini")
EMBED_MODEL = os.getenv("EMBED_MODEL", "text-embedding-3-small")  # "local" = offline hashing; "ollama:<model>" = local semantic embeddings



def llm_available() -> bool:
    """True when a chat model can be called: a local Ollama server, or a real-looking OpenAI key."""
    if LLM_PROVIDER == "ollama":
        return True
    key = os.getenv("OPENAI_API_KEY", "")
    return bool(key) and not key.endswith("...")


# USD per 1M tokens (input, output). Check current OpenAI pricing and edit if it changed.
PRICES = {
    "gpt-4o-mini": (0.15, 0.60),
    "gpt-4o": (2.50, 10.00),
    "gpt-3.5-turbo": (0.50, 1.50),
    "gpt-4.1-mini": (0.40, 1.60),
    "text-embedding-3-small": (0.02, 0.0),
}

# --- retrieval ------------------------------------------------------------------
VECTOR_BACKEND = os.getenv("VECTOR_BACKEND", "faiss")  # faiss | qdrant
QDRANT_URL = os.getenv("QDRANT_URL", "")                # empty = embedded local Qdrant under index/qdrant
DOC_TOP_K = int(os.getenv("DOC_TOP_K", "4"))

# --- evaluation -----------------------------------------------------------------
N_POSTMORTEM = int(os.getenv("N_POSTMORTEM", "300"))
N_RISK = int(os.getenv("N_RISK", "60"))
N_DOC = int(os.getenv("N_DOC", "40"))
# A response counts as "unsupported" if less than this share of its claims is backed by evidence.
GROUNDED_THRESHOLD = float(os.getenv("GROUNDED_THRESHOLD", "0.8"))

MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", f"sqlite:///{ROOT / 'mlflow.db'}")

for d in (DATA_DIR, MODEL_DIR, INDEX_DIR, RESULTS_DIR):
    d.mkdir(parents=True, exist_ok=True)
