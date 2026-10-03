"""Point every path at a temp dir and use offline embeddings BEFORE src.config is imported."""
import os
import tempfile
from pathlib import Path

_tmp = Path(tempfile.mkdtemp(prefix="rca_test_"))
os.environ.update({
    "DATA_DIR": str(_tmp / "data"), "MODEL_DIR": str(_tmp / "models"), "INDEX_DIR": str(_tmp / "index"),
    "RESULTS_DIR": str(_tmp / "results"), "MLFLOW_TRACKING_URI": f"sqlite:///{_tmp / 'mlflow.db'}",
    "EMBED_MODEL": "local", "LLM_PROVIDER": "openai", "MLFLOW_DISABLE_AGENT_HINT": "1",
    "N_POSTMORTEM": "20", "N_RISK": "6", "N_DOC": "6",
})
os.environ.pop("OPENAI_API_KEY", None)
