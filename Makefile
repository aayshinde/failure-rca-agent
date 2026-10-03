.PHONY: export-demo setup data demo eval eval-llm benchmark baselines lead-time test docker
PY ?= python

setup:          ## create venv + install deps
	python3 -m venv .venv && .venv/bin/pip install -r requirements.txt && cp -n .env.example .env || true

data:           ## simulate fleet, train both models, index docs (~2 min, no API key needed with EMBED_MODEL=local)
	EMBED_MODEL=local $(PY) run_pipeline.py

demo:           ## open the live dashboard at http://localhost:8000
	EMBED_MODEL=local $(PY) -m uvicorn api:app --port 8000

eval:           ## offline evaluation of the rules agent (free)
	EMBED_MODEL=local $(PY) -m src.evaluate --variants rules --no-judge

test:           ## 13 offline tests
	$(PY) -m pytest -q

docker:
	docker compose up --build

eval-llm:       ## compare react vs graph agents on a LOCAL Ollama model: free, ~1-2 h (see run_llm_eval.sh)
	./run_llm_eval.sh

benchmark:      ## validate the modelling approach on NASA C-MAPSS (downloads ~12 MB)
	$(PY) -m src.benchmark_cmapss

baselines:    ## ablations and linear baseline
	$(PY) -m src.baselines

lead-time:      ## warning lead time and false-alert burden per threshold
	$(PY) -m src.lead_time

export-demo:    ## rebuild the static GitHub Pages demo in docs/ (~4 min)
	$(PY) -m src.export_static
