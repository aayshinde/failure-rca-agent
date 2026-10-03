# Free live demo on Hugging Face Spaces (Docker SDK). No API keys: simulated data, local embeddings, `rules` agent.
# The whole pipeline (simulate -> train XGBoost + LSTM -> index docs) runs at image build time, so the Space starts instantly.
FROM python:3.12-slim
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 && rm -rf /var/lib/apt/lists/*
RUN useradd -m -u 1000 user
USER user
ENV PATH=/home/user/.local/bin:$PATH PYTHONUNBUFFERED=1 MLFLOW_DISABLE_AGENT_HINT=1 EMBED_MODEL=local
WORKDIR /home/user/app
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu
COPY --chown=user requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY --chown=user src ./src
COPY --chown=user dashboard ./dashboard
COPY --chown=user api.py run_pipeline.py ./
RUN python run_pipeline.py && python -m src.lead_time
EXPOSE 7860
CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "7860"]
