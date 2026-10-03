FROM python:3.12-slim
WORKDIR /app
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 MLFLOW_DISABLE_AGENT_HINT=1
# CPU-only PyTorch keeps the image small (no CUDA libraries)
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY src ./src
COPY api.py run_pipeline.py ./
COPY dashboard ./dashboard
EXPOSE 8000
# data/, models/, index/ and results/ are mounted from the host (see docker-compose.yml).
# The docs index is (re)built at start-up so it lands in whichever vector store is configured.
CMD ["sh", "-c", "python -m src.retrieval build && uvicorn api:app --host 0.0.0.0 --port 8000"]
