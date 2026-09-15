FROM python:3.12-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY src ./src
COPY data/.gitkeep ./data/.gitkeep

RUN python -c "from huggingface_hub import snapshot_download; snapshot_download(repo_id='sentence-transformers/all-MiniLM-L6-v2', local_dir='/app/models/embedding')" \
    && python -c "from huggingface_hub import snapshot_download; snapshot_download(repo_id='cross-encoder/ms-marco-MiniLM-L-6-v2', local_dir='/app/models/reranker')"

EXPOSE 8000

CMD ["sh", "-c", "uvicorn src.main:app --host 0.0.0.0 --port ${PORT:-8000}"]