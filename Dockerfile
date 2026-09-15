FROM python:3.12-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY src ./src
COPY data/.gitkeep ./data/.gitkeep
COPY models/embedding /app/models/embedding
COPY models/reranker /app/models/reranker

EXPOSE 8000

CMD ["sh", "-c", "uvicorn src.main:app --host 0.0.0.0 --port ${PORT:-8000}"]