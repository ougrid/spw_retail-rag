FROM python:3.11.12-slim-bookworm

ENV PYTHONDONTWRITEBYTECODE=1 \
  PYTHONUNBUFFERED=1 \
  PYTHONPATH=/app

WORKDIR /app

RUN apt-get update \
  && apt-get install -y --no-install-recommends build-essential curl \
  && rm -rf /var/lib/apt/lists/*

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY app ./app
COPY data ./data
COPY scripts ./scripts
COPY ui ./ui
COPY .env.example ./
COPY README.md ./README.md

EXPOSE 8000 7860

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
