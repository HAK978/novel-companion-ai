FROM python:3.11-slim

WORKDIR /app

COPY services/retrieval/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY shared/ /app/shared/
COPY services/retrieval/ /app/

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8002"]
