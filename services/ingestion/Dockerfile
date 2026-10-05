FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    ANONYMIZED_TELEMETRY=False \
    HOME=/home/app

WORKDIR /app

COPY services/ingestion/requirements.txt .
RUN pip install -r requirements.txt

# Bake the pinned embedding model into the image. Otherwise every fresh container downloads
# it on first use, into a filesystem thrown away on the next recreate. vector_store.py is the
# single place the model is named, so it alone is copied here: the layer stays cached across
# ordinary code changes. Readable by any user, because the ingestion worker runs as the host
# user in order to read mounted novel files.
RUN useradd --create-home --home-dir /home/app --shell /usr/sbin/nologin app
COPY --chmod=755 services/ingestion/vector_store.py /app/vector_store.py
RUN python -c "from vector_store import embedding_function; embedding_function()(['warm up'])" \
 && chmod -R a+rX /home/app

# --chmod: host files may be owner/group-only (a umask of 007 makes them 660), and the
# code must be readable by the non-root user the container runs as
COPY --chmod=755 services/ingestion/ /app/
USER app

EXPOSE 8001
HEALTHCHECK --interval=10s --timeout=5s --start-period=30s --retries=5 \
    CMD ["python", "-c", "import urllib.request; urllib.request.urlopen('http://localhost:8001/health', timeout=4)"]

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8001"]
