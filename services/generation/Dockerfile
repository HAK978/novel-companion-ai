FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

# Locked dependencies first, so code changes don't reinstall them
COPY services/generation/requirements.txt .
RUN pip install -r requirements.txt

# --chmod: host files may be owner/group-only (a umask of 007 makes them 660), and the
# code must be readable by the non-root user the container runs as
COPY --chmod=755 services/generation/ /app/

RUN useradd --create-home --home-dir /home/app --shell /usr/sbin/nologin app
USER app

EXPOSE 8003
HEALTHCHECK --interval=10s --timeout=5s --start-period=20s --retries=5 \
    CMD ["python", "-c", "import urllib.request; urllib.request.urlopen('http://localhost:8003/health', timeout=4)"]

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8003"]
