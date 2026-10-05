# One-shot job: apply pending migrations, record them, exit. Services start after it succeeds.
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app
COPY migrations/requirements.txt /app/migrations/requirements.txt
RUN pip install -r /app/migrations/requirements.txt

# --chmod: host files may be owner/group-only; the non-root user must be able to read these
COPY --chmod=755 scripts/migrate.py /app/scripts/migrate.py
COPY --chmod=755 migrations/*.sql /app/migrations/

RUN useradd --shell /usr/sbin/nologin app
USER app

CMD ["python", "/app/scripts/migrate.py"]
