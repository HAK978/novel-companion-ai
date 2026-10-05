FROM python:3.12-slim
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 PIP_DISABLE_PIP_VERSION_CHECK=1
WORKDIR /app
# same locked web stack as the generation service
COPY services/generation/requirements.txt .
RUN pip install -r requirements.txt
COPY --chmod=755 tests/stub_llm/app.py /app/app.py
RUN useradd --shell /usr/sbin/nologin app
USER app
HEALTHCHECK --interval=5s --timeout=3s --retries=10 \
    CMD ["python", "-c", "import urllib.request; urllib.request.urlopen('http://localhost:8000/health', timeout=2)"]
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
