#!/usr/bin/env bash
# Bring up the full Novel Companion AI stack after a reboot.
# Usage: bash scripts/start_services.sh
set -e

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV="$HOME/miniconda3/bin"
LOGS="$ROOT/logs"
mkdir -p "$LOGS"

echo "[1/4] Docker infra (postgres, redis, chromadb)..."
docker compose -f "$ROOT/docker-compose.yml" up -d postgres redis chromadb

echo "[2/4] Waiting for postgres to be healthy..."
until docker exec novel-companion-ai-postgres-1 pg_isready -U novel -d novel_companion -q 2>/dev/null; do
  sleep 1
done

start_service () {
  local name=$1 port=$2 dir=$3
  if curl -s -m 2 "http://localhost:$port/health" > /dev/null 2>&1; then
    echo "  $name already running on :$port"
    return
  fi
  echo "  starting $name on :$port"
  (cd "$ROOT/services/$dir" && nohup "$VENV/uvicorn" main:app --host 0.0.0.0 --port "$port" \
    > "$LOGS/$name.log" 2>&1 &)
}

echo "[3/4] App services..."

# vLLM serves the LLM (OpenAI-compatible, port 8004). Must be up BEFORE the
# generation service starts, or generation falls back to loading its own
# 24GB transformers copy and the two fight over GPU memory.
VLLM_GPU="${VLLM_GPU:-0}"
if ! curl -s -m 3 "http://localhost:8004/v1/models" > /dev/null 2>&1; then
  echo "  starting vllm on :8004 (GPU $VLLM_GPU, loads Mistral Nemo, takes minutes)"
  (cd "$ROOT" && CUDA_VISIBLE_DEVICES="$VLLM_GPU" nohup "$VENV/vllm" serve \
    mistralai/Mistral-Nemo-Instruct-2407 --port 8004 \
    --max-model-len 16384 --gpu-memory-utilization 0.65 \
    > "$LOGS/vllm.log" 2>&1 &)
  echo "  waiting for vllm to come up..."
  for i in $(seq 1 120); do
    curl -s -m 3 "http://localhost:8004/v1/models" > /dev/null 2>&1 && break
    sleep 5
  done
  curl -s -m 3 "http://localhost:8004/v1/models" > /dev/null 2>&1 \
    && echo "  vllm ready" \
    || echo "  WARNING: vllm not ready after 10min; generation will fall back to transformers (check $LOGS/vllm.log)"
else
  echo "  vllm already running on :8004"
fi

start_service generation 8003 generation
start_service retrieval  8002 retrieval
start_service ingestion  8001 ingestion
start_service gateway    8000 gateway

if ! pgrep -f "celery.*ingestion" > /dev/null; then
  echo "  starting celery worker"
  (cd "$ROOT/services/ingestion" && nohup "$VENV/celery" -A tasks.celery_app worker \
    --loglevel=info --concurrency=2 > "$LOGS/celery.log" 2>&1 &)
else
  echo "  celery worker already running"
fi

echo "[4/4] Waiting for services (generation loads the LLM, be patient)..."
for i in $(seq 1 60); do
  ok=0
  for port in 8000 8001 8002; do
    curl -s -m 2 "http://localhost:$port/health" > /dev/null 2>&1 && ok=$((ok+1))
  done
  [ "$ok" -eq 3 ] && break
  sleep 2
done

echo
echo "Health:"
for port in 8000 8001 8002 8003; do
  printf "  :%s  %s\n" "$port" "$(curl -s -m 3 http://localhost:$port/health | head -c 200 || echo 'not ready yet')"
done
echo
echo "Generation LLM is served by vLLM on :8004; the :8003 service proxies to it."
echo "Frontend: cd frontend && npm run dev"
