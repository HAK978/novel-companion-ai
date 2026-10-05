#!/usr/bin/env bash
# Serve the language model with the vllm compose service (gpu profile) on an idle GPU.
# GPUs here are shared: refuse rather than take one somebody else is using.
#
#   scripts/start_vllm.sh            pick an idle GPU
#   VLLM_GPU=3 scripts/start_vllm.sh use GPU 3
#   docker compose stop vllm         release the GPU
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

if [ -z "${VLLM_GPU:-}" ]; then
  # awk reads every line: exiting at the first match would SIGPIPE nvidia-smi, and with
  # pipefail and set -e that ends this script silently
  VLLM_GPU=$(nvidia-smi --query-gpu=index,memory.used,utilization.gpu \
               --format=csv,noheader,nounits \
             | awk -F', ' '$2 < 1000 && $3 < 5 && !found { print $1; found = 1 }')
fi
if [ -z "$VLLM_GPU" ]; then
  echo "No idle GPU (under 1 GB in use and under 5% busy). Set VLLM_GPU to choose one." >&2
  exit 1
fi

echo "Serving the model on GPU $VLLM_GPU"
VLLM_GPU=$VLLM_GPU docker compose --profile gpu up -d vllm
echo "Loading takes a few minutes. Follow it with: docker compose logs -f vllm"
