#!/usr/bin/env bash
# Runs bench/suites/external-ollama.json against a *private* Ollama instance (port 11435, our
# own settings) so a shared/system Ollama on 11434 is neither used nor disturbed.
#   bench/run_ollama_comparison.sh results/external-ollama/run1
set -euo pipefail
OUT=${1:-results/external-ollama/latest}
export OLLAMA_HOST=127.0.0.1:11435 OLLAMA_NUM_PARALLEL=16 OLLAMA_MAX_LOADED_MODELS=1 OLLAMA_KEEP_ALIVE=30m
mkdir -p "$OUT"
ollama serve > "$OUT/ollama.log" 2>&1 &
OLLAMA_PID=$!
trap 'kill $OLLAMA_PID 2>/dev/null' EXIT
for _ in $(seq 1 60); do curl -fs http://127.0.0.1:11435/api/version >/dev/null && break; sleep 0.5; done
ollama --version >> "$OUT/ollama.log" 2>&1 || true
# load the model once so the first measured request does not include loading
curl -fs http://127.0.0.1:11435/api/generate -d '{"model":"smollm2:135m-instruct-fp16","prompt":"hi","options":{"num_predict":4}}' >/dev/null
uv run --no-sync python -u -m miniserve.bench suite bench/suites/external-ollama.json --out "$OUT"
