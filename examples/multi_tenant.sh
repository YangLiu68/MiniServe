#!/usr/bin/env bash
# Two tenants with their own API keys and a deliberately small batch (4 sequences) so that
# queueing — and therefore fairness — is visible. Alice floods the server with 12 requests;
# Bob then sends one. Compare Bob's time-to-first-token under both policies:
#
#   MINISERVE_SCHEDULING_POLICY=fcfs examples/multi_tenant.sh   # Bob waits behind Alice's backlog
#   MINISERVE_SCHEDULING_POLICY=fair examples/multi_tenant.sh   # Bob is admitted next
#
# Uses the toy backend by default (no download); watch it live at http://127.0.0.1:$PORT/dashboard
set -euo pipefail
PORT=${MINISERVE_PORT:-8000}
export MINISERVE_PORT=$PORT
export MINISERVE_API_KEYS="alice:key-alice,bob:key-bob"
export MINISERVE_SCHEDULING_POLICY=${MINISERVE_SCHEDULING_POLICY:-fair}
export MINISERVE_MAX_NUM_SEQS=4
export MINISERVE_BACKEND=${MINISERVE_BACKEND:-toy}
export MINISERVE_TOY_STEP_OVERHEAD_MS=${MINISERVE_TOY_STEP_OVERHEAD_MS:-30}

# The repo's virtualenv binary when the venv is not activated (no `uv run` wrapper, so the
# EXIT trap's kill reaches the server itself).
MINISERVE_BIN=$(command -v miniserve || echo "$(dirname "$0")/../.venv/bin/miniserve")
"$MINISERVE_BIN" >/dev/null 2>&1 &
SERVER=$!
trap 'kill $SERVER 2>/dev/null' EXIT
for _ in $(seq 1 120); do
  curl -fs "localhost:$PORT/readyz" >/dev/null && break
  kill -0 $SERVER 2>/dev/null || { echo "server exited (port $PORT in use?)"; exit 1; }
  sleep 0.5
done

ask() {  # name key prompt
  curl -s "localhost:$PORT/v1/completions" -H "Authorization: Bearer $2" \
    -H 'Content-Type: application/json' -d "{\"prompt\": \"$3\", \"max_tokens\": 64}" |
    python3 -c 'import json, sys
t = json.load(sys.stdin)["miniserve"]["timings"]
print("%8s  queue %8.0f ms   ttft %8.0f ms" % (sys.argv[1], t["queue_ms"], t["ttft_ms"]))' "$1"
}

echo "policy: $MINISERVE_SCHEDULING_POLICY, 4 running slots"
pids=()
for i in $(seq 1 12); do
  ask "alice$i" key-alice "alice request $i: lorem ipsum dolor sit amet consectetur adipiscing" &
  pids+=($!)
done
sleep 0.3
ask "BOB" key-bob "bob: a short question that should not wait for all of alice's work" &
pids+=($!)
wait "${pids[@]}"   # only the clients; the server keeps running until the EXIT trap
