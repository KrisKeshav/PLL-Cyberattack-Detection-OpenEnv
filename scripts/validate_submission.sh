#!/usr/bin/env bash
# Pre-submission checks for the PLL Cyber-Guard OpenEnv environment.
#
#   ./scripts/validate_submission.sh [SPACE_URL]
#
# 1. unit tests            2. openenv validate (structure + runtime)
# 3. docker build + run    4. inference.py completes and prints valid logs
# 5. (optional) the live HF Space answers /health and POST /reset
set -uo pipefail
cd "$(dirname "$0")/.."
SPACE_URL="${1:-}"
PORT="${PORT:-7861}"
fail=0
ok()   { printf '  \033[32mPASS\033[0m %s\n' "$1"; }
bad()  { printf '  \033[31mFAIL\033[0m %s\n' "$1"; fail=1; }

echo "== 1. unit tests"
python -m pytest -q >/tmp/pll_tests.log 2>&1 && ok "pytest" || { bad "pytest (see /tmp/pll_tests.log)"; tail -5 /tmp/pll_tests.log; }

echo "== 2. openenv validate"
openenv validate . >/dev/null 2>&1 && ok "structure" || bad "structure"
python -m uvicorn server.app:app --port "$PORT" >/tmp/pll_server.log 2>&1 &
SRV=$!
for _ in $(seq 1 30); do curl -sf "localhost:$PORT/health" >/dev/null && break; sleep 1; done
openenv validate --url "http://localhost:$PORT" >/tmp/pll_runtime.json 2>/dev/null
python -c "import json,sys; sys.exit(0 if json.load(sys.stdin)['passed'] else 1)" </tmp/pll_runtime.json \
  && ok "runtime contract" || bad "runtime contract (see /tmp/pll_runtime.json)"

echo "== 3. inference.py"
ENV_URL="http://localhost:$PORT" EPISODES_PER_TASK="${EPISODES_PER_TASK:-3}" python inference.py >/tmp/pll_inf.log 2>/tmp/pll_inf.err
rc=$?
kill $SRV 2>/dev/null
starts=$(grep -c '^\[START\] ' /tmp/pll_inf.log); ends=$(grep -c '^\[END\] success=true ' /tmp/pll_inf.log)
bad_lines=$(grep -E '^\[' /tmp/pll_inf.log | grep -vcE '^\[(START|STEP|END)\] ')
if [ $rc -eq 0 ] && [ "$starts" -ge 3 ] && [ "$starts" -eq "$ends" ] && [ "$bad_lines" -eq 0 ]; then
  ok "inference ($starts episodes) — $(grep '^overall' /tmp/pll_inf.log)"
else
  bad "inference rc=$rc starts=$starts successful_ends=$ends (see /tmp/pll_inf.log /tmp/pll_inf.err)"
fi

echo "== 4. docker"
if command -v docker >/dev/null && docker info >/dev/null 2>&1; then
  if docker build -q -t pll-cyberguard:validate . >/dev/null; then
    ok "docker build"
    cid=$(docker run -d -p 17860:7860 pll-cyberguard:validate)
    for _ in $(seq 1 40); do curl -sf localhost:17860/health >/dev/null && break; sleep 1; done
    curl -sf -X POST localhost:17860/reset -H 'content-type: application/json' -d '{"task_id":0,"seed":1}' >/dev/null \
      && ok "docker run + POST /reset" || bad "docker run + POST /reset"
    docker rm -f "$cid" >/dev/null
  else bad "docker build"; fi
else
  echo "  SKIP docker (daemon not available)"
fi

if [ -n "$SPACE_URL" ]; then
  echo "== 5. HF Space $SPACE_URL"
  curl -sf "$SPACE_URL/health" >/dev/null && ok "/health" || bad "/health"
  code=$(curl -s -o /dev/null -w '%{http_code}' -X POST "$SPACE_URL/reset" -H 'content-type: application/json' -d '{}')
  [ "$code" = "200" ] && ok "POST /reset → 200" || bad "POST /reset → $code"
fi

[ $fail -eq 0 ] && echo "ALL CHECKS PASSED" || echo "SOME CHECKS FAILED"
exit $fail
