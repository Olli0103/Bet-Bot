#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

PY="$ROOT/.venv/bin/python"
LOG="$ROOT/logs/reddit_sentiment_15m.log"
PIDFILE="$ROOT/run/reddit_sentiment_daemon.pid"
mkdir -p "$ROOT/run" "$ROOT/logs"

if [[ -f "$PIDFILE" ]]; then
  oldpid=$(cat "$PIDFILE" || true)
  if [[ -n "${oldpid}" ]] && kill -0 "$oldpid" 2>/dev/null; then
    echo "daemon already running pid=$oldpid"
    exit 0
  fi
fi

echo $$ > "$PIDFILE"
trap 'rm -f "$PIDFILE"' EXIT

while true; do
  ts=$(date -u +"%Y-%m-%dT%H:%M:%SZ")
  out=$($PY - <<'PY'
from src.integrations.reddit_sentiment_pipeline import run_sentiment_pipeline
res = run_sentiment_pipeline(max_items_per_run=30)
feeds = res.get('feeds', {})
processed = int(res.get('processed', 0))
t = res.get('tier_counts', {})
w = res.get('weighted', {})
n200 = int(feeds.get('http_200', 0))
n304 = int(feeds.get('http_304', 0))
other = int(feeds.get('other', 0))
total = n200 + n304 + other
ratio304 = (100.0 * n304 / total) if total else 0.0
print(
    f"reddit_sentiment_15m: processed={processed} req200={n200} req304={n304} ratio304={ratio304:.1f}% "
    f"delta={int(feeds.get('delta_items', 0))} "
    f"tiers(core={int(t.get('core',0))},fact={int(t.get('fact_only',0))},noise={int(t.get('high_noise',0))}) "
    f"sentiment_delta={float(w.get('sentiment_delta',0.0)):.4f} hype={float(w.get('public_hype_index',0.0)):.4f}"
)
PY
  )
  echo "$ts $out" | tee -a "$LOG"
  sleep 900
done
