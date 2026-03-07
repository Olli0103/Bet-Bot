#!/usr/bin/env bash
set -euo pipefail

BOT_DIR="/Users/olli/.openclaw/workspace/Bet-Bot"
PY="$BOT_DIR/.venv/bin/python"
LOG_DIR="$BOT_DIR/logs"
WD_LOG="$LOG_DIR/watchdog.log"

mkdir -p "$LOG_DIR"

ts() { date '+%Y-%m-%d %H:%M:%S %z'; }

# Is bot already running?
if pgrep -f "$PY -m src.bot.app" >/dev/null 2>&1; then
  echo "[$(ts)] OK: bot process running" >> "$WD_LOG"
  exit 0
fi

# Fallback match in case process argv differs
if pgrep -f "src.bot.app" >/dev/null 2>&1; then
  echo "[$(ts)] OK: bot process running (fallback match)" >> "$WD_LOG"
  exit 0
fi

# Start bot
if [[ ! -x "$PY" ]]; then
  echo "[$(ts)] ERROR: python not found at $PY" >> "$WD_LOG"
  exit 1
fi

cd "$BOT_DIR"
BOT_LOG="$LOG_DIR/bot_$(date +%Y%m%d_%H%M%S).log"
nohup "$PY" -m src.bot.app >> "$BOT_LOG" 2>&1 &

echo "[$(ts)] STARTED: bot launched, log=$BOT_LOG" >> "$WD_LOG"
