#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   scripts/db_backup.sh daily
#   scripts/db_backup.sh weekly

MODE="${1:-daily}"
if [[ "$MODE" != "daily" && "$MODE" != "weekly" ]]; then
  echo "Usage: $0 [daily|weekly]" >&2
  exit 1
fi

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

# Read only required vars from .env (avoid executing arbitrary shell lines)
if [[ -z "${POSTGRES_DSN:-}" && -f .env ]]; then
  POSTGRES_DSN="$(grep -E '^POSTGRES_DSN=' .env | tail -n 1 | cut -d'=' -f2- || true)"
fi
if [[ -z "${PG_DUMP_BIN:-}" && -f .env ]]; then
  PG_DUMP_BIN="$(grep -E '^PG_DUMP_BIN=' .env | tail -n 1 | cut -d'=' -f2- || true)"
fi

POSTGRES_DSN="${POSTGRES_DSN:-postgresql+psycopg://postgres:postgres@localhost:5432/signalbot}"
# pg_dump understands postgresql:// URI, not sqlalchemy's +psycopg form
DSN="${POSTGRES_DSN/postgresql+psycopg:/postgresql:}"

PG_DUMP_BIN="${PG_DUMP_BIN:-}"
if [[ -z "$PG_DUMP_BIN" ]]; then
  if command -v pg_dump >/dev/null 2>&1; then
    PG_DUMP_BIN="$(command -v pg_dump)"
  elif [[ -x /opt/homebrew/opt/libpq/bin/pg_dump ]]; then
    PG_DUMP_BIN="/opt/homebrew/opt/libpq/bin/pg_dump"
  elif [[ -x /usr/local/opt/libpq/bin/pg_dump ]]; then
    PG_DUMP_BIN="/usr/local/opt/libpq/bin/pg_dump"
  else
    echo "pg_dump not found. Set PG_DUMP_BIN or install libpq." >&2
    exit 1
  fi
fi

BACKUP_BASE="$ROOT/backups"
OUT_DIR="$BACKUP_BASE/$MODE"
mkdir -p "$OUT_DIR"

STAMP="$(date -u +"%Y%m%d-%H%M%S")"
DB_NAME="$(echo "$DSN" | sed -E 's#.*/([^/?]+).*#\1#')"
OUT_FILE="$OUT_DIR/${MODE}_${DB_NAME}_${STAMP}.dump"

"$PG_DUMP_BIN" \
  --format=custom \
  --compress=9 \
  --no-owner \
  --no-privileges \
  --dbname="$DSN" \
  --file="$OUT_FILE"

if command -v shasum >/dev/null 2>&1; then
  shasum -a 256 "$OUT_FILE" > "$OUT_FILE.sha256"
elif command -v sha256sum >/dev/null 2>&1; then
  sha256sum "$OUT_FILE" > "$OUT_FILE.sha256"
fi

# Retention policies
if [[ "$MODE" == "daily" ]]; then
  RETENTION_DAYS=7
else
  RETENTION_DAYS=28
fi

find "$OUT_DIR" -type f -name "*.dump" -mtime +"$RETENTION_DAYS" -delete || true
find "$OUT_DIR" -type f -name "*.sha256" -mtime +"$RETENTION_DAYS" -delete || true

echo "backup_ok mode=$MODE file=$OUT_FILE retention_days=$RETENTION_DAYS"
