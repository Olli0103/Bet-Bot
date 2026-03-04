#!/usr/bin/env python3
"""Backfill owner_chat_id on existing placed_bets rows.

Idempotent — safe to run multiple times. Assigns the primary
TELEGRAM_CHAT_ID to all rows where owner_chat_id is NULL.

Usage:
    python scripts/backfill_owner_chat_id.py --dry-run   # preview
    python scripts/backfill_owner_chat_id.py              # execute
"""
from __future__ import annotations

import argparse
import sys

from dotenv import load_dotenv
load_dotenv(override=True)

from sqlalchemy import text
from src.core.settings import settings
from src.data.postgres import engine


def main():
    parser = argparse.ArgumentParser(description="Backfill owner_chat_id on placed_bets")
    parser.add_argument("--dry-run", action="store_true", help="Preview only, no changes")
    args = parser.parse_args()

    primary_chat_id = settings.telegram_chat_id
    if not primary_chat_id:
        print("ERROR: TELEGRAM_CHAT_ID not set. Cannot determine default owner.")
        sys.exit(1)

    with engine.connect() as conn:
        # Count rows needing backfill
        result = conn.execute(text(
            "SELECT COUNT(*) FROM placed_bets WHERE owner_chat_id IS NULL"
        ))
        null_count = result.scalar()

        print(f"Found {null_count} rows with owner_chat_id = NULL")
        print(f"Default owner: {primary_chat_id}")

        if null_count == 0:
            print("Nothing to backfill.")
            return

        if args.dry_run:
            print(f"DRY RUN: Would set owner_chat_id = '{primary_chat_id}' on {null_count} rows")
            return

        # Execute backfill
        conn.execute(text(
            "UPDATE placed_bets SET owner_chat_id = :owner WHERE owner_chat_id IS NULL"
        ), {"owner": primary_chat_id})
        conn.commit()

        print(f"Backfilled {null_count} rows with owner_chat_id = '{primary_chat_id}'")


if __name__ == "__main__":
    main()
