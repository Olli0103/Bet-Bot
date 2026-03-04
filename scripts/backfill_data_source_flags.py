#!/usr/bin/env python3
"""Backfill ``is_training_data`` and ``data_source`` columns for existing rows.

Idempotent — safe to run multiple times.

Recognition heuristics for historical imports:
  1. notes contains 'source=historical_import' (explicit tag)
  2. notes contains 'source=paper_signal' (paper signals)
  3. stake == 1.0 AND notes IS NULL AND status IN ('won','lost')
     AND no live-trade metadata (no sentiment_delta, no meta_features
     with 'signal_mode') — likely bulk-imported rows with flat stake=1

Usage:
  python scripts/backfill_data_source_flags.py --dry-run     # preview
  python scripts/backfill_data_source_flags.py --force        # apply
"""
from __future__ import annotations

import argparse
import logging
import sys

from sqlalchemy import select, update, func, and_, or_, text

# Ensure project root is importable
sys.path.insert(0, ".")

from src.data.models import PlacedBet
from src.data.postgres import SessionLocal

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)


def _count_by_source(db) -> dict:
    """Count rows grouped by current data_source value."""
    rows = db.execute(
        select(PlacedBet.data_source, func.count(PlacedBet.id))
        .group_by(PlacedBet.data_source)
    ).all()
    return {r[0]: r[1] for r in rows}


def _count_training(db) -> int:
    return db.scalar(
        select(func.count()).select_from(PlacedBet).where(
            PlacedBet.is_training_data.is_(True)
        )
    ) or 0


def backfill(dry_run: bool = True) -> dict:
    """Mark existing rows with correct data_source and is_training_data flags."""
    stats = {
        "historical_import_by_notes": 0,
        "paper_signal_by_notes": 0,
        "historical_import_by_heuristic": 0,
        "already_correct": 0,
        "live_trade_default": 0,
    }

    with SessionLocal() as db:
        log.info("Before backfill:")
        for src, cnt in _count_by_source(db).items():
            log.info("  data_source='%s': %d rows", src, cnt)
        log.info("  is_training_data=True: %d rows", _count_training(db))

        # 1) Mark rows with notes containing 'source=historical_import'
        q1 = (
            update(PlacedBet)
            .where(
                PlacedBet.notes.isnot(None),
                PlacedBet.notes.like("%source=historical_import%"),
                PlacedBet.data_source != "historical_import",
            )
            .values(is_training_data=True, data_source="historical_import")
        )
        if not dry_run:
            r1 = db.execute(q1)
            stats["historical_import_by_notes"] = r1.rowcount
        else:
            cnt = db.scalar(
                select(func.count()).select_from(PlacedBet).where(
                    PlacedBet.notes.isnot(None),
                    PlacedBet.notes.like("%source=historical_import%"),
                    PlacedBet.data_source != "historical_import",
                )
            ) or 0
            stats["historical_import_by_notes"] = cnt

        # 2) Mark rows with notes containing 'source=paper_signal'
        q2 = (
            update(PlacedBet)
            .where(
                PlacedBet.notes.isnot(None),
                PlacedBet.notes.like("%source=paper_signal%"),
                PlacedBet.data_source != "paper_signal",
            )
            .values(is_training_data=False, data_source="paper_signal")
        )
        if not dry_run:
            r2 = db.execute(q2)
            stats["paper_signal_by_notes"] = r2.rowcount
        else:
            cnt = db.scalar(
                select(func.count()).select_from(PlacedBet).where(
                    PlacedBet.notes.isnot(None),
                    PlacedBet.notes.like("%source=paper_signal%"),
                    PlacedBet.data_source != "paper_signal",
                )
            ) or 0
            stats["paper_signal_by_notes"] = cnt

        # 3) Heuristic: stake=1 + status in won/lost + no notes = likely historical import
        #    These are the bulk-imported rows from CSV data.
        q3 = (
            update(PlacedBet)
            .where(
                PlacedBet.stake == 1.0,
                PlacedBet.status.in_(["won", "lost"]),
                or_(PlacedBet.notes.is_(None), PlacedBet.notes == ""),
                PlacedBet.data_source == "live_trade",  # only rows not yet tagged
            )
            .values(is_training_data=True, data_source="historical_import")
        )
        if not dry_run:
            r3 = db.execute(q3)
            stats["historical_import_by_heuristic"] = r3.rowcount
        else:
            cnt = db.scalar(
                select(func.count()).select_from(PlacedBet).where(
                    PlacedBet.stake == 1.0,
                    PlacedBet.status.in_(["won", "lost"]),
                    or_(PlacedBet.notes.is_(None), PlacedBet.notes == ""),
                    PlacedBet.data_source == "live_trade",
                )
            ) or 0
            stats["historical_import_by_heuristic"] = cnt

        if not dry_run:
            db.commit()
            log.info("After backfill:")
            for src, cnt in _count_by_source(db).items():
                log.info("  data_source='%s': %d rows", src, cnt)
            log.info("  is_training_data=True: %d rows", _count_training(db))

    total_marked = (
        stats["historical_import_by_notes"]
        + stats["paper_signal_by_notes"]
        + stats["historical_import_by_heuristic"]
    )
    stats["total_marked"] = total_marked

    mode = "DRY-RUN" if dry_run else "APPLIED"
    log.info("")
    log.info("=== Backfill Summary (%s) ===", mode)
    log.info("  Historical import (by notes):      %d", stats["historical_import_by_notes"])
    log.info("  Paper signals (by notes):           %d", stats["paper_signal_by_notes"])
    log.info("  Historical import (by heuristic):   %d", stats["historical_import_by_heuristic"])
    log.info("  Total rows marked:                  %d", total_marked)

    return stats


def main():
    parser = argparse.ArgumentParser(description="Backfill data_source flags on placed_bets")
    parser.add_argument("--dry-run", action="store_true", default=True,
                        help="Preview changes without applying (default)")
    parser.add_argument("--force", action="store_true",
                        help="Apply changes to database")
    args = parser.parse_args()

    dry_run = not args.force
    backfill(dry_run=dry_run)


if __name__ == "__main__":
    main()
