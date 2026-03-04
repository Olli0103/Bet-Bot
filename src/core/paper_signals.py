"""Paper Signal Capture: records ALL signal candidates for continuous learning.

Every signal candidate is persisted as a paper record, regardless of whether
it passes trading gates. This enables:
- Learning from signals that were rejected (confidence, EV, stake caps)
- Tracking model accuracy across all predictions, not just placed bets
- Comparing paper PnL vs trading PnL for gate tuning
"""
from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from src.core.settings import settings
from src.data.redis_cache import cache

log = logging.getLogger(__name__)

PAPER_SIGNALS_CACHE_KEY = "paper:signals:latest"


class PaperSignalRecord:
    """Lightweight record for a paper signal candidate."""

    def __init__(
        self,
        event_id: str,
        sport: str,
        market: str,
        selection: str,
        bookmaker_odds: float,
        model_probability: float,
        expected_value: float,
        recommended_stake: float,
        confidence_gate_passed: bool,
        reject_reason: str = "",
        signal_mode: str = "PAPER_ONLY",
        meta: Optional[Dict[str, Any]] = None,
    ):
        self.event_id = event_id
        self.sport = sport
        self.market = market
        self.selection = selection
        self.bookmaker_odds = bookmaker_odds
        self.model_probability = model_probability
        self.expected_value = expected_value
        self.recommended_stake = recommended_stake
        self.confidence_gate_passed = confidence_gate_passed
        self.reject_reason = reject_reason
        self.signal_mode = signal_mode
        self.meta = meta or {}
        self.timestamp = datetime.now(timezone.utc).isoformat()

    def to_dict(self) -> Dict[str, Any]:
        return {
            "event_id": self.event_id,
            "sport": self.sport,
            "market": self.market,
            "selection": self.selection,
            "bookmaker_odds": self.bookmaker_odds,
            "model_probability": self.model_probability,
            "expected_value": self.expected_value,
            "recommended_stake": self.recommended_stake,
            "confidence_gate_passed": self.confidence_gate_passed,
            "reject_reason": self.reject_reason,
            "signal_mode": self.signal_mode,
            "timestamp": self.timestamp,
        }


def capture_paper_signal(
    event_id: str,
    sport: str,
    market: str,
    selection: str,
    bookmaker_odds: float,
    model_probability: float,
    expected_value: float,
    recommended_stake: float,
    confidence_gate_passed: bool,
    reject_reason: str = "",
    features: Optional[Dict[str, Any]] = None,
) -> PaperSignalRecord:
    """Capture a paper signal record for learning purposes.

    Called for EVERY signal candidate, including those rejected by
    trading gates. Persists to database for later grading.
    """
    signal_mode = "PLAYABLE" if (confidence_gate_passed and recommended_stake > 0) else "PAPER_ONLY"

    record = PaperSignalRecord(
        event_id=event_id,
        sport=sport,
        market=market,
        selection=selection,
        bookmaker_odds=bookmaker_odds,
        model_probability=model_probability,
        expected_value=expected_value,
        recommended_stake=recommended_stake,
        confidence_gate_passed=confidence_gate_passed,
        reject_reason=reject_reason,
        signal_mode=signal_mode,
        meta=features,
    )

    # Persist to database if learning mode is enabled
    if settings.learning_capture_all_signals:
        try:
            _persist_paper_signal(record, features)
        except Exception as exc:
            log.warning("paper_signal_persist_failed: %s %s: %s",
                        event_id, selection, str(exc)[:100])

    return record


def _persist_paper_signal(record: PaperSignalRecord, features: Optional[Dict[str, Any]] = None):
    """Write paper signal to placed_bets with source tag for separation."""
    from src.data.postgres import SessionLocal
    from src.data.models import PlacedBet
    from src.core.ghost_trading import _safe_meta

    features = features or {}

    # Build notes with signal mode and reject reason
    notes_parts = [f"signal_mode={record.signal_mode}"]
    if record.reject_reason:
        notes_parts.append(f"reject_reason={record.reject_reason}")
    notes_parts.append("source=paper_signal")

    meta = _safe_meta(features)
    meta["signal_mode"] = record.signal_mode
    meta["reject_reason"] = record.reject_reason
    meta["is_paper"] = True

    try:
        with SessionLocal() as db:
            # Check for duplicates
            from sqlalchemy import select
            exists = db.scalar(
                select(PlacedBet.id).where(
                    PlacedBet.event_id == record.event_id,
                    PlacedBet.selection == record.selection,
                    PlacedBet.market == record.market,
                )
            )
            if exists:
                return

            bet = PlacedBet(
                event_id=record.event_id,
                sport=record.sport,
                market=record.market,
                selection=record.selection,
                odds=record.bookmaker_odds,
                odds_open=record.bookmaker_odds,
                odds_close=record.bookmaker_odds,
                clv=float(features.get("clv", 0.0)),
                form_winrate_l5=float(features.get("form_winrate_l5", 0.5)),
                form_games_l5=int(features.get("form_games_l5", 0)),
                stake=record.recommended_stake if record.signal_mode == "PLAYABLE" else 0.0,
                status="open",
                sharp_implied_prob=float(features.get("sharp_implied_prob", 0.0)),
                sharp_vig=float(features.get("sharp_vig", 0.0)),
                sentiment_delta=float(features.get("sentiment_delta", 0.0)),
                injury_delta=float(features.get("injury_delta", 0.0)),
                meta_features=meta,
                notes="; ".join(notes_parts),
            )
            db.add(bet)
            db.commit()
    except Exception as exc:
        # Don't let paper signal persistence crash the pipeline
        log.debug("paper_signal_db_error: %s", str(exc)[:100])


def get_paper_signal_stats() -> Dict[str, Any]:
    """Get stats about paper signals for monitoring."""
    from src.data.postgres import SessionLocal
    from src.data.models import PlacedBet
    from sqlalchemy import select, func

    try:
        with SessionLocal() as db:
            total = db.scalar(
                select(func.count()).select_from(PlacedBet).where(
                    PlacedBet.notes.like("%source=paper_signal%")
                )
            ) or 0
            playable = db.scalar(
                select(func.count()).select_from(PlacedBet).where(
                    PlacedBet.notes.like("%signal_mode=PLAYABLE%"),
                    PlacedBet.notes.like("%source=paper_signal%"),
                )
            ) or 0
            graded = db.scalar(
                select(func.count()).select_from(PlacedBet).where(
                    PlacedBet.notes.like("%source=paper_signal%"),
                    PlacedBet.status.in_(["won", "lost"]),
                )
            ) or 0
            return {
                "total_paper_signals": total,
                "playable": playable,
                "paper_only": total - playable,
                "graded": graded,
                "pending": total - graded,
            }
    except Exception:
        return {"total_paper_signals": 0, "playable": 0, "paper_only": 0, "graded": 0, "pending": 0}
