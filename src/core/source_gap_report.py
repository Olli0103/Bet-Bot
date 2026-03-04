"""Source Gap Report: tracks where markets are lost at each pipeline stage.

Generates both JSON (machine-readable) and Markdown (human-readable)
reports explaining the drop from raw fetched events to final displayed signals.
"""
from __future__ import annotations

import json
import logging
import os
import time
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from src.data.redis_cache import cache

log = logging.getLogger(__name__)

REPORT_CACHE_KEY = "source_gap:latest_report"
ARTIFACTS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "artifacts")


class StageDropTracker:
    """Tracks events/signals through each pipeline stage."""

    def __init__(self):
        self.per_sport: Dict[str, Dict[str, Any]] = {}
        self.global_counts = {
            "fetched_keys": 0,
            "request_status_counts": {},
            "raw_events_count": 0,
            "parsed_events_count": 0,
            "with_tipico_count": 0,
            "with_sharp_count": 0,
            "with_tipico_and_sharp_count": 0,
            "dropped_no_bookmaker_overlap": 0,
            "dropped_outright_filter": 0,
            "dropped_time_window": 0,
            "dropped_invalid_market": 0,
            "dropped_confidence": 0,
            "dropped_ev": 0,
            "dropped_stake_zero": 0,
            "signals_generated": 0,
            "signals_playable": 0,
            "signals_paper_only": 0,
            "final_displayed_count": 0,
        }
        self.timestamp = datetime.now(timezone.utc).isoformat()

    def init_sport(self, sport: str):
        if sport not in self.per_sport:
            self.per_sport[sport] = {
                "raw_events": 0,
                "parsed_events": 0,
                "with_tipico": 0,
                "with_sharp": 0,
                "with_both": 0,
                "dropped_no_overlap": 0,
                "dropped_time_window": 0,
                "dropped_confidence": 0,
                "dropped_ev": 0,
                "dropped_stake_zero": 0,
                "signals_generated": 0,
                "signals_playable": 0,
            }

    def record_raw_event(self, sport: str):
        self.init_sport(sport)
        self.per_sport[sport]["raw_events"] += 1
        self.global_counts["raw_events_count"] += 1

    def record_parsed_event(self, sport: str, has_tipico: bool, has_sharp: bool):
        self.init_sport(sport)
        self.per_sport[sport]["parsed_events"] += 1
        self.global_counts["parsed_events_count"] += 1
        if has_tipico:
            self.per_sport[sport]["with_tipico"] += 1
            self.global_counts["with_tipico_count"] += 1
        if has_sharp:
            self.per_sport[sport]["with_sharp"] += 1
            self.global_counts["with_sharp_count"] += 1
        if has_tipico and has_sharp:
            self.per_sport[sport]["with_both"] += 1
            self.global_counts["with_tipico_and_sharp_count"] += 1

    def record_drop(self, sport: str, reason: str):
        self.init_sport(sport)
        sport_key = f"dropped_{reason}"
        global_key = f"dropped_{reason}"
        if sport_key in self.per_sport[sport]:
            self.per_sport[sport][sport_key] += 1
        if global_key in self.global_counts:
            self.global_counts[global_key] += 1

    def record_signal(self, sport: str, playable: bool):
        self.init_sport(sport)
        self.per_sport[sport]["signals_generated"] += 1
        self.global_counts["signals_generated"] += 1
        if playable:
            self.per_sport[sport]["signals_playable"] += 1
            self.global_counts["signals_playable"] += 1
        else:
            self.global_counts["signals_paper_only"] += 1

    def set_fetch_stats(self, fetch_stats: Dict[str, Any]):
        self.global_counts["fetched_keys"] = len(fetch_stats.get("fetched_keys", []))
        self.global_counts["request_status_counts"] = fetch_stats.get("status_counts", {})
        self.global_counts["dropped_outright_filter"] = len(fetch_stats.get("skipped_outright", []))

    def set_final_count(self, count: int):
        self.global_counts["final_displayed_count"] = count

    def to_dict(self) -> Dict[str, Any]:
        return {
            "timestamp": self.timestamp,
            "global": self.global_counts,
            "per_sport": self.per_sport,
        }

    def save_report(self):
        """Save report to cache, JSON file, and markdown file."""
        report = self.to_dict()
        cache.set_json(REPORT_CACHE_KEY, report, ttl_seconds=24 * 3600)

        os.makedirs(ARTIFACTS_DIR, exist_ok=True)

        # JSON report
        json_path = os.path.join(ARTIFACTS_DIR, "source_gap_report.json")
        try:
            with open(json_path, "w") as f:
                json.dump(report, f, indent=2)
        except Exception as exc:
            log.warning("Could not write JSON report: %s", exc)

        # Markdown report
        md_path = os.path.join(os.path.dirname(ARTIFACTS_DIR), "SOURCE_GAP_REPORT.md")
        try:
            md = self._render_markdown(report)
            with open(md_path, "w") as f:
                f.write(md)
        except Exception as exc:
            log.warning("Could not write MD report: %s", exc)

        log.info("source_gap_report: saved to %s and %s", json_path, md_path)

    @staticmethod
    def _render_markdown(report: Dict[str, Any]) -> str:
        g = report["global"]
        lines = [
            "# Source Gap Report",
            "",
            f"**Generated:** {report['timestamp']}",
            "",
            "## Global Pipeline Funnel",
            "",
            "| Stage | Count |",
            "|-------|-------|",
            f"| Fetched sport keys | {g['fetched_keys']} |",
            f"| Raw events fetched | {g['raw_events_count']} |",
            f"| Parsed events | {g['parsed_events_count']} |",
            f"| With Tipico odds | {g['with_tipico_count']} |",
            f"| With sharp odds | {g['with_sharp_count']} |",
            f"| With Tipico + sharp | {g['with_tipico_and_sharp_count']} |",
            f"| Dropped: outright filter | {g['dropped_outright_filter']} |",
            f"| Dropped: no bookmaker overlap | {g['dropped_no_bookmaker_overlap']} |",
            f"| Dropped: outside time window | {g['dropped_time_window']} |",
            f"| Dropped: invalid market | {g['dropped_invalid_market']} |",
            f"| Dropped: confidence gate | {g['dropped_confidence']} |",
            f"| Dropped: negative EV | {g['dropped_ev']} |",
            f"| Dropped: stake = 0 | {g['dropped_stake_zero']} |",
            f"| **Signals generated** | **{g['signals_generated']}** |",
            f"| Signals playable (trading) | {g['signals_playable']} |",
            f"| Signals paper-only (learning) | {g['signals_paper_only']} |",
            f"| **Final displayed** | **{g['final_displayed_count']}** |",
            "",
            "### Request Status Codes",
            "",
        ]

        status = g.get("request_status_counts", {})
        if status:
            lines.append("| Code | Count |")
            lines.append("|------|-------|")
            for code, count in sorted(status.items()):
                lines.append(f"| {code} | {count} |")
        else:
            lines.append("No request data available.")

        lines.extend(["", "## Per-Sport Breakdown", ""])
        per_sport = report.get("per_sport", {})
        if per_sport:
            lines.append(
                "| Sport | Raw | Parsed | Tipico | Sharp | Both | Signals | Playable |"
            )
            lines.append(
                "|-------|-----|--------|--------|-------|------|---------|----------|"
            )
            for sport, s in sorted(per_sport.items()):
                lines.append(
                    f"| {sport} | {s['raw_events']} | {s['parsed_events']} | "
                    f"{s['with_tipico']} | {s['with_sharp']} | {s['with_both']} | "
                    f"{s['signals_generated']} | {s['signals_playable']} |"
                )

        lines.append("")
        return "\n".join(lines)


def get_latest_report() -> Optional[Dict[str, Any]]:
    """Return the latest cached source gap report."""
    return cache.get_json(REPORT_CACHE_KEY)
