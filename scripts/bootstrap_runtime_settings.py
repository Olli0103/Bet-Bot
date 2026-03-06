#!/usr/bin/env python3
"""Bootstrap reproducible runtime dynamic settings.

Purpose:
- Make Redis-backed runtime settings reproducible (not only ad-hoc manual toggles).
- Apply the current sports-mix policy to global + selected owner scopes.

Usage:
  PYTHONPATH=. ./.venv/bin/python scripts/bootstrap_runtime_settings.py
  PYTHONPATH=. ./.venv/bin/python scripts/bootstrap_runtime_settings.py --owners 381129865,-1002602109508
"""
from __future__ import annotations

import argparse
from typing import List

from src.core.dynamic_settings import DynamicSettingsManager


TARGET_ACTIVE_SPORTS = [
    "soccer_germany_bundesliga",
    "soccer_epl",
    "soccer_spain_la_liga",
    "soccer_italy_serie_a",
    "soccer_france_ligue_one",
    "soccer_uefa_champs_league",
    "americanfootball_nfl",
    "basketball_nba",
    "icehockey_nhl",
]


def apply_scope(owner_chat_id: str = "") -> List[str]:
    m = DynamicSettingsManager(owner_chat_id=owner_chat_id)
    current = m.get_all()

    # Keep non-sport settings untouched; only enforce sports mix.
    m.set("active_sports", list(TARGET_ACTIVE_SPORTS))

    after = m.get_all().get("active_sports") or []
    scope = owner_chat_id or "global"
    print(f"[{scope}] active_sports={after}")
    return after


def main() -> None:
    p = argparse.ArgumentParser(description="Bootstrap runtime dynamic settings")
    p.add_argument(
        "--owners",
        default="381129865,-1002602109508",
        help="Comma-separated owner chat IDs to update in addition to global scope",
    )
    args = p.parse_args()

    # Global scope first
    apply_scope("")

    # Owner scopes
    owners = [x.strip() for x in (args.owners or "").split(",") if x.strip()]
    for oid in owners:
        apply_scope(oid)

    print("bootstrap_runtime_settings: done")


if __name__ == "__main__":
    main()
