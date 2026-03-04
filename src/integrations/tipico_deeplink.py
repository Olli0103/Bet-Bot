"""Tipico deep-link builder for one-tap bet-slip integration.

Constructs URLs that open the Tipico web/app directly to the event or
pre-populated betslip, eliminating the need for users to manually
navigate through sports > leagues > matches.

URL patterns
------------
Event link (fallback):
    https://sports.tipico.de/de/sports/events/{eventId}

Betslip link (ideal — pre-populates the betslip):
    https://sports.tipico.de/de/sports?options={eventId}-{marketId}-{selectionId}&stake={stake}&type=single

Combo betslip:
    ...&type=combo  with options joined by commas.
"""
from __future__ import annotations

from typing import List, Optional
from urllib.parse import quote, urlencode


_BASE = "https://sports.tipico.de/de/sports"


def event_link(event_id: str) -> str:
    """Return a deep link to the Tipico event page."""
    return f"{_BASE}/events/{quote(str(event_id))}"


def betslip_link(
    event_id: str,
    market_id: str = "",
    selection_id: str = "",
    stake: Optional[float] = None,
) -> str:
    """Return a deep link that pre-populates the Tipico betslip.

    If ``market_id`` and ``selection_id`` are available the link will
    directly add the selection to the slip.  Otherwise falls back to the
    event page.
    """
    if not market_id or not selection_id:
        return event_link(event_id)

    option = f"{event_id}-{market_id}-{selection_id}"
    params = {"options": option, "type": "single"}
    if stake is not None and stake > 0:
        params["stake"] = f"{stake:.2f}"
    return f"{_BASE}?{urlencode(params)}"


def combo_betslip_link(
    legs: List[dict],
    stake: Optional[float] = None,
) -> str:
    """Return a deep link for a combo bet with multiple legs.

    Each leg dict should contain ``event_id``, and optionally
    ``market_id`` and ``selection_id``.
    """
    options = []
    for leg in legs:
        eid = str(leg.get("event_id", ""))
        mid = str(leg.get("market_id", ""))
        sid = str(leg.get("selection_id", ""))
        if eid and mid and sid:
            options.append(f"{eid}-{mid}-{sid}")
        elif eid:
            options.append(eid)
    if not options:
        return _BASE

    params = {"options": ",".join(options), "type": "combo"}
    if stake is not None and stake > 0:
        params["stake"] = f"{stake:.2f}"
    return f"{_BASE}?{urlencode(params)}"
