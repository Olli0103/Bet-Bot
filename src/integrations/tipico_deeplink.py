"""Tipico deep-link builder for one-tap bet-slip integration.

Constructs URLs that open the Tipico web/app directly to the event or
pre-populated betslip, eliminating the need for users to manually
navigate through sports > leagues > matches.

IMPORTANT — ID mapping
----------------------
The Odds API (our data source) uses its own event IDs which do NOT
match Tipico's proprietary internal IDs.  Therefore ``event_link()``
falls back to a **search URL** using team names so that the user
lands on a relevant Tipico page regardless.

When Tipico-native ``market_id`` and ``selection_id`` are available
(e.g. scraped or mapped), ``betslip_link()`` can construct the
direct betslip pre-population URL:
    https://sports.tipico.de/de/sports?options={eventId}-{marketId}-{selectionId}&stake={stake}&type=single

Combo betslip:
    ...&type=combo  with options joined by commas.
"""
from __future__ import annotations

from typing import List, Optional
from urllib.parse import quote, urlencode


_BASE = "https://sports.tipico.de/de/sports"
_SEARCH = "https://sports.tipico.de/de/search"

# Odds API returns English names; Tipico's German app uses German names.
# This mapping covers the most common mismatches where Tipico's search
# would fail on the English variant.
_TIPICO_NAMES = {
    "bayern munich": "Bayern München",
    "borussia monchengladbach": "Borussia Mönchengladbach",
    "1. fc koln": "1. FC Köln",
    "fc koln": "1. FC Köln",
    "1. fc cologne": "1. FC Köln",
    "cologne": "1. FC Köln",
    "1. fc nurnberg": "1. FC Nürnberg",
    "fc nurnberg": "1. FC Nürnberg",
    "nuremberg": "1. FC Nürnberg",
    "fortuna dusseldorf": "Fortuna Düsseldorf",
    "dusseldorf": "Fortuna Düsseldorf",
    "ac milan": "AC Mailand",
    "inter milan": "Inter Mailand",
    "napoli": "SSC Neapel",
    "ssc napoli": "SSC Neapel",
    "juventus": "Juventus Turin",
    "as roma": "AS Rom",
    "roma": "AS Rom",
    "genoa": "CFC Genua",
    "atletico madrid": "Atlético Madrid",
    "real betis": "Real Betis Sevilla",
}


def _to_tipico_name(name: str) -> str:
    """Translate an Odds API team name to Tipico's German search form."""
    return _TIPICO_NAMES.get(name.lower().strip(), name)


def event_link(
    event_id: str,
    home_team: str = "",
    away_team: str = "",
) -> str:
    """Return a link to find the event on Tipico.

    Since the Odds API ``event_id`` does not match Tipico's internal
    IDs, we build a **search URL** using team names so the user lands
    on a relevant results page.  Falls back to the raw event URL only
    when no team names are provided.

    Team names are translated to German variants where needed (e.g.
    "Bayern Munich" → "Bayern München") to improve Tipico search hits.
    """
    parts = [_to_tipico_name(t) for t in [home_team, away_team] if t]
    query = " ".join(parts).strip()
    if query:
        return f"{_SEARCH}?query={quote(query)}"
    # Fallback: raw event URL (may 404 if ID doesn't match)
    return f"{_BASE}/events/{quote(str(event_id))}"


def betslip_link(
    event_id: str,
    market_id: str = "",
    selection_id: str = "",
    stake: Optional[float] = None,
) -> str:
    """Return a deep link that pre-populates the Tipico betslip.

    If ``market_id`` and ``selection_id`` are available (Tipico-native
    IDs) the link will directly add the selection to the slip.
    Otherwise falls back to the search-based event link.
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

    Caps at 15 legs to stay under browser/mobile URL length limits
    (~2000 chars).  Excess legs are silently dropped.
    """
    _MAX_URL_LEGS = 15

    options = []
    for leg in legs:
        eid = str(leg.get("event_id", ""))
        mid = str(leg.get("market_id", ""))
        sid = str(leg.get("selection_id", ""))
        if eid and mid and sid:
            options.append(f"{eid}-{mid}-{sid}")
        # Skip legs without full triple — raw event IDs cause Tipico's
        # frontend router to silently drop the parameter or error.
    if not options:
        return _BASE

    if len(options) > _MAX_URL_LEGS:
        options = options[:_MAX_URL_LEGS]

    params = {"options": ",".join(options), "type": "combo"}
    if stake is not None and stake > 0:
        params["stake"] = f"{stake:.2f}"
    return f"{_BASE}?{urlencode(params)}"
