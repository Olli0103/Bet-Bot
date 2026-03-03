"""Central sport mapping configuration.

Single source of truth bridging:
  - CSV division codes (e.g. 'D1', 'E1', 'SP1')
  - Odds API sport keys (e.g. 'soccer_germany_bundesliga')
  - Human-readable German display names for the Telegram UI
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional


@dataclass(frozen=True, slots=True)
class SportEntry:
    """One sport/league in the mapping."""
    csv_code: str           # Code used in CSV Div column or folder name
    odds_api_key: str       # The-Odds-API sport_key
    display_name: str       # German UI label (with flag emoji)
    category: str           # soccer / basketball / americanfootball / icehockey / tennis


# ── Master list ──────────────────────────────────────────────────────────────

_ENTRIES: List[SportEntry] = [
    # Soccer — 1st divisions
    SportEntry("D1",  "soccer_germany_bundesliga",        "🇩🇪 1. Bundesliga",          "soccer"),
    SportEntry("D2",  "soccer_germany_bundesliga2",       "🇩🇪 2. Bundesliga",          "soccer"),
    SportEntry("E0",  "soccer_epl",                       "🏴󠁧󠁢󠁥󠁮󠁧󠁿 Premier League",       "soccer"),
    SportEntry("E1",  "soccer_england_championship",      "🏴󠁧󠁢󠁥󠁮󠁧󠁿 Championship",         "soccer"),
    SportEntry("F1",  "soccer_france_ligue_one",          "🇫🇷 Ligue 1",                "soccer"),
    SportEntry("F2",  "soccer_france_ligue_two",          "🇫🇷 Ligue 2",                "soccer"),
    SportEntry("I1",  "soccer_italy_serie_a",             "🇮🇹 Serie A",                 "soccer"),
    SportEntry("I2",  "soccer_italy_serie_b",             "🇮🇹 Serie B",                 "soccer"),
    SportEntry("SP1", "soccer_spain_la_liga",             "🇪🇸 La Liga",                 "soccer"),
    SportEntry("SP2", "soccer_spain_segunda_division",    "🇪🇸 Segunda División",        "soccer"),
    SportEntry("N1",  "soccer_netherlands_eredivisie",    "🇳🇱 Eredivisie",              "soccer"),
    SportEntry("P1",  "soccer_portugal_primeira_liga",    "🇵🇹 Primeira Liga",           "soccer"),
    SportEntry("T1",  "soccer_turkey_super_league",       "🇹🇷 Süper Lig",               "soccer"),
    SportEntry("B1",  "soccer_belgium_first_div",         "🇧🇪 First Division A",        "soccer"),
    SportEntry("SC0", "soccer_spl",                       "🏴󠁧󠁢󠁳󠁣󠁴󠁿 Scottish Premiership", "soccer"),
    SportEntry("G1",  "soccer_greece_super_league",       "🇬🇷 Super League",            "soccer"),
    SportEntry("CL",  "soccer_uefa_champs_league",        "🏆 Champions League",         "soccer"),
    SportEntry("EL",  "soccer_uefa_europa_league",        "🏆 Europa League",            "soccer"),
    # Basketball
    SportEntry("nba",       "basketball_nba",             "🏀 NBA",                      "basketball"),
    SportEntry("euroleague","basketball_euroleague",       "🏀 EuroLeague",               "basketball"),
    # American football
    SportEntry("nfl",       "americanfootball_nfl",        "🏈 NFL",                      "americanfootball"),
    # Ice hockey
    SportEntry("nhl",       "icehockey_nhl",               "🏒 NHL",                      "icehockey"),
    # Tennis
    SportEntry("atp",       "tennis_atp",                  "🎾 ATP",                      "tennis"),
    SportEntry("wta",       "tennis_wta",                  "🎾 WTA",                      "tennis"),
    SportEntry("challenger","tennis_atp_challenger",        "🎾 Challenger",               "tennis"),
]


# ── Lookup indices (built once at import time) ───────────────────────────────

# csv_code (upper-cased) -> SportEntry
_BY_CSV: Dict[str, SportEntry] = {e.csv_code.upper(): e for e in _ENTRIES}

# odds_api_key -> SportEntry
_BY_API_KEY: Dict[str, SportEntry] = {e.odds_api_key: e for e in _ENTRIES}

# odds_api_key -> display_name  (convenience for settings UI)
SPORT_DISPLAY_NAMES: Dict[str, str] = {e.odds_api_key: e.display_name for e in _ENTRIES}


# ── Public API ───────────────────────────────────────────────────────────────

def csv_code_to_api_key(csv_code: str) -> Optional[str]:
    """Map a CSV division code (e.g. 'D1', 'E0') to its Odds API sport key.

    Returns ``None`` when the code is unknown.
    """
    entry = _BY_CSV.get(csv_code.upper().strip())
    return entry.odds_api_key if entry else None


def csv_code_to_entry(csv_code: str) -> Optional[SportEntry]:
    """Full :class:`SportEntry` for a CSV code, or ``None``."""
    return _BY_CSV.get(csv_code.upper().strip())


def api_key_to_display(api_key: str) -> str:
    """Human-readable display name for an Odds API key.

    Falls back to the raw key when unknown.
    """
    entry = _BY_API_KEY.get(api_key)
    return entry.display_name if entry else api_key


def api_key_to_entry(api_key: str) -> Optional[SportEntry]:
    """Full :class:`SportEntry` for an Odds API key, or ``None``."""
    return _BY_API_KEY.get(api_key)


def all_entries() -> List[SportEntry]:
    """Return a copy of the master entry list."""
    return list(_ENTRIES)


def all_api_keys() -> List[str]:
    """All known Odds API sport keys."""
    return [e.odds_api_key for e in _ENTRIES]
