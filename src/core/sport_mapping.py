"""Central sport mapping configuration.

Single source of truth bridging:
  - CSV division codes (e.g. 'D1', 'E1', 'SP1')
  - Odds API sport keys (e.g. 'soccer_germany_bundesliga')
  - Human-readable German display names for the Telegram UI
  - Team name alias resolution across different APIs
"""
from __future__ import annotations

import re
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
    # Soccer — 1st & 2nd divisions
    SportEntry("D1",  "soccer_germany_bundesliga",        "🇩🇪 1. Bundesliga",          "soccer"),
    SportEntry("D2",  "soccer_germany_bundesliga2",       "🇩🇪 2. Bundesliga",          "soccer"),
    SportEntry("E0",  "soccer_epl",                       "🏴󠁧󠁢󠁥󠁮󠁧󠁿 1. Liga England",      "soccer"),
    SportEntry("E1",  "soccer_england_championship",      "🏴󠁧󠁢󠁥󠁮󠁧󠁿 2. Liga England",      "soccer"),
    SportEntry("F1",  "soccer_france_ligue_one",          "🇫🇷 1. Liga Frankreich",     "soccer"),
    SportEntry("F2",  "soccer_france_ligue_two",          "🇫🇷 2. Liga Frankreich",     "soccer"),
    SportEntry("I1",  "soccer_italy_serie_a",             "🇮🇹 1. Liga Italien",        "soccer"),
    SportEntry("I2",  "soccer_italy_serie_b",             "🇮🇹 2. Liga Italien",        "soccer"),
    SportEntry("SP1", "soccer_spain_la_liga",             "🇪🇸 1. Liga Spanien",        "soccer"),
    SportEntry("SP2", "soccer_spain_segunda_division",    "🇪🇸 2. Liga Spanien",        "soccer"),
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
    SportEntry("atp",       "tennis_atp",                  "🎾 Tennis ATP",               "tennis"),
    SportEntry("wta",       "tennis_wta",                  "🎾 Tennis WTA",               "tennis"),
    SportEntry("challenger","tennis_atp_challenger",        "🎾 Tennis Challenger",        "tennis"),
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


# ── Team name alias resolution ───────────────────────────────────────────────
# Maps common abbreviations, alternate spellings, and API-specific names
# to a single canonical (already-normalized) form.
# Keys MUST be lowercase, alphanumeric only (as produced by the strip regex).

TEAM_ALIASES: Dict[str, str] = {
    # --- Germany (Bundesliga) ---
    "bayernmunich": "bayernmnchen",
    "bayernmuenchen": "bayernmnchen",
    "fcbayern": "bayernmnchen",
    "bayern": "bayernmnchen",
    "bvb": "borussiadortmund",
    "dortmund": "borussiadortmund",
    "bmg": "borussiamnchengladbach",
    "gladbach": "borussiamnchengladbach",
    "mgladbach": "borussiamnchengladbach",
    "monchengladbach": "borussiamnchengladbach",
    "schalke": "fcschalke04",
    "s04": "fcschalke04",
    "rbleipzig": "rbrasenballsportleipzig",
    "leipzg": "rbrasenballsportleipzig",
    "leipzig": "rbrasenballsportleipzig",
    "leverkusen": "bayerleverkusen",
    "bayer04": "bayerleverkusen",
    "wolfsburg": "vflwolfsburg",
    "freiburg": "scfreiburg",
    "mainz": "1fsvmainz05",
    "mainz05": "1fsvmainz05",
    "hoffenheim": "tsg1899hoffenheim",
    "tsghoffenheim": "tsg1899hoffenheim",
    "union": "1fcunionberlin",
    "unionberlin": "1fcunionberlin",
    "koln": "1fckoln",
    "koeln": "1fckoln",
    "effzeh": "1fckoln",
    "frankfurt": "eintrachtfrankfurt",
    "sge": "eintrachtfrankfurt",
    "stuttgart": "vfbstuttgart",
    "augsburg": "fcaugsburg",
    "bremen": "svwerderbremen",
    "werder": "svwerderbremen",
    "werderbremen": "svwerderbremen",
    "heidenheim": "1fcheidenheim",
    "stpauli": "fcstpauli",

    # --- England (Premier League / Championship) ---
    "manutd": "manchesterunited",
    "manunited": "manchesterunited",
    "mufc": "manchesterunited",
    "mancity": "manchestercity",
    "mcfc": "manchestercity",
    "spurs": "tottenhamhotspur",
    "tottenham": "tottenhamhotspur",
    "thfc": "tottenhamhotspur",
    "arsenal": "arsenalfc",
    "afc": "arsenalfc",
    "gunners": "arsenalfc",
    "chelsea": "chelseafc",
    "cfc": "chelseafc",
    "liverpool": "liverpoolfc",
    "lfc": "liverpoolfc",
    "wolves": "wolverhamptonwanderers",
    "wolverhampton": "wolverhamptonwanderers",
    "westham": "westhamunited",
    "whu": "westhamunited",
    "newcastle": "newcastleunited",
    "nufc": "newcastleunited",
    "toon": "newcastleunited",
    "palace": "crystalpalace",
    "cpfc": "crystalpalace",
    "everton": "evertonfc",
    "efc": "evertonfc",
    "toffees": "evertonfc",
    "brighton": "brightonhovealbifc",
    "bhafc": "brightonhovealbifc",
    "brightonhovealbion": "brightonhovealbifc",
    "villa": "astonvilla",
    "avfc": "astonvilla",
    "fulham": "fulhamfc",
    "ffc": "fulhamfc",
    "bournemouth": "afcbournemouth",
    "brentford": "brentfordfc",
    "forest": "nottinghamforest",
    "nffc": "nottinghamforest",
    "nottmforest": "nottinghamforest",
    "luton": "lutontown",
    "burnley": "burnleyfc",
    "sheffield": "sheffieldunited",
    "sheffutd": "sheffieldunited",
    "blades": "sheffieldunited",
    "ipswich": "ipswichtown",
    "leicester": "leicestercity",
    "lcfc": "leicestercity",
    "southampton": "southamptonfc",

    # --- Spain (La Liga) ---
    "barca": "fcbarcelona",
    "barcelona": "fcbarcelona",
    "fcb": "fcbarcelona",
    "realmadrid": "realmadridcf",
    "real": "realmadridcf",
    "rmcf": "realmadridcf",
    "atletico": "atleticodemadrid",
    "atleticomadrid": "atleticodemadrid",
    "atleti": "atleticodemadrid",
    "sevilla": "sevillafc",
    "betis": "realbetis",
    "realsociedad": "realsociedaddefutbol",
    "rsociedad": "realsociedaddefutbol",
    "villarreal": "villarrealcf",
    "valencia": "valenciacf",
    "bilbao": "athleticbilbao",
    "athletic": "athleticbilbao",
    "athleticclub": "athleticbilbao",

    # --- Italy (Serie A) ---
    "juve": "juventusfc",
    "juventus": "juventusfc",
    "inter": "internazionalemilano",
    "intermilan": "internazionalemilano",
    "acmilan": "acmilan",
    "milan": "acmilan",
    "roma": "asroma",
    "asroma": "asroma",
    "napoli": "sscnapoli",
    "sscnapoli": "sscnapoli",
    "lazio": "sslazio",
    "fiorentina": "acffiorentina",
    "viola": "acffiorentina",
    "atalanta": "atalantabc",
    "torino": "torinofc",
    "toro": "torinofc",

    # --- France (Ligue 1) ---
    "psg": "parisstgermain",
    "parissaintgermain": "parisstgermain",
    "om": "olympiquedemarseille",
    "marseille": "olympiquedemarseille",
    "ol": "olympiquelyonnais",
    "lyon": "olympiquelyonnais",
    "monaco": "asmonaco",
    "lille": "lilleosc",
    "losc": "lilleosc",
    "lens": "rclens",
    "nice": "ogcnice",

    # --- NBA ---
    "lal": "losangeleslakers",
    "lakers": "losangeleslakers",
    "gsw": "goldenstatewarriors",
    "warriors": "goldenstatewarriors",
    "bos": "bostonceltics",
    "celtics": "bostonceltics",
    "mil": "milwaukeebucks",
    "bucks": "milwaukeebucks",
    "den": "denvernuggets",
    "nuggets": "denvernuggets",
    "phx": "phoenixsuns",
    "suns": "phoenixsuns",
    "dal": "dallasmavericks",
    "mavs": "dallasmavericks",
    "mavericks": "dallasmavericks",
    "nyknicks": "newyorkknicks",
    "knicks": "newyorkknicks",
    "nyk": "newyorkknicks",
    "phi": "philadelphia76ers",
    "sixers": "philadelphia76ers",
    "76ers": "philadelphia76ers",
    "mia": "miamiheat",
    "heat": "miamiheat",
    "lac": "losangelesclippers",
    "clippers": "losangelesclippers",
    "okc": "oklahomacitythunder",
    "thunder": "oklahomacitythunder",
    "min": "minnesotatimberwolves",
    "wolves": "minnesotatimberwolves",
    "timberwolves": "minnesotatimberwolves",

    # --- NFL ---
    "kc": "kansascitychiefs",
    "chiefs": "kansascitychiefs",
    "sf": "sanfrancisco49ers",
    "niners": "sanfrancisco49ers",
    "49ers": "sanfrancisco49ers",
    "buf": "buffalobills",
    "bills": "buffalobills",
    "bal": "baltimoreravens",
    "ravens": "baltimoreravens",
    "det": "detroitlions",
    "lions": "detroitlions",
    "gb": "greenbaypackers",
    "packers": "greenbaypackers",
    "phi": "philadelphiaeagles",
    "eagles": "philadelphiaeagles",
    "dal": "dallascowboys",
    "cowboys": "dallascowboys",

    # --- NHL ---
    "edm": "edmontonoilers",
    "oilers": "edmontonoilers",
    "fla": "floridapanthers",
    "panthers": "floridapanthers",
    "nyr": "newyorkrangers",
    "rangers": "newyorkrangers",
    "bos": "bostonbruins",
    "bruins": "bostonbruins",
    "col": "coloradoavalanche",
    "avs": "coloradoavalanche",
    "avalanche": "coloradoavalanche",
    "tor": "torontomapleleafs",
    "leafs": "torontomapleleafs",
    "mapleleafs": "torontomapleleafs",
    "van": "vancouvercanucks",
    "canucks": "vancouvercanucks",
    "wpg": "winnipegjets",
    "jets": "winnipegjets",
}

# Build reverse index at import time (canonical -> canonical, alias -> canonical)
_ALIAS_INDEX: Dict[str, str] = {}
for _alias, _canonical in TEAM_ALIASES.items():
    _ALIAS_INDEX[_alias] = _canonical
    # Also index the canonical form to itself
    _ALIAS_INDEX[_canonical] = _canonical


def normalize_team(name: str) -> str:
    """Normalize a team name: strip non-alphanumeric, lowercase, resolve aliases.

    This is the single source of truth for team name normalization.
    All modules should use this instead of local _normalize_team functions.
    """
    stripped = re.sub(r"[^a-z0-9]", "", (name or "").lower())
    return _ALIAS_INDEX.get(stripped, stripped)
