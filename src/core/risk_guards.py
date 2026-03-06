"""Risk Guards: Confidence gates + stake caps.

Central module so both live_feed signal generation and the executioner agent
apply identical rules.
"""
from __future__ import annotations

import logging
from typing import Optional, Tuple

from src.core.settings import settings

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Confidence gate
# ---------------------------------------------------------------------------

def get_min_confidence(sport: str, market: str = "h2h") -> float:
    """Return the minimum model_probability required to allow a bet.

    Resolves by sport prefix + market type, falling back to the default.
    Applies hard safety floors so accidental permissive env overrides
    (e.g. MIN_CONF_*=0.45) cannot silently weaken production guards.
    """
    s = sport.lower()
    m = market.lower()

    floors = {
        "soccer_h2h": 0.55,
        "soccer_totals": 0.56,
        "soccer_spread": 0.56,
        "tennis": 0.57,
        "basketball": 0.55,
        "icehockey": 0.55,
        "americanfootball": 0.55,
        "default": 0.55,
    }

    if s.startswith(("soccer", "football")):
        if "totals" in m:
            return max(settings.min_confidence_soccer_totals, floors["soccer_totals"])
        if "spread" in m:
            return max(settings.min_confidence_soccer_spread, floors["soccer_spread"])
        return max(settings.min_confidence_soccer_h2h, floors["soccer_h2h"])
    if s.startswith("tennis"):
        return max(settings.min_confidence_tennis, floors["tennis"])
    if s.startswith("basketball"):
        return max(settings.min_confidence_basketball, floors["basketball"])
    if s.startswith("icehockey"):
        return max(settings.min_confidence_icehockey, floors["icehockey"])
    if s.startswith("americanfootball"):
        return max(settings.min_confidence_americanfootball, floors["americanfootball"])
    return max(settings.min_confidence_default, floors["default"])


def passes_confidence_gate(
    model_probability: float,
    sport: str,
    market: str = "h2h",
) -> Tuple[bool, float]:
    """Check if model_probability meets the confidence gate.

    Returns (passed, min_conf) so callers can log the reason.
    """
    min_conf = get_min_confidence(sport, market)
    return model_probability >= min_conf, min_conf


# ---------------------------------------------------------------------------
# Stake caps
# ---------------------------------------------------------------------------

def apply_stake_cap(
    stake: float,
    bankroll: float,
    odds: float = 1.0,
    market: str = "h2h",
    selection: str = "",
) -> Tuple[float, bool]:
    """Enforce hard stake caps.

    Returns (capped_stake, was_capped).

    Rules:
    - General cap: max_stake_pct * bankroll (default 1.5%)
    - Draw/longshot cap: max_stake_longshot_pct * bankroll (default 0.75%)
      applies when odds >= longshot_odds_threshold OR selection contains 'Draw'
    """
    is_draw = "draw" in selection.lower() or "unentschieden" in selection.lower()
    is_longshot = odds >= settings.longshot_odds_threshold

    # Hard safety floors to prevent accidental env drift loosening caps.
    max_stake_pct = min(settings.max_stake_pct, 0.015)
    max_longshot_pct = min(settings.max_stake_longshot_pct, 0.0075)

    if is_draw or is_longshot:
        cap = bankroll * max_longshot_pct
    else:
        cap = bankroll * max_stake_pct

    if stake > cap:
        return round(cap, 2), True
    return round(stake, 2), False


# ---------------------------------------------------------------------------
# "Why this bet?" — natural-language explanation for non-technical users
# ---------------------------------------------------------------------------

def explain_signal(
    model_probability: float,
    expected_value: float,
    bookmaker_odds: float,
    sport: str,
    market_momentum: float = 0.0,
    public_bias: float = 0.0,
    is_steam_move: bool = False,
    sport_group: str = "general",
) -> str:
    """Build a short German-language explanation of why this bet was flagged.

    Uses reliability bins from the ML model (if available) to assess
    calibration quality, and combines key signal drivers into a
    1-3 sentence human-readable summary.
    """
    from src.core.ml_trainer import get_reliability_adjustment

    parts: list[str] = []

    # Edge size
    ev_pct = expected_value * 100
    if ev_pct >= 5:
        parts.append(f"Starker Edge: +{ev_pct:.1f}% erwarteter Profit")
    elif ev_pct >= 2:
        parts.append(f"Solider Edge: +{ev_pct:.1f}% erwarteter Profit")
    else:
        parts.append(f"Kleiner Edge: +{ev_pct:.1f}% erwarteter Profit")

    # Probability assessment
    if model_probability >= 0.70:
        parts.append("Hohe Modell-Konfidenz")
    elif model_probability >= 0.55:
        parts.append("Mittlere Modell-Konfidenz")
    else:
        parts.append("Spekulative Wette")

    # Key drivers
    drivers: list[str] = []
    if is_steam_move:
        drivers.append("Sharp-Money-Bewegung erkannt")
    if market_momentum > 0.02:
        drivers.append("Linie bewegt sich zugunsten")
    elif market_momentum < -0.02:
        drivers.append("Linie bewegt sich dagegen")
    if public_bias > 0.02:
        drivers.append("Tipico-Preis angehoben (Oeffentlichkeit wettet dagegen)")
    if drivers:
        parts.append(". ".join(drivers))

    # Calibration quality from reliability bins — include historical accuracy
    kelly_adj = get_reliability_adjustment(model_probability, sport_group)
    actual_rate = _get_historical_accuracy(model_probability, sport_group)
    if actual_rate is not None:
        acc_pct = actual_rate * 100
        parts.append(
            f"Historisch {acc_pct:.0f}% Trefferquote in diesem Bereich"
        )
    if kelly_adj < 0.85:
        parts.append("Modell neigt zu Ueberschaetzung — Einsatz reduziert")
    elif kelly_adj > 1.15:
        parts.append("Modell historisch treffsicher in diesem Bereich")

    return ". ".join(parts) + "."


def _get_historical_accuracy(
    model_prob: float, sport_group: str = "general"
) -> Optional[float]:
    """Return the historical actual win rate for the reliability bin
    containing ``model_prob``, or None if no calibration data exists."""
    from src.core.ml_trainer import load_model

    model_data = load_model(sport_group)
    if model_data is None:
        model_data = load_model("general")
    if model_data is None:
        return None

    bins = model_data.get("metrics", {}).get("reliability_bins", {})
    for bin_key, info in bins.items():
        try:
            low, high = (float(x) for x in bin_key.split("_"))
        except (ValueError, TypeError):
            continue
        if low <= model_prob < high:
            return float(info.get("actual", info.get("predicted")))
    return None


# ---------------------------------------------------------------------------
# Dynamic Margin of Safety (EV threshold)
# ---------------------------------------------------------------------------

def get_dynamic_min_ev(sport: str, market: str = "h2h") -> float:
    """Return a Brier-score-dependent EV threshold for the given sport.

    Better-calibrated models (lower Brier) can tolerate smaller edges,
    while poorly calibrated models need a larger safety margin.

    Formula: min_ev = clamp(brier_score * 0.15, 0.005, 0.05)
        Brier 0.18 -> min_ev 0.027 (2.7%)
        Brier 0.22 -> min_ev 0.033 (3.3%)
        Brier 0.25 -> min_ev 0.038 (3.8%)

    Falls back to ``settings.min_ev_default`` when no model metrics exist.
    """
    from src.core.ml_trainer import load_model

    s = sport.lower()
    if s.startswith(("soccer", "football")):
        group = "soccer"
    elif s.startswith("basketball"):
        group = "basketball"
    elif s.startswith("tennis"):
        group = "tennis"
    elif s.startswith("icehockey"):
        group = "icehockey"
    elif s.startswith("americanfootball"):
        group = "americanfootball"
    else:
        group = "general"

    model_data = load_model(group)
    if model_data is None:
        model_data = load_model("general")
    if model_data is None:
        return settings.min_ev_default

    brier = model_data.get("metrics", {}).get("brier_score")
    if brier is None or brier <= 0:
        return settings.min_ev_default

    dynamic_ev = max(0.005, min(0.05, brier * 0.15))
    return dynamic_ev


def get_liquidity_cap(sport: str, market: str = "h2h") -> float:
    """Return a stake multiplier based on league liquidity tier.

    Low-tier leagues have thin markets where large stakes move the line
    against us and increase the risk of getting limited.  This caps the
    maximum stake as a fraction of what Kelly would recommend.

    Tier mapping:
        Tier 1 (EPL, NFL, NBA, La Liga, Champions League): 1.0
        Tier 2 (Bundesliga, Serie A, Ligue 1, NHL):        0.7
        Tier 3 (2nd leagues, MLS, domestic cups):           0.3
        Tier 4 (Unknown / niche leagues):                   0.1
    """
    s = sport.lower()

    # Tier 1: highest liquidity markets
    _TIER_1 = (
        "soccer_epl", "soccer_spain_la_liga", "soccer_uefa_champs_league",
        "americanfootball_nfl", "basketball_nba",
        "soccer_uefa_europa_league",
    )
    # Tier 2: major European leagues, NHL
    _TIER_2 = (
        "soccer_germany_bundesliga", "soccer_italy_serie_a",
        "soccer_france_ligue_one", "soccer_brazil_serie_a",
        "icehockey_nhl", "basketball_euroleague",
        "soccer_portugal_primeira_liga", "soccer_netherlands_eredivisie",
    )
    # Tier 3: second divisions, domestic cups, MLS
    _TIER_3 = (
        "soccer_germany_bundesliga2", "soccer_england_league1",
        "soccer_england_league2", "soccer_efl_champ",
        "soccer_usa_mls", "soccer_turkey_super_league",
        "soccer_scotland_premiership", "soccer_belgium_first_div",
    )

    # Check most specific tiers first (Tier 3 before Tier 2)
    # to avoid "bundesliga" matching before "bundesliga2"
    for key in _TIER_3:
        if key in s:
            return 0.3
    for key in _TIER_1:
        if key in s:
            return 1.0
    for key in _TIER_2:
        if key in s:
            return 0.7

    # Tier 4: unknown / niche — maximum caution
    return 0.1


def get_dynamic_kelly_frac(sport: str, base_frac: float = 0.20) -> float:
    """Return a Brier-score-dependent Kelly fraction for the given sport.

    Better-calibrated models deserve higher Kelly sizing; uncertain
    models should be shrunk aggressively.

    Mapping:
        Brier < 0.18:  full base_frac (model is well-calibrated)
        Brier 0.18-0.22: 75% of base_frac
        Brier >= 0.22: 40% of base_frac
        No model:      50% of base_frac (maximum caution)
    """
    from src.core.ml_trainer import load_model

    s = sport.lower()
    if s.startswith(("soccer", "football")):
        group = "soccer"
    elif s.startswith("basketball"):
        group = "basketball"
    elif s.startswith("tennis"):
        group = "tennis"
    elif s.startswith("icehockey"):
        group = "icehockey"
    elif s.startswith("americanfootball"):
        group = "americanfootball"
    else:
        group = "general"

    model_data = load_model(group)
    if model_data is None:
        model_data = load_model("general")
    if model_data is None:
        return base_frac * 0.50

    brier = model_data.get("metrics", {}).get("brier_score")
    if brier is None or brier <= 0:
        return base_frac * 0.50

    if brier < 0.18:
        return base_frac
    elif brier < 0.22:
        return base_frac * 0.75
    else:
        return base_frac * 0.40


# ---------------------------------------------------------------------------
# Data source health gate
# ---------------------------------------------------------------------------

# Sources that MUST be online for betting to proceed.  Without odds data
# the model is flying blind and all features degrade to defaults.
CRITICAL_SOURCES = ["odds_api"]


def check_data_source_health() -> Tuple[bool, str]:
    """Return (healthy, reason).

    If critical data sources (odds_api) are down (circuit breaker open),
    betting should be paused automatically rather than falling back to
    stale / default features.
    """
    from src.core.source_health import is_available, SOURCE_CONFIG

    for src in CRITICAL_SOURCES:
        if not is_available(src):
            label = SOURCE_CONFIG.get(src, {}).get("label", src)
            return False, f"{label} ist offline — Wetten pausiert"
    return True, ""
