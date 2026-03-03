"""Unified Injury Aggregator: merges API-Sports and Rotowire RSS signals.

Collects injury/lineup intelligence from both sources in parallel,
deduplicates, and uses Gemma 3 4B to extract a structured JSON list of
confirmed missing key players.

Replaces the legacy Twitter/X integration with 100% free data sources.
"""
from __future__ import annotations

import asyncio
import logging
from typing import Any, Dict, List

from src.data.redis_cache import cache

log = logging.getLogger(__name__)

AGGREGATOR_CACHE_TTL = 15 * 60  # 15 minutes


async def get_match_injuries_async(
    home_team: str,
    away_team: str,
    sport: str,
    event_date: str = "",
) -> Dict[str, Any]:
    """Aggregate injury intelligence for a match from all sources.

    Calls API-Sports and Rotowire RSS in parallel, merges text snippets,
    and passes them through Gemma 3 4B for structured extraction.

    Parameters
    ----------
    home_team, away_team : str
        Team names as provided by OddsAPI.
    sport : str
        OddsAPI sport key (e.g. "soccer_epl", "basketball_nba").
    event_date : str
        ISO date string for the event.

    Returns
    -------
    dict with keys:
        - injuries: list of {"player": str, "team": str, "status": str}
        - raw_snippets: int (count of raw text snippets collected)
        - sources: list of source names that contributed data
    """
    cache_key = f"injury_agg:{home_team}:{away_team}:{sport}".lower().replace(" ", "_")
    cached = cache.get_json(cache_key)
    if cached is not None:
        return cached

    team_names = [home_team, away_team]

    # Collect from both sources in parallel
    api_sports_task = _fetch_apisports(home_team, away_team, sport, event_date)
    rss_task = _fetch_rss(sport, team_names)

    api_results, rss_results = await asyncio.gather(
        api_sports_task, rss_task, return_exceptions=True
    )

    # Safely unpack results (exceptions become empty lists)
    if isinstance(api_results, BaseException):
        log.warning("API-Sports aggregation failed: %s", api_results)
        api_results = []
    if isinstance(rss_results, BaseException):
        log.warning("RSS aggregation failed: %s", rss_results)
        rss_results = []

    # Build combined text dump for LLM
    snippets: List[str] = []
    sources_used: List[str] = []

    # API-Sports results (already structured)
    if api_results:
        sources_used.append("api-sports")
        for inj in api_results:
            snippets.append(
                f"[API-Sports] {inj.get('player', '?')} ({inj.get('team', '?')}): {inj.get('status', '?')}"
            )

    # RSS results
    if rss_results:
        sources_used.append("rotowire-rss")
        for item in rss_results:
            snippets.append(f"[Rotowire] {item.get('title', '')} — {item.get('summary', '')[:200]}")

    # If no data from any source, return empty
    if not snippets:
        result: Dict[str, Any] = {"injuries": [], "raw_snippets": 0, "sources": []}
        cache.set_json(cache_key, result, AGGREGATOR_CACHE_TTL)
        return result

    # Pass combined text to Gemma 3 4B for structured extraction
    injuries = await _extract_with_llm(home_team, away_team, snippets)

    result = {
        "injuries": injuries,
        "raw_snippets": len(snippets),
        "sources": sources_used,
    }
    cache.set_json(cache_key, result, AGGREGATOR_CACHE_TTL)
    return result


# Keep backward-compatible alias
aggregate_injury_intel = get_match_injuries_async


async def _fetch_apisports(
    home: str, away: str, sport: str, event_date: str
) -> List[Dict[str, str]]:
    """Fetch from API-Sports (async)."""
    from src.integrations.apisports_fetcher import APISportsFetcher
    ap = APISportsFetcher()
    return await ap.get_injuries_for_event_async(home, away, sport, event_date)


async def _fetch_rss(sport: str, team_names: List[str]) -> List[Dict[str, str]]:
    """Fetch from Rotowire RSS (sync, run in thread)."""
    from src.integrations.rss_fetcher import RSSFetcher
    rss = RSSFetcher()
    return await asyncio.to_thread(rss.fetch_injury_news, sport, team_names, 24)


async def _extract_with_llm(
    home: str, away: str, snippets: List[str]
) -> List[Dict[str, str]]:
    """Use Gemma 3 4B to extract structured injury data from raw text.

    Returns a list of {"player": "...", "team": "...", "status": "Out|Doubtful|..."}.
    """
    try:
        from src.integrations.ollama_sentiment import OllamaSentimentClient
        nlp = OllamaSentimentClient()

        combined_text = "\n".join(snippets[:30])  # cap at 30 snippets
        example_json = (
            '{"injuries": [{"player": "LeBron James", '
            f'"team": "{home}", "status": "Out"}}]}}'
        )
        prompt = (
            "Extract critical injury news from the <text>. "
            f"The ONLY valid teams are: '{home}' and '{away}'. "
            "Map any abbreviations (e.g., 'LAL', 'Man Utd', 'BVB', 'Spurs') "
            "strictly to one of these two teams. "
            "Answer ONLY with a valid JSON object. Do not include markdown formatting.\n\n"
            "Status must be one of: Out, Doubtful, Questionable, Day-to-Day.\n"
            "If no injuries are found, return: {\"injuries\": []}\n\n"
            f"Expected JSON format:\n{example_json}\n\n"
            f"<text>\n{combined_text}\n</text>"
        )

        result = await asyncio.to_thread(nlp.generate_json, prompt)

        injuries_raw = result.get("injuries", [])
        if not isinstance(injuries_raw, list):
            return []

        # Validate structure
        validated: List[Dict[str, str]] = []
        for item in injuries_raw:
            if isinstance(item, dict) and item.get("player"):
                validated.append({
                    "player": str(item.get("player", "")),
                    "team": str(item.get("team", "")),
                    "status": str(item.get("status", "Unknown")),
                })
        return validated

    except Exception as exc:
        log.warning("LLM injury extraction failed: %s", exc)
        return []


def get_injury_impact_score(
    injuries: List[Dict[str, str]], team: str
) -> float:
    """Compute a negative impact score for a team based on confirmed injuries.

    Returns a float: 0.0 (no injuries) to -1.0 (severe: 3+ key players out).
    Used to penalize model probability / EV for the affected team.
    """
    team_injuries = [
        inj for inj in injuries
        if inj.get("team", "").lower() == team.lower()
    ]

    if not team_injuries:
        return 0.0

    score = 0.0
    for inj in team_injuries:
        status = (inj.get("status") or "").lower()
        if status == "out":
            score -= 0.35
        elif status == "doubtful":
            score -= 0.20
        elif status == "questionable":
            score -= 0.10
        elif status == "day-to-day":
            score -= 0.08

    # Clamp to [-1.0, 0.0]
    return max(-1.0, score)
