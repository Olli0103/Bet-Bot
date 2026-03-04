from __future__ import annotations

import logging
from datetime import datetime
from typing import Dict, List, Optional, Tuple

from src.core.settings import settings
from src.data.redis_cache import cache
from src.integrations.apisports_fetcher import APISportsFetcher
from src.integrations.news_fetcher import NewsFetcher
from src.integrations.ollama_sentiment import OllamaSentimentClient
from src.integrations.reddit_fetcher import RedditFetcher

log = logging.getLogger(__name__)


def _norm_label(label: str) -> float:
    l = (label or "neutral").lower()
    if l == "positive":
        return 1.0
    if l == "negative":
        return -1.0
    return 0.0


def team_sentiment_score(team: str, limit_articles: int = 8) -> float:
    k = f"enrich:sentiment:{team.lower()}"
    cached = cache.get_json(k)
    if cached is not None:
        return float(cached)

    try:
        news = NewsFetcher()
        nlp = OllamaSentimentClient()
        payload = news.search_news(query=team)
        articles = (payload.get("articles") or [])[:limit_articles]

        vals: List[float] = []
        for a in articles:
            text = "\n".join(
                [a.get("title") or "", a.get("description") or "", a.get("content") or ""]
            ).strip()
            if not text:
                continue
            s = nlp.analyze(text=text, context=f"Team={team}")
            vals.append(_norm_label(s.label) * float(s.confidence))

        # ── Reddit sentiment (supplementary source) ──────────────
        try:
            reddit = RedditFetcher()
            reddit_text = reddit.fetch_team_sentiment_posts_sync(team)
            if reddit_text:
                rs = nlp.analyze(text=reddit_text, context=f"Reddit r/sportsbook Team={team}")
                vals.append(_norm_label(rs.label) * float(rs.confidence))
        except Exception as exc:
            log.debug("reddit_sentiment_skip: %s for %s", exc, team)

        if not vals:
            cache.set_json(k, 0.0, 6 * 3600)
            return 0.0

        score = float(sum(vals) / max(1, len(vals)))
        cache.set_json(k, score, 6 * 3600)
        return score
    except Exception as exc:
        log.warning("Sentiment fetch failed for %s: %s", team, exc)
        cache.set_json(k, 0.0, 6 * 3600)
        return 0.0


def batch_team_sentiment(teams: List[str], max_teams: int = 24) -> Dict[str, float]:
    out: Dict[str, float] = {}
    uniq = []
    for t in teams:
        if t and t not in uniq:
            uniq.append(t)
    for t in uniq[:max_teams]:
        try:
            out[t] = team_sentiment_score(t)
        except Exception:
            out[t] = 0.0
    return out


def soccer_injury_delta(home_team: str, away_team: str, event_date_iso: str) -> Tuple[int, int]:
    """Best-effort API-Sports enrichment for football injuries by team/date."""
    from src.integrations.base_fetcher import _safe_sync_run

    ap = APISportsFetcher()
    try:
        d = datetime.fromisoformat(event_date_iso.replace("Z", "+00:00")).date().isoformat()
    except Exception:
        d = datetime.utcnow().date().isoformat()

    def _count_for_team(team: str) -> int:
        try:
            # Use _safe_sync_run for the async .get() call — loop-safe
            fixtures = _safe_sync_run(ap.get("fixtures", params={"date": d, "team": team})) or {}
            rows = fixtures.get("response") or []
            ids = [int(x.get("fixture", {}).get("id")) for x in rows if x.get("fixture", {}).get("id")]
            if not ids:
                return 0
            # get_injuries_by_fixture_ids is now loop-safe via _safe_sync_run
            inj = ap.get_injuries_by_fixture_ids(ids[:10])
            return len(inj.get("response") or [])
        except Exception:
            return 0

    return _count_for_team(home_team), _count_for_team(away_team)
