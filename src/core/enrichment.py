from __future__ import annotations

import logging
import signal
import time
from contextlib import contextmanager
from datetime import datetime
from typing import Dict, List, Optional, Tuple

from src.core.settings import settings
from src.core.source_health import record_failure, record_success, is_available
from src.data.redis_cache import cache
from src.integrations.apisports_fetcher import APISportsFetcher
from src.integrations.news_fetcher import NewsFetcher
from src.integrations.ollama_sentiment import OllamaSentimentClient
from src.integrations.reddit_fetcher import RedditFetcher

log = logging.getLogger(__name__)

# Track aggregated enrichment errors per cycle (rate-limit noisy logs)
_enrichment_error_counts: Dict[str, int] = {}
_enrichment_cycle_start: float = 0.0

ENRICHMENT_TIMEOUT = int(settings.enrichment_timeout_per_team)


class _EnrichmentTimeout(Exception):
    pass


@contextmanager
def _timeout_guard(seconds: int):
    """Context manager that raises _EnrichmentTimeout after `seconds`.

    Falls back to a no-op on platforms without SIGALRM (Windows).
    """
    def _handler(signum, frame):
        raise _EnrichmentTimeout(f"Enrichment timed out after {seconds}s")

    if hasattr(signal, "SIGALRM"):
        old_handler = signal.signal(signal.SIGALRM, _handler)
        signal.alarm(seconds)
        try:
            yield
        finally:
            signal.alarm(0)
            signal.signal(signal.SIGALRM, old_handler)
    else:
        yield


def _log_enrichment_summary():
    """Log aggregated enrichment errors once per cycle instead of per-team."""
    global _enrichment_error_counts
    if _enrichment_error_counts:
        summary = ", ".join(f"{k}={v}" for k, v in sorted(_enrichment_error_counts.items()))
        log.warning("enrichment_cycle_summary: errors=%s", summary)
        _enrichment_error_counts.clear()


def _record_enrichment_error(category: str):
    """Increment error count for rate-limited logging."""
    _enrichment_error_counts[category] = _enrichment_error_counts.get(category, 0) + 1


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

    # Check if news source is available
    if not is_available("newsapi"):
        _record_enrichment_error("newsapi_breaker_open")
        cache.set_json(k, 0.0, 3600)
        return 0.0

    try:
        with _timeout_guard(ENRICHMENT_TIMEOUT):
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
                try:
                    s = nlp.analyze(text=text, context=f"Team={team}")
                    vals.append(_norm_label(s.label) * float(s.confidence))
                except Exception:
                    _record_enrichment_error("ollama_analyze")
                    continue

            # ── Reddit sentiment (supplementary source) ──────────────
            try:
                reddit = RedditFetcher()
                reddit_text = reddit.fetch_team_sentiment_posts_sync(team)
                if reddit_text:
                    rs = nlp.analyze(text=reddit_text, context=f"Reddit r/sportsbook Team={team}")
                    vals.append(_norm_label(rs.label) * float(rs.confidence))
            except Exception as exc:
                _record_enrichment_error("reddit_sentiment")

            if not vals:
                cache.set_json(k, 0.0, 6 * 3600)
                record_success("newsapi")
                return 0.0

            score = float(sum(vals) / max(1, len(vals)))
            cache.set_json(k, score, 6 * 3600)
            record_success("newsapi")
            return score

    except _EnrichmentTimeout:
        _record_enrichment_error("timeout")
        cache.set_json(k, 0.0, 3600)
        return 0.0
    except Exception as exc:
        _record_enrichment_error("sentiment_general")
        record_failure("newsapi", str(exc)[:100])
        cache.set_json(k, 0.0, 6 * 3600)
        return 0.0


def batch_team_sentiment(teams: List[str], max_teams: int = 24) -> Dict[str, float]:
    global _enrichment_cycle_start, _enrichment_error_counts
    _enrichment_cycle_start = time.time()
    _enrichment_error_counts.clear()

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

    # Log aggregated summary instead of per-team noise
    _log_enrichment_summary()
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
