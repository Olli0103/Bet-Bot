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
from src.integrations.multi_news_fetcher import MultiNewsFetcher
from src.integrations.ollama_sentiment import OllamaSentimentClient
from src.integrations.reddit_fetcher import RedditFetcher

# Backward-compat for tests/patches still referencing NewsFetcher symbol.
NewsFetcher = MultiNewsFetcher

log = logging.getLogger(__name__)

# Track aggregated enrichment errors/metrics per cycle (rate-limit noisy logs)
_enrichment_error_counts: Dict[str, int] = {}
_enrichment_metrics: Dict[str, int] = {}
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


def _log_enrichment_summary(teams_processed: int = 0):
    """Log aggregated enrichment cycle summary instead of per-team noise."""
    global _enrichment_error_counts, _enrichment_metrics, _enrichment_cycle_start
    elapsed = round(time.time() - _enrichment_cycle_start, 2) if _enrichment_cycle_start else 0
    error_count = sum(_enrichment_error_counts.values())
    error_detail = ", ".join(f"{k}={v}" for k, v in sorted(_enrichment_error_counts.items()))

    # Source health snapshot
    source_health_parts = []
    for source_name in ("newsapi", "rotowire_rss", "ollama"):
        avail = is_available(source_name)
        source_health_parts.append(f"{source_name}={'ok' if avail else 'DOWN'}")

    # Budget snapshot from multi_news_fetcher
    budget_parts = []
    try:
        from src.integrations.multi_news_fetcher import MultiNewsFetcher
        health = MultiNewsFetcher().get_source_health()
        for src_name, info in health.items():
            budget_parts.append(f"{src_name}={info.get('used', 0)}/{info.get('budget', '?')}")
    except Exception:
        pass

    metric_detail = ", ".join(f"{k}={v}" for k, v in sorted(_enrichment_metrics.items()))

    log.info(
        "enrichment_cycle_summary: teams=%d elapsed=%.1fs errors=%d (%s) metrics=[%s] "
        "sources=[%s] budget=[%s]",
        teams_processed, elapsed, error_count,
        error_detail or "none",
        metric_detail or "none",
        ", ".join(source_health_parts),
        ", ".join(budget_parts) or "n/a",
    )
    _enrichment_error_counts.clear()
    _enrichment_metrics.clear()


def _record_enrichment_error(category: str):
    """Increment error count for rate-limited logging."""
    _enrichment_error_counts[category] = _enrichment_error_counts.get(category, 0) + 1


def _record_enrichment_metric(name: str, amount: int = 1):
    """Increment enrichment telemetry counters for observability."""
    _enrichment_metrics[name] = _enrichment_metrics.get(name, 0) + int(amount)


def _norm_label(label: str) -> float:
    l = (label or "neutral").lower()
    if l == "positive":
        return 1.0
    if l == "negative":
        return -1.0
    return 0.0


def team_sentiment_score(team: str, limit_articles: int = 0) -> float:
    if limit_articles <= 0:
        limit_articles = settings.enrichment_news_articles_per_team
    k = f"enrich:sentiment:{team.lower()}"
    cached = cache.get_json(k)
    if cached is not None:
        return float(cached)

    # Check if news source is available
    if not is_available("newsapi"):
        _record_enrichment_error("newsapi_breaker_open")
        _record_enrichment_metric("source_breaker_open")
        cache.set_json(k, 0.0, 3600)
        return 0.0

    try:
        with _timeout_guard(ENRICHMENT_TIMEOUT):
            fetcher = MultiNewsFetcher(min_articles=3, max_articles=max(5, limit_articles))
            nlp = OllamaSentimentClient()
            query = f'"{team}"'
            articles = fetcher.search(query=query, language="en", team_names=[team])[:limit_articles]
            _record_enrichment_metric("articles_found_total", len(articles))

            if not articles:
                _record_enrichment_metric("zero_articles")
                log.warning("enrichment_starvation: team='%s' returned 0 articles", team)

            vals: List[float] = []
            for a in articles:
                text = "\n".join(
                    [
                        a.get("title") or "",
                        a.get("summary") or a.get("description") or "",
                        a.get("content") or "",
                    ]
                ).strip()
                if not text:
                    _record_enrichment_metric("empty_article_text")
                    continue
                try:
                    s = nlp.analyze(text=text, context=f"Team={team}")
                    vals.append(_norm_label(s.label) * float(s.confidence))
                except Exception:
                    _record_enrichment_error("ollama_analyze")
                    _record_enrichment_metric("llm_parse_fail")
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
                _record_enrichment_metric("neutral_fallback")
                cache.set_json(k, 0.0, 6 * 3600)
                record_success("newsapi")
                return 0.0

            score = float(sum(vals) / max(1, len(vals)))
            cache.set_json(k, score, 6 * 3600)
            record_success("newsapi")
            return score

    except _EnrichmentTimeout:
        _record_enrichment_error("timeout")
        _record_enrichment_metric("timeout_fallback")
        cache.set_json(k, 0.0, 3600)
        return 0.0
    except Exception as exc:
        _record_enrichment_error("sentiment_general")
        _record_enrichment_metric("exception_fallback")
        record_failure("newsapi", str(exc)[:100])
        cache.set_json(k, 0.0, 6 * 3600)
        return 0.0


def batch_team_sentiment(teams: List[str], max_teams: int = 0) -> Dict[str, float]:
    """Compute sentiment for a batch of teams with configurable limit.

    max_teams defaults to settings.enrichment_max_teams when 0.
    Best-effort: individual team failures don't block the pipeline.
    """
    global _enrichment_cycle_start, _enrichment_error_counts, _enrichment_metrics
    _enrichment_cycle_start = time.time()
    _enrichment_error_counts.clear()
    _enrichment_metrics.clear()

    if max_teams <= 0:
        max_teams = settings.enrichment_max_teams

    out: Dict[str, float] = {}
    uniq = []
    for t in teams:
        if t and t not in uniq:
            uniq.append(t)

    processed = 0
    selected = uniq[:max_teams]
    _record_enrichment_metric("teams_selected", len(selected))
    _record_enrichment_metric("teams_total_unique", len(uniq))

    for t in selected:
        try:
            out[t] = team_sentiment_score(t)
            processed += 1
        except Exception:
            out[t] = 0.0
            _record_enrichment_error("team_sentiment_exception")
            processed += 1

    # Log aggregated summary instead of per-team noise
    _log_enrichment_summary(teams_processed=processed)
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
