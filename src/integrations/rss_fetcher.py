"""RSS news scraper for injury/lineup news from Rotowire.

Fetches sport-specific RSS feeds and filters for injury-related entries
published within the last 24 hours. Zero API cost.

Rotowire RSS source reliability (researched 2026-03):
  - NBA:    https://www.rotowire.com/rss/news.htm?sport=nba     ✅ reliable
  - NFL:    https://www.rotowire.com/rss/news.htm?sport=nfl     ✅ reliable (off-season: sparse)
  - NHL:    https://www.rotowire.com/rss/news.htm?sport=nhl     ✅ reliable
  - Soccer: https://www.rotowire.com/rss/news.htm?sport=soccer  ✅ reliable (US-centric)

  Rotowire officially offers RSS feeds for all four sports at no cost for
  blogs/personal use (must link back to rotowire.com, which is embedded in
  the feed items). The feeds are XML/RSS 2.0, no API key needed.

  Known issues:
    - Rotowire may return 403 if the User-Agent looks like a bot scraper.
      We set a browser-like User-Agent to avoid this.
    - During off-season (NFL: March-August, NHL: June-September), feeds
      may return few or no entries. This is expected, not an error.
    - Some entries lack structured_parsed dates; we fall back to string
      parsing with multiple formats.

  Fallback alternatives if Rotowire is unreliable:
    - RotoBaller RSS (https://www.rotoballer.com/) — similar coverage
    - ESPN RSS feeds (https://www.espn.com/espn/rss/)
    - NBC Sports Rotoworld (https://www.nbcsports.com/fantasy/)

Cache strategy (prevents IP bans):
  - Each feed URL is fetched AT MOST once per 30 minutes.
  - Redis stores the parsed entries; a per-feed threading.Lock prevents
    concurrent cache-miss stampedes.

Retry strategy:
  - Up to 3 attempts with exponential backoff (2s, 4s).
  - Timeout: 15 seconds per attempt.
  - On total failure: return empty list (graceful degradation).
"""
from __future__ import annotations

import logging
import threading
import time
import urllib.request
import urllib.error
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Optional

from src.data.redis_cache import cache

log = logging.getLogger(__name__)

# Rotowire RSS feed URLs per sport category
ROTOWIRE_FEEDS: Dict[str, str] = {
    "nba": "https://www.rotowire.com/rss/news.htm?sport=nba",
    "nfl": "https://www.rotowire.com/rss/news.htm?sport=nfl",
    "nhl": "https://www.rotowire.com/rss/news.htm?sport=nhl",
    "soccer": "https://www.rotowire.com/rss/news.htm?sport=soccer",
}

# Map OddsAPI sport keys to Rotowire feed categories
SPORT_TO_FEED: Dict[str, str] = {
    "basketball_nba": "nba",
    "americanfootball_nfl": "nfl",
    "icehockey_nhl": "nhl",
}

# Injury-related keywords for filtering
INJURY_KEYWORDS = [
    "injury", "injured", "out", "questionable", "doubtful",
    "ruled out", "day-to-day", "dnp", "inactive", "suspended",
    "concussion", "hamstring", "knee", "ankle", "acl", "mcl",
    "torn", "fracture", "strain", "sprain", "sidelined",
    "lineup", "starting", "bench", "rest", "load management",
    "probable", "game-time decision",
]

RSS_CACHE_TTL = 30 * 60  # 30 minutes

# Browser-like User-Agent to avoid 403 from Rotowire
USER_AGENT = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/120.0.0.0 Safari/537.36"
)

# Retry config
MAX_RETRIES = 3
RETRY_BACKOFF = [2, 4]  # seconds between retries
FETCH_TIMEOUT = 15  # seconds per attempt

# Per-feed lock: ensures only ONE thread fetches a given feed URL at a time.
_feed_locks: Dict[str, threading.Lock] = {k: threading.Lock() for k in ROTOWIRE_FEEDS}

# Feed health tracking
_feed_health: Dict[str, Dict] = {}


def _sport_key_to_feed(sport_key: str) -> str:
    """Map an OddsAPI sport key to the Rotowire feed category."""
    if sport_key in SPORT_TO_FEED:
        return SPORT_TO_FEED[sport_key]
    if sport_key.startswith("soccer"):
        return "soccer"
    return ""


def _parse_published(entry) -> Optional[datetime]:
    """Parse the published date from a feedparser entry."""
    import time as time_mod

    published_parsed = getattr(entry, "published_parsed", None)
    if published_parsed:
        try:
            return datetime.fromtimestamp(time_mod.mktime(published_parsed), tz=timezone.utc)
        except (ValueError, OverflowError, OSError):
            pass

    published_str = getattr(entry, "published", "")
    if published_str:
        for fmt in (
            "%a, %d %b %Y %H:%M:%S %Z",
            "%a, %d %b %Y %H:%M:%S %z",
            "%Y-%m-%dT%H:%M:%S%z",
            "%Y-%m-%d %H:%M:%S",
        ):
            try:
                dt = datetime.strptime(published_str, fmt)
                if dt.tzinfo is None:
                    dt = dt.replace(tzinfo=timezone.utc)
                return dt
            except ValueError:
                continue
    return None


def _matches_team(text: str, team_names: List[str]) -> bool:
    """Check if any team name appears in the text (case-insensitive)."""
    text_lower = text.lower()
    for name in team_names:
        if name.lower() in text_lower:
            return True
    return False


def _has_injury_keyword(text: str) -> bool:
    """Check if any injury keyword appears in the text."""
    text_lower = text.lower()
    return any(kw in text_lower for kw in INJURY_KEYWORDS)


def get_feed_health() -> Dict[str, Dict]:
    """Return health status of each RSS feed (for diagnostics)."""
    return dict(_feed_health)


class RSSFetcher:
    """Fetches and filters injury/lineup news from Rotowire RSS feeds.

    All errors are caught and logged — this class NEVER raises exceptions
    that could crash the Core pipeline.
    """

    def fetch_injury_news(
        self,
        sport_key: str,
        team_names: List[str],
        max_age_hours: int = 24,
    ) -> List[Dict[str, str]]:
        """Fetch injury-related RSS entries for given teams.

        Returns empty list on any error (graceful degradation).
        """
        try:
            return self._fetch_injury_news_inner(sport_key, team_names, max_age_hours)
        except Exception as exc:
            log.warning("RSS fetch_injury_news failed (sport=%s): %s", sport_key, exc)
            return []

    def _fetch_injury_news_inner(
        self,
        sport_key: str,
        team_names: List[str],
        max_age_hours: int,
    ) -> List[Dict[str, str]]:
        feed_key = _sport_key_to_feed(sport_key)
        if not feed_key:
            return []

        feed_url = ROTOWIRE_FEEDS.get(feed_key)
        if not feed_url:
            return []

        # Check Redis cache first (fast path — no lock needed)
        cache_key = f"rss:rotowire:{feed_key}"
        cached_entries = cache.get_json(cache_key)

        if cached_entries is None:
            lock = _feed_locks.get(feed_key) or threading.Lock()
            with lock:
                cached_entries = cache.get_json(cache_key)
                if cached_entries is None:
                    cached_entries = self._fetch_feed_with_retry(feed_url, feed_key)
                    if cached_entries is not None:
                        cache.set_json(cache_key, cached_entries, RSS_CACHE_TTL)
                    else:
                        cached_entries = []

        # Filter entries
        cutoff = datetime.now(timezone.utc) - timedelta(hours=max_age_hours)
        results: List[Dict[str, str]] = []

        for entry in cached_entries:
            pub_str = entry.get("published", "")
            try:
                pub_dt = datetime.fromisoformat(pub_str)
            except (ValueError, TypeError):
                pub_dt = datetime.now(timezone.utc)

            if pub_dt < cutoff:
                continue

            title = entry.get("title", "")
            summary = entry.get("summary", "")
            combined_text = f"{title} {summary}"

            if _matches_team(combined_text, team_names) and _has_injury_keyword(combined_text):
                matched_team = ""
                for name in team_names:
                    if name.lower() in combined_text.lower():
                        matched_team = name
                        break
                results.append({
                    "title": title,
                    "summary": summary[:500],
                    "published": pub_str,
                    "source": "rotowire",
                    "team_match": matched_team,
                })

        return results

    def _fetch_feed_with_retry(
        self, url: str, feed_key: str
    ) -> Optional[List[Dict[str, str]]]:
        """Fetch RSS feed with retry + exponential backoff."""
        last_error = None
        for attempt in range(MAX_RETRIES):
            try:
                result = self._fetch_feed(url)
                # Track health
                _feed_health[feed_key] = {
                    "status": "ok",
                    "entries": len(result) if result else 0,
                    "ts": datetime.now(timezone.utc).isoformat(),
                    "attempts": attempt + 1,
                }
                return result
            except Exception as exc:
                last_error = exc
                log.warning(
                    "RSS fetch attempt %d/%d failed for %s: %s",
                    attempt + 1, MAX_RETRIES, feed_key, exc,
                )
                if attempt < len(RETRY_BACKOFF):
                    time.sleep(RETRY_BACKOFF[attempt])

        # All retries exhausted
        _feed_health[feed_key] = {
            "status": "degraded",
            "error": str(last_error),
            "ts": datetime.now(timezone.utc).isoformat(),
            "attempts": MAX_RETRIES,
        }
        log.error(
            "RSS feed %s DEGRADED after %d attempts: %s",
            feed_key, MAX_RETRIES, last_error,
        )
        return []

    def _fetch_feed(self, url: str) -> List[Dict[str, str]]:
        """Parse an RSS feed URL with User-Agent header.

        Raises on hard failures (caller handles retry).
        Returns empty list on parse failures (bad XML but reachable).
        """
        try:
            import feedparser
        except ImportError:
            log.warning("feedparser not installed. pip install feedparser")
            return []

        # Fetch with User-Agent to avoid 403
        req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
        try:
            with urllib.request.urlopen(req, timeout=FETCH_TIMEOUT) as resp:
                raw_bytes = resp.read()
        except urllib.error.HTTPError as exc:
            if exc.code == 403:
                log.warning("RSS 403 Forbidden for %s (bot protection?)", url)
            raise
        except (urllib.error.URLError, OSError) as exc:
            raise RuntimeError(f"RSS network error for {url}: {exc}") from exc

        # Parse the fetched bytes
        try:
            feed = feedparser.parse(raw_bytes)
        except Exception as exc:
            log.warning("RSS feedparser.parse() crashed for %s: %s", url, exc)
            return []

        if feed.bozo and not feed.entries:
            log.warning("RSS bozo error for %s: %s", url, getattr(feed, "bozo_exception", "unknown"))
            return []

        entries: List[Dict[str, str]] = []
        for entry in feed.entries:
            try:
                pub_dt = _parse_published(entry)
                entries.append({
                    "title": str(getattr(entry, "title", "") or ""),
                    "summary": str(getattr(entry, "summary", "") or "")[:500],
                    "published": pub_dt.isoformat() if pub_dt else "",
                    "link": str(getattr(entry, "link", "") or ""),
                })
            except Exception as exc:
                log.debug("Skipping malformed RSS entry: %s", exc)
                continue

        return entries

    def fetch_all_sports_news(
        self,
        active_sports: List[str],
        team_names: List[str],
        max_age_hours: int = 24,
    ) -> List[Dict[str, str]]:
        """Fetch injury news across all active sports at once."""
        all_results: List[Dict[str, str]] = []
        seen_feeds: set = set()

        for sport in active_sports:
            feed_key = _sport_key_to_feed(sport)
            if not feed_key or feed_key in seen_feeds:
                continue
            seen_feeds.add(feed_key)

            try:
                results = self.fetch_injury_news(sport, team_names, max_age_hours)
                all_results.extend(results)
            except Exception as exc:
                log.warning("RSS fetch_all failed for sport %s: %s", sport, exc)

        return all_results
