"""RSS news scraper for injury/lineup news from Rotowire.

Fetches sport-specific RSS feeds and filters for injury-related entries
published within the last 24 hours. Zero API cost.
"""
from __future__ import annotations

import logging
from datetime import datetime, timezone, timedelta
from typing import Dict, List

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

RSS_CACHE_TTL = 15 * 60  # 15 minutes


def _sport_key_to_feed(sport_key: str) -> str:
    """Map an OddsAPI sport key to the Rotowire feed category."""
    if sport_key in SPORT_TO_FEED:
        return SPORT_TO_FEED[sport_key]
    if sport_key.startswith("soccer"):
        return "soccer"
    return ""


def _parse_published(entry) -> datetime | None:
    """Parse the published date from a feedparser entry."""
    import time

    published_parsed = getattr(entry, "published_parsed", None)
    if published_parsed:
        try:
            return datetime.fromtimestamp(time.mktime(published_parsed), tz=timezone.utc)
        except (ValueError, OverflowError, OSError):
            pass

    published_str = getattr(entry, "published", "")
    if published_str:
        for fmt in ("%a, %d %b %Y %H:%M:%S %Z", "%a, %d %b %Y %H:%M:%S %z"):
            try:
                return datetime.strptime(published_str, fmt).replace(tzinfo=timezone.utc)
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


class RSSFetcher:
    """Fetches and filters injury/lineup news from Rotowire RSS feeds."""

    def fetch_injury_news(
        self,
        sport_key: str,
        team_names: List[str],
        max_age_hours: int = 24,
    ) -> List[Dict[str, str]]:
        """Fetch injury-related RSS entries for given teams.

        Parameters
        ----------
        sport_key : str
            OddsAPI sport key (e.g. "basketball_nba", "soccer_epl").
        team_names : list of str
            Team names to filter for (e.g. ["Lakers", "Celtics"]).
        max_age_hours : int
            Only keep entries published within this many hours.

        Returns
        -------
        list of dict
            Each dict has keys: title, summary, published, source, team_match.
        """
        feed_key = _sport_key_to_feed(sport_key)
        if not feed_key:
            return []

        feed_url = ROTOWIRE_FEEDS.get(feed_key)
        if not feed_url:
            return []

        # Check cache first
        cache_key = f"rss:rotowire:{feed_key}"
        cached_entries = cache.get_json(cache_key)

        if cached_entries is None:
            cached_entries = self._fetch_feed(feed_url)
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

            # Must mention a team AND have an injury keyword
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

    def _fetch_feed(self, url: str) -> List[Dict[str, str]] | None:
        """Parse an RSS feed URL. Returns list of serializable entry dicts."""
        try:
            import feedparser
        except ImportError:
            log.warning("feedparser not installed. RSS integration disabled. pip install feedparser")
            return None

        try:
            feed = feedparser.parse(url)
            if feed.bozo and not feed.entries:
                log.warning("RSS feed parse error for %s: %s", url, feed.bozo_exception)
                return []

            entries: List[Dict[str, str]] = []
            for entry in feed.entries:
                pub_dt = _parse_published(entry)
                entries.append({
                    "title": getattr(entry, "title", ""),
                    "summary": getattr(entry, "summary", "")[:500],
                    "published": pub_dt.isoformat() if pub_dt else "",
                    "link": getattr(entry, "link", ""),
                })
            return entries

        except Exception as exc:
            log.warning("RSS fetch failed for %s: %s", url, exc)
            return []

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
                log.warning("RSS fetch failed for sport %s: %s", sport, exc)

        return all_results
