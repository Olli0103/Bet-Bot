"""Reddit integration for crowdsourced injury/lineup intelligence.

Scans sport-specific subreddits for breaking injury news using PRAW.
Filters for injury-related keywords and official flair in recent posts.
"""
from __future__ import annotations

import logging
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Set

from src.core.settings import settings
from src.data.redis_cache import cache

log = logging.getLogger(__name__)

# Map OddsAPI sport keys to relevant subreddits
SPORT_SUBREDDITS: Dict[str, List[str]] = {
    "basketball_nba": ["nba", "fantasybball"],
    "americanfootball_nfl": ["nfl", "fantasyfootball"],
    "icehockey_nhl": ["hockey"],
}

# Catch-all patterns for soccer and tennis
SOCCER_SUBREDDITS = ["soccer"]
TENNIS_SUBREDDITS = ["tennis"]

# Injury-related keywords for title filtering
INJURY_KEYWORDS = [
    "injury", "injured", "out", "questionable", "doubtful",
    "ruled out", "day-to-day", "dnp", "inactive", "suspended",
    "concussion", "torn", "fracture", "strain", "sidelined",
    "miss", "will not play", "game-time", "load management",
    "lineup", "starting", "rest",
]

# Official flair patterns
FLAIR_KEYWORDS = ["news", "out", "injury", "report", "official", "breaking"]

REDDIT_CACHE_TTL = 10 * 60  # 10 minutes


def _get_subreddits_for_sport(sport_key: str) -> List[str]:
    """Map an OddsAPI sport key to subreddit names."""
    if sport_key in SPORT_SUBREDDITS:
        return SPORT_SUBREDDITS[sport_key]
    if sport_key.startswith("soccer"):
        return SOCCER_SUBREDDITS
    if sport_key.startswith("tennis"):
        return TENNIS_SUBREDDITS
    return []


def _has_injury_signal(title: str, flair: str) -> bool:
    """Check if a post title or flair suggests injury/lineup news."""
    text = title.lower()
    fl = (flair or "").lower()

    if any(kw in text for kw in INJURY_KEYWORDS):
        return True
    if any(kw in fl for kw in FLAIR_KEYWORDS):
        return True
    return False


class RedditFetcher:
    """Fetches injury/lineup intel from sport subreddits via PRAW."""

    def __init__(self):
        self._reddit = None
        self._enabled = bool(
            settings.reddit_client_id and settings.reddit_client_secret
        )

    def _get_reddit(self):
        """Lazy-init PRAW Reddit instance."""
        if self._reddit is not None:
            return self._reddit
        if not self._enabled:
            return None
        try:
            import praw
            self._reddit = praw.Reddit(
                client_id=settings.reddit_client_id,
                client_secret=settings.reddit_client_secret,
                user_agent=settings.reddit_user_agent,
            )
            return self._reddit
        except ImportError:
            log.warning("praw not installed. Reddit integration disabled. pip install praw")
            self._enabled = False
            return None
        except Exception as exc:
            log.warning("Reddit client init failed: %s", exc)
            self._enabled = False
            return None

    def fetch_injury_posts(
        self,
        sport_key: str,
        team_names: List[str],
        max_age_hours: int = 12,
        limit_per_sub: int = 50,
    ) -> List[Dict[str, str]]:
        """Scan subreddits for injury-related posts mentioning given teams.

        Parameters
        ----------
        sport_key : str
            OddsAPI sport key.
        team_names : list of str
            Team names to filter for.
        max_age_hours : int
            Only keep posts from the last N hours.
        limit_per_sub : int
            Max posts to scan per subreddit (new + hot combined).

        Returns
        -------
        list of dict with keys: title, text, subreddit, created, url, source
        """
        reddit = self._get_reddit()
        if reddit is None:
            return []

        subreddit_names = _get_subreddits_for_sport(sport_key)
        if not subreddit_names:
            return []

        # Check cache
        cache_key = f"reddit:injuries:{sport_key}"
        cached = cache.get_json(cache_key)
        if cached is not None:
            return self._filter_for_teams(cached, team_names)

        cutoff = datetime.now(timezone.utc) - timedelta(hours=max_age_hours)
        all_posts: List[Dict[str, str]] = []
        seen_ids: Set[str] = set()

        for sub_name in subreddit_names:
            try:
                subreddit = reddit.subreddit(sub_name)

                # Scan both 'new' and 'hot' feeds
                for listing in [subreddit.new(limit=limit_per_sub), subreddit.hot(limit=limit_per_sub)]:
                    for post in listing:
                        if post.id in seen_ids:
                            continue
                        seen_ids.add(post.id)

                        created_utc = datetime.fromtimestamp(post.created_utc, tz=timezone.utc)
                        if created_utc < cutoff:
                            continue

                        title = post.title or ""
                        flair = post.link_flair_text or ""

                        if not _has_injury_signal(title, flair):
                            continue

                        selftext = (post.selftext or "")[:500]
                        all_posts.append({
                            "title": title,
                            "text": selftext,
                            "subreddit": sub_name,
                            "flair": flair,
                            "created": created_utc.isoformat(),
                            "url": f"https://reddit.com{post.permalink}" if hasattr(post, "permalink") else "",
                            "source": "reddit",
                        })

            except Exception as exc:
                log.warning("Reddit fetch failed for r/%s: %s", sub_name, exc)
                continue

        # Cache all posts (unfiltered by team) so we don't re-fetch per team
        cache.set_json(cache_key, all_posts, REDDIT_CACHE_TTL)

        return self._filter_for_teams(all_posts, team_names)

    def _filter_for_teams(
        self, posts: List[Dict[str, str]], team_names: List[str]
    ) -> List[Dict[str, str]]:
        """Filter posts to only those mentioning given team names."""
        if not team_names:
            return posts

        results: List[Dict[str, str]] = []
        for post in posts:
            combined = f"{post.get('title', '')} {post.get('text', '')}".lower()
            for name in team_names:
                if name.lower() in combined:
                    post_copy = dict(post)
                    post_copy["team_match"] = name
                    results.append(post_copy)
                    break
        return results
