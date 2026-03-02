"""Twitter/X API integration for breaking injury news detection."""
from __future__ import annotations

import logging
from typing import Dict, List, Optional

from src.core.settings import settings
from src.data.redis_cache import cache

log = logging.getLogger(__name__)

TWITTER_CACHE_TTL = 30 * 60  # 30 minutes

# Injury-related keywords for search
INJURY_KEYWORDS = [
    "injury", "injured", "out", "doubtful", "ruled out",
    "lineup", "starting XI", "bench", "suspended", "hamstring",
    "ankle", "knee", "concussion", "muscle", "groin",
    "verletzt", "ausfall", "fehlt", "Aufstellung",  # German
]


class TwitterFetcher:
    """Fetches breaking injury/lineup news from Twitter/X API v2."""

    def __init__(self):
        self._enabled = settings.twitter_enabled and bool(settings.twitter_bearer_token)
        self._client = None

    def _get_client(self):
        if self._client is not None:
            return self._client
        if not self._enabled:
            return None
        try:
            import tweepy
            self._client = tweepy.Client(bearer_token=settings.twitter_bearer_token)
            return self._client
        except ImportError:
            log.warning("tweepy not installed. Twitter integration disabled.")
            self._enabled = False
            return None
        except Exception as exc:
            log.warning("Twitter client init failed: %s", exc)
            self._enabled = False
            return None

    def search_team_news(self, team: str, max_results: int = 10) -> List[Dict]:
        """Search for recent tweets about a team with injury/lineup keywords."""
        cache_key = f"twitter:news:{team.lower().replace(' ', '_')}"
        cached = cache.get_json(cache_key)
        if cached is not None:
            return cached

        client = self._get_client()
        if client is None:
            return []

        try:
            # Build query: team name + injury keywords
            keyword_part = " OR ".join(f'"{kw}"' for kw in INJURY_KEYWORDS[:6])
            query = f'"{team}" ({keyword_part}) -is:retweet lang:en'
            # Truncate to Twitter's 512 char limit
            if len(query) > 512:
                query = f'"{team}" ("injury" OR "out" OR "lineup") -is:retweet'

            response = client.search_recent_tweets(
                query=query,
                max_results=min(max_results, 100),
                tweet_fields=["created_at", "text", "author_id"],
            )
            tweets = []
            if response and response.data:
                for tweet in response.data:
                    tweets.append({
                        "text": tweet.text,
                        "created_at": str(tweet.created_at) if tweet.created_at else "",
                        "id": str(tweet.id),
                    })

            cache.set_json(cache_key, tweets, ttl_seconds=TWITTER_CACHE_TTL)
            return tweets
        except Exception as exc:
            log.warning("Twitter search failed for %s: %s", team, exc)
            cache.set_json(cache_key, [], ttl_seconds=TWITTER_CACHE_TTL)
            return []

    def check_breaking_injury(self, team: str) -> Optional[str]:
        """Check if there's a breaking injury report for a team. Returns text or None."""
        tweets = self.search_team_news(team, max_results=5)
        if not tweets:
            return None

        # Look for high-signal injury tweets
        for tweet in tweets:
            text = (tweet.get("text") or "").lower()
            high_signal = ["ruled out", "will miss", "sidelined", "torn",
                           "broken", "ausfall", "fehlt"]
            if any(kw in text for kw in high_signal):
                return tweet.get("text", "")

        return None
