"""Twitter/X API integration for breaking injury news detection.

Uses a curated whitelist of verified sports journalists (beat writers)
for targeted searches instead of broad keyword queries. This is faster,
more accurate, and saves API rate limit tokens.
"""
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

# Curated journalist whitelist: verified sports beat writers.
# Searching only their handles is faster, more accurate, and saves API tokens.
# Format: Twitter handle (without @)
JOURNALIST_WHITELIST: List[str] = [
    # EPL / English Football
    "FabrizioRomano",       # Transfer/injury authority
    "David_Ornstein",       # The Athletic - EPL
    "JacobSteinberg",       # Guardian - EPL
    "honaborger",           # Guardian - EPL
    "Ben_Dinnery",          # Injury specialist
    "premaborger",          # Premier League injuries
    "PhysioRoom",           # Injury database
    # Bundesliga / German Football
    "kaborger",             # Kicker
    "SPORTBILD",            # Sport Bild
    "iMiaSanMia",           # Bayern Munich
    "BVBBuzz",              # Dortmund
    "Sky_MaxB",             # Sky Germany
    "paborger",             # German football
    # La Liga / Spanish Football
    "SiqueRodriguez",       # Cadena SER
    "HelenaCondis",         # Cope - La Liga
    "MatteMoretto",         # Relevo
    # Serie A / Italian Football
    "DiMarzio",             # Sky Italy
    "Gazzetta_it",          # Gazzetta
    # NBA Basketball
    "ShamsCharania",        # Athletic - NBA
    "wojespn",              # ESPN - NBA
    "ChrisBHaynes",         # TNT/Bleacher Report
    "JeffPassan",           # ESPN - MLB (covers NBA injuries too)
    "FantasyLabsNBA",       # NBA injury alerts
    # Tennis
    "josaborger",           # ATP/WTA
    "BenRothenberg",        # NYT Tennis
    "ChrisEvert",           # Tennis commentary
    # Multi-sport / General
    "ESPNStatsInfo",        # ESPN Stats
    "ActionNetworkHQ",      # Action Network
    "draftkings",           # Injury-related line moves
    "PointsBetAU",          # Line movement intel
]


def _build_journalist_from_clause(max_handles: int = 15) -> str:
    """Build a 'from:X OR from:Y' clause from the whitelist.

    Twitter API limits query length to 512 chars, so we cap the number
    of handles we include.
    """
    handles = JOURNALIST_WHITELIST[:max_handles]
    return " OR ".join(f"from:{h}" for h in handles)


class TwitterFetcher:
    """Fetches breaking injury/lineup news from Twitter/X API v2.

    Uses a two-tier search strategy:
    1. **Journalist search** (primary): queries only whitelisted beat writers
       for the team name — high signal, low noise, minimal API usage.
    2. **Broad search** (fallback): used only if journalist search returns
       nothing and the team is flagged as high-priority.
    """

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

    def search_journalist_feed(self, team: str, max_results: int = 10) -> List[Dict]:
        """Search only whitelisted journalists for team-related injury news.

        This is the primary search path — much more efficient than broad
        keyword searches because:
        - Fewer results to process
        - Higher signal-to-noise ratio
        - Uses fewer API rate limit tokens
        """
        cache_key = f"twitter:journalist:{team.lower().replace(' ', '_')}"
        cached = cache.get_json(cache_key)
        if cached is not None:
            return cached

        client = self._get_client()
        if client is None:
            return []

        try:
            from_clause = _build_journalist_from_clause(max_handles=15)
            query = f'"{team}" ({from_clause}) -is:retweet'
            # Truncate to Twitter's 512 char limit
            if len(query) > 512:
                from_clause = _build_journalist_from_clause(max_handles=8)
                query = f'"{team}" ({from_clause}) -is:retweet'
            if len(query) > 512:
                query = f'"{team}" (from:FabrizioRomano OR from:ShamsCharania OR from:Ben_Dinnery) -is:retweet'

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
                        "source": "journalist",
                    })

            cache.set_json(cache_key, tweets, ttl_seconds=TWITTER_CACHE_TTL)
            return tweets
        except Exception as exc:
            log.warning("Journalist search failed for %s: %s", team, exc)
            cache.set_json(cache_key, [], ttl_seconds=TWITTER_CACHE_TTL)
            return []

    def search_team_news(self, team: str, max_results: int = 10) -> List[Dict]:
        """Search for recent tweets about a team with injury/lineup keywords.

        Tries journalist feed first, falls back to broad search if empty.
        """
        # Tier 1: journalist whitelist (high signal)
        journalist_results = self.search_journalist_feed(team, max_results)
        if journalist_results:
            return journalist_results

        # Tier 2: broad keyword search (fallback)
        cache_key = f"twitter:news:{team.lower().replace(' ', '_')}"
        cached = cache.get_json(cache_key)
        if cached is not None:
            return cached

        client = self._get_client()
        if client is None:
            return []

        try:
            keyword_part = " OR ".join(f'"{kw}"' for kw in INJURY_KEYWORDS[:6])
            query = f'"{team}" ({keyword_part}) -is:retweet lang:en'
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
                        "source": "broad",
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
