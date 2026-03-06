"""Structured Reddit RSS sentiment sources with tier weights.

Provides machine-readable source config for the sentiment pipeline.
See CHANGELOG for tier semantics (core/fact_only/high_noise).
"""

SENTIMENT_SOURCES = {
    "core": {
        "default_weight": 1.00,
        "mode": "full_sentiment",
        "required_keywords": [],
        "feeds": [
            {"url": "https://www.reddit.com/r/soccer/.rss"},
            {"url": "https://www.reddit.com/r/championsleague/.rss"},
            {"url": "https://www.reddit.com/r/EuropaLeague/.rss"},
            {"url": "https://www.reddit.com/r/PremierLeague/.rss"},
            {"url": "https://www.reddit.com/r/Bundesliga/.rss"},
            {"url": "https://www.reddit.com/r/LaLiga/.rss"},
            {"url": "https://www.reddit.com/r/seriea/.rss"},
            {"url": "https://www.reddit.com/r/Ligue1/.rss"},
            {"url": "https://www.reddit.com/r/nba/.rss"},
            {"url": "https://www.reddit.com/r/nfl/.rss"},
            {"url": "https://www.reddit.com/r/hockey/.rss"},
            {"url": "https://www.reddit.com/r/tennis/.rss"},
        ],
    },
    "fact_only": {
        "default_weight": 0.35,
        "mode": "fact_only",
        "required_keywords": [
            "out", "injury", "doubtful", "suspended",
            "lineup", "ruled out", "questionable",
            "verletzt", "verletzung", "ausfall", "fällt aus",
            "gesperrt", "sperre", "aufstellung", "fraglich", "rückkehr"
        ],
        "feeds": [
            {"url": "https://www.reddit.com/r/fcbayern/.rss"},
            {"url": "https://www.reddit.com/r/borussiadortmund/.rss"},
            {"url": "https://www.reddit.com/r/coys/.rss"},
            {"url": "https://www.reddit.com/r/reddevils/.rss"},
            {"url": "https://www.reddit.com/r/LiverpoolFC/.rss"},
            {"url": "https://www.reddit.com/r/Gunners/.rss"},
            {"url": "https://www.reddit.com/r/chelseafc/.rss"},
            {"url": "https://www.reddit.com/r/MCFC/.rss"},
            {"url": "https://www.reddit.com/r/realmadrid/.rss"},
            {"url": "https://www.reddit.com/r/Barca/.rss"},
            {"url": "https://www.reddit.com/r/Juve/.rss"},
            {"url": "https://www.reddit.com/r/ACMilan/.rss"},
            {"url": "https://www.reddit.com/r/FCInterMilan/.rss"},
            {"url": "https://www.reddit.com/r/psg/.rss"},
            {"url": "https://www.reddit.com/r/footballhighlights/.rss"},
            {"url": "https://www.reddit.com/r/nbadiscussion/.rss"},
        ],
    },
    "high_noise": {
        "default_weight": 0.15,
        "mode": "contrarian",
        "required_keywords": [],
        "feeds": [
            {"url": "https://www.reddit.com/r/sportsbook/.rss"},
            {"url": "https://www.reddit.com/r/sportsbetting/.rss"},
        ],
    },
}

# Flat list for convenience
ALL_FEED_URLS = [
    f["url"]
    for tier in SENTIMENT_SOURCES.values()
    for f in tier["feeds"]
]
