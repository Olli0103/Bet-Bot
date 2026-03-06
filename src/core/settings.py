from pydantic import BaseModel
from dotenv import load_dotenv
import os

load_dotenv(override=True)


class Settings(BaseModel):
    app_env: str = os.getenv("APP_ENV", "local")
    log_level: str = os.getenv("LOG_LEVEL", "INFO")
    timezone: str = os.getenv("TIMEZONE", "Europe/Berlin")

    apisports_api_key: str = os.getenv("APISPORTS_API_KEY", "")
    apisports_base_url: str = os.getenv("APISPORTS_BASE_URL", "https://v3.football.api-sports.io")

    newsapi_key: str = os.getenv("NEWSAPI_KEY", "")
    newsapi_base_url: str = os.getenv("NEWSAPI_BASE_URL", "https://newsapi.org/v2")

    gnews_api_key: str = os.getenv("GNEWS_API_KEY", "")
    gnews_base_url: str = os.getenv("GNEWS_BASE_URL", "https://gnews.io/api/v4")

    newsdata_api_key: str = os.getenv("NEWSDATA_API_KEY", "")
    newsdata_base_url: str = os.getenv("NEWSDATA_BASE_URL", "https://newsdata.io/api/1")

    # Reddit
    reddit_user_agent: str = os.getenv(
        "REDDIT_USER_AGENT",
        "python:bet-bot-sentiment-scraper:v1.0 (by /u/Olli0103)",
    )

    postgres_dsn: str = os.getenv("POSTGRES_DSN", "postgresql+psycopg://postgres:postgres@localhost:5432/signalbot")
    redis_url: str = os.getenv("REDIS_URL", "redis://localhost:6379/0")

    ollama_base_url: str = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    ollama_model: str = os.getenv("OLLAMA_MODEL", "gemma:4b")

    odds_api_key: str = os.getenv("ODDS_API_KEY", "")
    odds_api_base_url: str = os.getenv("ODDS_API_BASE_URL", "https://api.the-odds-api.com/v4")

    telegram_bot_token: str = os.getenv("TELEGRAM_BOT_TOKEN", "")
    telegram_chat_id: str = os.getenv("TELEGRAM_CHAT_ID", "")
    telegram_chat_ids: str = os.getenv("TELEGRAM_CHAT_IDS", "")  # CSV of allowed IDs

    live_sports: str = os.getenv("LIVE_SPORTS", "basketball_nba,soccer_germany_bundesliga,soccer_epl,tennis_atp")

    # Enrichment
    enrichment_enabled: bool = os.getenv("ENRICHMENT_ENABLED", "true").lower() == "true"
    # Reddit RSS (subreddit feeds, comma-separated full URLs)
    reddit_rss_feeds: str = os.getenv(
        "REDDIT_RSS_FEEDS",
        "https://www.reddit.com/r/soccer/.rss,"
        "https://www.reddit.com/r/championsleague/.rss,"
        "https://www.reddit.com/r/EuropaLeague/.rss,"
        "https://www.reddit.com/r/Bundesliga/.rss,"
        "https://www.reddit.com/r/PremierLeague/.rss,"
        "https://www.reddit.com/r/LaLiga/.rss,"
        "https://www.reddit.com/r/seriea/.rss,"
        "https://www.reddit.com/r/Ligue1/.rss,"
        "https://www.reddit.com/r/nba/.rss,"
        "https://www.reddit.com/r/nfl/.rss,"
        "https://www.reddit.com/r/hockey/.rss,"
        "https://www.reddit.com/r/tennis/.rss,"
        "https://www.reddit.com/r/fcbayern/.rss,"
        "https://www.reddit.com/r/borussiadortmund/.rss,"
        "https://www.reddit.com/r/coys/.rss,"
        "https://www.reddit.com/r/reddevils/.rss,"
        "https://www.reddit.com/r/LiverpoolFC/.rss,"
        "https://www.reddit.com/r/Gunners/.rss,"
        "https://www.reddit.com/r/chelseafc/.rss,"
        "https://www.reddit.com/r/MCFC/.rss,"
        "https://www.reddit.com/r/realmadrid/.rss,"
        "https://www.reddit.com/r/Barca/.rss,"
        "https://www.reddit.com/r/Juve/.rss,"
        "https://www.reddit.com/r/ACMilan/.rss,"
        "https://www.reddit.com/r/FCInterMilan/.rss,"
        "https://www.reddit.com/r/psg/.rss,"
        "https://www.reddit.com/r/nbadiscussion/.rss,"
        "https://www.reddit.com/r/footballhighlights/.rss,"
        "https://www.reddit.com/r/sportsbook/.rss,"
        "https://www.reddit.com/r/sportsbetting/.rss"
    )
    enrichment_timeout_per_team: int = int(os.getenv("ENRICHMENT_TIMEOUT", "30"))
    enrichment_max_teams: int = int(os.getenv("ENRICHMENT_MAX_TEAMS", "24"))
    enrichment_news_articles_per_team: int = int(os.getenv("ENRICHMENT_NEWS_ARTICLES_PER_TEAM", "8"))

    # SSL
    insecure_ssl_fallback: bool = os.getenv("INSECURE_SSL_FALLBACK", "false").lower() in ("true", "1", "yes")

    # Bankroll
    initial_bankroll: float = float(os.getenv("INITIAL_BANKROLL", "1000.0"))

    # Tipico tax (5.3% Wettsteuer as of 2025 — GlüStV 2021 updated rate)
    tipico_tax_rate: float = float(os.getenv("TIPICO_TAX_RATE", "0.053"))
    tax_free_mode: bool = os.getenv("TAX_FREE_MODE", "false").lower() == "true"

    # Data sources (TheSportsDB, football-data.org)
    sportsdb_api_key: str = os.getenv("SPORTSDB_API_KEY", "3")
    football_data_api_key: str = os.getenv("FOOTBALL_DATA_API_KEY", "")

    # Stats ingestion
    stats_ingestion_enabled: bool = os.getenv("STATS_INGESTION_ENABLED", "true").lower() == "true"
    stats_ingestion_interval_hours: int = int(os.getenv("STATS_INGESTION_INTERVAL_HOURS", "6"))

    # --- Risk Guards ---
    # Confidence gates per market type (model_probability minimum to allow a bet)
    min_confidence_soccer_h2h: float = float(os.getenv("MIN_CONF_SOCCER_H2H", "0.55"))
    min_confidence_soccer_totals: float = float(os.getenv("MIN_CONF_SOCCER_TOTALS", "0.56"))
    min_confidence_soccer_spread: float = float(os.getenv("MIN_CONF_SOCCER_SPREAD", "0.56"))
    min_confidence_tennis: float = float(os.getenv("MIN_CONF_TENNIS", "0.57"))
    min_confidence_basketball: float = float(os.getenv("MIN_CONF_BASKETBALL", "0.55"))
    min_confidence_icehockey: float = float(os.getenv("MIN_CONF_ICEHOCKEY", "0.55"))
    min_confidence_americanfootball: float = float(os.getenv("MIN_CONF_AMERICANFOOTBALL", "0.55"))
    min_confidence_default: float = float(os.getenv("MIN_CONF_DEFAULT", "0.55"))

    # Stake caps (fraction of bankroll)
    max_stake_pct: float = float(os.getenv("MAX_STAKE_PCT", "0.015"))        # 1.5%
    max_stake_longshot_pct: float = float(os.getenv("MAX_STAKE_LONGSHOT_PCT", "0.0075"))  # 0.75%
    longshot_odds_threshold: float = float(os.getenv("LONGSHOT_ODDS_THRESHOLD", "3.5"))

    # Combo leg minimum confidence (model_probability)
    min_combo_leg_confidence: float = float(os.getenv("MIN_COMBO_LEG_CONFIDENCE", "0.40"))

    # EV (Expected Value) thresholds
    min_ev_default: float = float(os.getenv("MIN_EV_DEFAULT", "0.01"))
    min_ev_losing_streak: float = float(os.getenv("MIN_EV_LOSING_STREAK", "0.02"))
    min_ev_drawdown: float = float(os.getenv("MIN_EV_DRAWDOWN", "0.02"))
    min_ev_degradation: float = float(os.getenv("MIN_EV_DEGRADATION", "0.015"))
    min_ev_good_run: float = float(os.getenv("MIN_EV_GOOD_RUN", "0.005"))

    # Signal modes: Trading vs Learning
    learning_capture_all_signals: bool = os.getenv("LEARNING_CAPTURE_ALL_SIGNALS", "true").lower() == "true"
    allow_watchlist_signals: bool = os.getenv("ALLOW_WATCHLIST_SIGNALS", "true").lower() == "true"

    # Fetch scheduler
    fetch_min_delay_ms: int = int(os.getenv("FETCH_MIN_DELAY_MS", "800"))
    fetch_max_delay_ms: int = int(os.getenv("FETCH_MAX_DELAY_MS", "1500"))
    fetch_max_retries: int = int(os.getenv("FETCH_MAX_RETRIES", "3"))

    # Kelly fraction (used by executioner agent)
    kelly_fraction_default: float = float(os.getenv("KELLY_FRACTION_DEFAULT", "0.20"))
    kelly_fraction_reactive: float = float(os.getenv("KELLY_FRACTION_REACTIVE", "0.15"))

    # Circuit breaker thresholds
    losing_streak_threshold: int = int(os.getenv("LOSING_STREAK_THRESHOLD", "7"))
    daily_loss_limit_pct: float = float(os.getenv("DAILY_LOSS_LIMIT_PCT", "0.05"))
    drawdown_max_pct: float = float(os.getenv("DRAWDOWN_MAX_PCT", "0.10"))
    drawdown_lookback_days: int = int(os.getenv("DRAWDOWN_LOOKBACK_DAYS", "7"))

    # Combo correlation penalty
    combo_correlation_penalty: float = float(os.getenv("COMBO_CORRELATION_PENALTY", "0.80"))
    combo_correlation_floor: float = float(os.getenv("COMBO_CORRELATION_FLOOR", "0.20"))

    # Calibration
    calibration_method: str = os.getenv("CALIBRATION_METHOD", "beta")  # "isotonic", "platt", or "beta"
    calibration_enabled: bool = os.getenv("CALIBRATION_ENABLED", "true").lower() == "true"
    ev_diagnostics_enabled: bool = os.getenv("EV_DIAGNOSTICS_ENABLED", "true").lower() == "true"

    # Poisson model
    poisson_rho: float = float(os.getenv("POISSON_RHO", "-0.13"))


settings = Settings()
