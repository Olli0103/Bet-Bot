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
    enrichment_timeout_per_team: int = int(os.getenv("ENRICHMENT_TIMEOUT", "30"))

    # Bankroll
    initial_bankroll: float = float(os.getenv("INITIAL_BANKROLL", "1000.0"))

    # Tipico tax
    tipico_tax_rate: float = float(os.getenv("TIPICO_TAX_RATE", "0.05"))
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


settings = Settings()
