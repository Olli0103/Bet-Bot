# Bet-Bot

A multi-sport betting signal bot with machine learning, Telegram integration, and an autonomous agent framework. Built for value detection across soccer, basketball, and tennis markets on Tipico (DE) with sharp-book benchmarking via Pinnacle, Betfair, and bet365.

## Architecture

```
Telegram Bot (UI)
      |
Agent Orchestrator (adaptive: 60s pre-kickoff / 5min normal)
      |
  Scout Agent        -->  odds monitoring, steam move detection, Twitter/X alerts
      |
  Analyst Agent      -->  enrichment (sentiment, injuries, Elo, Poisson, form, H2H)
      |                   feature engineering (20+ features)
      |                   ML model prediction (XGBoost + calibration)
      |
  Executioner Agent  -->  circuit breakers, calibration-adjusted Kelly, virtual bets
      |
  Performance Monitor --> ROI tracking, daily reports, self-evaluation
```

### Core Pipeline

1. **Data Ingestion** -- The-Odds-API provides real-time odds across h2h, spreads, and totals markets from 6+ bookmakers
2. **Enrichment** -- News sentiment (NewsAPI + Ollama), injury data (API-Sports), weather (Open-Meteo), Twitter/X breaking news (journalist whitelist)
3. **Feature Engineering** -- 21 features including CLV, Elo differential, Poisson-derived probabilities, form tracking, H2H history, odds volatility, public bias
4. **Model Prediction** -- XGBoost with isotonic calibration, sport-specific models (soccer/basketball/tennis), TimeSeriesSplit validation, reliability diagrams
5. **Value Detection** -- Tax-adjusted EV calculation (Tipico 5% tax + combo tax-free mode), consensus sharp line from weighted Pinnacle/Betfair/bet365, public bias detection
6. **Bet Sizing** -- Fractional Kelly criterion with performance-based multipliers, calibration-adjusted scaling, and circuit breakers
7. **Combo Construction** -- Constraint-optimized 5/10/20/30-leg combos with dynamic pairwise correlation penalties
8. **Virtual Trading** -- Ghost-trades all signals for tracking without real money

## Features

### Multi-Sport Support
- **Soccer**: Bundesliga, EPL, La Liga, Serie A, Ligue 1, Champions League
- **Basketball**: NBA, EuroLeague
- **Tennis**: ATP, WTA
- Configurable via `LIVE_SPORTS` env var

### Market Coverage
- **h2h** (moneyline / 1X2)
- **spreads** (handicaps / point spreads)
- **totals** (over/under)
- Cross-market value detection via Poisson model (soccer)

### ML Pipeline
- **XGBoost** classifier with probability calibration (isotonic regression)
- Sport-specific models with fallback to general model
- 20+ engineered features:

| Feature | Source |
|---------|--------|
| `sharp_implied_prob` | Consensus sharp line (Pinnacle/Betfair/bet365) |
| `clv` | Closing line value: `(target / sharp) - 1.0` |
| `sharp_vig` | Sharp book overround |
| `sentiment_delta` | Home - away news sentiment |
| `injury_delta` | Home - away injury impact |
| `form_winrate_l5` | Last-5-games win rate |
| `elo_diff` | Elo rating differential |
| `elo_expected` | Elo-derived expected win probability |
| `h2h_home_winrate` | Historical head-to-head win rate |
| `home_volatility` / `away_volatility` | Historical odds movement std dev |
| `poisson_true_prob` | Poisson-derived probability (soccer only) |
| `line_staleness` | Minutes since Tipico last updated vs sharp |
| `weather_rain` / `weather_wind_high` | Weather conditions (outdoor sports) |
| `home_advantage` | Binary: selection is home team |
| `market_consensus` | Average implied prob across all books |
| `public_bias` | Tipico market shading vs sharp (retail over-bet detection) |

- Weekly automatic retraining with post-training validation (Brier score, calibration check, feature importance audit)
- **Reliability diagrams**: per-bucket actual-vs-predicted calibration with automatic Kelly adjustment multipliers

### Tipico Tax Handling
Tipico applies a 5% tax on gross winnings. All EV and Kelly calculations account for this:
```
net_profit = gross_profit * (1 - tax_rate)
EV = model_prob * net_profit - (1 - model_prob)
```
**Tax-free mode**: Tipico offers tax-free betting for qualifying combo bets (3+ legs) and mobile promotions. The `effective_tax_rate()` function automatically applies 0% tax for these cases.

**Public bias detection**: When Tipico shades favorites (lowers odds) more than sharp books, the `public_bias` feature captures this retail-driven vig. Heavily shaded favorites (bias > 0.02) require a stronger EV edge before the bot recommends a bet.

Configurable via `TIPICO_TAX_RATE` and `TAX_FREE_MODE` env vars.

### Combo System
Four combo tiers with constraint-based optimization:

| Size | Type | Stake | Constraints |
|------|------|-------|-------------|
| 5-leg | EV-optimal | 2.00 EUR | min 2 sports, max 2/league, prob > 0.55, max 2 heavy favs/league |
| 10-leg | Lotto | 1.00 EUR | min 3 sports, max 3/league, prob > 0.52, max 2 heavy favs/league |
| 20-leg | Lotto | 1.00 EUR | min 4 sports, max 4/league, prob > 0.50, max 3 heavy favs/league |
| 30-leg | Lotto | 0.50 EUR | min 5 sports, max 5/league, prob > 0.48, max 3 heavy favs/league |

**Heavy favorite cap**: No more than 2-3 selections with odds < 1.30 per league. Prevents a single league-wide upset (e.g. rainy EPL matchday) from killing the entire ticket.

Dynamic pairwise correlation penalties:
- Same event, different market: 0.80
- Same league: 0.92
- Same sport, different league: 0.97
- Cross-sport: 1.00 (independent)

### Agentic Framework

The bot uses a multi-agent architecture with **adaptive polling**:

- **Scout Agent** -- Monitors odds for steam moves (price changes exceeding 2x historical volatility) and Twitter/X for breaking injury news via a curated journalist whitelist (30+ verified beat writers)
- **Analyst Agent** -- Triggered by Scout alerts; performs full enrichment, feature engineering, ML prediction, public bias detection, and optional LLM reasoning (Ollama/Claude). Uses Poisson as primary driver (60% weight) for soccer Draw and Over/Under markets.
- **Executioner Agent** -- Applies circuit breakers, computes **calibration-adjusted** Kelly sizing (reliability bins trim stakes for over-predicting probability buckets), sends Telegram alerts, places virtual bets
- **Orchestrator** -- Adaptive polling: **60-second intervals** when events kick off within 1 hour (steam move window), **5-minute intervals** during quiet periods. Daily self-evaluation at 22:00.

### Circuit Breakers

| Breaker | Trigger | Action |
|---------|---------|--------|
| Losing streak | 7+ consecutive losses | Kelly x0.5, min EV raised to 0.02 |
| Daily loss limit | > 5% of bankroll lost today | Kelly x0.5, min EV raised to 0.02 |
| Model degradation | Hit rate < 40% over 14 days | Kelly x0.7, min EV raised to 0.015 |

### Backtesting

Walk-forward backtesting engine with strategy comparison:

```bash
# Single backtest with defaults
python scripts/run_backtest.py

# Compare multiple strategies
python scripts/run_backtest.py --compare

# Custom parameters
python scripts/run_backtest.py --kelly 0.15 --min-ev 0.01 --tax 0.05
```

Outputs: equity curve, ROI, hit rate, Sharpe ratio, Brier score, max drawdown, per-sport breakdown.

## Setup

### Prerequisites
- Python 3.11+
- PostgreSQL
- Redis

### Installation

```bash
git clone <repo-url>
cd Bet-Bot
pip install -r requirements.txt
```

### Environment Variables

Create a `.env` file in the project root:

```env
# Required
TELEGRAM_BOT_TOKEN=your_telegram_bot_token
TELEGRAM_CHAT_ID=your_chat_id
ODDS_API_KEY=your_odds_api_key
POSTGRES_DSN=postgresql+psycopg://postgres:postgres@localhost:5432/signalbot
REDIS_URL=redis://localhost:6379/0

# Enrichment APIs (optional but recommended)
NEWSAPI_KEY=your_newsapi_key
APISPORTS_API_KEY=your_apisports_key
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=gemma2

# Sports to track (comma-separated OddsAPI sport keys)
LIVE_SPORTS=soccer_germany_bundesliga,soccer_epl,basketball_nba,tennis_atp

# Bankroll & Tax
INITIAL_BANKROLL=1000.0
TIPICO_TAX_RATE=0.05
TAX_FREE_MODE=false

# Enrichment toggle
ENRICHMENT_ENABLED=true
ENRICHMENT_TIMEOUT=30

# Twitter/X (optional, opt-in)
TWITTER_ENABLED=false
TWITTER_BEARER_TOKEN=
TWITTER_API_KEY=
TWITTER_API_SECRET=
```

### Running

```bash
# Start the Telegram bot
python -m src.bot.app

# Run a backtest
python scripts/run_backtest.py --compare

# Import historical data
python scripts/import_historical_results.py

# Backfill features
python scripts/backfill_form_features.py
python scripts/backfill_odds_from_imports.py
```

## Project Structure

```
Bet-Bot/
├── src/
│   ├── agents/               # Multi-agent framework
│   │   ├── scout_agent.py    # Odds monitoring & steam move detection
│   │   ├── analyst_agent.py  # Deep event analysis & ML prediction
│   │   ├── executioner_agent.py  # Circuit breakers & bet execution
│   │   └── orchestrator.py   # Agent coordination
│   ├── bot/
│   │   ├── app.py            # Telegram bot setup & job scheduling
│   │   └── handlers.py       # Command handlers (value bets, combos, settings)
│   ├── core/
│   │   ├── backtester.py     # Walk-forward backtesting engine
│   │   ├── bankroll.py       # Dynamic bankroll management from DB
│   │   ├── betting_engine.py # Signal generation & combo building
│   │   ├── betting_math.py   # EV, Kelly criterion, tax-adjusted math
│   │   ├── combo_optimizer.py    # Constraint-based combo construction
│   │   ├── correlation.py    # Pairwise correlation penalties
│   │   ├── elo_ratings.py    # Redis-backed Elo power ratings
│   │   ├── enrichment.py     # News sentiment & injury enrichment
│   │   ├── feature_engineering.py  # 20+ feature builder
│   │   ├── form_tracker.py   # Redis-backed last-5-games form
│   │   ├── ghost_trading.py  # Virtual bet placement
│   │   ├── h2h_tracker.py    # Head-to-head history tracker
│   │   ├── live_feed.py      # Main signal pipeline orchestration
│   │   ├── ml_trainer.py     # XGBoost training with calibration
│   │   ├── performance_monitor.py  # ROI tracking & circuit breakers
│   │   ├── poisson_model.py  # Poisson goal model for soccer
│   │   ├── pricing_model.py  # True probability estimation (XGBoost)
│   │   ├── settings.py       # Configuration from environment
│   │   └── volatility_tracker.py   # Odds volatility & steam moves
│   ├── data/
│   │   └── venue_coordinates.py    # Stadium lat/lng for weather
│   ├── integrations/
│   │   ├── odds_fetcher.py   # The-Odds-API client (h2h/spreads/totals)
│   │   ├── news_fetcher.py   # NewsAPI client
│   │   ├── weather_fetcher.py    # Open-Meteo weather client
│   │   ├── twitter_fetcher.py    # Twitter/X API for injury news
│   │   ├── apisports_fetcher.py  # API-Sports injuries/lineups
│   │   └── ollama_sentiment.py   # Ollama LLM sentiment analysis
│   ├── models/
│   │   └── betting.py        # Pydantic models (BetSignal, ComboBet, etc.)
│   └── utils/
│       └── telegram_md.py    # Telegram markdown formatting
├── scripts/
│   ├── run_backtest.py       # Backtest CLI
│   ├── import_historical_results.py
│   ├── backfill_odds_from_imports.py
│   └── backfill_form_features.py
├── models/                   # Trained .joblib model files (gitignored)
├── config/
├── requirements.txt
└── ml_strategy_weights.json  # Legacy model weights (JSON fallback)
```

## Telegram Commands

| Command / Button | Description |
|------------------|-------------|
| `/start` | Show main keyboard menu |
| `/status` | Current bankroll, open bets, performance |
| `Heutige Value Bets` | Today's value bet signals |
| `Kombi-Vorschlage` | Optimized combo suggestions (5/10/20/30 legs) |
| `Daten aktualisieren` | Manually refresh odds & signals |
| `Kontostand` | Current bankroll and PnL |
| `Einstellungen` | Bot settings |
| `Hilfe` | Help menu |

## Scheduled Jobs

| Time | Job | Description |
|------|-----|-------------|
| 06:30 | Morning briefing | Notification that signals are ready |
| 07:00 | Data fetch + push | Fetch odds, build signals, push to Telegram |
| 13:00 | Data fetch | Afternoon odds refresh |
| 20:00 | Learning status | Win/loss stats and PnL summary |
| 20:05 | API health check | Status of all external API connections |
| 22:00 | Daily performance | Full performance report with circuit breaker status |
| 03:15 (Sat) | Weekly retrain | Retrain all XGBoost models on latest data |
| 60s / 5 min | Agent cycle | Scout/Analyst/Executioner (adaptive: fast pre-kickoff) |
| Every 30 min | Auto-grading | Settle open bets against API results |

## Dependencies

See `requirements.txt`. Key dependencies:
- `xgboost` -- Gradient boosted trees for probability estimation
- `scikit-learn` -- Calibration, preprocessing, metrics
- `python-telegram-bot` -- Telegram bot framework
- `SQLAlchemy` / `psycopg` -- PostgreSQL ORM
- `redis` -- Caching (form, Elo, volatility, odds snapshots)
- `pandas` / `numpy` -- Data processing
- `scipy` -- Poisson distribution for soccer modeling
- `aiohttp` / `httpx` -- Async HTTP clients for API integrations

## License

Private repository. All rights reserved.
