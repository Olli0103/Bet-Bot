# Bet-Bot

A multi-sport betting signal bot with machine learning, Telegram integration, and an autonomous agent framework. Built for high-win-rate, stress-free accumulator betting on Tipico (DE) with sharp-book benchmarking via Pinnacle, Betfair, and bet365.

## Architecture

```
Telegram Bot (UI)
      |
Agent Orchestrator (adaptive: 60s pre-kickoff / 5min normal)
      |
  Scout Agent        -->  odds monitoring, steam move detection, Twitter/X alerts
      |                   12h market momentum tracking (Pro API)
  Analyst Agent      -->  enrichment (sentiment, injuries, Elo, Poisson, form, H2H)
      |                   feature engineering (22+ features incl. momentum)
      |                   ML model prediction (XGBoost + calibration)
      |                   market expansion (Double Chance, DNB derivations)
  Executioner Agent  -->  circuit breakers, calibration-adjusted Kelly, virtual bets
      |
  Performance Monitor --> ROI tracking, daily reports, self-evaluation
```

### Core Pipeline

1. **Data Ingestion** -- The-Odds-API (Pro Tier) provides real-time and historical odds across h2h, spreads, totals, double_chance, and draw_no_bet markets from 6+ bookmakers
2. **Enrichment** -- News sentiment (NewsAPI + Ollama Gemma 3 4B), injury data (API-Sports), weather (Open-Meteo), Twitter/X breaking news (journalist whitelist)
3. **Feature Engineering** -- 22+ features including CLV, Elo differential, Poisson-derived probabilities, form tracking, H2H history, odds volatility, public bias, market momentum (12h delta)
4. **Model Prediction** -- XGBoost with isotonic calibration, sport-specific models (soccer/basketball/tennis), TimeSeriesSplit validation, reliability diagrams
5. **Value Detection** -- Tax-adjusted EV calculation (Tipico 5% tax + combo tax-free mode), consensus sharp line from weighted Pinnacle/Betfair/bet365, public bias detection
6. **Bet Sizing** -- Fractional Kelly criterion with performance-based multipliers, calibration-adjusted scaling, and circuit breakers
7. **Combo Construction** -- Constraint-optimized 10/20/30-leg lotto combos with dynamic pairwise correlation penalties and market-type-aware scoring
8. **Virtual Trading** -- Ghost-trades all signals for tracking without real money

## Features

### Telegram UI/UX

The bot provides a premium, interactive Telegram experience:

- **Top 10 Singles** -- "Heutige Top 10 Einzelwetten" button shows the best daily singles ranked by hit probability (70%) + EV (30%)
- **Lotto Combos** -- "10/20/30 Kombis" button shows Tipico-friendly accumulators with market-type-aware leg selection
- **Interactive Settings Dashboard** -- Toggle sports, markets, min odds, and combo sizes via inline keyboard buttons
- **Inline Keyboard Pagination** -- Value bets display one card at a time with Prev/Next navigation buttons, eliminating chat spam
- **Visual PnL Dashboards** -- Matplotlib-generated charts (equity curve, win/loss pie, stats) sent as Telegram photos with dark theme
- **Progress Bars** -- Model probabilities rendered as `[████████░░] 80%` visual bars
- **Calibration Badges** -- Well-calibrated / moderate / high variance badges, derived from reliability bin data
- **Retail Trap Indicator** -- `Retail Trap` or `Public Bias` badges on signals where Tipico is shading favorites
- **Tax-Free Badge** -- `Steuerfrei` on qualifying 3+ leg combo suggestions
- **Interactive Agent Alerts** -- Executioner alerts include inline buttons: Deep Dive (triggers Analyst re-analysis), Ghost Bet (places virtual bet), Ignorieren
- **NLP Intent Routing** -- Type natural language commands like "Gib mir die Top 5" or "Zeig mir 10er Kombis" and the bot routes via Gemma 3 4B intent classification
- **Per-User State** -- Pagination and session data stored in `user_data`, not global `bot_data`, so concurrent users don't corrupt each other's views

### Multi-Sport Support

- **Soccer**: Bundesliga, EPL, La Liga, Serie A, Ligue 1, Champions League
- **Basketball**: NBA, EuroLeague
- **American Football**: NFL
- **Ice Hockey**: NHL
- **Tennis**: ATP, WTA

Configurable via Telegram settings dashboard or `LIVE_SPORTS` env var.

### Market Coverage

- **h2h** (moneyline / 1X2)
- **spreads** (handicaps / point spreads)
- **totals** (over/under 0.5, 1.5, 2.5, 3.5)
- **double_chance** (1X, X2, 12) -- mathematically derived from 1X2 sharp odds (soccer)
- **draw_no_bet** (DNB) -- derived from 1X2 via draw-removed redistribution (soccer)
- Cross-market value detection via Poisson model (soccer: over/under 0.5/1.5/2.5/3.5, BTTS)

### Dynamic Settings (Redis-Backed)

All settings are toggleable via Telegram inline keyboard and persist in Redis:

```
Settings Dashboard:
━━━━━━━━━━━━━━━━━━━━

Sportarten:
[✅ Bundesliga] [✅ EPL] [❌ La Liga]
[✅ NBA] [❌ NFL] [✅ ATP]

Markte:
[✅ H2H] [✅ Totals] [✅ Spreads]
[✅ Double Chance] [✅ DNB]

Min Quote: 1.20
[1.10] [1.20 ✓] [1.30] [1.50]

Kombi-Grossen:
[✅ 10er] [✅ 20er] [✅ 30er]
```

### ML Pipeline

- **XGBoost** classifier with probability calibration (isotonic regression)
- Sport-specific models with fallback to general model
- 22+ engineered features:

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
| `public_bias` | Tipico market shading vs sharp (retail over-bet detection) |
| `market_momentum` | Implied probability delta over 12h (Pro API history) |
| `twitter_sentiment_delta` | Twitter/X breaking news sentiment |
| `time_to_kickoff_hours` | Hours until event starts |

- Weekly automatic retraining with post-training validation (Brier score, calibration check, feature importance audit)
- **Reliability diagrams**: per-bucket actual-vs-predicted calibration with automatic Kelly adjustment multipliers

### Market Momentum (Pro API)

Uses the Odds API Pro tier `/odds-history` endpoint to fetch historical odds from 12 hours ago:

```
current_ip = 1.0 / current_odds
historical_ip = odds_12h_ago.get(event_id, {}).get(selection, current_ip)
momentum = current_ip - historical_ip  # positive = market moving toward selection
```

Momentum is passed through the entire pipeline: feature engineering, analyst reasoning, and model prediction adjustment (`model_p + momentum * 0.15`).

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

Three lotto combo tiers with constraint-based optimization and market-type-aware scoring:

| Size | Type | Stake | Constraints |
|------|------|-------|-------------|
| 10-leg | Lotto | 1.00 EUR | min 3 sports, max 3/league, prob > 0.52, max 2 heavy favs/league |
| 20-leg | Lotto | 1.00 EUR | min 4 sports, max 4/league, prob > 0.50, max 3 heavy favs/league |
| 30-leg | Lotto | 0.50 EUR | min 5 sports, max 5/league, prob > 0.48, max 3 heavy favs/league |

**Market-type scoring boosts**: Double Chance and DNB legs get a 1.20x scoring boost; high-probability totals (prob >= 0.80) get 1.15x. This prioritizes high-hit-rate legs in lotto combos.

**Heavy favorite cap**: No more than 2-3 selections with odds < 1.30 per league. Prevents a single league-wide upset (e.g. rainy EPL matchday) from killing the entire ticket.

Dynamic pairwise correlation penalties:
- Same event, different market: 0.80
- Same league: 0.92
- Same sport, different league: 0.97
- Cross-sport: 1.00 (independent)

### Poisson Model (Soccer)

Independent Poisson goal-distribution model with per-team attack/defense strengths (cached in Redis):

- **Score matrix**: P(home=i, away=j) for i,j in 0..6
- **1X2**: Home/Draw/Away probabilities
- **Totals**: Over/Under 0.5, 1.5, 2.5, 3.5
- **BTTS**: Both teams to score yes/no
- **Online learning**: Strengths update after each result via multiplicative adjustment
- **Poisson/XGBoost blending**: Base weight 60-70% for totals, 30% for H2H, with dynamic bonus when xG differential is lopsided

### NLP Intent Routing

Free-text messages are classified by Gemma 3 4B into intents:

| Intent | Example | Action |
|--------|---------|--------|
| `get_top_bets` | "Gib mir die Top 5 fur heute" | Shows top N bets from signals |
| `get_combos` | "Zeig mir 10er Kombis" | Shows combos filtered by size |
| `explain_bet` | "Erklare mir den Lakers Tipp" | LLM Q&A with signal context |
| fallback | Any other text | General LLM answer |

### Agentic Framework

The bot uses a multi-agent architecture with **adaptive polling**:

- **Scout Agent** -- Monitors odds for steam moves (price changes exceeding 2x historical volatility), Twitter/X for breaking injury news, and 12h market momentum via Pro API
- **Analyst Agent** -- Triggered by Scout alerts; performs full enrichment, feature engineering, ML prediction, public bias detection, market momentum integration, and Gemma 3 4B reasoning. Uses **dynamic Poisson/XGBoost blending** with momentum adjustment.
- **Executioner Agent** -- Applies circuit breakers, computes **calibration-adjusted** Kelly sizing, sends Telegram alerts with interactive inline buttons, places virtual bets
- **Orchestrator** -- Adaptive polling: **60-second intervals** when events kick off within 1 hour, **5-minute intervals** during quiet periods. Daily self-evaluation at 22:00.

### Process Safety

- **Singleton Guard** -- PID-file based process guard. On startup, checks if an old instance is still running, sends SIGTERM (5s grace), then SIGKILL if needed.
- **Async-Safe Sync Wrappers** -- All API fetchers use `_safe_sync_run()` which detects a running event loop and offloads to a `ThreadPoolExecutor`.
- **Non-Blocking DB Calls** -- All database operations in Telegram handlers are wrapped in `asyncio.to_thread()`.

### Circuit Breakers

| Breaker | Trigger | Action |
|---------|---------|--------|
| Losing streak | 7+ consecutive losses | Kelly x0.5, min EV raised to 0.02 |
| Daily loss limit | > 5% of bankroll lost today | Kelly x0.5, min EV raised to 0.02 |
| Model degradation | Hit rate < 40% over 14 days | Kelly x0.7, min EV raised to 0.015 |

### Historical Data Importer

Universal data importer supporting multiple sports and formats:

```bash
# Import soccer (football-data.co.uk format)
python scripts/import_historical_results.py --football-dir data/imports/football

# Import tennis (ATP/WTA/Challenger, csv + xlsx)
python scripts/import_historical_results.py --tennis-dir data/imports/tennis

# Import US sports (aussportsbetting.com format)
python scripts/import_historical_results.py --nba-dir data/imports/nba
python scripts/import_historical_results.py --nfl-dir data/imports/nfl
python scripts/import_historical_results.py --nhl-dir data/imports/nhl

# Limit rows for testing
python scripts/import_historical_results.py --nba-dir data/imports/nba --max-rows 500

# All at once
python scripts/import_historical_results.py \
    --football-dir data/imports/football \
    --tennis-dir data/imports/tennis \
    --nba-dir data/imports/nba \
    --nfl-dir data/imports/nfl \
    --nhl-dir data/imports/nhl
```

**Soccer imports** generate H2H + totals (Over 1.5, Over 2.5) + BTTS rows from FTHG/FTAG columns.

**US sports imports** (aussportsbetting format) generate Moneyline + Spreads + Totals rows with real historical odds parsed from CSV columns.

**Tennis imports** handle `.csv`, `.xlsx`, `.xls` files with automatic ATP/WTA/Challenger detection.

### Backtesting

Walk-forward backtesting engine with strategy comparison:

```bash
python scripts/run_backtest.py
python scripts/run_backtest.py --compare
python scripts/run_backtest.py --kelly 0.15 --min-ev 0.01 --tax 0.05
```

## Setup

### Prerequisites

- Python 3.11+
- PostgreSQL
- Redis
- Ollama with `gemma:4b` model (for sentiment analysis and NLP intent routing)

### Ollama Setup

```bash
# Install Ollama
curl -fsSL https://ollama.com/install.sh | sh

# Pull Gemma 3 4B model
ollama pull gemma3:4b

# Verify
ollama run gemma3:4b "Hello"
```

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
ODDS_API_KEY=your_odds_api_key          # Pro Tier required for /odds-history
POSTGRES_DSN=postgresql+psycopg://postgres:postgres@localhost:5432/signalbot
REDIS_URL=redis://localhost:6379/0

# Enrichment APIs (optional but recommended)
NEWSAPI_KEY=your_newsapi_key
APISPORTS_API_KEY=your_apisports_key
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=gemma:4b                    # Gemma 3 4B (optimized prompts)

# Sports to track (comma-separated OddsAPI sport keys)
# Note: Sports are also configurable via Telegram settings dashboard
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
python scripts/import_historical_results.py --football-dir data/imports/football

# Bootstrap seed data
python scripts/bootstrap_history.py

# Backfill features
python scripts/backfill_form_features.py
python scripts/backfill_odds_from_imports.py
```

## Project Structure

```
Bet-Bot/
├── src/
│   ├── agents/                         # Multi-agent framework
│   │   ├── scout_agent.py              # Odds monitoring, steam moves, momentum
│   │   ├── analyst_agent.py            # Deep analysis, ML prediction, Gemma reasoning
│   │   ├── executioner_agent.py        # Circuit breakers & bet execution
│   │   └── orchestrator.py             # Agent coordination (adaptive polling)
│   ├── bot/
│   │   ├── app.py                      # Telegram bot setup & job scheduling
│   │   └── handlers.py                 # UI handlers, settings dashboard, NLP routing
│   ├── core/
│   │   ├── backtester.py               # Walk-forward backtesting engine
│   │   ├── bankroll.py                 # Dynamic bankroll management from DB
│   │   ├── betting_engine.py           # Signal generation & combo building
│   │   ├── betting_math.py             # EV, Kelly criterion, tax-adjusted math
│   │   ├── combo_optimizer.py          # Constraint-based combo construction
│   │   ├── correlation.py              # Pairwise correlation penalties
│   │   ├── dynamic_settings.py         # Redis-backed dynamic settings (Telegram toggles)
│   │   ├── elo_ratings.py              # Redis-backed Elo power ratings
│   │   ├── enrichment.py               # News sentiment & injury enrichment
│   │   ├── feature_engineering.py      # 22+ feature builder (incl. momentum)
│   │   ├── form_tracker.py             # Redis-backed last-5-games form
│   │   ├── ghost_trading.py            # Virtual bet placement
│   │   ├── h2h_tracker.py              # Head-to-head history tracker
│   │   ├── live_feed.py                # Main signal pipeline (DC/DNB, momentum)
│   │   ├── ml_trainer.py               # XGBoost training with calibration
│   │   ├── performance_monitor.py      # ROI tracking & circuit breakers
│   │   ├── poisson_model.py            # Poisson goal model (0.5/1.5/2.5/3.5)
│   │   ├── pricing_model.py            # True probability estimation (XGBoost)
│   │   ├── settings.py                 # Configuration from environment
│   │   └── volatility_tracker.py       # Odds volatility & steam moves
│   ├── data/
│   │   ├── __init__.py                 # Package init
│   │   ├── models.py                   # PlacedBet SQLAlchemy model
│   │   ├── postgres.py                 # Engine + SessionLocal factory
│   │   ├── redis_cache.py              # Redis cache wrapper (get/set JSON)
│   │   └── venue_coordinates.py        # Stadium lat/lng for weather
│   ├── integrations/
│   │   ├── base_fetcher.py             # Async HTTP base + _safe_sync_run()
│   │   ├── odds_fetcher.py             # The-Odds-API client (Pro tier + history)
│   │   ├── news_fetcher.py             # NewsAPI client
│   │   ├── weather_fetcher.py          # Open-Meteo weather client
│   │   ├── twitter_fetcher.py          # Twitter/X API (journalist whitelist)
│   │   ├── apisports_fetcher.py        # API-Sports injuries/lineups
│   │   └── ollama_sentiment.py         # Ollama Gemma 3 4B (sentiment + intents)
│   ├── models/
│   │   └── betting.py                  # Pydantic models (BetSignal, ComboBet)
│   └── utils/
│       ├── charts.py                   # Matplotlib dashboards (equity, pie)
│       └── telegram_md.py              # Telegram markdown formatting
├── scripts/
│   ├── run_backtest.py                 # Backtest CLI
│   ├── import_historical_results.py    # Universal importer (soccer/tennis/US sports)
│   ├── bootstrap_history.py            # Seed data download helper
│   ├── backfill_odds_from_imports.py
│   └── backfill_form_features.py
├── models/                             # Trained .joblib model files (gitignored)
├── config/
├── requirements.txt
└── ml_strategy_weights.json            # Legacy model weights (JSON fallback)
```

## Telegram Commands

| Command / Button | Description |
|------------------|-------------|
| `/start` | Show main keyboard menu |
| `/status` | Bankroll dashboard with PnL chart |
| `Heutige Top 10 Einzelwetten` | Top 10 singles ranked by hit probability |
| `10/20/30 Kombis` | Lotto combos (10, 20, 30-leg) |
| `Daten aktualisieren` | Manually refresh odds & signals |
| `Kontostand` | Visual PnL dashboard (equity curve + pie chart) |
| `Einstellungen` | Interactive toggle dashboard (sports, markets, min odds, combos) |
| `Hilfe` | Help menu with feature overview and NL examples |
| *Free text* | NLP intent routing via Gemma 3 4B |

### Natural Language Examples

| Input | Routed To |
|-------|-----------|
| "Gib mir die Top 5 fur heute" | Top bets handler (limit=5) |
| "Zeig mir 10er Kombis" | Combo handler (size=10) |
| "Erklare mir den Lakers Tipp" | LLM Q&A with signal context |
| "Zeig mir die besten 3 Fussball Wetten" | Top bets handler (limit=3) |

### Inline Button Actions (on Agent Alerts)

| Button | Action |
|--------|--------|
| Deep Dive | Re-runs Analyst with full enrichment |
| Ghost Bet | Places a virtual bet for tracking |
| Ignorieren | Dismisses the alert |
| Als platziert | Marks a value bet as placed |

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
- `matplotlib` -- PnL dashboards and visual charts (Agg backend, dark theme)
- `SQLAlchemy` / `psycopg` -- PostgreSQL ORM
- `redis` -- Caching (form, Elo, volatility, odds snapshots, dynamic settings)
- `pandas` / `numpy` -- Data processing
- `scipy` -- Poisson distribution for soccer modeling
- `httpx` -- Async HTTP client with retry (Tenacity) for all API integrations

## API Requirements

| API | Tier | Used For |
|-----|------|----------|
| The-Odds-API | **Pro** | Real-time odds + `/odds-history` for 12h momentum |
| NewsAPI | Free/Dev | News sentiment enrichment |
| API-Sports | Free | Injury data and lineups |
| Open-Meteo | Free | Weather conditions for outdoor sports |
| Ollama (local) | -- | Gemma 3 4B for sentiment, reasoning, NLP intents |
| Twitter/X | Optional | Breaking injury news via journalist whitelist |

## License

Private repository. All rights reserved.
