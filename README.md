# Bet-Bot

A multi-sport betting signal bot with machine learning, Telegram integration, and an autonomous agent framework. Built for high-win-rate, stress-free accumulator betting on Tipico (DE) with sharp-book benchmarking via Pinnacle, Betfair, and bet365.

---

## Table of Contents

1. [Quick Start (Full Install to First Signal)](#quick-start-full-install-to-first-signal)
2. [Prerequisites](#prerequisites)
3. [Installation](#installation)
4. [Database Setup](#database-setup)
5. [Historical Data Import](#historical-data-import)
6. [Initial ML Training](#initial-ml-training)
7. [Running the Bot](#running-the-bot)
8. [Architecture](#architecture)
9. [Features](#features)
10. [Configuration Reference](#configuration-reference)
11. [CLI Scripts Reference](#cli-scripts-reference)
12. [Telegram Commands](#telegram-commands)
13. [Scheduled Jobs](#scheduled-jobs)
14. [Project Structure](#project-structure)
15. [API Requirements](#api-requirements)
16. [Troubleshooting](#troubleshooting)

---

## Quick Start (Full Install to First Signal)

```bash
# 1. Clone & install
git clone <repo-url> && cd Bet-Bot
python3.11 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# 2. Infrastructure
#    Start PostgreSQL and Redis (see Database Setup below)

# 3. Configure
cp .env.example .env   # then edit with your API keys (see Configuration Reference)

# 4. Seed historical data
python scripts/bootstrap_history.py                        # download free datasets
cp -r data/raw/* data/imports/                             # stage into import dirs
python scripts/import_historical_results.py                # import all sports

# 5. Backfill engineered features
python scripts/backfill_form_features.py
python scripts/backfill_odds_from_imports.py

# 6. Train ML models
python -c "from src.core.ml_trainer import auto_train_all_models; print(auto_train_all_models(min_samples=100))"

# 7. Launch
python -m src.bot.app
```

---

## Prerequisites

| Dependency | Version | Purpose |
|------------|---------|---------|
| Python | 3.11+ | Runtime |
| PostgreSQL | 14+ | Bet history, ML training data |
| Redis | 7+ | Caching (form, Elo, odds snapshots, settings) |
| Ollama | latest | Local LLM for sentiment & NLP intent routing |

### Ollama Setup

```bash
# Install Ollama
curl -fsSL https://ollama.com/install.sh | sh

# Pull Gemma 3 4B model (used for sentiment, reasoning, and intent routing)
ollama pull gemma3:4b

# Verify
ollama run gemma3:4b "Hello"
```

---

## Installation

```bash
git clone <repo-url>
cd Bet-Bot

# Create virtual environment (Python 3.11 recommended)
python3.11 -m venv .venv
source .venv/bin/activate

# Install all dependencies
pip install -r requirements.txt
```

Key packages installed:
- `xgboost`, `scikit-learn` -- ML pipeline
- `pandas`, `numpy`, `scipy` -- data processing + Poisson model
- `SQLAlchemy`, `psycopg` -- PostgreSQL ORM
- `redis` -- caching layer
- `python-telegram-bot` -- Telegram UI
- `openpyxl` -- Excel file support for Tennis/NFL imports
- `httpx`, `tenacity` -- async HTTP with retry
- `matplotlib` -- PnL charts
- `feedparser` -- RSS injury feeds

---

## Database Setup

### PostgreSQL

```bash
# Create the database
createdb signalbot

# Or via psql:
psql -U postgres -c "CREATE DATABASE signalbot;"
```

The default connection string is:
```
postgresql+psycopg://postgres:postgres@localhost:5432/signalbot
```

Override via `POSTGRES_DSN` in your `.env` file.

**Tables are created automatically** on first run of any script that imports `src.data.postgres`. The importer also runs `_ensure_schema()` to add any missing columns to existing tables (safe to run repeatedly).

### Redis

```bash
# Start Redis (default port 6379)
redis-server

# Or via Docker:
docker run -d --name redis -p 6379:6379 redis:7-alpine
```

Default URL: `redis://localhost:6379/0` (override via `REDIS_URL`).

### Schema: `placed_bets` Table

This is the central table. All imports, virtual bets, and live signals write here.

| Column | Type | Description |
|--------|------|-------------|
| `id` | SERIAL PK | Auto-increment |
| `event_id` | VARCHAR(128) | Deterministic MD5 hash (dedup key) |
| `sport` | VARCHAR(64) | Odds API sport key (e.g. `soccer_epl`) |
| `market` | VARCHAR(32) | `h2h`, `spreads`, `totals`, `btts` |
| `selection` | VARCHAR(256) | Team name or "Over 2.5", "Draw", etc. |
| `odds` | FLOAT | Closing decimal odds |
| `odds_open` | FLOAT | Opening odds (for CLV calc) |
| `odds_close` | FLOAT | Closing odds |
| `clv` | FLOAT | Closing line value: `(open/close) - 1.0` |
| `stake` | FLOAT | Stake amount (default 1.0 for imports) |
| `status` | VARCHAR(16) | `open`, `won`, `lost`, `void` |
| `pnl` | FLOAT | Profit/loss for this bet |
| `sharp_implied_prob` | FLOAT | Sharp consensus probability |
| `sentiment_delta` | FLOAT | Home - away sentiment |
| `injury_delta` | FLOAT | Home - away injury impact |
| `form_winrate_l5` | FLOAT | Last-5 win rate |
| `form_games_l5` | FLOAT | Games in last-5 window |
| `meta_features` | JSONB | Sport-specific advanced stats |
| `notes` | TEXT | Free text |
| `created_at` | TIMESTAMPTZ | Row creation time |
| `updated_at` | TIMESTAMPTZ | Last update time |

If you hit `meta_features column missing` or similar schema errors, the importer's `_ensure_schema()` handles this automatically via `ALTER TABLE ... ADD COLUMN IF NOT EXISTS`.

---

## Historical Data Import

The ML models need historical bet outcomes to train on. The more data you feed, the better the models.

### Step 1: Download Free Datasets

```bash
python scripts/bootstrap_history.py
```

This downloads from:
- **football-data.co.uk** -- Soccer (EPL, Bundesliga, etc.) with full odds columns
- **tennis-data.co.uk** -- ATP/WTA with Pinnacle/bet365 odds
- **aussportsbetting.com** -- NBA, NFL, NHL with moneyline, spreads, totals

Files land in `data/raw/`.

### Step 2: Stage Files for Import

Create the import directory structure and copy/move your data files:

```bash
mkdir -p data/imports/{football,tennis,nba,nfl,nhl}

# Copy downloaded data (adjust paths to your files):
cp data/raw/football/*.csv   data/imports/football/
cp data/raw/tennis/*         data/imports/tennis/
cp data/raw/nba/*.csv        data/imports/nba/
cp data/raw/nfl/*            data/imports/nfl/
cp data/raw/nhl/*.csv        data/imports/nhl/
```

### Step 3: Run the Importer

```bash
# Import everything at once
python scripts/import_historical_results.py

# Or import sport by sport
python scripts/import_historical_results.py --football-dir data/imports/football
python scripts/import_historical_results.py --tennis-dir data/imports/tennis
python scripts/import_historical_results.py --nba-dir data/imports/nba
python scripts/import_historical_results.py --nfl-dir data/imports/nfl
python scripts/import_historical_results.py --nhl-dir data/imports/nhl

# Limit rows for testing
python scripts/import_historical_results.py --nba-dir data/imports/nba --max-rows 500

# Limit football CSV files
python scripts/import_historical_results.py --football-dir data/imports/football --football-files 3
```

The importer is **idempotent** -- it tracks `(event_id, selection)` pairs and skips duplicates. Safe to re-run.

### Supported File Formats per Sport

| Sport | Formats | Source Layout |
|-------|---------|---------------|
| **Football** | `.csv` | football-data.co.uk: `HomeTeam`, `AwayTeam`, `FTR`, `FTHG/FTAG`, B365/PS odds |
| **Tennis** | `.csv`, `.xlsx`, `.xls` | tennis-data.co.uk: `Winner`, `Loser`, `WRank`, `LRank`, B365/PS odds |
| **NBA** | `.csv` | Kaggle/aussportsbetting: `home`/`away` or `team_home`/`team_away`, `score_home`/`score_away`, American moneyline |
| **NFL** | `.csv`, `.xlsx`, `.xls` | aussportsbetting: `Home Team`, `Away Team`, decimal odds, spread lines |
| **NHL** | `.csv` | Three formats auto-detected: per-game rows, per-team with `home_away` column, or team-perspective with `is_home` + `team_name`/`opp_team_name` |

### What Gets Imported

Each sport extracts multiple market types:

- **Football**: H2H (1X2), Totals (Over/Under 2.5, 1.5), BTTS. Meta: half-time scores, shots, corners, fouls, cards, AHC lines.
- **Tennis**: H2H (winner + loser rows). Meta: rankings, points, surface, tournament, round, set scores.
- **NBA**: Moneyline, Spreads, Totals (Over/Under). Meta: quarter scores (Q1-Q4), OT, playoffs, favourite indicator.
- **NFL**: Moneyline, Spreads, Totals (Over/Under). Meta: line momentum (open vs close), playoff flag, neutral venue.
- **NHL**: Moneyline (American -> decimal), Totals. Meta: shots, power play, faceoffs, hits, PIM, rolling averages (3-game, 10-game).

### Step 4: Backfill Engineered Features

After importing, populate the form and odds features:

```bash
# Backfill last-5-game form (form_winrate_l5, form_games_l5)
python scripts/backfill_form_features.py

# Backfill open/close odds and CLV from imported data
python scripts/backfill_odds_from_imports.py
```

### Verify Import

```bash
# Quick row count
python -c "
from sqlalchemy import select, func
from src.data.models import PlacedBet
from src.data.postgres import SessionLocal
with SessionLocal() as db:
    total = db.scalar(select(func.count()).select_from(PlacedBet))
    print(f'Total rows in placed_bets: {total}')
"
```

---

## Initial ML Training

Once you have imported data (minimum ~200 rows per sport group), train the models:

```bash
# Train all models (general + sport-specific)
python -c "
from src.core.ml_trainer import auto_train_all_models
result = auto_train_all_models(min_samples=100)
print(result)
"
```

This trains:
1. **general** -- all data combined
2. **soccer** -- all soccer leagues
3. **basketball** -- NBA + EuroLeague
4. **tennis** -- ATP/WTA/Challenger
5. **americanfootball** -- NFL
6. **icehockey** -- NHL

Each model is an XGBoost classifier with isotonic calibration, saved to `models/xgb_{group}.joblib`.

### Expected Output

```
general: 270021 samples, brier=0.2479 | soccer: 70860 samples, brier=0.2484 | basketball: 48000 samples, brier=0.2412 | tennis: 41462 samples, brier=0.2313 | americanfootball: 5200 samples, brier=0.2501 | icehockey: 8400 samples, brier=0.2455
```

A Brier score under 0.25 is the baseline (random = 0.25). Lower is better.

### Feature Variance Fallback

If a sport group has no engineered features (e.g. raw imports without `sharp_implied_prob`), the trainer automatically derives `sharp_implied_prob = 1/odds`. This prevents "no feature variance" skips on imported datasets.

### Retraining

The bot automatically retrains every Saturday at 03:15 (Berlin time). Manual retrain:

```bash
python -c "from src.core.ml_trainer import auto_train_all_models; print(auto_train_all_models())"
```

### Model Files

Trained models are saved to `models/` (gitignored):
```
models/
├── xgb_general.joblib
├── xgb_soccer.joblib
├── xgb_basketball.joblib
├── xgb_tennis.joblib
├── xgb_americanfootball.joblib
└── xgb_icehockey.joblib
```

A legacy `ml_strategy_weights.json` is also generated for backward compatibility.

---

## Running the Bot

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
OLLAMA_MODEL=gemma:4b

# Sports to track (comma-separated Odds API sport keys)
# Also configurable via Telegram settings dashboard
LIVE_SPORTS=soccer_germany_bundesliga,soccer_epl,basketball_nba,tennis_atp

# Bankroll & Tax
INITIAL_BANKROLL=1000.0
TIPICO_TAX_RATE=0.05
TAX_FREE_MODE=false

# Enrichment toggle
ENRICHMENT_ENABLED=true
ENRICHMENT_TIMEOUT=30
```

### Start the Bot

```bash
python -m src.bot.app
```

### macOS LaunchAgent (Auto-Start)

A plist file is provided at `config/com.clawy.bettingbot.plist`. To install:

```bash
# Edit paths in the plist to match your installation
cp config/com.clawy.bettingbot.plist ~/Library/LaunchAgents/
launchctl load ~/Library/LaunchAgents/com.clawy.bettingbot.plist
```

### Other CLI Commands

```bash
# Run a backtest
python scripts/run_backtest.py --compare

# Grade open bets against results
python scripts/auto_grade_once.py

# One-off feature backfill
python scripts/backfill_form_features.py
```

---

## Architecture

```
Telegram Bot (UI)
      |
Agent Orchestrator (adaptive: 60s pre-kickoff / 5min normal)
      |
  Scout Agent        -->  odds monitoring, steam move detection, injury aggregation
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
2. **Enrichment** -- News sentiment (NewsAPI + Ollama Gemma 3 4B), injury data (API-Sports + Rotowire RSS), weather (Open-Meteo), LLM-structured injury extraction
3. **Feature Engineering** -- 22+ features including CLV, Elo differential, Poisson-derived probabilities, form tracking, H2H history, odds volatility, public bias, market momentum (12h delta)
4. **Model Prediction** -- XGBoost with isotonic calibration, sport-specific models (soccer/basketball/tennis/americanfootball/icehockey), TimeSeriesSplit validation, reliability diagrams
5. **Value Detection** -- Tax-adjusted EV calculation (Tipico 5% tax + combo tax-free mode), consensus sharp line from weighted Pinnacle/Betfair/bet365, public bias detection
6. **Bet Sizing** -- Fractional Kelly criterion with performance-based multipliers, calibration-adjusted scaling, and circuit breakers
7. **Combo Construction** -- Constraint-optimized 10/20/30-leg lotto combos with dynamic pairwise correlation penalties and market-type-aware scoring
8. **Virtual Trading** -- Ghost-trades all signals for tracking without real money

---

## Features

### Multi-Sport Support

- **Soccer**: Bundesliga (1+2), EPL, Championship, La Liga, Serie A, Ligue 1, Eredivisie, Champions League, Europa League, and more
- **Basketball**: NBA, EuroLeague
- **American Football**: NFL
- **Ice Hockey**: NHL
- **Tennis**: ATP, WTA, Challenger

Configurable via Telegram settings dashboard or `LIVE_SPORTS` env var.

### Market Coverage

- **h2h** (moneyline / 1X2)
- **spreads** (handicaps / point spreads)
- **totals** (over/under 0.5, 1.5, 2.5, 3.5)
- **double_chance** (1X, X2, 12) -- mathematically derived from 1X2 sharp odds (soccer)
- **draw_no_bet** (DNB) -- derived from 1X2 via draw-removed redistribution (soccer)
- Cross-market value detection via Poisson model (soccer: over/under 0.5/1.5/2.5/3.5, BTTS)

### Telegram UI/UX

The bot provides a premium, interactive Telegram experience:

- **Top 10 Singles** -- "Heutige Top 10 Einzelwetten" button shows the best daily singles ranked by hit probability (70%) + EV (30%)
- **Lotto Combos** -- "10/20/30 Kombis" button shows Tipico-friendly accumulators with market-type-aware leg selection
- **Interactive Settings Dashboard** -- Toggle sports, markets, min odds, and combo sizes via inline keyboard buttons
- **Inline Keyboard Pagination** -- Value bets display one card at a time with Prev/Next navigation buttons
- **Visual PnL Dashboards** -- Matplotlib-generated charts (equity curve, win/loss pie, stats) sent as Telegram photos
- **Progress Bars** -- Model probabilities rendered as `[████████░░] 80%` visual bars
- **Calibration Badges** -- Well-calibrated / moderate / high variance badges
- **Retail Trap Indicator** -- `Retail Trap` or `Public Bias` badges on signals where Tipico is shading favorites
- **Tax-Free Badge** -- `Steuerfrei` on qualifying 3+ leg combo suggestions
- **Interactive Agent Alerts** -- Executioner alerts include inline buttons: Deep Dive, Ghost Bet, Ignorieren
- **NLP Intent Routing** -- Natural language commands via Gemma 3 4B intent classification
- **Per-User State** -- Pagination and session data stored per-user, concurrent-safe

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
| `injury_news_delta` | Aggregated injury news sentiment (Rotowire RSS) |
| `time_to_kickoff_hours` | Hours until event starts |

- Weekly automatic retraining with post-training validation (Brier score, calibration check, feature importance audit)
- **Reliability diagrams**: per-bucket actual-vs-predicted calibration with automatic Kelly adjustment multipliers

### Sport Group Mapping

The trainer maps sport keys to groups for sport-specific models:

| Sport Key Pattern | Group | Model File |
|---|---|---|
| `soccer_*`, `football_*` | soccer | `xgb_soccer.joblib` |
| `basketball_*` | basketball | `xgb_basketball.joblib` |
| `tennis_*` | tennis | `xgb_tennis.joblib` |
| `americanfootball_*` | americanfootball | `xgb_americanfootball.joblib` |
| `icehockey_*` | icehockey | `xgb_icehockey.joblib` |
| everything else | general | `xgb_general.joblib` |

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

**Public bias detection**: When Tipico shades favorites (lowers odds) more than sharp books, the `public_bias` feature captures this retail-driven vig.

Configurable via `TIPICO_TAX_RATE` and `TAX_FREE_MODE` env vars.

### Combo System

Three lotto combo tiers with constraint-based optimization and market-type-aware scoring:

| Size | Type | Stake | Constraints |
|------|------|-------|-------------|
| 10-leg | Lotto | 1.00 EUR | min 3 sports, max 3/league, prob > 0.52, max 2 heavy favs/league |
| 20-leg | Lotto | 1.00 EUR | min 4 sports, max 4/league, prob > 0.50, max 3 heavy favs/league |
| 30-leg | Lotto | 0.50 EUR | min 5 sports, max 5/league, prob > 0.48, max 3 heavy favs/league |

**Market-type scoring boosts**: Double Chance and DNB legs get a 1.20x scoring boost; high-probability totals (prob >= 0.80) get 1.15x.

**Heavy favorite cap**: No more than 2-3 selections with odds < 1.30 per league.

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

- **Scout Agent** -- Monitors odds for steam moves (price changes exceeding 2x historical volatility), aggregated injury intel (API-Sports + Rotowire RSS + LLM), and 12h market momentum via Pro API
- **Analyst Agent** -- Triggered by Scout alerts; performs full enrichment, injury aggregation (with EV penalty for confirmed absentees), feature engineering, ML prediction, public bias detection, market momentum integration, and Gemma 3 4B reasoning. Uses **dynamic Poisson/XGBoost blending** with momentum adjustment.
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

### Backtesting

Walk-forward backtesting engine with strategy comparison:

```bash
python scripts/run_backtest.py
python scripts/run_backtest.py --compare
python scripts/run_backtest.py --kelly 0.15 --min-ev 0.01 --tax 0.05
```

---

## Configuration Reference

All settings are loaded from environment variables (`.env` file) via `src/core/settings.py`:

| Variable | Default | Required | Description |
|----------|---------|----------|-------------|
| `TELEGRAM_BOT_TOKEN` | -- | Yes | Telegram Bot API token |
| `TELEGRAM_CHAT_ID` | -- | Yes | Target Telegram chat ID |
| `ODDS_API_KEY` | -- | Yes | The-Odds-API key (Pro tier for momentum) |
| `POSTGRES_DSN` | `postgresql+psycopg://postgres:postgres@localhost:5432/signalbot` | No | PostgreSQL connection string |
| `REDIS_URL` | `redis://localhost:6379/0` | No | Redis connection URL |
| `NEWSAPI_KEY` | -- | No | NewsAPI key for sentiment |
| `APISPORTS_API_KEY` | -- | No | API-Sports key for injuries |
| `OLLAMA_BASE_URL` | `http://localhost:11434` | No | Ollama server URL |
| `OLLAMA_MODEL` | `gemma:4b` | No | LLM model name |
| `LIVE_SPORTS` | `basketball_nba,soccer_germany_bundesliga,soccer_epl,tennis_atp` | No | Comma-separated Odds API sport keys |
| `INITIAL_BANKROLL` | `1000.0` | No | Starting bankroll |
| `TIPICO_TAX_RATE` | `0.05` | No | Tax rate on gross winnings |
| `TAX_FREE_MODE` | `false` | No | Set `true` for combo tax-free mode |
| `ENRICHMENT_ENABLED` | `true` | No | Enable/disable API enrichment |
| `ENRICHMENT_TIMEOUT` | `30` | No | Timeout per enrichment call (seconds) |

---

## CLI Scripts Reference

| Script | Purpose | Usage |
|--------|---------|-------|
| `scripts/import_historical_results.py` | Import historical results into `placed_bets` | See [Historical Data Import](#historical-data-import) |
| `scripts/bootstrap_history.py` | Download free datasets from public sources | `python scripts/bootstrap_history.py` |
| `scripts/backfill_form_features.py` | Populate `form_winrate_l5` and `form_games_l5` | `python scripts/backfill_form_features.py` |
| `scripts/backfill_odds_from_imports.py` | Backfill `odds_open`/`odds_close`/`clv` from CSVs | `python scripts/backfill_odds_from_imports.py` |
| `scripts/run_backtest.py` | Walk-forward backtesting engine | `python scripts/run_backtest.py --compare` |
| `scripts/auto_grade_once.py` | Settle open bets against API results | `python scripts/auto_grade_once.py` |
| `scripts/run_bot.py` | Alternative bot launcher | `python scripts/run_bot.py` |

### Importer CLI Arguments

```
--imports-dir         Base directory (default: data/imports)
--football-dir        Override football directory
--football-files      Limit number of CSV files (0=all)
--tennis-dir          Override tennis directory
--tennis-files        Limit tennis files (0=all)
--nba-dir             Override NBA directory
--nfl-dir             Override NFL directory
--nhl-dir             Override NHL directory
--max-rows            Max rows per US sport file (default: 50000, 0=all)
```

---

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

---

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

---

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
│   │   ├── sport_mapping.py            # Central sport/league registry (25+ leagues)
│   │   └── volatility_tracker.py       # Odds volatility & steam moves
│   ├── data/
│   │   ├── models.py                   # PlacedBet SQLAlchemy model
│   │   ├── postgres.py                 # Engine + SessionLocal factory
│   │   ├── redis_cache.py              # Redis cache wrapper (get/set JSON)
│   │   └── venue_coordinates.py        # Stadium lat/lng for weather
│   ├── integrations/
│   │   ├── base_fetcher.py             # Async HTTP base + _safe_sync_run()
│   │   ├── odds_fetcher.py             # The-Odds-API client (Pro tier + history)
│   │   ├── news_fetcher.py             # NewsAPI client
│   │   ├── weather_fetcher.py          # Open-Meteo weather client
│   │   ├── rss_fetcher.py              # Rotowire RSS injury/lineup scraper
│   │   ├── injury_aggregator.py        # Unified aggregator (API-Sports + RSS + LLM)
│   │   ├── apisports_fetcher.py        # API-Sports injuries/lineups
│   │   └── ollama_sentiment.py         # Ollama Gemma 3 4B (sentiment + intents)
│   ├── models/
│   │   └── betting.py                  # Pydantic models (BetSignal, ComboBet)
│   └── utils/
│       ├── charts.py                   # Matplotlib dashboards (equity, pie)
│       └── telegram_md.py              # Telegram markdown formatting
├── scripts/
│   ├── import_historical_results.py    # Universal importer (5 sports, multi-format)
│   ├── bootstrap_history.py            # Free dataset downloader
│   ├── backfill_form_features.py       # Populate form_winrate/games_l5
│   ├── backfill_odds_from_imports.py   # Backfill open/close odds + CLV
│   ├── run_backtest.py                 # Backtest CLI
│   ├── run_bot.py                      # Alternative bot launcher
│   ├── auto_grade_once.py              # Settle open bets
│   └── smoke_test_step2.py             # Integration smoke test
├── data/
│   └── imports/                        # Staged data for importer (gitignored)
│       ├── football/                   # Soccer CSVs (football-data.co.uk)
│       ├── tennis/                     # Tennis CSV/XLSX (tennis-data.co.uk)
│       ├── nba/                        # NBA CSVs
│       ├── nfl/                        # NFL CSV/XLSX
│       └── nhl/                        # NHL CSVs
├── models/                             # Trained .joblib model files (gitignored)
├── config/
│   └── com.clawy.bettingbot.plist      # macOS LaunchAgent
├── requirements.txt
├── ml_strategy_weights.json            # Legacy model weights (JSON fallback)
└── package.json
```

---

## API Requirements

| API | Tier | Used For |
|-----|------|----------|
| The-Odds-API | **Pro** | Real-time odds + `/odds-history` for 12h momentum |
| NewsAPI | Free/Dev | News sentiment enrichment |
| API-Sports | Free | Injury data and lineups |
| Open-Meteo | Free | Weather conditions for outdoor sports |
| Ollama (local) | -- | Gemma 3 4B for sentiment, reasoning, NLP intents |
| Rotowire RSS | Free | Player injury/lineup news feeds (NBA/NFL/NHL/Soccer) |

---

## Troubleshooting

### `meta_features column missing` (or any schema mismatch)

The importer runs `_ensure_schema()` automatically, which executes `ALTER TABLE placed_bets ADD COLUMN IF NOT EXISTS ...` for all columns. Just re-run the importer:

```bash
python scripts/import_historical_results.py
```

If you prefer a nuclear reset:

```bash
python -c "
from src.data.models import Base
from src.data.postgres import engine
Base.metadata.drop_all(bind=engine)
Base.metadata.create_all(bind=engine)
print('Tables recreated')
"
```

Then re-import your data.

### `no feature variance` during training (basketball/NFL/NHL)

This happens when imported data has no engineered features (e.g. `sharp_implied_prob` is all zeros). The trainer now auto-derives `sharp_implied_prob = 1/odds` as a fallback. Re-run training:

```bash
python -c "from src.core.ml_trainer import auto_train_all_models; print(auto_train_all_models(min_samples=100))"
```

### NFL files not importing (`.xlsx` format)

The NFL importer now supports `.csv`, `.xlsx`, and `.xls` files. Make sure `openpyxl` is installed:

```bash
pip install openpyxl==3.1.5
```

### NHL `is_home` format not recognized

The NHL importer auto-detects three formats:
1. `is_home` column with `team_name`/`opp_team_name` (team-perspective)
2. `home_away` / `HoA` column (per-team with H/A indicator)
3. One row per game with `Home Team`/`Away Team` columns

Make sure your CSV has a `game_id` column for formats 1 and 2.

### NBA lowercase columns (`home`, `away`, `score_home`, `score_away`)

Supported. The NBA importer recognizes both camelCase (`HomeTeam`) and lowercase (`home`, `score_home`) column names.

### Redis connection refused

```bash
# Check if Redis is running
redis-cli ping   # Should return PONG

# Start Redis
redis-server
```

### PostgreSQL connection refused

```bash
# Check PostgreSQL status
pg_isready

# Create the database if missing
createdb signalbot
```

---

## License

Private repository. All rights reserved.
