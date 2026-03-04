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

# 5. Run DB migrations + backfill engineered features
alembic upgrade head                                    # apply schema migrations
python scripts/backfill_form_features.py
python scripts/backfill_odds_from_imports.py
python scripts/backfill_ml_features.py --force          # backfill ALL ML features (phase 1-4)

# 6. Train ML models (generates ML_FEATURE_COVERAGE_REPORT.md)
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

### Schema

#### `placed_bets` -- Central bet tracking table

All imports, virtual bets, and live signals write here.

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
| `sharp_vig` | FLOAT | Sharp book overround |
| `sentiment_delta` | FLOAT | Home - away sentiment |
| `injury_delta` | FLOAT | Home - away injury impact |
| `form_winrate_l5` | FLOAT | Last-5 win rate |
| `form_games_l5` | FLOAT | Games in last-5 window |
| `meta_features` | JSONB | Sport-specific advanced stats |
| `notes` | TEXT | Free text |
| `created_at` | TIMESTAMPTZ | Row creation time |
| `updated_at` | TIMESTAMPTZ | Last update time |

#### `team_match_stats` -- Per-team per-match statistics

Ingested from TheSportsDB and football-data.org. One row per team per match.

| Column | Type | Description |
|--------|------|-------------|
| `source_match_id` | VARCHAR(128) | External match ID |
| `team` / `opponent` | VARCHAR(256) | Team names |
| `is_home` | BOOLEAN | Whether this team was home |
| `goals_for` / `goals_against` | INTEGER | Goals scored / conceded |
| `result` | VARCHAR(4) | W / D / L |
| `shots`, `possession_pct`, `corners`, etc. | various | Extended stats (nullable) |

#### `event_stats_snapshots` -- Pre-match feature snapshots

Rolling features computed from `team_match_stats` before each event (data leakage prevention).

| Column | Type | Description |
|--------|------|-------------|
| `event_id` | VARCHAR(128) | Links to the upcoming event |
| `team` | VARCHAR(256) | Team name |
| `attack_strength` / `defense_strength` | FLOAT | Goals vs league average |
| `form_trend_slope` | FLOAT | Linear regression over recent form |
| `over25_rate` / `btts_rate` | FLOAT | Historical rates |
| `rest_days` | INTEGER | Days since last match |
| `schedule_congestion` | FLOAT | Match density (30-day window) |
| `league_position` | INTEGER | Current league position |

### Database Migrations

Schema changes are managed via Alembic:

```bash
# Run pending migrations
alembic upgrade head

# Create a new migration
alembic revision -m "description"
```

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
TELEGRAM_CHAT_ID=your_chat_id              # Primary chat (always receives all messages)
TELEGRAM_CHAT_IDS=id1,id2,id3              # Additional allowed chat IDs (CSV, optional)
ODDS_API_KEY=your_odds_api_key             # Pro Tier required for /odds-history
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

### Running Modes

The bot supports two deployment modes:

#### Mode 1: Monolithic (simple, backward-compatible)

Single process runs both Telegram I/O and the Core pipeline:

```bash
python -m src.bot.app
```

#### Mode 2: Split Architecture (recommended for production)

Two independent workers connected via Redis queues. Telegram failures
cannot crash or block the Core pipeline:

```bash
# Terminal 1: Core Worker (signals, agents, ML, grading)
python -m src.bot.core_worker

# Terminal 2: Telegram Worker (UI, polling, message delivery)
python -m src.bot.telegram_worker
```

Benefits:
- Core keeps running if Telegram is down/blocked
- Telegram worker can restart without losing signals
- Redis outbox queue buffers messages during Telegram outages
- Independent health monitoring per worker
- Circuit breaker on Telegram sends (5 failures -> 60s cooldown)

### Multi-Chat-ID Routing

| ENV Variable | Purpose |
|---|---|
| `TELEGRAM_CHAT_ID` | Primary chat ID (receives everything) |
| `TELEGRAM_CHAT_IDS` | CSV of additional allowed IDs |

Routing rules:
- **Broadcast events** (daily push, agent alerts, combos) -> all IDs
- **Primary-only events** (diagnostics, retrain, API health) -> primary only
- **Incoming messages** are only accepted from allowed IDs

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

### Split Worker Architecture

```
┌─────────────────────────────────────────────┐
│ CORE WORKER (python -m src.bot.core_worker) │
│                                             │
│  Orchestrator (adaptive 60s / 5min)         │
│    ├── Scout Agent    → odds, steam, RSS    │
│    ├── Analyst Agent  → ML, enrichment      │
│    ├── Executioner    → Kelly, circuits      │
│    └── Performance    → ROI, self-eval      │
│                                             │
│  Scheduled: fetch, grading, retrain, health │
│                                             │
│  Writes → Redis Outbox Queue                │
│  Reads  ← Redis Inbox Queue                 │
└──────────────────┬──────────────────────────┘
                   │ Redis
┌──────────────────┴──────────────────────────┐
│ TELEGRAM WORKER (python -m src.bot.telegram_worker) │
│                                             │
│  Polling / Webhook                          │
│  Commands, Buttons, NLP Intent Router       │
│  Multi-Chat-ID Broadcast + Access Control   │
│  Circuit Breaker (retry + backoff)          │
│                                             │
│  Reads  ← Redis Outbox Queue               │
│  Writes → Redis Inbox Queue                 │
└─────────────────────────────────────────────┘
```

### Pipeline Overview

```
Agent Orchestrator (adaptive: 60s pre-kickoff / 5min normal)
      |
  Scout Agent        -->  odds monitoring, steam move detection, injury aggregation
      |                   12h market momentum tracking (Pro API)
  Analyst Agent      -->  enrichment (sentiment, injuries, Elo, Poisson, form, H2H)
      |                   feature engineering (34 features incl. stats-based)
      |                   ML model prediction (XGBoost + calibration)
      |                   market expansion (Double Chance, DNB derivations)
  Executioner Agent  -->  circuit breakers, calibration-adjusted Kelly, virtual bets
      |
  Performance Monitor --> ROI tracking, daily reports, self-evaluation
```

### Core Pipeline

1. **Data Ingestion** -- The-Odds-API (Pro Tier) provides real-time and historical odds across h2h, spreads, totals, double_chance, and draw_no_bet markets from 6+ bookmakers. TheSportsDB and football-data.org provide match results and standings for rolling feature computation.
2. **Stats Ingestion** -- Periodic pipeline (every 6h) fetches past match results from TheSportsDB + football-data.org, stores per-team per-match stats (`team_match_stats`), and computes rolling feature snapshots (`event_stats_snapshots`) with data leakage prevention.
3. **Enrichment** -- News sentiment (NewsAPI + Ollama Gemma 3 4B), injury data (API-Sports + Rotowire RSS), weather (Open-Meteo), LLM-structured injury extraction
4. **Feature Engineering** -- 34+ features including CLV, Elo differential, Poisson-derived probabilities, form tracking, H2H history, odds volatility, public bias, market momentum, attack/defense strength, form trend slope, rest fatigue, schedule congestion, over 2.5/BTTS rates, home/away splits, league position delta
5. **Model Prediction** -- XGBoost with isotonic calibration, sport-specific models (soccer/basketball/tennis/americanfootball/icehockey), TimeSeriesSplit validation, reliability diagrams, quality gates (Brier > 0.25 rejection, min 50 samples)
6. **Value Detection** -- Tax-adjusted EV calculation (Tipico 5% tax + combo tax-free mode), consensus sharp line from weighted Pinnacle/Betfair/bet365, public bias detection
7. **Bet Sizing** -- Fractional Kelly criterion with performance-based multipliers, calibration-adjusted scaling, and circuit breakers
8. **Combo Construction** -- Constraint-optimized 10/20/30-leg lotto combos with dynamic pairwise correlation penalties and market-type-aware scoring
9. **Virtual Trading** -- Ghost-trades all signals for tracking without real money
10. **Source Health** -- Per-source circuit breakers with failure counting, cooldown timers, and half-open recovery for all 8 data sources

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

### Signal Deduplication & Card Format

- **One pick per leg**: Enforces one signal per `(event_id, canonical_market_group)`. Market groups: `h2h`, `double_chance`, `draw_no_bet`, `spreads`, `totals`.
- **Selection rule**: `model_probability DESC → expected_value DESC → bookmaker_odds ASC`
- **Status badges**: 🟢 PLAYABLE (positive EV, stake > 0) | 🟡 WATCHLIST (EV ≤ 0) | 🔴 BLOCKED (rejected by gate)
- **Summary header**: Shows raw vs deduped count, status breakdown, active EV-cut and confidence gates
- **Card format**: Compact, card-like layout with sport emoji, status badge, and all transparency fields preserved

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
- **Stats Card** -- Tipico-style comparison card in Deep Dive showing form blocks (🟩🟨🟥), league position, goals, O/U rates, BTTS, attack/defense strength, rest days, and home/away splits
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
- Quality gates: Brier > 0.25 rejection, minimum 50 training samples
- 34 engineered features:

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
| `team_attack_strength` | Goals scored / league avg (rolling window) |
| `team_defense_strength` | Goals conceded / league avg (rolling window) |
| `opp_attack_strength` | Opponent attack strength |
| `opp_defense_strength` | Opponent defense strength |
| `expected_total_proxy` | Predicted total goals from strength ratings |
| `form_trend_slope` | Linear regression slope over recent form points |
| `rest_fatigue_score` | Fatigue from short rest (0-1 scale) |
| `schedule_congestion` | Matches per 30-day window (normalized) |
| `over25_rate` | Historical over 2.5 goals rate |
| `btts_rate` | Historical both-teams-to-score rate |
| `home_away_split_delta` | Home win rate minus away win rate |
| `league_position_delta` | Opponent position minus team position |
| `goals_scored_avg` | Average goals scored per match |
| `goals_conceded_avg` | Average goals conceded per match |

- Weekly automatic retraining with Champion/Challenger gating (challenger must beat champion's log loss)
- Data-driven retraining: auto-triggers when 500+ new graded bets accumulate
- **Permutation-importance feature pruning**: two-pass training drops noisy features, retrains, only promotes if Brier score doesn't regress
- **NaN spike detection**: warns when feature NaN rates exceed 10% (catches upstream schema changes)
- **Reliability diagrams**: per-bucket actual-vs-predicted calibration with automatic Kelly adjustment multipliers
- Temporal ordering enforced in training queries to prevent data leakage in TimeSeriesSplit
- **Semantically correct defaults**: `FEATURE_DEFAULTS` dict provides neutral values (e.g. `elo_expected=0.5`, not `0.0`) for missing features

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

### Risk Guards

Central module (`risk_guards.py`) enforced in both `BettingEngine.make_signal()` and `ExecutionerAgent.execute()`:

**Confidence Gate** -- Hard-blocks signals below a per-sport/market minimum `model_probability`:

| Sport / Market | Min model_probability |
|---|---|
| Soccer H2H | 0.55 |
| Soccer Totals / Spread | 0.56 |
| Tennis | 0.57 |
| Basketball / Ice Hockey / NFL | 0.55 |
| Default | 0.55 |

Steam moves and other triggers cannot override the gate.

**Stake Caps** -- Prevents Kelly from over-sizing:

| Cap | Applies When | Default |
|---|---|---|
| General | Always | 1.5% of bankroll |
| Draw / Longshot | odds >= 3.5 OR selection contains "Draw" | 0.75% of bankroll |

**Data Source Health Gate** -- If the Odds API circuit breaker is open, all new signals are blocked automatically. The bot won't generate bets while flying blind.

**Signal Explanations** -- Every accepted signal includes a German-language explanation with edge size, model confidence tier, key drivers (steam, momentum, public bias), historical accuracy from reliability bins, and calibration warnings.

### ML Feature Pruning

During training, a two-pass strategy reduces overfitting on small datasets:

1. Train XGBoost on all active features
2. Compute permutation importance on the holdout set
3. Prune features with zero or negative impact on Brier score (stricter threshold for datasets < 3000 samples)
4. Retrain on the pruned feature set
5. Only promote the pruned model if Brier score doesn't regress

Feature importance and pruning decisions are stored in the model's `metrics` dict.

**NaN Spike Detection** -- `_clean_frame()` logs warnings when any feature exceeds 10% NaN rate (50%+ = higher severity), catching upstream schema changes or data source failures before they silently corrupt training.

### Circuit Breakers

| Breaker | Trigger | Action |
|---------|---------|--------|
| Losing streak | `LOSING_STREAK_THRESHOLD` consecutive losses (default 7) | Kelly x0.5, min EV raised (`MIN_EV_LOSING_STREAK`, default 0.02) |
| Daily loss limit | > `DAILY_LOSS_LIMIT_PCT` of bankroll lost today (default 5%) | Kelly x0.5, min EV raised (`MIN_EV_LOSING_STREAK`) |
| Drawdown | `DRAWDOWN_LOOKBACK_DAYS`-day PnL loss > `DRAWDOWN_MAX_PCT` of bankroll (default 10%) | Kelly x0.5, min EV raised (`MIN_EV_DRAWDOWN`, default 0.02) |
| Model degradation | Hit rate < 40% over 14 days | Kelly x0.7, min EV raised (`MIN_EV_DEGRADATION`, default 0.015) |
| Data source offline | Odds API circuit breaker open | All betting halted |

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
| `TELEGRAM_CHAT_ID` | -- | Yes | Primary Telegram chat ID |
| `TELEGRAM_CHAT_IDS` | -- | No | CSV of additional allowed chat IDs |
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
| `SPORTSDB_API_KEY` | `3` | No | TheSportsDB API key (free tier = `3`) |
| `FOOTBALL_DATA_API_KEY` | -- | No | football-data.org API key (free tier) |
| `STATS_INGESTION_ENABLED` | `true` | No | Enable/disable stats ingestion pipeline |
| `STATS_INGESTION_INTERVAL_HOURS` | `6` | No | How often stats ingestion runs |
| **EV Thresholds** | | | |
| `MIN_EV_DEFAULT` | `0.01` | No | Min expected value for normal operation |
| `MIN_EV_LOSING_STREAK` | `0.02` | No | Min EV when losing streak or daily loss breaker trips |
| `MIN_EV_DRAWDOWN` | `0.02` | No | Min EV when drawdown breaker trips |
| `MIN_EV_DEGRADATION` | `0.015` | No | Min EV when model degradation detected |
| `MIN_EV_GOOD_RUN` | `0.005` | No | Min EV during good performance (ROI > 5%) |
| **Confidence Gates** | | | |
| `MIN_CONF_SOCCER_H2H` | `0.55` | No | Min model_probability for soccer H2H bets |
| `MIN_CONF_SOCCER_TOTALS` | `0.56` | No | Min model_probability for soccer totals |
| `MIN_CONF_SOCCER_SPREAD` | `0.56` | No | Min model_probability for soccer spreads |
| `MIN_CONF_TENNIS` | `0.57` | No | Min model_probability for tennis |
| `MIN_CONF_BASKETBALL` | `0.55` | No | Min model_probability for basketball |
| `MIN_CONF_ICEHOCKEY` | `0.55` | No | Min model_probability for ice hockey |
| `MIN_CONF_AMERICANFOOTBALL` | `0.55` | No | Min model_probability for NFL |
| `MIN_CONF_DEFAULT` | `0.55` | No | Fallback min model_probability |
| `MAX_STAKE_PCT` | `0.015` | No | Max stake as fraction of bankroll (1.5%) |
| `MAX_STAKE_LONGSHOT_PCT` | `0.0075` | No | Max stake for draws/longshots (0.75%) |
| `LONGSHOT_ODDS_THRESHOLD` | `3.5` | No | Odds >= this trigger longshot cap |
| `MIN_COMBO_LEG_CONFIDENCE` | `0.40` | No | Min model_probability per combo leg |
| **Kelly Fraction** | | | |
| `KELLY_FRACTION_DEFAULT` | `0.20` | No | Standard Kelly fraction for bet sizing |
| `KELLY_FRACTION_REACTIVE` | `0.15` | No | Reduced Kelly fraction for reactive bets (steam moves) |
| **Circuit Breakers** | | | |
| `LOSING_STREAK_THRESHOLD` | `7` | No | Consecutive losses to trigger losing streak breaker |
| `DAILY_LOSS_LIMIT_PCT` | `0.05` | No | Daily loss as fraction of bankroll to trigger breaker (5%) |
| `DRAWDOWN_MAX_PCT` | `0.10` | No | Rolling drawdown as fraction of bankroll to trigger breaker (10%) |
| `DRAWDOWN_LOOKBACK_DAYS` | `7` | No | Days lookback for drawdown calculation |
| **Combo Correlation** | | | |
| `COMBO_CORRELATION_PENALTY` | `0.80` | No | Probability penalty per correlated pair in combos |
| `COMBO_CORRELATION_FLOOR` | `0.20` | No | Minimum correlation penalty floor (prevents near-zero) |
| **Signal Modes / Learning** | | | |
| `LEARNING_CAPTURE_ALL_SIGNALS` | `true` | No | Capture all signal candidates as paper records for learning |
| `ALLOW_WATCHLIST_SIGNALS` | `true` | No | Include watchlist/paper-only signals in output |
| **Fetch Scheduler** | | | |
| `FETCH_MIN_DELAY_MS` | `800` | No | Min delay between sequential odds API requests (ms) |
| `FETCH_MAX_DELAY_MS` | `1500` | No | Max delay between sequential odds API requests (ms) |
| `FETCH_MAX_RETRIES` | `3` | No | Max retries per sport key on 429/5xx errors |

---

## CLI Scripts Reference

| Script | Purpose | Usage |
|--------|---------|-------|
| `scripts/import_historical_results.py` | Import historical results into `placed_bets` | See [Historical Data Import](#historical-data-import) |
| `scripts/bootstrap_history.py` | Download free datasets from public sources | `python scripts/bootstrap_history.py` |
| `scripts/backfill_form_features.py` | Populate `form_winrate_l5` and `form_games_l5` | `python scripts/backfill_form_features.py` |
| `scripts/backfill_odds_from_imports.py` | Backfill `odds_open`/`odds_close`/`clv` from CSVs | `python scripts/backfill_odds_from_imports.py` |
| `scripts/backfill_ml_features.py` | Backfill 6 critical ML features in `meta_features` | `python scripts/backfill_ml_features.py` |
| `scripts/run_backtest.py` | Walk-forward backtesting engine | `python scripts/run_backtest.py --compare` |
| `scripts/auto_grade_once.py` | Settle open bets against API results | `python scripts/auto_grade_once.py` |
| `scripts/run_bot.py` | Alternative bot launcher | `python scripts/run_bot.py` |

### backfill_ml_features.py CLI Arguments

```
--limit N       Max rows to process (0=all)
--sport SPORT   Filter by sport (e.g. soccer, tennis)
--dry-run       Preview changes without writing to DB
--force         Rewrite all rows, not only missing
```

Fills missing critical features: `sentiment_delta`, `injury_delta`, `sharp_implied_prob`,
`sharp_vig`, `form_winrate_l5`, `form_games_l5`. Batchwise (200 rows), idempotent.
Derives `sharp_implied_prob` from odds when missing, computes form from `TeamMatchStats`.

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

### Daily (Berlin time)

| Time | Job | Description |
|------|-----|-------------|
| 06:00 | Morning briefing | Fetch daily schedule (1 API call/sport), build initial signals |
| 07:00 | Daily push | Push top singles + combos to Telegram |
| 20:00 | Learning status | Win/loss stats and PnL summary |
| 20:05 | API health check | Status of all external API connections |
| 22:00 | Daily performance | Full performance report with circuit breaker status |

### Weekly

| Time | Job | Description |
|------|-----|-------------|
| 03:15 (Sat) | Weekly retrain | Retrain all XGBoost models with Champion/Challenger gating |

### Recurring

| Interval | Job | Description |
|----------|-----|-------------|
| Every 60s | JIT signal fetch | Targeted fetch for events kicking off within 75 min (only relevant sports) |
| Every 30s | CLV snapshot | Log Pinnacle closing lines for events at kickoff (T-1 to T+3 min) |
| Every 5 min | Agent cycle | Scout/Analyst/Executioner (reads from cache, no API calls) |
| Every 5 min | Source health check | Push summary to Telegram if any data source is degraded/open |
| Every 30 min | Auto-grading | Settle open bets against API results |
| After grading | Data-driven retrain | Trigger retraining when 500+ new graded bets accumulate |

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
│   │   ├── app.py                      # Monolithic entry point (backward compat)
│   │   ├── core_worker.py              # Core worker (signals, agents, ML — no Telegram)
│   │   ├── telegram_worker.py          # Telegram worker (UI, polling, outbox consumer)
│   │   ├── message_queue.py            # Redis-backed outbox/inbox queues
│   │   ├── chat_router.py              # Multi-chat-ID routing (broadcast/primary/ACL)
│   │   ├── handlers.py                 # UI handlers, settings dashboard, NLP routing
│   │   └── __main__.py                 # python -m src.bot entry point
│   ├── core/
│   │   ├── api_health.py              # Daily API connectivity check
│   │   ├── autograding.py             # Settle open bets against results
│   │   ├── backtester.py               # Walk-forward backtesting engine
│   │   ├── bankroll.py                 # Dynamic bankroll management from DB
│   │   ├── betting_engine.py           # Signal generation & combo building
│   │   ├── betting_math.py             # EV, Kelly criterion, tax-adjusted math
│   │   ├── clv_logger.py              # Closing line value logging at kickoff
│   │   ├── combo_optimizer.py          # Constraint-based combo construction
│   │   ├── correlation.py              # Pairwise correlation penalties
│   │   ├── dynamic_settings.py         # Redis-backed dynamic settings (Telegram toggles)
│   │   ├── elo_ratings.py              # Redis-backed Elo power ratings
│   │   ├── enrichment.py               # News sentiment & injury enrichment
│   │   ├── feature_engineering.py      # 34 feature builder (core + stats-based)
│   │   ├── form_tracker.py             # Redis-backed last-5-games form
│   │   ├── ghost_trading.py            # Virtual bet placement
│   │   ├── h2h_tracker.py              # Head-to-head history (TeamMatchStats primary)
│   │   ├── learning_monitor.py        # Win/loss tracking and PnL health
│   │   ├── live_feed.py                # Main signal pipeline (DC/DNB, momentum)
│   │   ├── ml_trainer.py               # XGBoost training with calibration
│   │   ├── performance_monitor.py      # ROI tracking & circuit breakers
│   │   ├── poisson_model.py            # Poisson goal model (0.5/1.5/2.5/3.5)
│   │   ├── pricing_model.py            # True probability estimation (XGBoost)
│   │   ├── risk_guards.py             # Confidence gates, stake caps, data source gate
│   │   ├── settings.py                 # Configuration from environment
│   │   ├── source_health.py            # Per-source circuit breakers & health reports
│   │   ├── sport_mapping.py            # Central sport/league registry (25+ leagues)
│   │   ├── stats_ingester.py           # Stats ingestion pipeline (TheSportsDB + football-data.org)
│   │   └── volatility_tracker.py       # Odds volatility & steam moves
│   ├── data/
│   │   ├── models.py                   # PlacedBet, TeamMatchStats, EventStatsSnapshot models
│   │   ├── postgres.py                 # Engine + SessionLocal factory
│   │   ├── redis_cache.py              # Redis cache wrapper (get/set JSON)
│   │   └── venue_coordinates.py        # Stadium lat/lng for weather
│   ├── integrations/
│   │   ├── base_fetcher.py             # Async HTTP base + _safe_sync_run()
│   │   ├── odds_fetcher.py             # The-Odds-API client (Pro tier + history)
│   │   ├── football_data_fetcher.py    # football-data.org v4 API client
│   │   ├── sportsdb_fetcher.py         # TheSportsDB API client
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
├── alembic/                            # Database migrations
│   ├── env.py                          # Alembic config (imports Base metadata)
│   └── versions/                       # Migration scripts
├── config/
│   └── com.clawy.bettingbot.plist      # macOS LaunchAgent
├── tests/                              # Pytest test suite (153+ tests)
├── .env.example                        # Environment variable template
├── requirements.txt
├── ml_strategy_weights.json            # Legacy model weights (JSON fallback)
└── alembic.ini                         # Alembic configuration
```

---

## API Requirements

| API | Tier | Used For |
|-----|------|----------|
| The-Odds-API | **Pro** | Real-time odds + `/odds-history` for 12h momentum |
| TheSportsDB | Free (`3`) | Match results, standings, team data for stats ingestion |
| football-data.org | Free | Match results, standings, home/away splits (12 competitions) |
| NewsAPI | Free/Dev | News sentiment enrichment |
| API-Sports | Free | Injury data and lineups |
| Open-Meteo | Free | Weather conditions for outdoor sports |
| Ollama (local) | -- | Gemma 3 4B for sentiment, reasoning, NLP intents |
| Rotowire RSS | Free | Player injury/lineup news feeds (NBA/NFL/NHL/Soccer) |

---

## RSS Source Reliability & Fallbacks

**Researched: March 2026**

### Rotowire RSS Feeds

| Sport | URL | Status | Notes |
|-------|-----|--------|-------|
| NBA | `rotowire.com/rss/news.php?sport=NBA` | Reliable | Year-round coverage |
| NFL | `rotowire.com/rss/news.php?sport=NFL` | Reliable | Sparse during off-season (March-August) |
| NHL | `rotowire.com/rss/news.php?sport=NHL` | Reliable | Sparse during off-season (June-September) |
| Soccer | `rotowire.com/rss/news.php?sport=SOCCER` | Reliable | US-centric; EPL/La Liga/Bundesliga coverage |

**Access**: Free RSS 2.0 feeds. No API key required. Must link back to rotowire.com (embedded in feed items).

**Known issues**:
- May return HTTP 403 if User-Agent looks like a bot scraper. The fetcher uses a browser-like User-Agent.
- Off-season feeds return few or no entries. This is expected, not an error.
- Some entries lack `published_parsed`; the parser tries 4 date formats.

**Error handling**:
- 3 retries with exponential backoff (2s, 4s) per feed
- 15-second timeout per attempt
- 30-minute Redis cache (prevents IP bans)
- Per-feed threading.Lock (prevents stampede on cache miss)
- Total failure returns empty list (graceful degradation, never crashes Core)
- Feed health tracked per-sport via `get_feed_health()`

**Fallback alternatives** (if Rotowire becomes unreliable):
- [RotoBaller](https://www.rotoballer.com/) -- Similar coverage, XML/RSS + JSON feeds
- [ESPN RSS](https://www.espn.com/espn/rss/) -- NFL/NBA/NHL injury reports
- [NBC Sports Rotoworld](https://www.nbcsports.com/fantasy/) -- NFL/NBA player news

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

### Bets not appearing (confidence gate rejection)

If signals are generated but have `recommended_stake=0`, they were blocked by the confidence gate. Check the logs for:
```
Confidence gate blocked: soccer_epl <event_id> <selection> — reject_confidence_below_min: model_prob=0.48 < gate=0.55
```

Lower the gate via env vars (e.g. `MIN_CONF_SOCCER_H2H=0.50`) or wait for the model to improve with more training data.

### Data source circuit breaker tripped

When a data source fails repeatedly, the circuit breaker trips and you'll see a Telegram alert:
```
🔴 Circuit Breaker: The Odds API
5 aufeinanderfolgende Fehler — Quelle pausiert (300s Cooldown).
```

If the Odds API is down, **all betting is automatically paused** (data source health gate). Check source status:
```bash
python -c "from src.core.source_health import get_health_report; print(get_health_report())"
```

The breaker auto-recovers after the cooldown period. Manual reset:
```bash
python -c "from src.core.source_health import record_success; record_success('odds_api')"
```

### Drawdown circuit breaker halting bets

If the 7-day PnL loss exceeds 10% of bankroll, the drawdown breaker trips and halves all Kelly multipliers. Check status:
```bash
python -c "from src.core.performance_monitor import PerformanceMonitor; m = PerformanceMonitor(); print(m.check_circuit_breakers())"
```

The breaker auto-resets when the 7-day PnL recovers. The bot continues generating signals but with reduced stakes.

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
