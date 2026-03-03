# Changelog

## [2026-03-03] Architecture Split + Multi-Chat + Combo UX + RSS Hardening

### A) Telegram / Core Architecture Separation

**New files:**
- `src/bot/core_worker.py` -- Independent Core process: runs signal pipeline, agent orchestrator, scheduled jobs (fetch, grading, retrain, health). Writes outbound messages to Redis outbox queue. Has NO Telegram dependency.
- `src/bot/telegram_worker.py` -- Independent Telegram process: handles polling, commands, inline buttons, NLP routing. Reads outbox queue and sends. Writes user actions to inbox queue.
- `src/bot/message_queue.py` -- Redis-backed outbox/inbox queue (LPUSH/BRPOP pattern). Dedup keys prevent double-sends on restart. 1-hour dedup window.
- `src/bot/__main__.py` -- `python -m src.bot` entry point (monolithic fallback).

**Key design:**
- Core worker runs a simple cron-like scheduler (no APScheduler dependency)
- Telegram worker polls outbox every 5 seconds
- Telegram send has retry (4 attempts, exponential backoff) + circuit breaker (5 failures -> 60s cooldown)
- Monolithic `app.py` still works as before (backward compatible)

**Entry points:**
```bash
# Split mode (recommended)
python -m src.bot.core_worker
python -m src.bot.telegram_worker

# Monolithic mode (backward compatible)
python -m src.bot.app
```

### B) Multi-Chat-ID Support

**New files:**
- `src/bot/chat_router.py` -- `ChatRouter` class with `primary_id`, `all_ids`, `broadcast_ids()`, `primary_only_ids()`, `is_authorized(chat_id)`.

**ENV semantics:**
- `TELEGRAM_CHAT_ID` = primary (always included)
- `TELEGRAM_CHAT_IDS` = CSV of additional allowed IDs

**Routing rules:**
- Broadcast events (daily push, agent alerts, combos) -> all IDs
- Primary-only events (diagnostics, retrain report, API health) -> primary only
- Incoming messages only accepted from allowed IDs

**Updated:** `app.py` all scheduled jobs now use `chat_router` for broadcasting instead of hardcoded `settings.telegram_chat_id`.

### C) Combo Output UX Fix

**Problem:** Combo legs showed naked selections like "Over 6.5" or "Draw" without event context -- completely useless.

**Before:**
```
 1. Over 6.5                        2.15   51%
 2. Draw                            3.40   32%
```

**After:**
```
 1. 🏒 NHL | Sabres vs Jets | totals 6.5 Over 6.5 | @2.15 | 51%
 2. ⚽ GERMANY BUNDESLIGA | Dortmund vs Bayern | h2h Draw | @3.40 | 32%
```

**Changes:**
- `live_feed.py`: All 5 combo_legs code paths (h2h, spreads, totals, double_chance, draw_no_bet) now include `home_team`, `away_team`, `market` fields.
- `models/betting.py`: `ComboLeg` model gains `home_team`, `away_team`, `market` fields.
- `betting_engine.py`: `build_combo()` now passes all fields through to `ComboLeg`.
- `handlers.py`: New `_format_leg_line()` renders each leg with sport emoji, league name, event, market, selection, odds, probability. Legs without event context are filtered out with warning. Combo header includes risk notes for very low probabilities.

**Validation:** Legs without `home_team`/`away_team` are rejected from the output with a `⚠️ N Legs ohne Event-Kontext verworfen` note.

### D) Importer Fixes (Verified)

Already in repo from previous work:
- NBA: lowercase aliases (`home/away/score_home/score_away`)
- NFL: `.xlsx` + `.xls` support via openpyxl
- NHL: 3 format variants (`is_home`, `home_away`, per-game)
- `openpyxl==3.1.5` in requirements.txt
- `_ensure_schema()` for automatic column migration

### E) Training Pipeline (Verified)

Already in repo:
- Sport groups: soccer, basketball, tennis, americanfootball, icehockey
- `sharp_implied_prob = 1/odds` fallback for raw imports without engineered features
- `auto_train_all_models()` trains general + all sport-specific models
- Training report: `{group}: {N} samples, brier={score}` per model

### F) RSS Error Handling + Rotowire Research

**Hardened `rss_fetcher.py`:**
- 3 retries with exponential backoff (2s, 4s) per feed
- 15-second timeout per HTTP attempt
- Browser-like User-Agent header (avoids Rotowire 403)
- Feed health tracking via `get_feed_health()` (status, entries, attempts, timestamp)
- All exceptions caught at outer boundary -- never crashes Core
- Separate `_fetch_feed_with_retry()` method with structured error propagation

**Rotowire Research (March 2026):**
- All 4 feeds (NBA, NFL, NHL, Soccer) are officially offered as free RSS 2.0
- URL pattern: `https://www.rotowire.com/rss/news.htm?sport={sport}`
- No API key needed. Must link back to rotowire.com (embedded in feed items).
- NFL/NHL feeds are sparse during off-season (expected, not an error).
- Known bot protection: returns 403 without browser User-Agent.

**Fallback alternatives documented:**
- RotoBaller (rotoballer.com) -- XML/RSS + JSON feeds
- ESPN RSS (espn.com/espn/rss/) -- injury reports
- NBC Sports Rotoworld -- NFL/NBA player news

### Documentation

- README updated: new architecture diagram, split worker docs, multi-chat-ID docs, RSS reliability section, updated config reference, updated project structure
- CHANGELOG created
