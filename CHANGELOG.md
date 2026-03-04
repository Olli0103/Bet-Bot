# Changelog

## [2026-03-04] ML Feature Health Check — Audit Report

### Critical Finding: 29/35 features train on zeros

Only 6 of 35 XGBoost features are persisted to the `PlacedBet` table via
`ghost_trading.py`. The remaining 29 features (Phase 2-4) are correctly computed
at signal time but never written to the database. When `ml_trainer.py` reads via
`pd.read_sql()`, `_clean_frame()` creates missing columns as `0.0`, causing the
model to train on ~83% zero-variance data.

### Deliverable

- **`ML_FEATURE_AUDIT.md`** — Full audit report with:
  - Data coverage tables per feature (DB column, write path, null rate, variance)
  - Preprocessing chain walkthrough with exact file:line references
  - Leakage/timing verification (CLV removed, temporal split correct)
  - Root-cause classification (A=empty, B=constant, C=circular, D=redundant)
  - Prioritized KEEP/FIX/DROP action plan (P0-P3)
  - Architecture diagram showing signal-time vs training-time data flow

### Key recommendations

| Priority | Action |
|----------|--------|
| P0 | Persist all features via `meta_features` JSONB in `ghost_trading.py` |
| P0 | Unpack `meta_features` JSONB in `ml_trainer.py::_clean_frame()` |
| P1 | Use `FEATURE_DEFAULTS` dict instead of blanket `0.0` for missing features |
| P1 | Decouple `form_winrate_l5` and `h2h_home_winrate` from PlacedBet (circular) |
| P2 | Wire Phase 4 stats pipeline into feature dict in `live_feed.py` |

---

## [2026-03-03] Unified Confidence Model + Combo Leg Gate + Sorting Fix

### Root cause: "Model 28% + Conf 100%" contradiction

The BetSignal model had TWO separate fields that the UI both called "confidence":
- `model_probability` — the actual model prediction (e.g. 28%)
- `confidence` — the **odds source reliability** (e.g. 100% for primary Tipico/Pinnacle pair)

These were displayed side-by-side as "Modell: 28%" and "Conf: 100%", creating a
contradiction. Ranking used `confidence` (source quality) which meant a 28% pick
from a "reliable" source could outrank a 72% pick from a fallback source.

### Fix: Single source of truth

- `model_probability` is now THE confidence for ranking, gates, and UI display
- The old `confidence` field is set to `model_probability` (backward-compat)
- Source reliability is renamed to `source_quality` (separate field, shown as "SrcQ")
- `BettingEngine.make_signal()` enforces: `confidence = model_probability`
- No more contradictory display possible

### Ranking change

- **Before:** sort by `confidence` DESC (= source reliability, not model output!)
- **After:** sort by `model_probability` DESC -> `expected_value` DESC -> `odds` ASC
- Both `live_feed.py` (global ranking) and `handlers.py` (per-sport re-sort) updated

### Combo leg confidence gate

- New setting `MIN_COMBO_LEG_CONFIDENCE` (default 0.40)
- Every combo leg must have `model_probability >= 0.40` before combo construction
- `max_per_league` reduced to 2 across all combo profiles for diversification
- Cross-sport combos remain allowed

### Logging

- Every accepted bet logs: `model_prob`, `ev`, `trigger`, `stake`
- Every rejected bet logs: `reject_confidence_below_min: model_prob=X < gate=Y`

### New env vars

- `MIN_COMBO_LEG_CONFIDENCE=0.40` — minimum model_probability for combo legs

### Files changed

| File | Change |
|------|--------|
| `src/models/betting.py` | + `source_quality` field; `confidence` = `model_probability` |
| `src/core/betting_engine.py` | `confidence=` param renamed to `source_quality=`; sets `confidence=model_probability` |
| `src/core/live_feed.py` | All `confidence=conf` -> `source_quality=conf`; ranking by `model_probability` |
| `src/bot/handlers.py` | Sort by `model_probability`; card shows "Confidence:" + "SrcQ:" |
| `src/core/combo_optimizer.py` | `max_per_league=2` across all profiles |
| `src/core/settings.py` | + `min_combo_leg_confidence` |
| `tests/test_confidence_unified.py` | NEW: 14 tests covering all 5 mandatory cases |
| `tests/test_risk_guards.py` | Updated for new API (source_quality, model_probability sort) |

### Tests

- `tests/test_confidence_unified.py` — 14 tests:
  1. `test_single_tips_sorted_by_model_conf_desc`
  2. `test_single_tip_confidence_gate_blocks_low_conf`
  3. `test_steam_move_cannot_bypass_conf_gate`
  4. `test_combo_leg_min_confidence_40_applied`
  5. `test_ui_confidence_consistency_single_source`
- 76 total tests passing

---

## [2026-03-03] Risk Guards + Confidence Gate + Stake Caps + Top10 Fix

### Why false picks slipped through (root cause)

Before this change, there was **no confidence gate** anywhere in the pipeline.
Signals with model_probability as low as 18% could get high stakes because:
1. Kelly fraction was applied to raw model_probability without a floor check
2. `steam_move` triggers could amplify weak signals into actionable bets
3. No stake cap existed -- Kelly alone decided the size, sometimes 3-5% bankroll
4. Top10 was ranked by a hybrid formula `(prob*0.7 + ev*0.3)` that let high-EV/
   low-confidence picks bubble up

### A) Confidence Gate (hard block)

**New file:** `src/core/risk_guards.py`

| Sport / Market            | Min model_probability |
|---------------------------|-----------------------|
| Soccer h2h / draw         | 0.55                  |
| Soccer totals / spread    | 0.56                  |
| Tennis h2h                | 0.57                  |
| NBA / NHL / NFL sides     | 0.55                  |
| Default                   | 0.55                  |

- Gate enforced in **BettingEngine.make_signal()** -- signals below gate get
  `recommended_stake=0` and `rejected_reason` explaining why.
- Gate also enforced in **ExecutionerAgent.execute()** -- steam_move / totals_steam
  cannot override the gate. If confidence < gate, the bet is blocked regardless
  of trigger.
- All thresholds configurable via env vars (`MIN_CONF_SOCCER_H2H`, etc.).

### B) Stake Caps

- **General cap:** 1.5% of bankroll (`MAX_STAKE_PCT`)
- **Draw / longshot cap:** 0.75% of bankroll (`MAX_STAKE_LONGSHOT_PCT`)
  - Applies when odds >= 3.5 OR selection contains "Draw"
- Applied in both BettingEngine (signal generation) and Executioner (agent bets)
- `stake_cap_applied=true` flag on every capped signal for full transparency

### C) steam_move is booster-only

- `steam_move` and `totals_steam` still use reduced Kelly (0.15 vs 0.20)
- But the confidence gate runs first -- if model_probability < threshold, the
  bet is blocked regardless of how strong the steam move is
- This prevents the scenario where a 44% confidence pick got a high-stake
  recommendation purely because of a steam trigger

### D) Top10 sorting fixed

- **Before:** `(model_probability * 0.7 + min(ev*10, 1.0) * 0.3)` -- mixed formula
- **After:** `confidence DESC -> expected_value DESC -> odds ASC`
- Item #1 per sport is always the highest-confidence signal
- Re-sort happens per-sport after filtering, not just globally

### E) Output transparency

Every signal card now shows:
- `Kelly: <raw_fraction>` -- the raw Kelly value before any cap
- `Stake: <before_cap> -> <final> EUR` -- before and after cap
- `[CAP]` tag when cap was applied
- `trigger=<reason>` when signal was agent-triggered
- ASCII progress bars (`[########--]`) instead of Unicode block characters

### F) Settings additions

New env vars in `src/core/settings.py`:
- `MIN_CONF_SOCCER_H2H` (0.55), `MIN_CONF_SOCCER_TOTALS` (0.56), etc.
- `MAX_STAKE_PCT` (0.015), `MAX_STAKE_LONGSHOT_PCT` (0.0075)
- `LONGSHOT_ODDS_THRESHOLD` (3.5)

### G) Tests

- `tests/test_risk_guards.py` -- 19 tests covering all 5 mandatory test cases:
  1. `confidence_gate_blocks_low_confidence_even_with_steam_move`
  2. `stake_cap_applied_for_draw_and_longshot`
  3. `top10_per_sport_sorted_by_confidence_desc`
  4. `top10_item_1_is_best_confidence`
  5. `no_global_top10_leak_into_sport_filter`
- All 62 tests pass (19 new + 13 from previous Top10 fix + 30 existing)

### Files changed

| File | Change |
|------|--------|
| `src/core/settings.py` | + confidence gate + stake cap env vars |
| `src/core/risk_guards.py` | NEW: confidence gate + stake cap logic |
| `src/core/betting_engine.py` | Enforce gate + cap in make_signal, add transparency fields |
| `src/models/betting.py` | + kelly_raw, stake_before_cap, stake_cap_applied, trigger, rejected_reason |
| `src/core/live_feed.py` | Ranking changed to confidence DESC; gate-rejected signals excluded |
| `src/bot/handlers.py` | Per-sport re-sort by confidence; ASCII bars; full transparency card |
| `src/agents/executioner_agent.py` | Confidence gate before Kelly; stake cap after Kelly |
| `tests/test_risk_guards.py` | NEW: 19 tests |

---

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
