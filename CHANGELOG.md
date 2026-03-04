# Changelog

## [2026-03-04] Improve EV Quality: Calibration Layer + Sharp/Target Price Audit

### A) Calibration Layer (P0)

**Problem:** High confidence values but systematically negative EV across multiple
markets (e.g. basketball h2h). Model probabilities were miscalibrated — raw model
output didn't match actual win rates.

**Fix: Per-sport/market calibration layer**

New module `src/core/calibration.py`:
- **Isotonic regression** (default) and **Platt scaling** (configurable)
- Per sport/market calibrators (e.g. `basketball_h2h`, `soccer_h2h`)
- Global fallback when sport/market has < 30 samples
- Raw passthrough with warning when no calibrator is available
- Calibration report generation: `artifacts/calibration_report.json` + `CALIBRATION_REPORT.md`
- Metrics: ECE, MCE, Brier score, log-loss per sport/market

**New fields on BetSignal:**
- `model_probability_raw` — raw model output before calibration
- `model_probability_calibrated` — calibrated probability (= `model_probability`)
- `calibration_source` — "sport_market", "global", or "raw_passthrough"

**EV computation** now uses calibrated probability by default.

### B) Sharp/Target Price Logic Audit (P0)

**Problem:** Potential systematic negative EV from odds normalization, vig/tax
handling, or incorrect target/sharp mapping.

**Verified:**
- Selection matching: target and sharp use identical selection keys (no team-side flip)
- Vig removal happens exactly once in `FeatureEngineer.calculate_vig()`
- Tax applied exactly once in `expected_value()` on gross payout
- No double tax/vig deductions in the EV chain
- Sharp implied probability computed without tax
- CLV proxy sign consistency (positive = target better than sharp)

### C) EV Diagnostics (Debug Mode)

New module `src/core/ev_diagnostics.py`:
- Per-signal diagnostic payload written to `artifacts/ev_diagnostics.jsonl`
- Each entry contains: `raw_prob`, `calibrated_prob`, `target_odds`, `sharp_odds`,
  `implied_prob_target`, `implied_prob_sharp`, `vig`, `tax_rate`, `EV_final`
- Per-cycle calibration stats: `avg_raw_prob`, `avg_calibrated_prob`,
  `calibration_adjustment_mean`

### D) No Functional Regressions

- PLAYABLE flow unchanged — signals still flow through confidence gates + stake caps
- All existing signal generation paths updated to pass calibration metadata
- Backward-compatible: new BetSignal fields have safe defaults

### E) Tests (27 new, 6 required + 21 supporting)

| # | Test | Validates |
|---|------|-----------|
| 1 | `test_calibrated_probability_used_for_ev` | EV uses calibrated prob, not raw |
| 2 | `test_probability_scale_consistency_0_1` | All probs on 0-1 scale |
| 3 | `test_target_sharp_selection_alignment` | Selection keys match, no team flip |
| 4 | `test_vig_and_tax_applied_once` | No double tax/vig in EV |
| 5 | `test_ev_diagnostics_contains_required_fields` | Full EV decomposition in diagnostics |
| 6 | `test_fallback_calibrator_when_low_samples` | Global fallback when sport has low samples |

### F) New env vars

| Variable | Default | Description |
|----------|---------|-------------|
| `CALIBRATION_METHOD` | `isotonic` | Calibration method: "isotonic" or "platt" |
| `CALIBRATION_ENABLED` | `true` | Enable/disable calibration layer |
| `EV_DIAGNOSTICS_ENABLED` | `true` | Write per-signal EV diagnostics |

### Files changed

| File | Change |
|------|--------|
| `src/core/calibration.py` | NEW: Calibration module (isotonic/Platt, per-sport/market, global fallback, report generation) |
| `src/core/ev_diagnostics.py` | NEW: EV diagnostics logger (per-signal + per-cycle stats) |
| `src/core/pricing_model.py` | Calibration integration: raw prob stored, calibrated prob returned |
| `src/core/betting_engine.py` | Propagate calibration fields (raw, calibrated, source) to BetSignal |
| `src/core/betting_math.py` | No change (verified: EV formula correct) |
| `src/core/live_feed.py` | Pass `market` to pricing model, log EV diagnostics, cycle calibration stats |
| `src/core/settings.py` | + `calibration_method`, `calibration_enabled`, `ev_diagnostics_enabled` |
| `src/models/betting.py` | + `model_probability_raw`, `model_probability_calibrated`, `calibration_source` |
| `src/agents/analyst_agent.py` | Pass `market` to pricing model, log EV diagnostics |
| `tests/test_ev_calibration.py` | NEW: 27 tests covering all 6 required + calibrator basics |
| `.env.example` | + calibration settings |
| `CHANGELOG.md` | This entry |
| `README.md` | Calibration documentation section |

---

## [2026-03-04] Fix: Training Blockers — Schema, Feature Coverage, Backfill

### A) CLV Regressor Schema Fix

**Problem:** `_build_clv_dataset()` crashed with `event_closing_lines.market does
not exist` because the `market` column was added to the ORM model but the DB
migration hadn't been run.

**Fix:**
- `_build_clv_dataset()` now probes the DB schema at runtime via `sqlalchemy.inspect()`.
  If the `market` column is absent, it falls back to a legacy join
  `(event_id, selection, sport)` with an `h2h` filter and logs a clear warning.
- New Alembic migration `d5a3c91e2b04`: adds `market` column to
  `event_closing_lines` (default `"h2h"`), updates unique constraint.

### B) Feature Coverage Fix — 100% NaN Eliminated

**Problem:** 14 Phase 2-4 features (`elo_diff`, `weather_rain`, `home_volatility`,
`is_steam_move`, `line_staleness`, `injury_news_delta`, `public_bias`,
`market_momentum`, `line_velocity`, `form_trend_slope`, `home_away_split_delta`,
`league_position_delta`, etc.) were 100% NaN because they were intentionally
left out of `FEATURE_DEFAULTS` to use XGBoost native missing handling. But with
ALL rows being NaN, XGBoost had zero variance and dropped them entirely.

**Fix:**
- ALL 35+ features now have explicit entries in `FEATURE_DEFAULTS`.
- No feature column is ever 100% NaN after `_clean_frame()`.
- `_get_active_features()` now logs which features were dropped and why.
- `auto_train_all_models()` emits detailed diagnostics (NaN rates, constant
  features) instead of the opaque `"no feature variance"` message.

### C) Missing-Indicator Features

5 binary `is_missing_*` indicator features added so XGBoost can distinguish
"value was 0.0 because measured" from "value was 0.0 because backfill default":
- `is_missing_elo`, `is_missing_weather`, `is_missing_volatility`,
  `is_missing_stats`, `is_missing_form_trend`
- Generated automatically in `_clean_frame()` before defaults are applied.

### D) Backfill Script Extended (Phase 2-4)

`scripts/backfill_ml_features.py` now backfills ALL features, not just the 6
critical ones:
- Phase 2 enrichment features (Elo, weather, volatility, etc.) → semantic defaults
- Phase 4 stats features (attack/defense strength, form trend, etc.) → league avg
- Prints coverage BEFORE and AFTER backfill for verification
- `_needs_backfill()` checks ALL features, not just critical 6

### E) Training Guards Enhanced

- `_get_active_features()` logs dropped features with reasons
- Sport-specific "no feature variance" now emits full diagnostic:
  NaN rates, constant features, and a hint to run backfill
- Coverage report already existed — now benefits from full defaults

### F) Tests (6 new, matching required test names)

| # | Test | Validates |
|---|------|-----------|
| 7 | `test_clv_regression_handles_missing_event_closing_lines_market` | Schema fallback |
| 8 | `test_feature_pipeline_persists_problem_features` | All 14 problem features in FeatureEngineer output |
| 9 | `test_clean_frame_unpacks_meta_features_consistently` | Zero NaN after _clean_frame + missing indicators |
| 10 | `test_backfill_fills_missing_features_idempotent` | ALL_BACKFILL_FEATURES covers phase 1-4 |
| 11 | `test_coverage_report_flags_nan_spikes` | 100% NaN critical features flagged |
| 12 | `test_training_does_not_fail_on_partial_schema` | _clean_frame handles missing columns |

### Files changed

| File | Change |
|------|--------|
| `src/core/ml_trainer.py` | FEATURE_DEFAULTS expanded to all features, MISSING_INDICATOR_GROUPS, defensive CLV join, diagnostic logging |
| `src/data/models.py` | `market` column on EventClosingLine (from previous commit) |
| `scripts/backfill_ml_features.py` | Phase 2-4 backfill, coverage before/after reporting |
| `alembic/versions/d5a3c91e2b04_*.py` | Migration: add `market` to `event_closing_lines` |
| `tests/test_ml_feature_pipeline.py` | 6 new tests (12 total including helpers) |

### Migration required

```bash
alembic upgrade head   # adds 'market' column to event_closing_lines
python scripts/backfill_ml_features.py --force   # backfill phase 2-4 features
```

---

## [2026-03-04] Fix: Training Features Pipeline — 6 Critical Features Restored

### Root Cause: `_clean_frame()` meta_features unpacking bug

The 6 critical features (`sentiment_delta`, `injury_delta`, `sharp_implied_prob`,
`sharp_vig`, `form_winrate_l5`, `form_games_l5`) were 100% NaN during training
because `_clean_frame()` skipped JSONB unpacking for columns that already existed
in the DataFrame. Since these features have dedicated DB columns (nullable, often
NULL for old rows), the JSONB values were never extracted.

**Fix:** Changed `_clean_frame()` to fill NaN cells from `meta_features` JSONB
even when dedicated columns exist. Added NaN→default fallback using
`FEATURE_DEFAULTS` for all 6 critical features.

### Live pipeline: enrichment fallback logging

- Sentiment/injury enrichment failures now log explicit reason codes
  (`sentiment_fallback_neutral`, `injury_fallback_neutral`, `form_fallback_neutral`)
- No silent drops — pipeline continues with neutral 0.0 defaults

### Historical backfill script

New `scripts/backfill_ml_features.py`:
- Finds rows with missing critical features in `meta_features`
- Backfills: `sentiment_delta`/`injury_delta` → 0.0, `sharp_implied_prob`/`sharp_vig`
  → derived from odds, `form_winrate_l5`/`form_games_l5` → from TeamMatchStats history
- Supports `--dry-run`, `--limit`, `--sport`, `--force`
- Batchwise (200 rows), idempotent

### Training coverage gate + report

- Pre-training feature coverage report per sport: non-null rate, zero rate,
  variance, unique count
- Hard warnings when critical features are 100% NaN with root-cause hints
- Artifacts: `ML_FEATURE_COVERAGE_REPORT.md` + `artifacts/feature_coverage.json`

### Tests

6 new tests in `tests/test_ml_feature_pipeline.py`:
1. `test_new_bets_persist_required_ml_features`
2. `test_sentiment_failure_sets_neutral_not_nan`
3. `test_injury_failure_sets_neutral_not_nan`
4. `test_clean_frame_unpacks_meta_features_no_nan_for_defaults`
5. `test_backfill_script_fills_missing_features_idempotent`
6. `test_training_report_detects_no_feature_variance`

### Files changed

| File | Change |
|------|--------|
| `src/core/ml_trainer.py` | Fixed `_clean_frame()` JSONB unpacking, explicit `FEATURE_DEFAULTS` for all 6 critical features, `generate_feature_coverage_report()`, `write_feature_coverage_artifacts()`, coverage gate in `auto_train_all_models()` |
| `src/core/live_feed.py` | Enrichment fallback logging with reason codes |
| `scripts/backfill_ml_features.py` | New: comprehensive feature backfill script |
| `tests/test_ml_feature_pipeline.py` | New: 15 tests for the feature pipeline fix |

---

## [2026-03-04] Production Hardening: Feature Pruning, Drawdown Protection, Source Gates

### ML: Permutation-importance feature pruning

Two-pass training in `ml_trainer.py`: after the initial XGBoost fit, compute
permutation importance on the holdout set.  Features with zero or negative
impact on Brier score are pruned, and the model is retrained on the reduced
feature set.  The pruned model is only promoted if it matches or beats the
original.  More aggressive pruning for small datasets (< 3000 samples)
to reduce overfitting.

### ML: NaN spike detection

`_clean_frame()` now logs warnings when any feature exceeds 10% NaN rate
(50%+ triggers a higher-severity warning).  Catches upstream schema changes
or data source outages before they silently degrade the model.

### Financial: Same-game parlay guard hardened

`_compute_correlation_penalty()` in `betting_engine.py` increased from 0.90
to 0.80 per correlated pair.  Logs explicit warnings when same-game parlays
are detected.  The combo optimizer already blocks same-event legs via
`no_same_event=True`; this is a defense-in-depth fallback.

### Financial: Drawdown circuit breaker

New `_check_drawdown()` in `performance_monitor.py`: trips when 7-day PnL
loss exceeds 10% of bankroll.  Catches multi-day losing runs that individual
breakers (daily cap, streak) might miss.  Halves Kelly multiplier and raises
min EV threshold when active.

### Architecture: Data source health gate

New `check_data_source_health()` in `risk_guards.py`: if the Odds API circuit
breaker is open, all new bets are automatically blocked.  Integrated into both
`BettingEngine.make_signal()` and `ExecutionerAgent.execute()`.  The bot won't
bet blind when its primary data source is down.

### Files changed

| File | Change |
|------|--------|
| `src/core/ml_trainer.py` | `_compute_feature_importance()`, `_prune_noisy_features()`, NaN spike detection, `_train_xgboost()` returns 3-tuple |
| `src/core/betting_engine.py` | Same-game parlay penalty 0.90→0.80, data source health gate in `make_signal()` |
| `src/core/risk_guards.py` | `check_data_source_health()` with critical source list |
| `src/core/performance_monitor.py` | `_check_drawdown()`, drawdown in `check_circuit_breakers()` and `get_adjustment_factors()` |
| `src/agents/executioner_agent.py` | Drawdown + data source gates in `execute()` |

---

## [2026-03-04] Signal Explanations + API Health Monitoring

### Signal explanations: historical accuracy from reliability bins

`explain_signal()` in `risk_guards.py` now surfaces the model's historical
accuracy for the relevant confidence bin, e.g. "Historisch 60% Trefferquote
in diesem Bereich". Adds calibration transparency to every signal card.

### Proactive API health alerts

- `source_health.py`: `record_failure()` now pushes a Telegram alert when a
  circuit breaker trips (transition to "open" only — no spam on subsequent failures)
- `core_worker.py`: New `_run_source_health_check()` runs every 5 minutes,
  pushes a summary to the primary chat whenever any source is degraded or open

### Files changed

| File | Change |
|------|--------|
| `src/core/risk_guards.py` | `explain_signal()` includes historical accuracy; new `_get_historical_accuracy()` helper |
| `src/core/source_health.py` | `_push_breaker_alert()` sends Telegram alert on breaker trip |
| `src/bot/core_worker.py` | `_run_source_health_check()` periodic source health push |

---

## [2026-03-04] ML Feature Pipeline Fix — All Audit Issues Resolved

### Critical fix: 29/35 features were training on zeros

Only 6 of 35 XGBoost features were persisted to the `PlacedBet` table.
All 29 Phase 2-4 features were computed at signal time but never stored,
causing the model to train on ~83% zero-variance data.

### All fixes applied

| Priority | Fix | File |
|----------|-----|------|
| P0 | Persist all features via `meta_features` JSONB | `ghost_trading.py` |
| P0 | Unpack `meta_features` in `_clean_frame()` | `ml_trainer.py` |
| P1 | `FEATURE_DEFAULTS` dict with correct neutral values | `ml_trainer.py` |
| P1 | Form tracking from TeamMatchStats (breaks circular dep) | `form_tracker.py` |
| P1 | H2H stats from TeamMatchStats (breaks circular dep) | `h2h_tracker.py` |
| P2 | Auto-compute + persist Phase 4 stats snapshots | `live_feed.py` |

### Files changed

| File | Change |
|------|--------|
| `src/core/ghost_trading.py` | Write full feature dict to `meta_features` JSONB in both `auto_place_virtual_bets()` and `place_virtual_bet()` |
| `src/core/ml_trainer.py` | Unpack `meta_features` JSONB in `_clean_frame()`; add `FEATURE_DEFAULTS` dict for semantically correct neutral values |
| `src/core/form_tracker.py` | Primary source changed to TeamMatchStats; Redis sliding window is now fallback only |
| `src/core/h2h_tracker.py` | Primary source changed to TeamMatchStats; returns 0.5 (neutral) when no data |
| `src/core/live_feed.py` | Auto-compute + persist `EventStatsSnapshot` via `compute_team_snapshot()` when no snapshot exists |
| `ML_FEATURE_AUDIT.md` | Updated with fix status |

---

## [2026-03-04] ML Feature Health Check — Audit Report

See `ML_FEATURE_AUDIT.md` for full audit details.

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
