# Bet-Bot Roadmap (v1)

Last updated: 2026-03-06

## Guiding Principle
Stability and risk controls first, then automation depth, then speed.

---

## Phase 0 — Stabilize & Verify (1–2 days)

### Goals
- Ensure current production setup is reliable and recoverable.
- Validate that recent hardening changes behave as intended.

### Work Items
- [ ] Verify DB backup/restore path end-to-end (daily + weekly)
- [ ] Confirm Reddit stealth metrics in steady state (`local_skip`, `ratio304eq`)
- [ ] Pin/verify scheduler timezone behavior (Berlin vs UTC expectations)
- [ ] Capture 24h KPI baseline after latest tuning

### Done Criteria
- [ ] Successful restore test from latest daily backup
- [ ] `ratio304eq` consistently high in low-news windows
- [ ] Morning fetch timing behavior documented and intentional
- [ ] KPI snapshot archived and reviewed

---

## Phase 1 — Interactive DSS Analyst (≈1 week)

### Goals
- Turn Telegram into a true decision-support interface.
- Explain *why* tips were accepted/rejected with evidence.

### Work Items
- [ ] Add intents: `why_rejected`, `explain_tip`, `why_no_tip`
- [ ] Query reject reasons + odds drift + feature snapshot for the event
- [ ] Return compact analyst answer (reason, risk, counterfactual, suggested action)
- [ ] Add audit-friendly trace ID in explanations

### Done Criteria
- [ ] User can ask “Warum abgelehnt?” and get deterministic evidence-backed response
- [ ] Response includes data source + key numeric factors
- [ ] No added instability to existing tip flow

---

## Phase 2 — Model Governance & Promotion (1–2 weeks)

### Goals
- Safe shadow-to-live model lifecycle.
- Promote only with measurable statistical advantage.

### Work Items
- [ ] Build shadow-vs-live scoreboard (Brier, ROI proxy, calibration bins)
- [ ] Add promotion gates (minimum sample size, significance, drawdown guard)
- [ ] Add human-confirmed promotion workflow
- [ ] Persist model metadata/version + deployment timestamp

### Done Criteria
- [ ] Promotion only possible when all gates pass
- [ ] Rollback path tested
- [ ] Promotion decision logged with evidence bundle

---

## Phase 3 — Feature Maturity (2–4 weeks)

### Goals
- Mature new NLP streams into train-ready features.
- Validate contribution before production model impact.

### Work Items
- [ ] Collect enough live rows for `sentiment_delta` and `public_hype_index`
- [ ] Monitor missingness/drift by sport and market
- [ ] Run shadow retrain with new features
- [ ] Evaluate SHAP impact and stability
- [ ] Validate experiment feature `smart_money_divergence`

### Done Criteria
- [ ] Feature quality report complete
- [ ] SHAP/importance confirms signal value (or rejects it)
- [ ] Decision documented: keep/drop each new feature

---

## Phase 4 — Portfolio Intelligence Upgrade (later)

### Goals
- Improve multi-bet risk handling and dependency modeling.

### Work Items
- [ ] Better correlation model for simultaneous bets/combos
- [ ] Stronger exposure constraints by event/league/time bucket
- [ ] Scenario stress tests (odds shock, model miss, news lag)

### Done Criteria
- [ ] Drawdown resilience improves in stress backtests
- [ ] Concentration risk reduced without killing edge

---

## Phase 5 — Halftime Value Engine (In-Play Program für den DE-Markt)

### Goals
- Ausnutzen von Live-Quoten-Ineffizienzen ohne das Risiko des gesetzlichen Live-Delays.
- Fokus auf strukturierte Updates zur Halbzeitpause, statt Millisekunden-Trading.

### Work Items
- [ ] Integration einer Live-Stats-API (z.B. xG, Ballbesitz, Rote Karten) für die 1. Halbzeit
- [ ] Modell-Erweiterung: Bayesianisches Update der Pre-Match XGBoost-Wahrscheinlichkeiten mit den 1. Halbzeit-Stats
- [ ] Aufbau einer Alert-Logik, die EV-Vorteile zur Halbzeitpause (15-Minuten-Zeitfenster) via Deeplink pusht
- [ ] Paper-Only Pilot für reine Halbzeitwetten (1X2 und Über/Unter)
- [ ] Guardrails: max. Alerts pro Match und Skip bei unvollständigen Live-Stats

### Done Criteria
- [ ] Bot ignoriert das laufende Spiel und triggert Berechnungen exakt beim Halbzeitpfiff
- [ ] Stabile Paper-Pilot-Ergebnisse zeigen einen messbaren Edge in den Halftime-Lines
- [ ] Execution ist regelkonform gemäß aktueller Anbieter-/Regel-Lage und stressfrei via Telegram-Deeplink klickbar

---

## Ongoing Monitoring (Always On)

- `at_cap_rate`
- stake p50/p90
- EV outlier count
- reject reason distribution
- sport/market split
- Reddit ingest efficiency (`local_skip`, `ratio304eq`, delta items/run)
- data source health/circuit breaker status

---

## Immediate Next Action
Proceed with **Phase 0 completion** and publish a short baseline report before any further architecture changes.
