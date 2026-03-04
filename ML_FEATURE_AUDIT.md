# ML Feature Health Check — Audit Report

**Datum:** 2026-03-04 (Audit) | 2026-03-04 (Fixes applied)
**Scope:** Alle 35 XGBoost-Features + 1 Soccer-Extra (`poisson_true_prob`)
**Methode:** Statische Code-Analyse des gesamten Feature-Datenpipeline-Pfads

---

## Executive Summary

**Kritischer Befund (BEHOBEN):** Nur **6 von 35 Features** wurden in der PlacedBet-Tabelle persistiert.
Die restlichen 29 Features wurden zur Signalzeit korrekt berechnet, aber **nie in die Datenbank
geschrieben**.

### Fixes Applied

| Fix | Datei | Status |
|-----|-------|--------|
| P0: Alle Features via `meta_features` JSONB persistieren | `ghost_trading.py` | **DONE** |
| P0: `meta_features` JSONB beim Laden auspacken | `ml_trainer.py` | **DONE** |
| P1: `FEATURE_DEFAULTS` statt pauschal 0.0 | `ml_trainer.py` | **DONE** |
| P1: Form-Tracking aus TeamMatchStats statt PlacedBet | `form_tracker.py` | **DONE** |
| P1: H2H aus TeamMatchStats statt PlacedBet | `h2h_tracker.py` | **DONE** |
| P2: Phase 4 Stats Pipeline anbinden (compute+save) | `live_feed.py` | **DONE** |

| Kategorie | Anzahl | Status |
|-----------|--------|--------|
| Korrekt persistiert & nutzbar | 4 | `sharp_implied_prob`, `sharp_vig`, `sentiment_delta`, `injury_delta` |
| Persistiert aber problematisch | 2 | `form_winrate_l5` (zirkulär → **FIXED**), `form_games_l5` (zirkulär → **FIXED**) |
| Nie persistiert → trainiert auf 0.0 | 29 | Alle Phase 2-4 Features → **FIXED via meta_features JSONB** |

---

## 1. Datenverfügbarkeit je Feature

### Legende

- **DB-Spalte:** Existiert als Spalte in `PlacedBet` (`src/data/models.py`)
- **Geschrieben von:** Wo wird der Wert beim Bet-Placement gesetzt
- **Non-Null-Rate (geschätzt):** Basierend auf Code-Analyse der Schreibpfade
- **Varianz im Training:** Erwartete Varianz wenn `_clean_frame()` den Wert erzeugt

### Phase 1: Core Features (6 Stück — in DB)

| Feature | DB-Spalte | Geschrieben | Non-Null | Varianz | Status |
|---------|-----------|-------------|----------|---------|--------|
| `sharp_implied_prob` | `models.py:49` | `ghost_trading.py:46` | ~95% | ✅ Gut | **KEEP** |
| `sharp_vig` | `models.py:50` | `ghost_trading.py:47` | ~95% | ✅ Gut | **KEEP** |
| `sentiment_delta` | `models.py:51` | `ghost_trading.py:48` | ~70% | ⚠️ Oft 0 | **FIX** |
| `injury_delta` | `models.py:52` | `ghost_trading.py:49` | ~70% | ⚠️ Oft 0 | **FIX** |
| `form_winrate_l5` | `models.py:53` | `ghost_trading.py:42` | ~60% | ⚠️ Zirkulär | **FIX** |
| `form_games_l5` | `models.py:54` | `ghost_trading.py:43` | ~60% | ⚠️ Zirkulär | **FIX** |

### Phase 2: Market & Enrichment Features (15 Stück — NICHT in DB)

| Feature | DB-Spalte | Berechnet in | Training-Wert | Status |
|---------|-----------|--------------|---------------|--------|
| `elo_diff` | ❌ | `feature_engineering.py:119` | 0.0 | **FIX** |
| `elo_expected` | ❌ | `feature_engineering.py:120` | 0.0 (≠ 0.5 Default!) | **FIX** |
| `h2h_home_winrate` | ❌ | `feature_engineering.py:121` | 0.0 (≠ 0.5 Default!) | **FIX** |
| `home_advantage` | ❌ | `feature_engineering.py:122` | 0.0 | **FIX** |
| `weather_rain` | ❌ | `feature_engineering.py:123` | 0.0 | **FIX** |
| `weather_wind_high` | ❌ | `feature_engineering.py:124` | 0.0 | **FIX** |
| `home_volatility` | ❌ | `feature_engineering.py:125` | 0.0 | **FIX** |
| `away_volatility` | ❌ | `feature_engineering.py:126` | 0.0 | **FIX** |
| `is_steam_move` | ❌ | `feature_engineering.py:127` | 0.0 | **FIX** |
| `line_staleness` | ❌ | `feature_engineering.py:128` | 0.0 | **FIX** |
| `injury_news_delta` | ❌ | `feature_engineering.py:129` | 0.0 | **FIX** |
| `time_to_kickoff_hours` | ❌ | `feature_engineering.py:130` | 0.0 (≠ 24h Default!) | **FIX** |
| `public_bias` | ❌ | `feature_engineering.py:131` | 0.0 | **FIX** |
| `market_momentum` | ❌ | `feature_engineering.py:132` | 0.0 | **FIX** |
| `line_velocity` | ❌ | `feature_engineering.py:133` | 0.0 | **FIX** |

### Phase 4: Stats-Based Features (14 Stück — NICHT in DB)

| Feature | DB-Spalte | Berechnet in | Training-Wert | Status |
|---------|-----------|--------------|---------------|--------|
| `team_attack_strength` | ❌ | `feature_engineering.py:135` | 0.0 (≠ 1.0 Default!) | **FIX** |
| `team_defense_strength` | ❌ | `feature_engineering.py:136` | 0.0 (≠ 1.0 Default!) | **FIX** |
| `opp_attack_strength` | ❌ | `feature_engineering.py:137` | 0.0 (≠ 1.0 Default!) | **FIX** |
| `opp_defense_strength` | ❌ | `feature_engineering.py:138` | 0.0 (≠ 1.0 Default!) | **FIX** |
| `expected_total_proxy` | ❌ | `feature_engineering.py:139` | 0.0 | **FIX** |
| `form_trend_slope` | ❌ | `feature_engineering.py:140` | 0.0 | **FIX** |
| `rest_fatigue_score` | ❌ | `feature_engineering.py:141` | 0.0 | **FIX** |
| `schedule_congestion` | ❌ | `feature_engineering.py:142` | 0.0 | **FIX** |
| `over25_rate` | ❌ | `feature_engineering.py:143` | 0.0 | **FIX** |
| `btts_rate` | ❌ | `feature_engineering.py:144` | 0.0 | **FIX** |
| `home_away_split_delta` | ❌ | `feature_engineering.py:145` | 0.0 | **FIX** |
| `league_position_delta` | ❌ | `feature_engineering.py:146` | 0.0 | **FIX** |
| `goals_scored_avg` | ❌ | `feature_engineering.py:147` | 0.0 | **FIX** |
| `goals_conceded_avg` | ❌ | `feature_engineering.py:148` | 0.0 | **FIX** |

### Soccer Extra (1 Stück — NICHT in DB)

| Feature | DB-Spalte | Berechnet in | Training-Wert | Status |
|---------|-----------|--------------|---------------|--------|
| `poisson_true_prob` | ❌ | `feature_engineering.py:151` | 0.0 | **FIX** |

---

## 2. Preprocessing-Kette (fillna / Defaults / Coercion)

### Schritt 1: `_clean_frame()` — Fehlende Spalten werden erzeugt

**Datei:** `src/core/ml_trainer.py:142-216`

```python
# Zeile 144-146: Jede fehlende Spalte wird als 0.0 erzeugt
for c in feature_list:
    if c not in out.columns:
        out[c] = 0.0
```

**Problem:** 29 Features sind nicht in der DB → werden als konstant 0.0 erzeugt.
XGBoost erkennt zero-variance Spalten und ignoriert sie via `_get_active_features()` (Zeile 219-222).
Das Modell trainiert effektiv nur auf den ~6 verbleibenden Features mit Varianz.

### Schritt 2: `apply(pd.to_numeric, errors="coerce")` — Zeile 152

Konvertiert alle Feature-Spalten zu numerisch. Nicht-numerische Werte → NaN.
XGBoost kann NaN nativ verarbeiten, aber die `0.0`-Initialisierung aus Schritt 1
verhindert, dass XGBoost seine optimale Missing-Value-Behandlung nutzen kann.

### Schritt 3: `sharp_implied_prob` Ableitung — Zeile 157-167

Wenn `sharp_implied_prob` nahezu null Varianz hat UND `odds` existiert, wird
`1/odds` (vig-bereinigt) als Ersatz berechnet. Sinnvoller Fallback für historische
Imports ohne Sharp-Odds-Daten.

### Schritt 4: Outlier Clamping — Zeile 170-215

Wertebereich-Clipping für alle Features. Korrekt implementiert, aber wirkungslos
für die 29 Features die ohnehin konstant 0.0 sind.

### Schritt 5: `_get_active_features()` — Zeile 219-222

```python
def _get_active_features(X, feature_list):
    variances = X[feature_list].var(axis=0)
    return [f for f in feature_list if float(variances.get(f, 0.0)) > EPS]
```

Filtert zero-variance Features aus. Bei 29 Features mit konstantem 0.0 bleiben
nur die 4-6 Features mit echten Werten übrig. **Das ist der Grund für die
near-zero Importance der meisten Features.**

---

## 3. Leakage / Timing Check

| Prüfpunkt | Ergebnis | Details |
|------------|----------|---------|
| CLV als Feature | ✅ Entfernt | `clv` wurde in vorherigem Commit aus `FEATURES` entfernt (data leakage) |
| `sharp_closing_odds/prob` | ✅ Sicher | Nicht in `FEATURES`, nur in `PlacedBet` für CLV-Audit |
| `odds_close` als Feature | ✅ Sicher | Nicht in `FEATURES` |
| `pnl` als Feature | ✅ Sicher | Nicht in `FEATURES` |
| `status` als Feature | ✅ Sicher | Nur als Target (`y`) verwendet |
| `form_winrate_l5` | ⚠️ Zirkulär | Nur aus PlacedBet-Ergebnissen berechnet (Redis-Cache). Neue Teams → immer 0.5. `form_tracker.py` aktualisiert nur bei Bet-Grading, nicht aus `TeamMatchStats`. |
| `h2h_home_winrate` | ⚠️ Zirkulär | `h2h_tracker.py` queries PlacedBet selbst. Neue Paarungen → immer 0.5. |
| Phase 4 Stats | ✅ Sicher (im Design) | `EventStatsSnapshot` nutzt nur Pre-Match-Daten. Aber: **nicht angebunden**. |
| Temporal split | ✅ Korrekt | `TimeSeriesSplit` + 20% Holdout (chronologisch). Zeile 241-245. |

---

## 4. Feature Importance (Modellbeitrag)

Da 29 von 35 Features konstant 0.0 sind, meldet `_validate_model()` (Zeile 356-363)
für diese Features `near-zero importance`. Das ist **kein Fehler im Modell**, sondern
die korrekte Reaktion auf fehlende Trainingsdaten.

### Erwartetes Importance-Ranking (nur nicht-null Features):

| Rang | Feature | Begründung |
|------|---------|------------|
| 1 | `sharp_implied_prob` | Stärkstes Signal — vig-bereinigte Sharp-Wahrscheinlichkeit |
| 2 | `sharp_vig` | Markt-Overround korreliert mit Liquidität/Unsicherheit |
| 3 | `sentiment_delta` | Sentiment-Differenz (wenn ≠ 0) |
| 4 | `injury_delta` | Verletzungsdifferenz (wenn ≠ 0) |
| 5-6 | `form_winrate_l5`, `form_games_l5` | Zirkulär, aber mit etwas Varianz |
| 7-35 | Alle anderen | 0.0 (zero variance → von `_get_active_features` gefiltert) |

---

## 5. Root-Cause-Klassifikation

| Code | Ursache | Betroffene Features |
|------|---------|---------------------|
| **A** | Spalte fehlt in DB → `_clean_frame` füllt mit 0.0 | `elo_diff`, `elo_expected`, `h2h_home_winrate`, `home_advantage`, `weather_rain`, `weather_wind_high`, `home_volatility`, `away_volatility`, `is_steam_move`, `line_staleness`, `injury_news_delta`, `time_to_kickoff_hours`, `public_bias`, `market_momentum`, `line_velocity` |
| **A** | Phase 4 Stats nie persistiert | `team_attack_strength`, `team_defense_strength`, `opp_attack_strength`, `opp_defense_strength`, `expected_total_proxy`, `form_trend_slope`, `rest_fatigue_score`, `schedule_congestion`, `over25_rate`, `btts_rate`, `home_away_split_delta`, `league_position_delta`, `goals_scored_avg`, `goals_conceded_avg`, `poisson_true_prob` |
| **B** | Meist 0 weil Enrichment oft keine Daten liefert | `sentiment_delta`, `injury_delta` |
| **C** | Zirkuläre Berechnung (aus PlacedBet abgeleitet) | `form_winrate_l5`, `form_games_l5`, `h2h_home_winrate` |
| **D** | Redundanz: `elo_expected` ≈ f(`elo_diff`) | `elo_expected` |
| **E** | Noch unklar (kein Training möglich) | Alle 29 Features (erst nach Fix von A bewertbar) |

---

## 6. Aktionsplan: KEEP / FIX / DROP

### P0 (Kritisch): Feature-Persistierung reparieren

**Problem:** `ghost_trading.py` schreibt nur 6 Features in die DB.
**Lösung:** Alle Features via `meta_features` JSONB-Spalte persistieren.

Die Spalte `meta_features` existiert bereits in `PlacedBet` (`models.py:58`)
aber wird **nie beschrieben**.

#### Vorgeschlagene Code-Änderung in `ghost_trading.py`:

```python
# ghost_trading.py — auto_place_virtual_bets()
# VORHER (Zeile 33-50): Nur 6 Features einzeln geschrieben
# NACHHER: Alle Features via meta_features JSONB

new_bet = PlacedBet(
    event_id=str(sig.event_id),
    sport=str(sig.sport),
    market=str(sig.market),
    selection=str(sig.selection),
    odds=float(sig.bookmaker_odds),
    odds_open=float(sig.bookmaker_odds),
    odds_close=float(sig.bookmaker_odds),
    # Einzelne Spalten bleiben für Abwärtskompatibilität
    sharp_implied_prob=float(feat.get("sharp_implied_prob", 0.0)),
    sharp_vig=float(feat.get("sharp_vig", 0.0)),
    sentiment_delta=float(feat.get("sentiment_delta", 0.0)),
    injury_delta=float(feat.get("injury_delta", 0.0)),
    form_winrate_l5=float(feat.get("form_winrate_l5", 0.5)),
    form_games_l5=int(feat.get("form_games_l5", 0)),
    stake=float(sig.recommended_stake),
    status="open",
    # NEU: Alle Features im JSONB-Blob
    meta_features=feat,  # Komplettes Feature-Dict
)
```

#### Vorgeschlagene Code-Änderung in `ml_trainer.py`:

```python
# ml_trainer.py — _clean_frame()
# NACHHER: meta_features JSONB auspacken

def _clean_frame(df, feature_list):
    out = df.copy()
    # Schritt 0: meta_features JSONB auspacken
    if "meta_features" in out.columns:
        meta_df = pd.json_normalize(out["meta_features"].dropna())
        for col in meta_df.columns:
            if col in feature_list and col not in out.columns:
                out.loc[meta_df.index, col] = meta_df[col]
    # Restliche Logik bleibt gleich
    for c in feature_list:
        if c not in out.columns:
            out[c] = 0.0
    ...
```

### P1 (Hoch): Zirkuläre Features entkoppeln

| Feature | Problem | Lösung |
|---------|---------|--------|
| `form_winrate_l5` | Nur aus Bet-Ergebnissen (Redis) | Aus `TeamMatchStats.result` berechnen |
| `form_games_l5` | Ebenfalls Redis-only | Aus `TeamMatchStats` COUNT berechnen |
| `h2h_home_winrate` | Queries PlacedBet | Aus `TeamMatchStats` berechnen (gleiche Teams, is_home) |

### P2 (Mittel): Feature-Defaults korrigieren

Einige Features haben semantisch falsche 0.0-Defaults wenn sie fehlen:

| Feature | Aktueller Default (in `_clean_frame`) | Korrekter Default | Begründung |
|---------|--------------------------------------|-------------------|------------|
| `elo_expected` | 0.0 | 0.5 | 50% = kein Elo-Vorteil |
| `h2h_home_winrate` | 0.0 | 0.5 | 50% = keine H2H-Daten |
| `home_advantage` | 0.0 | 0.5 | Unklar ob Heim oder Auswärts |
| `time_to_kickoff_hours` | 0.0 | 24.0 | 0 = Spiel läuft schon (falsch!) |
| `team_attack_strength` | 0.0 | 1.0 | 1.0 = Durchschnitt |
| `team_defense_strength` | 0.0 | 1.0 | 1.0 = Durchschnitt |
| `opp_attack_strength` | 0.0 | 1.0 | 1.0 = Durchschnitt |
| `opp_defense_strength` | 0.0 | 1.0 | 1.0 = Durchschnitt |
| `form_winrate_l5` | 0.0 | 0.5 | 50% = keine Form-Daten |

**Vorschlag:** In `_clean_frame()` einen `FEATURE_DEFAULTS`-Dict verwenden statt pauschal `0.0`:

```python
FEATURE_DEFAULTS = {
    "elo_expected": 0.5,
    "h2h_home_winrate": 0.5,
    "home_advantage": 0.5,
    "time_to_kickoff_hours": 24.0,
    "team_attack_strength": 1.0,
    "team_defense_strength": 1.0,
    "opp_attack_strength": 1.0,
    "opp_defense_strength": 1.0,
    "form_winrate_l5": 0.5,
}

def _clean_frame(df, feature_list):
    out = df.copy()
    for c in feature_list:
        if c not in out.columns:
            out[c] = FEATURE_DEFAULTS.get(c, 0.0)
    ...
```

### P3 (Niedrig): Zukünftige Evaluierung

| Feature | Aktion | Begründung |
|---------|--------|------------|
| `weather_rain` | KEEP (nach P0-Fix) | Regen korreliert mit Under-Goals im Fußball |
| `weather_wind_high` | KEEP (nach P0-Fix) | Wind beeinflusst Totals |
| `elo_expected` | Evaluieren ob DROP nach P0 | Möglicherweise redundant zu `elo_diff` |
| `poisson_true_prob` | KEEP (Soccer) | Nur sinnvoll wenn Poisson-Modell Daten hat |
| `league_position_delta` | KEEP (nach P0-Fix) | Nie automatisch befüllt — braucht Datenquelle |

---

## 7. Zusammenfassung & Priorisierung

| Priorität | Aufgabe | Aufwand | Impact |
|-----------|---------|---------|--------|
| **P0** | `ghost_trading.py`: Alle Features via `meta_features` JSONB persistieren | 1-2h | **Kritisch** — Modell trainiert auf 83% Nullen |
| **P0** | `ml_trainer.py`: `meta_features` JSONB beim Laden auspacken | 1h | **Kritisch** — Voraussetzung für Feature-Nutzung |
| **P1** | `_clean_frame()`: `FEATURE_DEFAULTS`-Dict statt pauschal 0.0 | 30min | **Hoch** — Verhindert semantisch falsche Defaults |
| **P1** | `form_tracker.py`: Form aus `TeamMatchStats` statt PlacedBet | 2-3h | **Hoch** — Entfernt zirkuläre Abhängigkeit |
| **P1** | `h2h_tracker.py`: H2H aus `TeamMatchStats` statt PlacedBet | 1-2h | **Hoch** — Entfernt zirkuläre Abhängigkeit |
| **P2** | Phase 4 Stats Pipeline an Feature-Dict anbinden (`live_feed.py`) | 3-4h | **Mittel** — Stats-Features werden berechnet aber nicht weitergereicht |
| **P3** | Feature-Importance nach P0-Fix neu evaluieren | 1h | **Niedrig** — Erst nach P0 sinnvoll |

---

## Sportmodell-spezifische Übersicht

### General Model

| Feature-Kategorie | Verfügbar | Trainiert auf echten Werten |
|--------------------|-----------|-----------------------------|
| Sharp-Odds (2) | ✅ | ✅ |
| Sentiment/Injury (2) | ✅ | ⚠️ Oft 0 |
| Form (2) | ✅ | ⚠️ Zirkulär |
| Phase 2 Market (9) | ✅ zur Signalzeit | ❌ Training = 0.0 |
| Phase 2 Enrichment (6) | ✅ zur Signalzeit | ❌ Training = 0.0 |
| Phase 4 Stats (14) | ⚠️ Pipeline unvollständig | ❌ Training = 0.0 |

### Soccer Model

Wie General, plus `poisson_true_prob` (ebenfalls nicht persistiert).
`EventStatsSnapshot` hat alle Stats-Spalten, aber kein FK von `PlacedBet` → Snapshot.

### Basketball / Tennis / NFL / NHL Models

Nur `FEATURES` (kein Extra). Gleiche Persistierungs-Lücke wie General.
Zusätzlich: weniger Trainingsdaten → oft `< min_samples` → Modell wird gar nicht trainiert.

---

## Architektur-Diagramm: Feature-Datenfluss

```
Signal-Zeit (korrekt):                    Training-Zeit (defekt):
═══════════════════════                   ═══════════════════════

OddsAPI ──┐                               PlacedBet (DB)
Enrichment┤                                    │
EloSystem ├─► build_core_features() ──┐        │  pd.read_sql()
Volatility│      (35 Features)        │        ▼
Stats     ┘                           │   _clean_frame()
                                      │        │
                                      ▼        │ 6 Spalten → echte Werte
                              BettingEngine    │ 29 Spalten fehlen → 0.0
                                      │        │
                                      ▼        ▼
                              ghost_trading   _get_active_features()
                                      │        │
                              Schreibt nur    Filtert zero-variance
                              6 Features!     → nur 4-6 Features aktiv
                                      │        │
                                      ▼        ▼
                              PlacedBet (DB)  XGBoost trainiert auf
                              meta_features   4-6 statt 35 Features
                              = NULL (leer!)
```

---

*Erstellt am 2026-03-04 durch automatisierten Code-Audit.*
