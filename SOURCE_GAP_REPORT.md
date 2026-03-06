# Source Gap Report

**Generated:** 2026-03-06T09:19:20.957959+00:00

## Global Pipeline Funnel

| Stage | Count |
|-------|-------|
| Fetched sport keys | 9 |
| Raw events fetched | 116 |
| Parsed events | 18 |
| With Tipico odds | 18 |
| With sharp odds | 18 |
| With Tipico + sharp | 18 |
| Dropped: outright filter | 2 |
| Dropped: no bookmaker overlap | 0 |
| Dropped: outside time window | 98 |
| Dropped: invalid market | 0 |
| Dropped: confidence gate | 19 |
| Dropped: negative EV | 0 |
| Dropped: stake = 0 | 0 |
| **Signals generated** | **62** |
| Signals playable (trading) | 13 |
| Signals paper-only (learning) | 49 |
| **Final displayed** | **2** |

### Request Status Codes

| Code | Count |
|------|-------|
| 200 | 9 |

## Per-Sport Breakdown

| Sport | Raw | Parsed | Tipico | Sharp | Both | Signals | Playable |
|-------|-----|--------|--------|-------|------|---------|----------|
| basketball_nba | 7 | 7 | 7 | 7 | 7 | 16 | 2 |
| icehockey_nhl | 7 | 7 | 7 | 7 | 7 | 18 | 11 |
| soccer_epl | 18 | 0 | 0 | 0 | 0 | 0 | 0 |
| soccer_france_ligue_one | 17 | 1 | 1 | 1 | 1 | 7 | 0 |
| soccer_germany_bundesliga | 18 | 1 | 1 | 1 | 1 | 4 | 0 |
| soccer_italy_serie_a | 20 | 1 | 1 | 1 | 1 | 7 | 0 |
| soccer_spain_la_liga | 21 | 1 | 1 | 1 | 1 | 10 | 0 |
| soccer_uefa_champs_league | 8 | 0 | 0 | 0 | 0 | 0 | 0 |
