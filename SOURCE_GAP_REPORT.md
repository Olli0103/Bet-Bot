# Source Gap Report

**Generated:** 2026-03-07T14:05:51.035059+00:00

## Global Pipeline Funnel

| Stage | Count |
|-------|-------|
| Fetched sport keys | 9 |
| Raw events fetched | 115 |
| Parsed events | 31 |
| With Tipico odds | 31 |
| With sharp odds | 31 |
| With Tipico + sharp | 31 |
| Dropped: outright filter | 2 |
| Dropped: no bookmaker overlap | 0 |
| Dropped: outside time window | 84 |
| Dropped: invalid market | 0 |
| Dropped: confidence gate | 50 |
| Dropped: negative EV | 1 |
| Dropped: stake = 0 | 0 |
| **Signals generated** | **154** |
| Signals playable (trading) | 13 |
| Signals paper-only (learning) | 141 |
| **Final displayed** | **1** |

### Request Status Codes

| Code | Count |
|------|-------|
| 200 | 9 |

## Per-Sport Breakdown

| Sport | Raw | Parsed | Tipico | Sharp | Both | Signals | Playable |
|-------|-----|--------|--------|-------|------|---------|----------|
| basketball_nba | 6 | 6 | 6 | 6 | 6 | 13 | 0 |
| icehockey_nhl | 11 | 11 | 11 | 11 | 11 | 32 | 13 |
| soccer_epl | 18 | 0 | 0 | 0 | 0 | 0 | 0 |
| soccer_france_ligue_one | 16 | 3 | 3 | 3 | 3 | 26 | 0 |
| soccer_germany_bundesliga | 17 | 6 | 6 | 6 | 6 | 46 | 0 |
| soccer_italy_serie_a | 19 | 2 | 2 | 2 | 2 | 13 | 0 |
| soccer_spain_la_liga | 20 | 3 | 3 | 3 | 3 | 24 | 0 |
| soccer_uefa_champs_league | 8 | 0 | 0 | 0 | 0 | 0 | 0 |
