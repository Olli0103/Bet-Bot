# Source Gap Report

**Generated:** 2026-03-08T11:24:48.255911+00:00

## Global Pipeline Funnel

| Stage | Count |
|-------|-------|
| Fetched sport keys | 8 |
| Raw events fetched | 108 |
| Parsed events | 108 |
| With Tipico odds | 105 |
| With sharp odds | 108 |
| With Tipico + sharp | 105 |
| Dropped: outright filter | 2 |
| Dropped: no bookmaker overlap | 1 |
| Dropped: outside time window | 0 |
| Dropped: invalid market | 0 |
| Dropped: confidence gate | 231 |
| Dropped: negative EV | 5 |
| Dropped: stake = 0 | 0 |
| **Signals generated** | **524** |
| Signals playable (trading) | 25 |
| Signals paper-only (learning) | 499 |
| **Final displayed** | **1** |

### Request Status Codes

| Code | Count |
|------|-------|
| 200 | 8 |

## Per-Sport Breakdown

| Sport | Raw | Parsed | Tipico | Sharp | Both | Signals | Playable |
|-------|-----|--------|--------|-------|------|---------|----------|
| basketball_nba | 10 | 10 | 10 | 10 | 10 | 20 | 15 |
| icehockey_nhl | 7 | 7 | 7 | 7 | 7 | 14 | 10 |
| soccer_epl | 18 | 18 | 18 | 18 | 18 | 123 | 0 |
| soccer_france_ligue_one | 14 | 14 | 13 | 14 | 13 | 76 | 0 |
| soccer_germany_bundesliga | 11 | 11 | 11 | 11 | 11 | 63 | 0 |
| soccer_italy_serie_a | 16 | 16 | 16 | 16 | 16 | 104 | 0 |
| soccer_spain_la_liga | 16 | 16 | 16 | 16 | 16 | 98 | 0 |
| tennis_atp_indian_wells | 16 | 16 | 14 | 16 | 14 | 26 | 0 |
