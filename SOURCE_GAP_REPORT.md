# Source Gap Report

**Generated:** 2026-03-05T18:35:28.133062+00:00

## Global Pipeline Funnel

| Stage | Count |
|-------|-------|
| Fetched sport keys | 11 |
| Raw events fetched | 192 |
| Parsed events | 50 |
| With Tipico odds | 50 |
| With sharp odds | 50 |
| With Tipico + sharp | 50 |
| Dropped: outright filter | 2 |
| Dropped: no bookmaker overlap | 0 |
| Dropped: outside time window | 142 |
| Dropped: invalid market | 0 |
| Dropped: confidence gate | 63 |
| Dropped: negative EV | 0 |
| Dropped: stake = 0 | 0 |
| **Signals generated** | **160** |
| Signals playable (trading) | 13 |
| Signals paper-only (learning) | 147 |
| **Final displayed** | **1** |

### Request Status Codes

| Code | Count |
|------|-------|
| 200 | 11 |

## Per-Sport Breakdown

| Sport | Raw | Parsed | Tipico | Sharp | Both | Signals | Playable |
|-------|-----|--------|--------|-------|------|---------|----------|
| basketball_nba | 10 | 9 | 9 | 9 | 9 | 24 | 3 |
| icehockey_nhl | 15 | 8 | 8 | 8 | 8 | 26 | 10 |
| soccer_epl | 19 | 1 | 1 | 1 | 1 | 10 | 0 |
| soccer_france_ligue_one | 17 | 0 | 0 | 0 | 0 | 0 | 0 |
| soccer_germany_bundesliga | 18 | 0 | 0 | 0 | 0 | 0 | 0 |
| soccer_italy_serie_a | 20 | 0 | 0 | 0 | 0 | 0 | 0 |
| soccer_spain_la_liga | 21 | 0 | 0 | 0 | 0 | 0 | 0 |
| soccer_uefa_champs_league | 8 | 0 | 0 | 0 | 0 | 0 | 0 |
| tennis_atp_indian_wells | 32 | 16 | 16 | 16 | 16 | 51 | 0 |
| tennis_wta_indian_wells | 32 | 16 | 16 | 16 | 16 | 49 | 0 |
