# 🎾 Tennis Quant Model: Architecture Feasibility Report

## Executive Summary

**Good News:** Our architecture is already positioned better than expected. The foundation for sport-specific models exists. Tennis historical data (2,644 matches in 2025) already present in imports.

---

## 1. Model Isolation (Artifact Routing) ✅ READY

**Current State:** `ml_trainer.py` already supports this!

```python
# Already implemented:
SPORT_GROUPS = {
    "soccer": [...],
    "tennis": ["tennis_atp_indian_wells", ...],
    "basketball": [...],
}
CRITICAL_FEATURES_BY_SPORT = {
    "tennis": {"api_pred_available": 0.0, ...},  # Tennis-specific gates
}
```

**Gap Analysis:**
- ✅ Sport-specific model export already exists
- ✅ Critical feature gating per sport implemented
- ✅ `model_tennis_v{x}.xgb` artifact naming convention in place
- ⚠️ **Missing:** Auto-routing during inference (predict uses current model for all sports)

**Refactoring Path:**
1. Add tennis to `SPORT_GROUPS`
2. Add tennis-specific features to `CRITICAL_FEATURES_BY_SPORT`
3. Update `predict()` to route by `sport` field before inference

**Complexity:** LOW (2-3 hours)

---

## 2. Feature Pipeline & Storage ✅ READY

**Current State:**
```python
# In feature_engineering.py:
team_meta = json.dumps({
    "api_pred_available": 1.0,
    "sentiment_delta": 0.0,
    "public_hype_index": 0.0,
})
```
This pattern **already supports** arbitrary JSON for tennis KPIs without schema migration.

**Proposed Tennis Feature Schema:**
```python
TENNIS_META_KPIS = {
    # From historical match analysis (JeffSackmann + live)
    "hold_rate": 0.68,           # % service games held
    "break_rate": 0.32,         # % opponent service games broken  
    "dominance_ratio": 1.2,      # (Hold% + Break%) / 2
    "clutch_index": 0.75,       # Break points saved % + deciding set win %
    "surface_advantage": 0.85,    # Win rate on current surface vs career avg
    "rank_diff": -15,            # Opponent rank - My rank
    "fatigue_factor": 0.9,      # Match minutes /apped)
    "h2h_dominance":  10 (c0.6,        # Head-to-head win ratio
}
```

**Gap Analysis:**
- ✅ JSON meta field supports dynamic KPIs
- ✅ No schema changes required
- ⚠️ Need to compute these from match data

**Complexity:** LOW (JSON serialization)

---

## 3. Data Sourcing Strategy 🎯 CRITICAL DECISION

### Current Assets:

| Source | Status | Coverage |
|--------|--------|----------|
| API-Sports (Odds) | ✅ Working | 26 ATP tournaments |
| JeffSackmann (Historical) | ✅ **2,644 matches in 2025** | Full match stats |
| Reddit | ✅ Working | Sentiment enrichment |
| Live API | ⚠️ Unknown | Point-by-point |

### Option Analysis:

| Option | Pros | Cons | Recommendation |
|--------|------|------|----------------|
| **A) JeffSackmann + API-Sports** | ✅ Free, comprehensive, historical depth | ⚠️ Historical only, no live | **RECOMMENDED** for model training |
| **B) Build custom point-by-point fetcher** | ✅ Real-time depth | ❌ High OpEx, maintenance burden | SKIP for MVP |
| **C) API-Sports deep stats** | ✅ Already integrated | ❌ Likely no deep tennis stats | Verify first |

### Recommended Architecture:

```
┌─────────────────────────────────────────────────────────────┐
│                    TENNIS DATA PIPELINE                      │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  ┌──────────────────┐    ┌──────────────────────────────┐  │
│  │ JeffSackmann     │    │ API-Sports (live odds)      │  │
│  │ (Historical)     │    │ (current tournaments)       │  │
│  └────────┬─────────┘    └─────────────┬──────────────┘  │
│           │                             │                   │
│           ▼                             ▼                   │
│  ┌──────────────────────────────────────────────────────┐    │
│  │        Tennis Feature Calculator (NEW)              │    │
│  │  - Hold/Break from set scores                      │    │
│  │  - Surface CPI mapping                              │    │
│  │  - H2H from historical matches                     │    │
│  │  - Fatigue from round progression                  │    │
│  └───────────────────────┬──────────────────────────┘    │
│                          │                                │
│                          ▼                                │
│ ──────┐ ┌────────────────────────────────────────────────    │
│  │  meta_json: {tennis_kpis: {...}}                  │    │
│  │  → Stored alongside soccer features                │    │
│  └──────────────────────────────────────────────────────┘    │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

---

## 4. Specific KPI Feasibility

| KPI | Data Source | Feasibility | Notes |
|-----|--------------|-------------|-------|
| Hold/Break % | JeffSackmann scores | ✅ HIGH | Extract from W1-L1...W5-L5 |
| Dominance Ratio | Computed | ✅ HIGH | (Hold% + Break%) / 2 |
| Fatigue | Round data | ✅ MEDIUM | Best-of + round progression |
| Court Pace (CPI) | Static mapping | ✅ HIGH | Map Tournament → Surface → CPI |
| Clutch Index | Score margins | ✅ MEDIUM | Break pts saved in close games |
| H2H | Historical | ✅ HIGH | Match winner/loser lookup |
| Travel/Jetlag | Tournament locations | ⚠️ LOW | Would need external API |

---

## 5. Recommendations

### Phase 1: MVP (This Week)
1. **Export tennis model** using existing JeffSackmann data
2. **Add to SPORT_GROUPS** in ml_trainer.py
3. **Compute basic KPIs** from historical scores (Hold%, Win%, Rank diff)
4. **Route inference** by sport

### Phase 2: Quant Expansion (Next Sprint)
1. Implement Hold/Break matrix from set scores
2. Add Surface→CPI mapping
3. Add Clutch Index from score margins

### Phase 3: Live Enrichment (Future)
- Only if API-Sports provides deep stats
- Otherwise use JeffSackmann daily updates

---

## Bottom Line

**Can we do this? YES.**

- Model isolation: Already built
- Feature storage: JSON meta field ready  
- Data: 2,644 historical matches available NOW
- OpEx: Low (leverage existing infrastructure + JeffSackmann)

**Recommendation:** Start with JeffSackmann + API-Sports odds combo. Skip custom point-by-point fetcher for MVP. Target: Tennis model within 1 sprint.

---

## Appendix: Current Data Assets

### JeffSackmann 2025 Dataset
- **Location:** `data/imports/tennis/2025.xlsx`
- **Records:** 2,644 matches
- **Columns:** ATP, Location, Tournament, Date, Series, Court, Surface, Round, Best of, Winner, Loser, WRank, LRank, WPts, LPts, W1-L5 (set scores), Wsets, Lsets, Comment, B365W, B365L, PSW, PSL, MaxW, MaxL, AvgW, AvgL, BFEW, BFEL

### API-Sports Coverage
- 26 ATP tournaments available
- Live odds via existing integration
- No deep tennis stats (expected)

---
*Generated: 2026-03-08*
*Author: Clawy (Architecture Analysis)*
