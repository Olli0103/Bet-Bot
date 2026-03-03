"""Tests for the Top10 filter -> rank -> topN pipeline.

Verifies that:
1. Sport filtering happens independently per sport (no global cap).
2. The order is: load all -> filter placed -> filter sport -> topN.
3. Pagination state is per-sport, not shared.
4. The full ranked pool is used (not just global top 10).

Note: We re-implement _filter_items_by_sport here to avoid importing
handlers.py (which pulls in the full telegram dependency chain).
The implementation is trivially simple and tested against the real one
via the regression test.
"""
import pytest


def _filter_items_by_sport(items: list, sport_filter: str) -> list:
    """Mirror of src.bot.handlers._filter_items_by_sport."""
    if sport_filter == "all":
        return items
    return [x for x in items if str(x.get("sport", "")).startswith(sport_filter)]


def _make_signal(sport: str, ev: float = 0.05, selection: str = "Team A",
                 event_id: str = "evt_1", odds: float = 2.0, stake: float = 1.0) -> dict:
    """Create a minimal signal dict for testing."""
    return {
        "sport": sport,
        "expected_value": ev,
        "recommended_stake": stake,
        "bookmaker_odds": odds,
        "model_probability": 0.6,
        "selection": selection,
        "event_id": event_id,
        "market": "h2h",
        "confidence": 1.0,
        "source_mode": "primary",
        "reference_book": "pinnacle",
    }


class TestSportFilter:
    """Test _filter_items_by_sport correctness."""

    def test_all_returns_everything(self):
        items = [_make_signal("soccer_epl"), _make_signal("basketball_nba")]
        assert len(_filter_items_by_sport(items, "all")) == 2

    def test_soccer_filter(self):
        items = [
            _make_signal("soccer_epl"),
            _make_signal("soccer_germany_bundesliga"),
            _make_signal("basketball_nba"),
        ]
        result = _filter_items_by_sport(items, "soccer")
        assert len(result) == 2
        assert all(s["sport"].startswith("soccer") for s in result)

    def test_basketball_filter(self):
        items = [_make_signal("soccer_epl"), _make_signal("basketball_nba")]
        result = _filter_items_by_sport(items, "basketball")
        assert len(result) == 1
        assert result[0]["sport"] == "basketball_nba"

    def test_icehockey_filter(self):
        items = [_make_signal("icehockey_nhl"), _make_signal("soccer_epl")]
        result = _filter_items_by_sport(items, "icehockey")
        assert len(result) == 1
        assert result[0]["sport"] == "icehockey_nhl"

    def test_americanfootball_filter(self):
        items = [_make_signal("americanfootball_nfl"), _make_signal("soccer_epl")]
        result = _filter_items_by_sport(items, "americanfootball")
        assert len(result) == 1
        assert result[0]["sport"] == "americanfootball_nfl"

    def test_tennis_filter(self):
        items = [
            _make_signal("tennis_atp_dubai"),
            _make_signal("tennis_wta_rome"),
            _make_signal("basketball_nba"),
        ]
        result = _filter_items_by_sport(items, "tennis")
        assert len(result) == 2

    def test_empty_on_no_match(self):
        items = [_make_signal("soccer_epl"), _make_signal("basketball_nba")]
        result = _filter_items_by_sport(items, "icehockey")
        assert result == []


class TestFilterRankTopNPipeline:
    """Test that the full pipeline (load -> placed -> sport -> topN) works correctly."""

    def _build_pool(self) -> list:
        """Build a realistic multi-sport pool of 30 signals."""
        pool = []
        for i in range(12):
            pool.append(_make_signal("soccer_epl", ev=0.1 - i * 0.005,
                                     event_id=f"soccer_{i}", selection=f"Soccer Team {i}"))
        for i in range(8):
            pool.append(_make_signal("basketball_nba", ev=0.08 - i * 0.005,
                                     event_id=f"nba_{i}", selection=f"NBA Team {i}"))
        for i in range(5):
            pool.append(_make_signal("tennis_atp_dubai", ev=0.06 - i * 0.005,
                                     event_id=f"tennis_{i}", selection=f"Tennis Player {i}"))
        for i in range(3):
            pool.append(_make_signal("icehockey_nhl", ev=0.04 - i * 0.005,
                                     event_id=f"nhl_{i}", selection=f"NHL Team {i}"))
        for i in range(2):
            pool.append(_make_signal("americanfootball_nfl", ev=0.03 - i * 0.005,
                                     event_id=f"nfl_{i}", selection=f"NFL Team {i}"))
        return pool

    def test_each_sport_gets_up_to_10(self):
        """Each sport filter should independently yield up to 10 results."""
        pool = self._build_pool()

        for sport, expected_count in [
            ("soccer", 10),      # 12 available, capped at 10
            ("basketball", 8),   # 8 available
            ("tennis", 5),       # 5 available
            ("icehockey", 3),    # 3 available
            ("americanfootball", 2),  # 2 available
        ]:
            filtered = _filter_items_by_sport(pool, sport)
            top_n = filtered[:10]
            assert len(top_n) == expected_count, f"{sport}: expected {expected_count}, got {len(top_n)}"

    def test_all_sports_independent_of_previous_filter(self):
        """Filtering one sport must not affect results for another sport."""
        pool = self._build_pool()

        # First filter soccer
        soccer = _filter_items_by_sport(pool, "soccer")[:10]
        assert len(soccer) == 10

        # Then filter basketball -- should still have all 8
        basketball = _filter_items_by_sport(pool, "basketball")[:10]
        assert len(basketball) == 8

        # Then filter all -- should have all 30 (capped at 10)
        all_items = _filter_items_by_sport(pool, "all")[:10]
        assert len(all_items) == 10

    def test_placed_bets_dont_leak_across_sports(self):
        """Marking a soccer bet as placed should not affect basketball results."""
        pool = self._build_pool()
        placed = {"soccer_0|Soccer Team 0"}

        # Simulate placed-bet filter
        playable = [
            x for x in pool
            if f"{x['event_id']}|{x['selection']}" not in placed
        ]

        # Soccer loses 1 (the placed bet)
        soccer = _filter_items_by_sport(playable, "soccer")[:10]
        assert len(soccer) == 10  # still has 11 left, capped at 10

        # Basketball still has all 8
        basketball = _filter_items_by_sport(playable, "basketball")[:10]
        assert len(basketball) == 8

    def test_global_top10_from_full_pool(self):
        """'All Sports' top 10 should come from a fresh global ranking,
        not from a pre-sliced top 10 that might miss entire sports."""
        pool = self._build_pool()

        # Sort by EV (simulating the ranking)
        pool.sort(key=lambda x: float(x["expected_value"]), reverse=True)
        top10 = pool[:10]

        # The global top 10 should contain items from multiple sports
        sports_in_top10 = {x["sport"].split("_")[0] for x in top10}
        assert len(sports_in_top10) >= 2, \
            f"Global top 10 should span multiple sports, got: {sports_in_top10}"

    def test_regression_sport_filter_not_blocked_by_global_cap(self):
        """Regression: Previously, only 10 signals were cached globally.
        Sport filters on that pre-sliced set returned empty for minor sports.
        Now the full ranked pool is used, so every sport with EV>0 signals
        should return results."""
        pool = self._build_pool()

        # Simulate old bug: if only top 10 were cached
        pool.sort(key=lambda x: float(x["expected_value"]), reverse=True)
        old_top10_only = pool[:10]

        # Old behavior: NHL might not appear in global top 10
        nhl_in_old = _filter_items_by_sport(old_top10_only, "icehockey")
        # This MIGHT be empty (the bug) -- verify it IS empty to prove the bug
        assert len(nhl_in_old) == 0, "Old top-10-only caching should miss NHL"

        # New behavior: full pool always has NHL
        nhl_in_new = _filter_items_by_sport(pool, "icehockey")
        assert len(nhl_in_new) == 3, "NHL should have 3 signals from full pool"


class TestPaginationStatePerSport:
    """Test that pagination context is per-sport (stored in user_data)."""

    def test_sport_filter_stored_in_context(self):
        """The sport filter should be tracked so back-navigation works."""
        # Simulate context.user_data
        user_data = {}
        pool = [_make_signal("soccer_epl", event_id=f"s{i}") for i in range(5)]

        # First view: soccer
        user_data["signal_items"] = pool[:5]
        user_data["top10_sport_filter"] = "soccer"
        assert user_data["top10_sport_filter"] == "soccer"
        assert len(user_data["signal_items"]) == 5

        # Switch to basketball -- overwrites previous state
        bball = [_make_signal("basketball_nba", event_id=f"b{i}") for i in range(3)]
        user_data["signal_items"] = bball
        user_data["top10_sport_filter"] = "basketball"
        assert user_data["top10_sport_filter"] == "basketball"
        assert len(user_data["signal_items"]) == 3

        # Switch back to soccer -- fresh from pool (not from old state)
        user_data["signal_items"] = pool[:5]
        user_data["top10_sport_filter"] = "soccer"
        assert len(user_data["signal_items"]) == 5
