#!/usr/bin/env python3
from src.core.betting_engine import BettingEngine


def main():
    engine = BettingEngine(bankroll=1000)

    s1 = engine.make_signal(
        sport="basketball",
        event_id="nba_001",
        market="h2h",
        selection="Lakers",
        bookmaker_odds=2.10,
        model_probability=0.56,
    )
    s2 = engine.make_signal(
        sport="tennis",
        event_id="atp_044",
        market="h2h",
        selection="Player A",
        bookmaker_odds=1.55,
        model_probability=0.71,
    )

    ranked = engine.rank_value_bets([s1, s2])
    print("Top value bets:")
    for s in ranked:
        print(s.model_dump())

    combo = engine.build_combo(
        [
            {"event_id": "atp_044", "selection": "Player A", "odds": 1.55, "probability": 0.71},
            {"event_id": "atp_045", "selection": "Player B", "odds": 1.42, "probability": 0.74},
            {"event_id": "nba_001", "selection": "Lakers", "odds": 2.10, "probability": 0.56},
        ],
        correlation_penalty=0.92,
        kelly_frac=0.05,
    )
    print("Combo:")
    print(combo.model_dump())


if __name__ == "__main__":
    main()
