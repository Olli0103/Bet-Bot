#!/usr/bin/env python3
from src.core.autograding import settle_bets_with_results


if __name__ == "__main__":
    # Placeholder input; wire real API result mapping in next iteration.
    results = {
        # "nba_001": {"won_selection": "Lakers"}
    }
    n = settle_bets_with_results(results)
    print(f"settled={n}")
