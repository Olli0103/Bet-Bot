#!/usr/bin/env python3
"""Run walk-forward backtest and strategy comparison.

Usage:
    python scripts/run_backtest.py                   # single backtest with defaults
    python scripts/run_backtest.py --compare          # compare multiple strategies
    python scripts/run_backtest.py --min-ev 0.01      # custom min EV threshold
    python scripts/run_backtest.py --kelly 0.15       # custom Kelly fraction
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.core.backtester import Backtester, BacktestConfig


def main():
    parser = argparse.ArgumentParser(description="Run betting strategy backtest")
    parser.add_argument("--compare", action="store_true", help="Compare multiple strategies")
    parser.add_argument("--bankroll", type=float, default=1000.0, help="Initial bankroll (default: 1000)")
    parser.add_argument("--kelly", type=float, default=0.2, help="Kelly fraction (default: 0.2)")
    parser.add_argument("--tax", type=float, default=0.05, help="Tax rate (default: 0.05)")
    parser.add_argument("--min-ev", type=float, default=0.0, help="Minimum EV threshold (default: 0.0)")
    parser.add_argument("--min-samples", type=int, default=50, help="Minimum samples to run (default: 50)")
    args = parser.parse_args()

    if args.compare:
        print("Running strategy comparison...\n")
        bt = Backtester()
        configs = Backtester.default_comparison_configs()
        results = bt.compare_strategies(configs)

        for i, result in enumerate(results):
            print(f"\n--- Strategy {i + 1} (Kelly={result.config.kelly_frac}, "
                  f"min_EV={result.config.min_ev}, tax={result.config.tax_rate}) ---")
            print(Backtester.format_report(result))

        # Summary table
        print("\n" + "=" * 70)
        print("STRATEGY COMPARISON SUMMARY (sorted by ROI)")
        print("=" * 70)
        print(f"{'Kelly':>6} {'MinEV':>7} {'Tax':>5} {'Bets':>6} {'ROI':>8} {'PnL':>10} {'DD':>8} {'Sharpe':>8}")
        print("-" * 70)
        for r in results:
            print(f"{r.config.kelly_frac:>6.2f} {r.config.min_ev:>7.3f} {r.config.tax_rate:>5.2f} "
                  f"{r.total_bets:>6} {r.roi:>7.2%} {r.total_pnl:>+9.2f} "
                  f"{r.max_drawdown:>7.2%} {r.sharpe_ratio:>8.2f}")
    else:
        config = BacktestConfig(
            initial_bankroll=args.bankroll,
            kelly_frac=args.kelly,
            tax_rate=args.tax,
            min_ev=args.min_ev,
            min_train_samples=args.min_samples,
        )
        bt = Backtester(config=config)
        print("Loading historical data...")
        result = bt.run()
        print(Backtester.format_report(result))


if __name__ == "__main__":
    main()
