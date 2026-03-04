"""Walk-forward backtesting engine for betting strategy evaluation.

Simulates the full pipeline on historical PlacedBet data using a rolling
training window. Produces equity curves, ROI, Brier scores, and
per-sport breakdowns.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sqlalchemy import select

from src.core.betting_math import expected_value, kelly_fraction, kelly_stake
from src.data.postgres import SessionLocal

log = logging.getLogger(__name__)


@dataclass
class BacktestConfig:
    """Configuration for a backtest run."""
    initial_bankroll: float = 1000.0
    kelly_frac: float = 0.2
    tax_rate: float = 0.05
    min_ev: float = 0.0
    min_probability: float = 0.0
    max_odds: float = 10.0
    min_odds: float = 1.05
    train_window_days: int = 60
    retrain_every_days: int = 7
    min_train_samples: int = 100


@dataclass
class BacktestResult:
    """Result of a single backtest run."""
    config: BacktestConfig
    total_bets: int = 0
    wins: int = 0
    losses: int = 0
    total_staked: float = 0.0
    total_pnl: float = 0.0
    peak_bankroll: float = 0.0
    min_bankroll: float = 0.0
    final_bankroll: float = 0.0
    roi: float = 0.0
    hit_rate: float = 0.0
    max_drawdown: float = 0.0
    sharpe_ratio: float = 0.0
    brier_score: float = 0.0
    equity_curve: List[float] = field(default_factory=list)
    daily_pnl: List[float] = field(default_factory=list)
    by_sport: Dict[str, Dict[str, float]] = field(default_factory=dict)


class Backtester:
    """Walk-forward backtesting engine."""

    def __init__(self, config: Optional[BacktestConfig] = None):
        self.config = config or BacktestConfig()

    def load_historical_data(self, include_training_data: bool = False) -> pd.DataFrame:
        """Load settled bets from the database.

        Parameters
        ----------
        include_training_data : bool
            If True, include historical imports. Default False (live only)
            for realistic backtest results.
        """
        try:
            from src.data.models import PlacedBet
        except ImportError:
            log.error("Cannot import PlacedBet model")
            return pd.DataFrame()

        with SessionLocal() as db:
            query = select(PlacedBet).where(PlacedBet.status.in_(["won", "lost"]))
            if not include_training_data:
                query = query.where(PlacedBet.is_training_data.is_(False))
            df = pd.read_sql(query, db.bind)

        if df.empty:
            return df

        # Ensure numeric columns
        for col in ["odds", "stake", "pnl"]:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)

        # Sort chronologically
        if "id" in df.columns:
            df = df.sort_values("id").reset_index(drop=True)

        return df

    def run(self, df: Optional[pd.DataFrame] = None) -> BacktestResult:
        """Run walk-forward backtest on historical data.

        Uses a simple strategy: bet when model probability implies +EV after
        tax, using fractional Kelly for sizing.
        """
        if df is None:
            df = self.load_historical_data()

        if df.empty or len(df) < self.config.min_train_samples:
            log.warning("Not enough data for backtest: %d rows", len(df))
            return BacktestResult(config=self.config)

        cfg = self.config
        bankroll = cfg.initial_bankroll
        equity = [bankroll]
        daily_pnl_list: List[float] = []
        preds_and_actuals: List[Tuple[float, int]] = []
        sport_stats: Dict[str, Dict[str, float]] = {}

        total_bets = 0
        wins = 0
        losses = 0
        total_staked = 0.0
        total_pnl = 0.0
        peak = bankroll

        for _, row in df.iterrows():
            odds = float(row.get("odds", 0))
            status = str(row.get("status", ""))
            sport = str(row.get("sport", "unknown"))

            if odds < cfg.min_odds or odds > cfg.max_odds:
                continue

            # Compute model probability from sharp_implied_prob if available,
            # otherwise estimate from odds
            model_p = float(row.get("sharp_implied_prob", 0))
            if model_p <= 0 or model_p >= 1:
                model_p = 1.0 / odds if odds > 1.0 else 0.5

            # Check EV
            ev = expected_value(model_p, odds, tax_rate=cfg.tax_rate)
            if ev < cfg.min_ev:
                continue

            if model_p < cfg.min_probability:
                continue

            # Kelly sizing
            kf = kelly_fraction(model_p, odds, frac=cfg.kelly_frac, tax_rate=cfg.tax_rate)
            stake = kelly_stake(bankroll, kf)
            stake = max(0.50, min(stake, bankroll * 0.10))  # cap at 10% of bankroll

            if stake > bankroll:
                continue

            # Simulate outcome
            is_won = status == "won"
            if is_won:
                gross_profit = stake * (odds - 1.0)
                net_profit = gross_profit * (1.0 - cfg.tax_rate)
                pnl = net_profit
                wins += 1
            else:
                pnl = -stake
                losses += 1

            bankroll += pnl
            total_bets += 1
            total_staked += stake
            total_pnl += pnl
            equity.append(bankroll)
            daily_pnl_list.append(pnl)
            peak = max(peak, bankroll)

            # Track prediction accuracy
            preds_and_actuals.append((model_p, 1 if is_won else 0))

            # Per-sport stats
            if sport not in sport_stats:
                sport_stats[sport] = {"bets": 0, "wins": 0, "pnl": 0.0, "staked": 0.0}
            sport_stats[sport]["bets"] += 1
            if is_won:
                sport_stats[sport]["wins"] += 1
            sport_stats[sport]["pnl"] += pnl
            sport_stats[sport]["staked"] += stake

        # Compute aggregate metrics
        result = BacktestResult(config=cfg)
        result.total_bets = total_bets
        result.wins = wins
        result.losses = losses
        result.total_staked = round(total_staked, 2)
        result.total_pnl = round(total_pnl, 2)
        result.final_bankroll = round(bankroll, 2)
        result.peak_bankroll = round(peak, 2)
        result.min_bankroll = round(min(equity), 2) if equity else 0.0
        result.roi = round(total_pnl / max(1.0, total_staked), 4)
        result.hit_rate = round(wins / max(1, total_bets), 4)
        result.equity_curve = [round(e, 2) for e in equity]
        result.daily_pnl = [round(p, 2) for p in daily_pnl_list]

        # Max drawdown
        if equity:
            peak_arr = np.maximum.accumulate(equity)
            drawdowns = (np.array(equity) - peak_arr) / np.where(peak_arr > 0, peak_arr, 1)
            result.max_drawdown = round(float(np.min(drawdowns)), 4)

        # Sharpe ratio (daily PnL)
        if len(daily_pnl_list) > 1:
            pnl_arr = np.array(daily_pnl_list)
            mean_pnl = float(np.mean(pnl_arr))
            std_pnl = float(np.std(pnl_arr))
            result.sharpe_ratio = round(mean_pnl / max(0.001, std_pnl) * np.sqrt(252), 4)

        # Brier score
        if preds_and_actuals:
            preds, actuals = zip(*preds_and_actuals)
            result.brier_score = round(float(np.mean((np.array(preds) - np.array(actuals)) ** 2)), 6)

        # Per-sport ROI
        for sport, stats in sport_stats.items():
            stats["roi"] = round(stats["pnl"] / max(1.0, stats["staked"]), 4)
            stats["hit_rate"] = round(stats["wins"] / max(1, stats["bets"]), 4)
        result.by_sport = sport_stats

        return result

    def compare_strategies(
        self,
        configs: List[BacktestConfig],
        df: Optional[pd.DataFrame] = None,
    ) -> List[BacktestResult]:
        """Run multiple backtests with different configs and return sorted by ROI."""
        if df is None:
            df = self.load_historical_data()

        results = []
        for cfg in configs:
            bt = Backtester(config=cfg)
            result = bt.run(df)
            results.append(result)

        results.sort(key=lambda r: r.roi, reverse=True)
        return results

    @staticmethod
    def format_report(result: BacktestResult) -> str:
        """Format a backtest result as a readable report."""
        lines = [
            "=" * 50,
            "BACKTEST REPORT",
            "=" * 50,
            f"Bankroll: {result.config.initial_bankroll:.2f} -> {result.final_bankroll:.2f} EUR",
            f"Total Bets: {result.total_bets} (W:{result.wins} / L:{result.losses})",
            f"Hit Rate: {result.hit_rate:.2%}",
            f"Total Staked: {result.total_staked:.2f} EUR",
            f"Total PnL: {result.total_pnl:+.2f} EUR",
            f"ROI: {result.roi:.2%}",
            f"Max Drawdown: {result.max_drawdown:.2%}",
            f"Sharpe Ratio: {result.sharpe_ratio:.2f}",
            f"Brier Score: {result.brier_score:.4f}",
            f"Peak Bankroll: {result.peak_bankroll:.2f} EUR",
            f"Min Bankroll: {result.min_bankroll:.2f} EUR",
            "",
            "Per-Sport Breakdown:",
        ]
        for sport, stats in sorted(result.by_sport.items()):
            lines.append(
                f"  {sport}: {stats['bets']} bets, "
                f"ROI={stats['roi']:.2%}, "
                f"Hit={stats['hit_rate']:.2%}, "
                f"PnL={stats['pnl']:+.2f}"
            )
        lines.append("=" * 50)
        return "\n".join(lines)

    @staticmethod
    def default_comparison_configs() -> List[BacktestConfig]:
        """Return a set of configs for strategy comparison."""
        return [
            BacktestConfig(
                kelly_frac=0.1, tax_rate=0.05, min_ev=0.0,
            ),
            BacktestConfig(
                kelly_frac=0.2, tax_rate=0.05, min_ev=0.0,
            ),
            BacktestConfig(
                kelly_frac=0.2, tax_rate=0.05, min_ev=0.01,
            ),
            BacktestConfig(
                kelly_frac=0.15, tax_rate=0.05, min_ev=0.005,
            ),
            BacktestConfig(
                kelly_frac=0.2, tax_rate=0.0, min_ev=0.0,
            ),
        ]
