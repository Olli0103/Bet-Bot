"""Executioner Agent: Final EV check, circuit breakers, and Telegram alerts.

The Executioner is the last agent in the pipeline. It applies circuit
breakers, performance-based Kelly adjustments, and sends formatted
alerts to Telegram. It can also auto-place virtual bets.
"""
from __future__ import annotations

import logging
from typing import Any, Dict, Optional

from src.core.bankroll import BankrollManager
from src.core.betting_math import kelly_fraction, kelly_stake
from src.core.performance_monitor import PerformanceMonitor
from src.core.settings import settings
from src.data.redis_cache import cache

log = logging.getLogger(__name__)

EXECUTED_KEY = "exec:alerts:today"


class ExecutionerAgent:
    """Applies circuit breakers, calculates final stakes, and sends alerts."""

    def __init__(self) -> None:
        self.monitor = PerformanceMonitor()
        self.bankroll_mgr = BankrollManager()

    async def execute(
        self,
        analysis: Dict[str, Any],
        bot=None,
        chat_id: str = "",
    ) -> Dict[str, Any]:
        """Process an analysis result: apply breakers, size stake, optionally send alert.

        Parameters
        ----------
        analysis : dict
            Output from AnalystAgent.analyze_event().
        bot : telegram.Bot, optional
            If provided, sends alert via Telegram.
        chat_id : str
            Telegram chat ID for sending alerts.

        Returns
        -------
        dict with keys: action ("bet", "skip", "halt"), stake, reasoning
        """
        result: Dict[str, Any] = {
            "event_id": analysis.get("event_id"),
            "selection": analysis.get("selection"),
            "action": "skip",
            "stake": 0.0,
            "reasoning": [],
        }

        # 1. Check circuit breakers
        breakers = self.monitor.check_circuit_breakers()
        if breakers.get("losing_streak"):
            result["action"] = "halt"
            result["reasoning"].append("Losing streak > 7: system halted")
            log.warning("Circuit breaker: losing streak > 7")
            return result

        if breakers.get("daily_loss_limit"):
            result["action"] = "halt"
            result["reasoning"].append("Daily loss > 5% of bankroll: system halted")
            log.warning("Circuit breaker: daily loss limit")
            return result

        # 2. Check EV threshold
        ev = float(analysis.get("expected_value", 0))
        model_p = float(analysis.get("model_probability", 0))
        recommendation = analysis.get("recommendation", "SKIP")

        adjustments = self.monitor.get_adjustment_factors()
        min_ev = adjustments.get("min_ev", 0.01)

        if ev < min_ev or recommendation == "SKIP":
            result["reasoning"].append(f"EV {ev:.4f} < min_ev {min_ev:.4f}")
            return result

        # 3. Calculate stake with performance-adjusted Kelly
        target_odds = float(analysis.get("features", {}).get("sharp_implied_prob", 0))
        # Reconstruct odds from the analysis context
        tax_rate = settings.tipico_tax_rate if not settings.tax_free_mode else 0.0

        bankroll = self.bankroll_mgr.get_current_bankroll()
        kelly_mult = adjustments.get("kelly_multiplier", 1.0)

        # Use reduced Kelly for reactive bets (steam moves)
        frac = 0.15 if analysis.get("trigger") == "steam_move" else 0.2
        frac *= kelly_mult

        # We need bookmaker_odds to compute kelly; extract from features or analysis
        bookmaker_odds = analysis.get("bookmaker_odds", 0)
        if bookmaker_odds and bookmaker_odds > 1.0:
            kf = kelly_fraction(model_p, bookmaker_odds, frac=frac, tax_rate=tax_rate)
            stake = round(kelly_stake(bankroll, kf), 2)
        else:
            stake = round(bankroll * 0.01, 2)  # fallback 1% of bankroll

        if stake < 0.50:
            result["reasoning"].append(f"Calculated stake {stake:.2f} < 0.50 minimum")
            return result

        result["action"] = "bet"
        result["stake"] = stake
        result["reasoning"].append(f"EV={ev:.4f}, Kelly frac={frac:.2f}, stake={stake:.2f}")

        # 4. Optionally send Telegram alert
        if bot and chat_id:
            try:
                await self._send_alert(bot, chat_id, analysis, stake)
            except Exception as exc:
                log.warning("Failed to send Telegram alert: %s", exc)

        # 5. Auto-place virtual bet
        try:
            from src.core.ghost_trading import place_virtual_bet
            place_virtual_bet(
                event_id=str(analysis.get("event_id", "")),
                sport=str(analysis.get("sport", "")),
                market="h2h",
                selection=str(analysis.get("selection", "")),
                odds=bookmaker_odds if bookmaker_odds else 2.0,
                stake=stake,
                features=analysis.get("features", {}),
            )
        except Exception:
            pass

        return result

    async def _send_alert(
        self,
        bot,
        chat_id: str,
        analysis: Dict[str, Any],
        stake: float,
    ) -> None:
        """Format and send a Telegram alert for a triggered bet."""
        sport = str(analysis.get("sport", "")).replace("_", " ").upper()
        home = analysis.get("home", "")
        away = analysis.get("away", "")
        selection = analysis.get("selection", "")
        trigger = analysis.get("trigger", "")
        model_p = float(analysis.get("model_probability", 0))
        ev = float(analysis.get("expected_value", 0))

        trigger_emoji = "⚡" if trigger == "steam_move" else "🏥" if trigger == "breaking_injury" else "🎯"

        msg = (
            f"{trigger_emoji} AGENT ALERT | {sport}\n"
            f"Match: {home} vs {away}\n"
            f"Tipp: {selection}\n"
            f"Trigger: {trigger}\n"
            f"Modell: {model_p:.2%} | EV: {ev:.4f}\n"
            f"Einsatz: {stake:.2f} EUR"
        )
        await bot.send_message(chat_id=chat_id, text=msg)

    def check_circuit_breakers(self) -> bool:
        """Return False if we should stop betting (convenience wrapper)."""
        breakers = self.monitor.check_circuit_breakers()
        return not any(breakers.values())
