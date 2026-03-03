"""Executioner Agent: Final EV check, circuit breakers, and Telegram alerts.

The Executioner is the last agent in the pipeline. It applies circuit
breakers, performance-based Kelly adjustments, and sends formatted
alerts to Telegram. It can also auto-place virtual bets.
"""
from __future__ import annotations

import hashlib
import json
import logging
from typing import Any, Dict, Optional

from src.core.bankroll import BankrollManager
from src.core.betting_math import kelly_fraction, kelly_stake
from src.core.ml_trainer import get_reliability_adjustment
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

        # 3. Calculate stake with performance-adjusted + calibration-adjusted Kelly
        tax_rate = settings.tipico_tax_rate if not settings.tax_free_mode else 0.0

        bankroll = self.bankroll_mgr.get_current_bankroll()
        kelly_mult = adjustments.get("kelly_multiplier", 1.0)

        # Calibration reliability adjustment: if the model over-predicts in
        # this probability bucket, trim the Kelly fraction proportionally.
        sport = str(analysis.get("sport", ""))
        sport_group = "general"
        if sport.startswith(("soccer", "football")):
            sport_group = "soccer"
        elif sport.startswith("basketball"):
            sport_group = "basketball"
        elif sport.startswith("tennis"):
            sport_group = "tennis"
        calib_adj = get_reliability_adjustment(model_p, sport_group)
        kelly_mult *= min(1.3, max(0.5, calib_adj))  # clamp to [0.5, 1.3]

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

    def _cache_alert(self, analysis: Dict[str, Any], stake: float) -> str:
        """Cache alert data and return a short ID for inline button callbacks.

        Caches the FULL analysis result so that Deep Dive can display the exact
        same data without re-running the analyst (which caused data mismatches
        when inputs like sharp_odds were lost).
        """
        # Core identification + inputs for potential re-analysis
        payload = {
            "event_id": str(analysis.get("event_id", "")),
            "sport": str(analysis.get("sport", "")),
            "home": str(analysis.get("home", "")),
            "away": str(analysis.get("away", "")),
            "selection": str(analysis.get("selection", "")),
            "commence_time": str(analysis.get("commence_time", "")),
            "target_odds": float(analysis.get("bookmaker_odds", 0)),
            "sharp_odds": float(analysis.get("sharp_odds", 0)),
            "sharp_market": analysis.get("sharp_market", {}),
            "market_momentum": float(analysis.get("market_momentum", 0)),
            "trigger": str(analysis.get("trigger", "")),
            "stake": stake,
            # Full analysis results (so Deep Dive shows identical data)
            "model_probability": float(analysis.get("model_probability", 0)),
            "expected_value": float(analysis.get("expected_value", 0)),
            "recommendation": str(analysis.get("recommendation", "SKIP")),
            "sentiment": analysis.get("sentiment", {}),
            "injuries": analysis.get("injuries", {}),
            "injury_details": analysis.get("injury_details", []),
            "form": analysis.get("form", {}),
            "elo": analysis.get("elo", {}),
            "poisson_prob": analysis.get("poisson_prob"),
            "public_bias": float(analysis.get("public_bias", 0)),
        }
        raw = json.dumps(payload, sort_keys=True, default=str)
        alert_id = hashlib.md5(raw.encode()).hexdigest()[:12]
        cache.set_json(f"agent_alert:{alert_id}", payload, ttl_seconds=6 * 3600)
        return alert_id

    async def _send_alert(
        self,
        bot,
        chat_id: str,
        analysis: Dict[str, Any],
        stake: float,
    ) -> None:
        """Format and send a Telegram alert with interactive inline buttons."""
        from telegram import InlineKeyboardButton, InlineKeyboardMarkup

        sport = str(analysis.get("sport", "")).replace("_", " ").upper()
        home = analysis.get("home", "")
        away = analysis.get("away", "")
        selection = analysis.get("selection", "")
        trigger = analysis.get("trigger", "")
        model_p = float(analysis.get("model_probability", 0))
        ev = float(analysis.get("expected_value", 0))
        commence = str(analysis.get("commence_time", ""))

        trigger_emoji = "⚡" if trigger == "steam_move" else "🏥" if trigger == "breaking_injury" else "🎯"

        # Format event time (ISO -> readable German format)
        event_time_str = ""
        if commence:
            try:
                from datetime import datetime as dt
                from zoneinfo import ZoneInfo
                ct = dt.fromisoformat(commence.replace("Z", "+00:00"))
                local = ct.astimezone(ZoneInfo("Europe/Berlin"))
                event_time_str = local.strftime("%d.%m. %H:%M")
            except Exception:
                event_time_str = commence[:16] if len(commence) >= 16 else commence

        # Progress bar for model probability
        filled = int(round(model_p * 10))
        bar = "█" * filled + "░" * (10 - filled)

        time_line = f"Anstoss: {event_time_str}\n" if event_time_str else ""

        msg = (
            f"{trigger_emoji} AGENT ALERT | {sport}\n"
            f"━━━━━━━━━━━━━━━━━━━━\n"
            f"Match: {home} vs {away}\n"
            f"{time_line}"
            f"Tipp: {selection}\n"
            f"Trigger: {trigger}\n"
            f"Modell: [{bar}] {model_p:.0%}\n"
            f"EV: {ev:+.4f} | Einsatz: {stake:.2f} EUR"
        )

        # Cache alert data for callback retrieval
        alert_id = self._cache_alert(analysis, stake)

        keyboard = InlineKeyboardMarkup([
            [
                InlineKeyboardButton("🔍 Deep Dive", callback_data=f"agent_analyze:{alert_id}"),
                InlineKeyboardButton("💰 Ghost Bet", callback_data=f"agent_ghost:{alert_id}"),
            ],
            [
                InlineKeyboardButton("🛑 Ignorieren", callback_data=f"agent_ignore:{alert_id}"),
            ],
        ])

        await bot.send_message(chat_id=chat_id, text=msg, reply_markup=keyboard)

    def check_circuit_breakers(self) -> bool:
        """Return False if we should stop betting (convenience wrapper)."""
        breakers = self.monitor.check_circuit_breakers()
        return not any(breakers.values())
