"""Agent Orchestrator: Coordinates Scout -> Analyst -> Executioner pipeline.

Replaces the fixed APScheduler approach with an event-driven, agent-based
architecture. Falls back to schedule-based mode if the agent framework fails.
"""
from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from src.agents.analyst_agent import AnalystAgent
from src.agents.executioner_agent import ExecutionerAgent
from src.agents.scout_agent import ScoutAgent
from src.core.performance_monitor import PerformanceMonitor
from src.core.settings import settings
from src.data.redis_cache import cache
from src.integrations.odds_fetcher import OddsFetcher

log = logging.getLogger(__name__)

ORCHESTRATOR_STATE_KEY = "orchestrator:state"


class AgentOrchestrator:
    """Coordinates all agents: Scout -> Analyst -> Executioner -> Learn.

    The orchestrator can run in two modes:
    1. Continuous monitoring (default) — polls every 5 minutes
    2. Scheduled mode (fallback) — runs at fixed times
    """

    def __init__(self, bot=None, chat_id: str = "") -> None:
        self.scout = ScoutAgent()
        self.analyst = AnalystAgent()
        self.executioner = ExecutionerAgent()
        self.monitor = PerformanceMonitor()
        self.bot = bot
        self.chat_id = chat_id or settings.telegram_chat_id
        self._running = False

    async def run_once(self) -> Dict[str, Any]:
        """Run one full Scout -> Analyst -> Executioner cycle.

        Returns a summary dict of actions taken.
        """
        summary: Dict[str, Any] = {
            "ts": datetime.now(timezone.utc).isoformat(),
            "scout_alerts": 0,
            "analyses": 0,
            "bets_placed": 0,
            "bets_skipped": 0,
            "halted": False,
        }

        # 1. Scout: detect steam moves and breaking news
        try:
            odds_alerts = await self.scout.monitor_odds()
            twitter_alerts = await self.scout.monitor_twitter()
            all_alerts = odds_alerts + twitter_alerts
            summary["scout_alerts"] = len(all_alerts)
        except Exception as exc:
            log.error("Scout failed: %s", exc)
            all_alerts = []

        if not all_alerts:
            return summary

        # 2. Analyst: deep analysis for each alert
        odds_fetcher = OddsFetcher()

        for alert in all_alerts:
            try:
                # Get current odds for this event
                event_id = alert.get("event_id", "")
                sport = alert.get("sport", "")
                home = alert.get("home", "")
                away = alert.get("away", "")
                selection = alert.get("selection", alert.get("team", ""))

                if not event_id or not selection:
                    continue

                # Fetch current odds for context
                target_odds = alert.get("current_odds", 2.0)
                sharp_odds = alert.get("prev_odds", target_odds)

                analysis = await self.analyst.analyze_event(
                    event_id=event_id,
                    sport=sport,
                    home=home,
                    away=away,
                    selection=selection,
                    target_odds=target_odds,
                    sharp_odds=sharp_odds,
                    sharp_market={selection: sharp_odds},
                    trigger=alert.get("type", "unknown"),
                )
                summary["analyses"] += 1

                # 3. Executioner: final decision
                exec_result = await self.executioner.execute(
                    analysis=analysis,
                    bot=self.bot,
                    chat_id=self.chat_id,
                )

                if exec_result.get("action") == "bet":
                    summary["bets_placed"] += 1
                elif exec_result.get("action") == "halt":
                    summary["halted"] = True
                    break  # Stop processing if circuit breaker tripped
                else:
                    summary["bets_skipped"] += 1

            except Exception as exc:
                log.warning("Analysis/execution failed for alert: %s", exc)
                continue

        # Save state
        cache.set_json(ORCHESTRATOR_STATE_KEY, summary, ttl_seconds=6 * 3600)
        return summary

    async def run_continuous(self, interval_minutes: int = 5) -> None:
        """Run the orchestrator continuously with a polling interval.

        This replaces the fixed schedule approach with real-time monitoring.
        """
        self._running = True
        log.info("Agent orchestrator starting (interval=%dm)", interval_minutes)

        while self._running:
            try:
                summary = await self.run_once()
                if summary.get("scout_alerts", 0) > 0:
                    log.info(
                        "Orchestrator cycle: %d alerts, %d analyses, %d bets, %d skipped",
                        summary["scout_alerts"],
                        summary["analyses"],
                        summary["bets_placed"],
                        summary["bets_skipped"],
                    )
            except Exception as exc:
                log.error("Orchestrator cycle failed: %s", exc)

            await asyncio.sleep(interval_minutes * 60)

    def stop(self) -> None:
        """Stop the continuous monitoring loop."""
        self._running = False
        log.info("Agent orchestrator stopping")

    async def daily_self_evaluation(self) -> str:
        """Daily self-evaluation: check calibration, ROI, weaknesses.

        Should be called at ~22:00 daily.
        """
        report = self.monitor.generate_report()

        # Send to Telegram if available
        if self.bot and self.chat_id:
            try:
                await self.bot.send_message(chat_id=self.chat_id, text=report)
            except Exception as exc:
                log.warning("Failed to send daily report: %s", exc)

        return report

    @staticmethod
    def get_last_state() -> Dict[str, Any]:
        """Retrieve the last orchestrator run state."""
        return cache.get_json(ORCHESTRATOR_STATE_KEY) or {}
