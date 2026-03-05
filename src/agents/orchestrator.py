"""Agent Orchestrator: Coordinates Scout -> Analyst -> Executioner pipeline.

Uses an event-driven architecture with asyncio.Queue: the Scout pushes
alerts the moment they are detected, and a dedicated worker loop consumes
them with zero polling delay.  Falls back to a fast-poll loop (5s near
kickoff, 60s otherwise) when the Scout isn't running a persistent
websocket connection yet.
"""
from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timedelta, timezone
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
KICKOFF_TIMES_KEY = "orchestrator:kickoff_times"

# Adaptive polling thresholds
PRE_KICKOFF_WINDOW = timedelta(hours=1)  # start fast-polling 1h before kickoff
FAST_INTERVAL_SECONDS = 5                # 5s during steam-move window (was 60s)
NORMAL_INTERVAL_SECONDS = 60             # 60s during normal periods (was 300s)


class AgentOrchestrator:
    """Coordinates all agents: Scout -> Analyst -> Executioner -> Learn.

    Architecture:
    - **Alert queue** (asyncio.Queue): Scout pushes alerts, worker consumes them
      with zero polling delay.  This eliminates the 60s latency gap that was
      causing the bot to buy Tipico closing lines instead of pre-move prices.
    - **Fast mode** (5s): when any tracked event kicks off within the next hour.
      Pinnacle steam moves last 1-2s, Tipico adjusts in 10-15s — we need to
      be inside that window.
    - **Normal mode** (60s): no imminent kickoffs.
    """

    def __init__(self, bot=None, chat_id: str = "") -> None:
        self.scout = ScoutAgent()
        self.analyst = AnalystAgent()
        self.executioner = ExecutionerAgent()
        self.monitor = PerformanceMonitor()
        self.bot = bot
        self.chat_id = chat_id or settings.telegram_chat_id
        self._running = False
        self.alert_queue: asyncio.Queue = asyncio.Queue()

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
            injury_alerts = await self.scout.monitor_injuries()
            all_alerts = odds_alerts + injury_alerts
            summary["scout_alerts"] = len(all_alerts)
        except Exception as exc:
            log.error("Scout failed: %s", exc)
            all_alerts = []

        if not all_alerts:
            return summary

        # Deduplicate: keep only the strongest alert per event_id.
        # Without this, the Scout can emit contradictory 1X2 picks for the
        # same match (e.g. Home win + Away win + Draw), all as separate alerts.
        # We pick the one with the largest odds movement per event.
        deduped: Dict[str, Dict[str, Any]] = {}
        for alert in all_alerts:
            eid = alert.get("event_id", "")
            if not eid:
                continue
            movement = float(alert.get("movement_pct", 0))
            existing = deduped.get(eid)
            if existing is None or movement > float(existing.get("movement_pct", 0)):
                deduped[eid] = alert
        all_alerts = list(deduped.values())
        log.info("Deduped alerts: %d events (from %d raw alerts)", len(all_alerts), summary["scout_alerts"])

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
                    market_momentum=float(alert.get("market_momentum", 0)),
                )
                # Attach event commence time so it flows into the alert cache
                analysis["commence_time"] = alert.get("commence_time", "")
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

    def _has_imminent_kickoffs(self) -> bool:
        """Check if any tracked event kicks off within PRE_KICKOFF_WINDOW.

        Uses cached kickoff times from the Scout's last odds poll to
        avoid an extra API call.
        """
        cached_times = cache.get_json(KICKOFF_TIMES_KEY) or []
        now = datetime.now(timezone.utc)
        cutoff = now + PRE_KICKOFF_WINDOW
        for ts_str in cached_times:
            try:
                ct = datetime.fromisoformat(str(ts_str).replace("Z", "+00:00"))
                if now < ct <= cutoff:
                    return True
            except (ValueError, TypeError):
                continue
        return False

    def _get_adaptive_interval(self) -> int:
        """Return the polling interval in seconds based on kickoff proximity."""
        if self._has_imminent_kickoffs():
            return FAST_INTERVAL_SECONDS
        return NORMAL_INTERVAL_SECONDS

    async def _process_single_alert(self, alert: Dict[str, Any]) -> Dict[str, Any]:
        """Process a single alert through the Analyst -> Executioner pipeline.

        Returns an exec_result dict.
        """
        event_id = alert.get("event_id", "")
        sport = alert.get("sport", "")
        home = alert.get("home", "")
        away = alert.get("away", "")
        selection = alert.get("selection", alert.get("team", ""))

        if not event_id or not selection:
            return {"action": "skip", "reasoning": ["missing event_id or selection"]}

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
            market_momentum=float(alert.get("market_momentum", 0)),
        )
        analysis["commence_time"] = alert.get("commence_time", "")

        return await self.executioner.execute(
            analysis=analysis,
            bot=self.bot,
            chat_id=self.chat_id,
        )

    async def _worker_loop(self) -> None:
        """Asynchronously consume alerts the moment they arrive in the queue.

        This is the core of the event-driven architecture: zero polling
        delay between Scout detection and Executioner action.  Each alert
        is processed independently so a slow analysis doesn't block
        fast-moving steam moves.
        """
        while self._running:
            try:
                alert = await asyncio.wait_for(self.alert_queue.get(), timeout=1.0)
            except asyncio.TimeoutError:
                continue

            try:
                exec_result = await self._process_single_alert(alert)
                action = exec_result.get("action", "skip")
                if action == "bet":
                    log.info("Worker executed bet: %s %s",
                             alert.get("sport", ""), alert.get("selection", ""))
                elif action == "halt":
                    log.warning("Circuit breaker tripped during worker processing")
            except Exception as exc:
                log.error("Worker alert processing failed: %s", exc)
            finally:
                self.alert_queue.task_done()

    async def run_continuous(self, interval_minutes: int = 5) -> None:
        """Run the orchestrator in event-driven mode with a fast-poll producer.

        Architecture:
        1. A dedicated worker task consumes alerts from the queue with 0ms delay.
        2. The Scout produces alerts via a fast-poll loop (5s near kickoff, 60s
           otherwise).  When the Scout is upgraded to websockets, it will push
           directly to self.alert_queue and the poll loop becomes a no-op.
        """
        self._running = True
        log.info(
            "Agent orchestrator starting in EVENT-DRIVEN mode "
            "(fast=%ds, normal=%ds, queue-based execution)",
            FAST_INTERVAL_SECONDS, NORMAL_INTERVAL_SECONDS,
        )

        # Start the consumer worker
        worker_task = asyncio.create_task(self._worker_loop())

        # Producer: Scout polls and pushes alerts to the queue
        while self._running:
            try:
                odds_alerts = await self.scout.monitor_odds()
                injury_alerts = await self.scout.monitor_injuries()
                all_alerts = odds_alerts + injury_alerts

                # Deduplicate: keep only the strongest alert per event_id
                deduped: Dict[str, Dict[str, Any]] = {}
                for alert in all_alerts:
                    eid = alert.get("event_id", "")
                    if not eid:
                        continue
                    movement = float(alert.get("movement_pct", 0))
                    existing = deduped.get(eid)
                    if existing is None or movement > float(existing.get("movement_pct", 0)):
                        deduped[eid] = alert

                # Push to execution queue (non-blocking)
                for alert in deduped.values():
                    self.alert_queue.put_nowait(alert)

                if deduped:
                    log.info("Scout pushed %d alerts to execution queue", len(deduped))

            except Exception as exc:
                log.error("Scout producer loop failed: %s", exc)

            interval = self._get_adaptive_interval()
            await asyncio.sleep(interval)

        # Drain remaining alerts before shutting down
        if not self.alert_queue.empty():
            log.info("Draining %d remaining alerts from queue", self.alert_queue.qsize())
            await self.alert_queue.join()

        worker_task.cancel()
        try:
            await worker_task
        except asyncio.CancelledError:
            pass

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
