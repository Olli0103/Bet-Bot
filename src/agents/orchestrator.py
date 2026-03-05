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
from src.agents.tip_publisher import (
    TipState, TipStatus, tip_flow,
    build_dss_alert, cache_stateful_tip,
)
from src.core.performance_monitor import PerformanceMonitor
from src.core.settings import settings
from src.data.redis_cache import cache
from src.integrations.odds_fetcher import OddsFetcher
from src.core.execution_jitter import apply_execution_jitter
from src.integrations.session_manager import ensure_session_fresh

log = logging.getLogger(__name__)

# Atomic lock prefix — prevents double-spend when the same signal arrives
# via multiple queue events within milliseconds of each other.
BET_LOCK_PREFIX = "bet_lock:"
BET_LOCK_TTL = 12 * 3600  # 12 hours — one lock per event:market per session

ORCHESTRATOR_STATE_KEY = "orchestrator:state"
KICKOFF_TIMES_KEY = "orchestrator:kickoff_times"

# Adaptive polling thresholds (Sniper Scheduler)
# Time-to-kickoff → polling interval mapping:
#   > 6h:   3600s (1 hour)   — markets barely move, conserve API quota
#   1-6h:     60s (1 minute) — lines firming, watch for sharp moves
#   15min-1h: 15s            — lineup drops, steam moves starting
#   < 15min:  10s            — maximum fire, Tipico lags Pinnacle by 10-15s
PRE_KICKOFF_WINDOW = timedelta(hours=6)  # earliest window to track
SNIPER_WINDOW = timedelta(minutes=15)    # maximum fire mode
FAST_WINDOW = timedelta(hours=1)         # fast polling mode

SNIPER_INTERVAL_SECONDS = 10             # < 15 min before kickoff
FAST_INTERVAL_SECONDS = 15               # 15 min - 1h before kickoff
NORMAL_INTERVAL_SECONDS = 60             # 1-6h before kickoff
IDLE_INTERVAL_SECONDS = 3600             # > 6h or no events tracked


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

        # Deduplicate: keep only the strongest alert per (event_id, market).
        # Using event_id alone would discard independent market signals — e.g.
        # a H2H steam move and a Totals edge on the same match are uncorrelated
        # and must both be evaluated.  Within the same market, we pick the
        # alert with the largest odds movement.
        deduped: Dict[str, Dict[str, Any]] = {}
        for alert in all_alerts:
            eid = alert.get("event_id", "")
            if not eid:
                continue
            market = alert.get("market", "h2h")
            dedup_key = f"{eid}:{market}"
            movement = float(alert.get("movement_pct", 0))
            existing = deduped.get(dedup_key)
            if existing is None or movement > float(existing.get("movement_pct", 0)):
                deduped[dedup_key] = alert
        all_alerts = list(deduped.values())
        log.info("Deduped alerts: %d event:market pairs (from %d raw alerts)", len(all_alerts), summary["scout_alerts"])

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

    def _nearest_kickoff_delta(self) -> Optional[timedelta]:
        """Return time until the nearest future kickoff, or None if no events.

        Uses cached kickoff times from the Scout's last odds poll to
        avoid an extra API call.
        """
        cached_times = cache.get_json(KICKOFF_TIMES_KEY) or []
        now = datetime.now(timezone.utc)
        nearest: Optional[timedelta] = None
        for ts_str in cached_times:
            try:
                ct = datetime.fromisoformat(str(ts_str).replace("Z", "+00:00"))
                if ct > now:
                    delta = ct - now
                    if nearest is None or delta < nearest:
                        nearest = delta
            except (ValueError, TypeError):
                continue
        return nearest

    def _get_adaptive_interval(self) -> int:
        """Sniper Scheduler: return polling interval based on time-to-kickoff.

        Granular schedule (conserves 95%+ of API quota while being HFT-ready):
            > 6h before kickoff:      3600s  (idle — markets barely move)
            1h - 6h before kickoff:     60s  (lines firming)
            15min - 1h before kickoff:  15s  (lineup drops, steam moves)
            < 15min before kickoff:     10s  (maximum fire — inside Tipico lag window)
            No events tracked:        3600s  (idle)
        """
        delta = self._nearest_kickoff_delta()
        if delta is None:
            return IDLE_INTERVAL_SECONDS

        if delta <= SNIPER_WINDOW:
            return SNIPER_INTERVAL_SECONDS
        elif delta <= FAST_WINDOW:
            return FAST_INTERVAL_SECONDS
        elif delta <= PRE_KICKOFF_WINDOW:
            return NORMAL_INTERVAL_SECONDS
        else:
            return IDLE_INTERVAL_SECONDS

    async def _process_single_alert(self, alert: Dict[str, Any]) -> Dict[str, Any]:
        """Process a single alert through the Analyst -> Executioner pipeline.

        Atomic SETNX lock prevents double-spend: if two queue events carry
        the same (event_id, market) signal within milliseconds, only the
        first one acquires the lock and executes.  The second sees the lock
        and returns immediately with ``action=skip``.

        Returns an exec_result dict.
        """
        event_id = alert.get("event_id", "")
        sport = alert.get("sport", "")
        home = alert.get("home", "")
        away = alert.get("away", "")
        selection = alert.get("selection", alert.get("team", ""))
        market = alert.get("market", "h2h")

        if not event_id or not selection:
            return {"action": "skip", "reasoning": ["missing event_id or selection"]}

        # --- Atomic Redis lock: one bet per (event_id, market) per session ---
        lock_key = f"{BET_LOCK_PREFIX}{event_id}:{market}"
        if not cache.setnx(lock_key, "locked", ttl_seconds=BET_LOCK_TTL):
            log.info(
                "Double-spend prevented: %s:%s already locked", event_id, market
            )
            return {"action": "skip", "reasoning": [f"bet_lock exists for {event_id}:{market}"]}

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
                # Humanized jitter: randomized delay between consecutive
                # executions to avoid WAF bot-detection (Akamai/Cloudflare).
                await apply_execution_jitter()

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
            "(sniper=%ds, fast=%ds, normal=%ds, idle=%ds, queue-based execution)",
            SNIPER_INTERVAL_SECONDS, FAST_INTERVAL_SECONDS,
            NORMAL_INTERVAL_SECONDS, IDLE_INTERVAL_SECONDS,
        )

        # Start the consumer worker
        worker_task = asyncio.create_task(self._worker_loop())

        # Producer: Scout polls and pushes alerts to the queue
        while self._running:
            # Proactive session refresh: if we're entering sniper mode
            # (< 15 min to kickoff), ensure the bookie session is hot.
            interval_preview = self._get_adaptive_interval()
            if interval_preview <= SNIPER_INTERVAL_SECONDS:
                ok, reason = ensure_session_fresh()
                if not ok:
                    log.warning("Session refresh failed before sniper window: %s", reason)

            try:
                odds_alerts = await self.scout.monitor_odds()
                injury_alerts = await self.scout.monitor_injuries()
                all_alerts = odds_alerts + injury_alerts

                # Deduplicate: keep only the strongest alert per (event_id, market)
                deduped: Dict[str, Dict[str, Any]] = {}
                for alert in all_alerts:
                    eid = alert.get("event_id", "")
                    if not eid:
                        continue
                    market = alert.get("market", "h2h")
                    dedup_key = f"{eid}:{market}"
                    movement = float(alert.get("movement_pct", 0))
                    existing = deduped.get(dedup_key)
                    if existing is None or movement > float(existing.get("movement_pct", 0)):
                        deduped[dedup_key] = alert

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

    async def _get_current_odds(self, event_id: str, selection: str) -> float:
        """Fetch the latest odds for a selection from the odds API.

        Used by the tip publisher to refresh odds before validation.
        """
        try:
            fetcher = OddsFetcher()
            odds_data = await fetcher.get_event_odds(event_id)
            if odds_data and selection in odds_data:
                return float(odds_data[selection])
        except Exception as exc:
            log.warning("Failed to fetch current odds for %s: %s", event_id, exc)
        return 0.0

    async def _publish_tip(self, state: TipState) -> None:
        """Publish a validated tip via Telegram with interactive DSS interface.

        Instead of placing a bet, formats the tip as an actionable
        recommendation with inline keyboards:
          [Mark as Placed at X.XX] | [Odds Dropped] | [Show Math]

        Creates a ``StatefulTip`` audit record that requires human review
        before any bankroll update occurs (GDPR Art. 22 compliance).
        """
        # Generate LLM reasoning for the tip card
        reasoning_text = ""
        try:
            reasoning_text = await self.analyst.reason_with_llm(state.analysis) or ""
        except Exception:
            pass

        # Build validated TipAlert and cache StatefulTip for audit
        alert = build_dss_alert(state, reasoning=reasoning_text or "Quantitatives Signal.")
        tip_id = cache_stateful_tip(state, alert)

        if not self.bot or not self.chat_id:
            log.info("DSS tip ready (no Telegram): tip_id=%s %s", tip_id, state.to_dict())
            return

        # Format the rich tip card
        text = alert.format_for_telegram()

        # Build interactive DSS inline keyboard
        try:
            from telegram import InlineKeyboardButton, InlineKeyboardMarkup
            keyboard = InlineKeyboardMarkup([
                [
                    InlineKeyboardButton(
                        f"\u2705 Platziert @ {state.current_odds:.2f}",
                        callback_data=f"dss_placed:{tip_id}",
                    ),
                    InlineKeyboardButton(
                        "\u274c Abgelehnt",
                        callback_data=f"dss_rejected:{tip_id}",
                    ),
                ],
                [
                    InlineKeyboardButton(
                        "\U0001f4ca Mathe zeigen",
                        callback_data=f"dss_math:{tip_id}",
                    ),
                    InlineKeyboardButton(
                        "\U0001f50d Deep Dive",
                        callback_data=f"agent_analyze:{tip_id}",
                    ),
                ],
            ])
            await self.bot.send_message(
                chat_id=self.chat_id,
                text=text,
                reply_markup=keyboard,
            )
        except ImportError:
            # Telegram not available — send plain text
            await self.bot.send_message(chat_id=self.chat_id, text=text)
        except Exception as exc:
            log.warning("Failed to send DSS tip to Telegram: %s", exc)

    async def run_tip_flow(self, alert: Dict[str, Any]) -> TipState:
        """Process a single alert through the stateful tip publisher.

        This is the Tippprovider alternative to ``_process_single_alert``
        that uses the state-graph flow with re-validation instead of the
        linear Executioner pipeline.
        """
        state = TipState(
            event_id=alert.get("event_id", ""),
            sport=alert.get("sport", ""),
            home=alert.get("home", ""),
            away=alert.get("away", ""),
            selection=alert.get("selection", alert.get("team", "")),
            market=alert.get("market", "h2h"),
            initial_odds=float(alert.get("current_odds", 2.0)),
        )

        result = await tip_flow(
            state=state,
            analyst=self.analyst,
            get_current_odds=self._get_current_odds,
            publish_fn=self._publish_tip,
        )

        # Cache tip state for monitoring
        cache.set_json(
            f"tip:{state.event_id}:{state.market}",
            state.to_dict(),
            ttl_seconds=6 * 3600,
        )

        return result

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
