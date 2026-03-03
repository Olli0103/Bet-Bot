"""Core Worker: runs the signal pipeline, agents, and scheduled jobs.

Independent of Telegram — writes outbound messages to the Redis outbox
queue. Telegram failures do NOT block or crash this worker.

Entry point:
    python -m src.bot.core_worker
"""
from __future__ import annotations

import asyncio
import atexit
import logging
import os
import signal
import sys
import time
from datetime import datetime, timezone, timedelta
from zoneinfo import ZoneInfo

from src.bot.chat_router import chat_router
from src.bot.message_queue import push_outbox, pop_inbox
from src.core.settings import settings
from src.data.models import Base
from src.data.postgres import engine

log = logging.getLogger(__name__)
logging.basicConfig(
    level=getattr(logging, settings.log_level, logging.INFO),
    format="%(asctime)s [CORE] %(levelname)s %(name)s: %(message)s",
)

PID_FILE = os.path.join(os.path.dirname(__file__), ".core_worker.pid")

_running = True


def _signal_handler(signum, frame):
    global _running
    log.info("Core worker received signal %s, shutting down...", signum)
    _running = False


def _write_pid():
    with open(PID_FILE, "w") as f:
        f.write(str(os.getpid()))


def _remove_pid():
    try:
        if os.path.isfile(PID_FILE):
            with open(PID_FILE) as f:
                if f.read().strip() == str(os.getpid()):
                    os.remove(PID_FILE)
    except OSError:
        pass


# ── Scheduled tasks (write to outbox instead of Telegram) ─────

def _now_berlin() -> datetime:
    return datetime.now(timezone.utc).astimezone(ZoneInfo("Europe/Berlin"))


def _run_scheduled_fetch(push: bool = False):
    """Fetch signals and push results to outbox."""
    from src.core.live_feed import fetch_and_build_signals
    try:
        items = fetch_and_build_signals()
        push_outbox("text", {"text": f"📡 Datenupdate fertig: {len(items)} Signals"}, target="broadcast")
        if push and items:
            _push_daily_signals()
    except Exception as exc:
        log.error("Scheduled fetch failed: %s", exc)


def _push_daily_signals():
    """Push top singles + combos to outbox."""
    from src.core.live_feed import get_cached_signals, get_cached_combos
    items, ts = get_cached_signals()
    if not items:
        push_outbox("text", {"text": "Keine spielbaren Einzelwetten heute."}, target="broadcast")
        return

    singles = [x for x in items if float(x.get("expected_value", 0)) > 0][:10]
    if singles:
        push_outbox("signal_push", {"signals": singles, "ts": ts}, target="broadcast")

    combos = get_cached_combos()
    if combos:
        push_outbox("combo_push", {"combos": combos}, target="broadcast")


def _run_auto_grading():
    from src.core.autograding import run_auto_grading
    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            settled = asyncio.run_coroutine_threadsafe(run_auto_grading(), loop).result(timeout=60)
        else:
            settled = asyncio.run(run_auto_grading())
        if settled > 0:
            push_outbox("text", {"text": f"✅ Auto-Grading: {settled} Wetten bewertet"}, target="broadcast")
    except Exception as exc:
        log.error("Auto-grading failed: %s", exc)


def _run_learning_status():
    from src.core.learning_monitor import learning_health
    try:
        h = learning_health()
        push_outbox("text", {
            "text": (
                f"🧠 Learning Check\n"
                f"Total: {h['total']} | Settled: {h['settled']} | Open: {h['open']}\n"
                f"W/L: {h['wins']}/{h['losses']} | Hit: {h['hit_rate_pct']}% | PnL: {h['pnl']:.2f} EUR"
            )
        }, target="broadcast")
    except Exception as exc:
        log.error("Learning status failed: %s", exc)


def _run_weekly_retrain():
    from src.core.ml_trainer import auto_train_all_models
    try:
        msg = auto_train_all_models(100)
        push_outbox("text", {"text": f"📈 Weekly Retrain: {msg}"}, target="primary")
    except Exception as exc:
        log.error("Weekly retrain failed: %s", exc)


def _run_api_health():
    from src.core.api_health import format_api_health_report, run_api_health_check
    try:
        loop = asyncio.new_event_loop()
        result = loop.run_until_complete(run_api_health_check())
        loop.close()
        push_outbox("text", {"text": format_api_health_report(result)}, target="primary")
    except Exception as exc:
        log.error("API health check failed: %s", exc)


def _run_daily_performance():
    from src.core.performance_monitor import PerformanceMonitor
    try:
        monitor = PerformanceMonitor()
        report = monitor.generate_report()
        push_outbox("text", {"text": report}, target="broadcast")
    except Exception as exc:
        log.error("Daily performance report failed: %s", exc)


async def _run_agent_cycle():
    """Run one Scout -> Analyst -> Executioner cycle."""
    from src.agents.orchestrator import AgentOrchestrator
    try:
        orchestrator = AgentOrchestrator(bot=None, chat_id=chat_router.primary_id)
        is_fast = orchestrator._has_imminent_kickoffs()
        summary = await orchestrator.run_once()
        if summary.get("scout_alerts", 0) > 0:
            mode_label = "FAST" if is_fast else "NORMAL"
            push_outbox("text", {
                "text": (
                    f"🤖 Agent [{mode_label}]: {summary['scout_alerts']} alerts, "
                    f"{summary['bets_placed']} bets, {summary['bets_skipped']} skipped"
                )
            }, target="broadcast")
    except Exception as exc:
        log.error("Agent cycle failed: %s", exc)


# ── Inbox processor ───────────────────────────────────────────

def _process_inbox():
    """Drain inbox queue and handle user actions."""
    for _ in range(10):  # process up to 10 messages per tick
        msg = pop_inbox(timeout=0)
        if msg is None:
            break
        action = msg.get("action", "")
        payload = msg.get("payload", {})
        log.info("Inbox action: %s", action)

        if action == "refresh_data":
            _run_scheduled_fetch(push=True)
        elif action == "retrain":
            _run_weekly_retrain()
        # Other actions are handled directly by telegram_worker


# ── Main loop ─────────────────────────────────────────────────

class Scheduler:
    """Simple cron-like scheduler for the core worker."""

    def __init__(self):
        self._last_run: dict[str, float] = {}

    def should_run(self, name: str, interval_sec: float) -> bool:
        now = time.time()
        last = self._last_run.get(name, 0)
        if now - last >= interval_sec:
            self._last_run[name] = now
            return True
        return False

    def should_run_daily(self, name: str, hour: int, minute: int = 0) -> bool:
        now = _now_berlin()
        key = f"{name}:{now.date().isoformat()}"
        if key in self._last_run:
            return False
        if now.hour == hour and now.minute == minute:
            self._last_run[key] = time.time()
            return True
        return False

    def should_run_weekly(self, name: str, weekday: int, hour: int, minute: int = 0) -> bool:
        now = _now_berlin()
        key = f"{name}:{now.isocalendar()[1]}"
        if key in self._last_run:
            return False
        if now.weekday() == weekday and now.hour == hour and now.minute == minute:
            self._last_run[key] = time.time()
            return True
        return False


async def main_loop():
    global _running
    sched = Scheduler()
    agent_counter = 0

    log.info("Core worker starting (pid=%d)", os.getpid())

    while _running:
        now = _now_berlin()

        # ── Daily scheduled jobs ──
        if sched.should_run_daily("morning_briefing", 6, 30):
            push_outbox("text", {"text": "🌅 Morning Briefing bereit. Tippe auf Heutige Top 10 Einzelwetten."}, target="broadcast")

        if sched.should_run_daily("fetch_0700", 7, 0):
            _run_scheduled_fetch(push=True)

        if sched.should_run_daily("fetch_1300", 13, 0):
            _run_scheduled_fetch(push=False)

        if sched.should_run_daily("learning_daily", 20, 0):
            _run_learning_status()

        if sched.should_run_daily("api_health", 20, 5):
            _run_api_health()

        if sched.should_run_daily("daily_report", 22, 0):
            _run_daily_performance()

        # Weekly retrain (Saturday = weekday 5)
        if sched.should_run_weekly("retrain_weekly", 5, 3, 15):
            _run_weekly_retrain()

        # ── Repeating jobs ──

        # Auto-grading every 30 minutes
        if sched.should_run("auto_grading", 1800):
            _run_auto_grading()

        # Agent cycle every 60s (internally adaptive: skip 4/5 in normal mode)
        if sched.should_run("agent_cycle", 60):
            agent_counter += 1
            # Check imminent kickoffs for adaptive polling
            try:
                from src.agents.orchestrator import AgentOrchestrator
                orch = AgentOrchestrator(bot=None, chat_id=chat_router.primary_id)
                is_fast = orch._has_imminent_kickoffs()
            except Exception:
                is_fast = False

            if is_fast or (agent_counter % 5) == 0:
                await _run_agent_cycle()

        # Process inbox (user actions forwarded from telegram worker)
        _process_inbox()

        # Health heartbeat
        if sched.should_run("heartbeat", 300):
            log.info("Core worker heartbeat: outbox=%d, inbox=%d",
                     __import__("src.bot.message_queue", fromlist=["outbox_len"]).outbox_len(),
                     __import__("src.bot.message_queue", fromlist=["inbox_len"]).inbox_len())

        await asyncio.sleep(10)  # tick every 10s


def main():
    global _running

    # Ensure DB tables
    Base.metadata.create_all(bind=engine)

    _write_pid()
    atexit.register(_remove_pid)

    signal.signal(signal.SIGTERM, _signal_handler)
    signal.signal(signal.SIGINT, _signal_handler)

    try:
        asyncio.run(main_loop())
    except KeyboardInterrupt:
        pass
    finally:
        _running = False
        _remove_pid()
        log.info("Core worker stopped.")


if __name__ == "__main__":
    main()
