"""Core Worker: runs the signal pipeline, agents, and scheduled jobs.

Independent of Telegram — writes outbound messages to the Redis outbox
queue. Telegram failures do NOT block or crash this worker.

Architecture (Targeted Polling / JIT):

    06:00  Morning Briefing Fetch — one API call per sport to get today's
           schedule (event IDs + commence times).  Results are cached.
    T-60m  JIT Signal Fetch — targeted API call for events kicking off
           in the next hour.  Builds final betting signals.
    T-1m   CLV Closing Line Snapshot — targeted API call to log Pinnacle's
           sharp closing odds at kickoff for continuous learning.

    NO blind 5-minute polling.  The Scout reads from Redis/DB, NOT from
    live API calls.

Entry point:
    python -m src.bot.core_worker
"""
from __future__ import annotations

import asyncio
import atexit
import logging
import os
import signal
import time
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Set, Tuple
from zoneinfo import ZoneInfo

from src.bot.chat_router import chat_router
from src.bot.message_queue import push_outbox, pop_inbox
from src.core.settings import settings
from src.data.models import Base
from src.data.postgres import engine
from src.data.redis_cache import cache

log = logging.getLogger(__name__)
logging.basicConfig(
    level=getattr(logging, settings.log_level, logging.INFO),
    format="%(asctime)s [CORE] %(levelname)s %(name)s: %(message)s",
)

PID_FILE = os.path.join(os.path.dirname(__file__), ".core_worker.pid")

_running = True

# Redis keys for JIT schedule
JIT_SCHEDULE_KEY = "jit:daily_schedule"       # {sport: [{event_id, commence_time, ...}]}
JIT_FETCHED_KEY = "jit:fetched_events"        # set of event_ids already fetched at T-60
JIT_CLV_LOGGED_KEY = "jit:clv_logged_events"  # set of event_ids with CLV logged


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


# ── Helpers ───────────────────────────────────────────────────

def _now_berlin() -> datetime:
    return datetime.now(timezone.utc).astimezone(ZoneInfo("Europe/Berlin"))


def _now_utc() -> datetime:
    return datetime.now(timezone.utc)


# ── Morning Briefing: fetch today's schedule (1 API call/sport) ──

def _run_morning_briefing() -> int:
    """Fetch daily schedule: event IDs and commence_times.

    Uses only the /v4/sports/{key}/odds endpoint with aggressive caching
    (6 hours) so the data is reused all day.  This is the ONE broad fetch
    per day — all later calls are targeted to specific sports.
    """
    from src.core.live_feed import fetch_and_build_signals
    from src.integrations.odds_fetcher import OddsFetcher

    odds = OddsFetcher()

    # Resolve active sports
    from src.core.dynamic_settings import dynamic_settings
    dyn_sports = dynamic_settings.get_active_sports()
    base_sports = dyn_sports or [s.strip() for s in settings.live_sports.split(",") if s.strip()]
    try:
        api_active = odds.get_active_sports_from_api()
    except Exception:
        api_active = []
    expanded = OddsFetcher.resolve_sport_keys(base_sports, api_active)

    schedule: Dict[str, List[Dict]] = {}
    total_events = 0

    for sport in expanded:
        try:
            # Long TTL (6h) — this is the daily schedule fetch
            events = odds.get_sport_odds(
                sport_key=sport, regions="eu", markets="h2h",
                ttl_seconds=6 * 3600,
            )
        except Exception:
            continue
        if not isinstance(events, list):
            continue

        sport_events = []
        for e in events:
            eid = str(e.get("id") or "")
            commence = str(e.get("commence_time") or "")
            if eid and commence:
                sport_events.append({
                    "event_id": eid,
                    "commence_time": commence,
                    "sport": sport,
                    "home_team": e.get("home_team", ""),
                    "away_team": e.get("away_team", ""),
                })
        if sport_events:
            schedule[sport] = sport_events
            total_events += len(sport_events)

    cache.set_json(JIT_SCHEDULE_KEY, schedule, ttl_seconds=18 * 3600)
    log.info("Morning briefing: %d events across %d sports", total_events, len(schedule))

    # Also build initial signals from the cached data
    try:
        items = fetch_and_build_signals()
        push_outbox("text", {
            "text": f"🌅 Morgen-Briefing: {total_events} Events, {len(items)} Signals"
        }, target="broadcast")
    except Exception as exc:
        log.error("Initial signal build failed: %s", exc)
        push_outbox("text", {
            "text": f"🌅 Morgen-Briefing: {total_events} Events geladen"
        }, target="broadcast")

    return total_events


# ── JIT: Targeted pre-kickoff fetch (T-60 min) ──────────────

def _get_events_in_window(
    minutes_from: int, minutes_to: int
) -> List[Tuple[str, Dict]]:
    """Return (sport, event_dict) pairs with kickoff in [now+from, now+to]."""
    schedule = cache.get_json(JIT_SCHEDULE_KEY) or {}
    now = _now_utc()
    window_start = now + timedelta(minutes=minutes_from)
    window_end = now + timedelta(minutes=minutes_to)
    results = []

    for sport, events in schedule.items():
        for ev in events:
            try:
                ct = datetime.fromisoformat(
                    str(ev["commence_time"]).replace("Z", "+00:00")
                )
                if window_start <= ct <= window_end:
                    results.append((sport, ev))
            except (ValueError, TypeError, KeyError):
                continue
    return results


def _run_jit_signal_fetch():
    """Targeted fetch for events kicking off in ~60 minutes.

    Only fetches sports that have imminent events, saving API quota.
    Skips events already fetched this cycle.
    """
    from src.core.live_feed import fetch_and_build_signals
    from src.integrations.odds_fetcher import OddsFetcher

    imminent = _get_events_in_window(0, 75)  # 0-75 minutes from now
    if not imminent:
        return

    already_fetched: Set[str] = set(cache.get_json(JIT_FETCHED_KEY) or [])
    new_events = [(s, e) for s, e in imminent if e["event_id"] not in already_fetched]
    if not new_events:
        return

    # Determine which sports need a fresh fetch
    sports_needed = list({s for s, _ in new_events})
    log.info("JIT T-60: %d new events across %s", len(new_events), sports_needed)

    # Fetch fresh odds for just these sports (short cache)
    odds = OddsFetcher()
    for sport in sports_needed:
        try:
            odds.get_sport_odds(
                sport_key=sport, regions="eu", markets="h2h,spreads,totals",
                ttl_seconds=120,
            )
        except Exception:
            log.warning("JIT fetch failed for %s", sport)

    # Rebuild signals with the freshly cached data
    try:
        items = fetch_and_build_signals()
        if items:
            push_outbox("text", {
                "text": f"🎯 JIT Update (T-60): {len(items)} Signals aktualisiert"
            }, target="broadcast")
    except Exception as exc:
        log.error("JIT signal rebuild failed: %s", exc)

    # Mark events as fetched
    already_fetched.update(e["event_id"] for _, e in new_events)
    cache.set_json(JIT_FETCHED_KEY, sorted(already_fetched), ttl_seconds=24 * 3600)


# ── JIT: CLV Closing Line Snapshot (T-1 min) ────────────────

def _run_clv_snapshot():
    """Log Pinnacle closing lines for events about to kick off.

    Called every tick; only fetches when events are within 2 minutes
    of kickoff and haven't been logged yet.
    """
    from src.core.clv_logger import fetch_and_log_closing_lines

    imminent = _get_events_in_window(-1, 3)  # -1 to +3 minutes
    if not imminent:
        return

    already_logged: Set[str] = set(cache.get_json(JIT_CLV_LOGGED_KEY) or [])
    new_events = [(s, e) for s, e in imminent if e["event_id"] not in already_logged]
    if not new_events:
        return

    sports_needed = list({s for s, _ in new_events})
    log.info("CLV snapshot: %d events at kickoff across %s", len(new_events), sports_needed)

    total_logged = 0
    for sport in sports_needed:
        try:
            n = fetch_and_log_closing_lines(sport)
            total_logged += n
        except Exception:
            log.error("CLV logging failed for %s", sport, exc_info=True)

    if total_logged > 0:
        push_outbox("text", {
            "text": f"📊 CLV Snapshot: {total_logged} Closing Lines geloggt"
        }, target="primary")

    already_logged.update(e["event_id"] for _, e in new_events)
    cache.set_json(JIT_CLV_LOGGED_KEY, sorted(already_logged), ttl_seconds=24 * 3600)


# ── Existing scheduled tasks ─────────────────────────────────

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
        msg = auto_train_all_models()
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
    """Run one Scout -> Analyst -> Executioner cycle.

    The Scout now reads from Redis/DB cache — NOT from live API calls.
    It only triggers the Analyst when cached data shows a steam move.
    """
    from src.agents.orchestrator import AgentOrchestrator
    try:
        orchestrator = AgentOrchestrator(bot=None, chat_id=chat_router.primary_id)
        summary = await orchestrator.run_once()
        if summary.get("scout_alerts", 0) > 0:
            push_outbox("text", {
                "text": (
                    f"🤖 Agent: {summary['scout_alerts']} alerts, "
                    f"{summary['bets_placed']} bets, {summary['bets_skipped']} skipped"
                )
            }, target="broadcast")
    except Exception as exc:
        log.error("Agent cycle failed: %s", exc)


# ── Inbox processor ───────────────────────────────────────────

def _process_inbox():
    """Drain inbox queue and handle user actions."""
    for _ in range(10):
        msg = pop_inbox(timeout=0)
        if msg is None:
            break
        action = msg.get("action", "")
        log.info("Inbox action: %s", action)

        if action == "refresh_data":
            _run_morning_briefing()
        elif action == "retrain":
            _run_weekly_retrain()


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

    log.info("Core worker starting (pid=%d) — JIT polling mode", os.getpid())

    while _running:
        # ── Daily scheduled jobs ──

        # 06:00 — Morning Briefing: fetch schedule + initial signals (1 broad fetch/day)
        if sched.should_run_daily("morning_briefing", 6, 0):
            _run_morning_briefing()

        # 07:00 — Push daily signal summary to Telegram
        if sched.should_run_daily("daily_push", 7, 0):
            _push_daily_signals()

        if sched.should_run_daily("learning_daily", 20, 0):
            _run_learning_status()

        if sched.should_run_daily("api_health", 20, 5):
            _run_api_health()

        if sched.should_run_daily("daily_report", 22, 0):
            _run_daily_performance()

        # Weekly retrain (Saturday = weekday 5)
        if sched.should_run_weekly("retrain_weekly", 5, 3, 15):
            _run_weekly_retrain()

        # ── JIT polling (every 60s) ──
        # These are lightweight: they only check cached schedule timestamps
        # and only make API calls when events are actually imminent.

        if sched.should_run("jit_signal_fetch", 60):
            _run_jit_signal_fetch()

        if sched.should_run("clv_snapshot", 30):
            _run_clv_snapshot()

        # ── Repeating jobs ──

        # Auto-grading every 30 minutes
        if sched.should_run("auto_grading", 1800):
            _run_auto_grading()

        # Agent cycle every 5 minutes — reads from cache only, no API calls
        if sched.should_run("agent_cycle", 300):
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
