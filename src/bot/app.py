from datetime import time as dtime
import atexit
import os
import signal
import asyncio

from telegram.ext import Application, CallbackQueryHandler, CommandHandler, MessageHandler, ContextTypes, filters, ApplicationHandlerStop

from src.bot.chat_router import chat_router
from src.bot.handlers import (
    start,
    menu_value_bets,
    balance,
    callback_handler,
    combo_suggestions,
    settings_menu,
    help_menu,
    refresh_data,
    push_daily_signals,
    agentic_chat,
)
from src.agents.orchestrator import AgentOrchestrator
from src.core.api_health import format_api_health_report, run_api_health_check
from src.core.autograding import run_auto_grading
from src.core.live_feed import fetch_and_build_signals, run_enrichment_pass
from src.core.learning_monitor import learning_health
from src.core.ml_trainer import auto_train_all_models, auto_train_model
from src.core.performance_monitor import PerformanceMonitor
from src.core.settings import settings
from src.data.models import Base
from src.data.postgres import engine

# ---------------------------------------------------------------------------
# Singleton process guard (PID file)
# ---------------------------------------------------------------------------
PID_FILE = os.path.join(os.path.dirname(__file__), ".bot.pid")


def _kill_stale_bot() -> None:
    """If a PID file exists, check whether the old process is still alive
    and kill it before we start a new instance."""
    if not os.path.isfile(PID_FILE):
        return
    try:
        with open(PID_FILE) as f:
            old_pid = int(f.read().strip())
    except (ValueError, OSError):
        return

    if old_pid == os.getpid():
        return

    # Check if the old process is still alive
    try:
        os.kill(old_pid, 0)  # signal 0 = existence check
    except OSError:
        # Process is already dead — stale PID file
        print(f"Stale PID file found (pid={old_pid}, already dead). Cleaning up.")
        return

    # Old process is alive — terminate it
    print(f"Killing stale bot instance (pid={old_pid})...")
    try:
        os.kill(old_pid, signal.SIGTERM)
        # Give it a moment to shut down gracefully
        import time
        for _ in range(10):
            time.sleep(0.5)
            try:
                os.kill(old_pid, 0)
            except OSError:
                break
        else:
            # Still alive after 5s — force kill
            print(f"Force-killing stale instance (pid={old_pid})...")
            os.kill(old_pid, signal.SIGKILL)
    except OSError:
        pass


def _write_pid() -> None:
    """Write our PID to the lock file."""
    with open(PID_FILE, "w") as f:
        f.write(str(os.getpid()))


def _remove_pid() -> None:
    """Remove the PID file on exit."""
    try:
        if os.path.isfile(PID_FILE):
            with open(PID_FILE) as f:
                if f.read().strip() == str(os.getpid()):
                    os.remove(PID_FILE)
    except OSError:
        pass


def build_app() -> Application:
    Base.metadata.create_all(bind=engine)

    app = Application.builder().token(settings.telegram_bot_token).build()

    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("status", balance))
    app.add_handler(MessageHandler(filters.Regex("^Heutige Top 10 Einzelwetten$"), menu_value_bets))
    app.add_handler(MessageHandler(filters.Regex("^10/20/30 Kombis$"), combo_suggestions))
    app.add_handler(MessageHandler(filters.Regex("^Daten aktualisieren$"), refresh_data))
    app.add_handler(MessageHandler(filters.Regex("^Kontostand$"), balance))
    app.add_handler(MessageHandler(filters.Regex("^Einstellungen$"), settings_menu))
    app.add_handler(MessageHandler(filters.Regex("^Hilfe$"), help_menu))

    # Inline keyboard callbacks (pagination, mark-as-placed, agent actions)
    app.add_handler(CallbackQueryHandler(callback_handler))

    # Agentic chat: low-priority fallback for free-text questions
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, agentic_chat))

    return app


async def _broadcast(bot, text: str, target: str = "broadcast"):
    """Send text to all applicable chat IDs."""
    ids = chat_router.broadcast_ids() if target == "broadcast" else chat_router.primary_only_ids()
    for cid in ids:
        try:
            await bot.send_message(chat_id=cid, text=text)
        except Exception as exc:
            import logging
            logging.getLogger(__name__).warning("Send to %s failed: %s", cid, exc)


async def scheduled_fetch(context: ContextTypes.DEFAULT_TYPE):
    import logging
    _log = logging.getLogger(__name__)
    try:
        # Phase 1: Core fetch (fast, no enrichment)
        items = await asyncio.to_thread(fetch_and_build_signals, skip_enrichment=True)
        await _broadcast(context.bot, f"📡 Data update done (core): {len(items)} Signals")
        push = bool((context.job.data or {}).get("push", False)) if context.job else False
        if push:
            for cid in chat_router.broadcast_ids():
                await push_daily_signals(context.bot, cid)

        # Phase 2: Enrichment in background
        try:
            result = await asyncio.to_thread(run_enrichment_pass)
            if result.get("status") == "done":
                n_sig = result.get("signals", 0)
                n_ev = result.get("events", 0)
                await _broadcast(
                    context.bot,
                    f"🧠 Enrichment done: {n_sig} Signals aus {n_ev} Events angereichert",
                )
            else:
                _log.info("Enrichment skipped: %s", result.get("reason"))
        except Exception as exc:
            _log.warning("Background enrichment failed: %s", exc)
    except Exception:
        pass


async def scheduled_grading(context: ContextTypes.DEFAULT_TYPE):
    try:
        settled = await run_auto_grading()
        if settled > 0:
            await _broadcast(context.bot, f"✅ Auto-Grading: {settled} Wetten bewertet")
    except Exception:
        pass


async def morning_briefing(context: ContextTypes.DEFAULT_TYPE):
    await _broadcast(context.bot, "🌅 Morning Briefing bereit. Tippe auf Heutige Top 10 Einzelwetten.")


async def learning_status_push(context: ContextTypes.DEFAULT_TYPE):
    try:
        h = learning_health()
        await _broadcast(context.bot, (
            f"🧠 Learning Check\n"
            f"Total: {h['total']} | Settled: {h['settled']} | Open: {h['open']}\n"
            f"W/L: {h['wins']}/{h['losses']} | Hit: {h['hit_rate_pct']}% | PnL: {h['pnl']:.2f} EUR"
        ))
    except Exception:
        pass


async def weekly_retrain(context: ContextTypes.DEFAULT_TYPE):
    try:
        msg = await asyncio.to_thread(auto_train_all_models, 100)
        await _broadcast(context.bot, f"📈 Weekly Retrain: {msg}", target="primary")
    except Exception:
        pass


async def api_health_push(context: ContextTypes.DEFAULT_TYPE):
    try:
        result = await run_api_health_check()
        await _broadcast(context.bot, format_api_health_report(result), target="primary")
    except Exception:
        pass


_agent_cycle_counter = 0


async def agent_cycle(context: ContextTypes.DEFAULT_TYPE):
    """Run one Scout -> Analyst -> Executioner cycle via the AgentOrchestrator."""
    global _agent_cycle_counter
    _agent_cycle_counter += 1

    try:
        orchestrator = AgentOrchestrator(bot=context.bot, chat_id=chat_router.primary_id)
        is_fast_mode = orchestrator._has_imminent_kickoffs()

        if not is_fast_mode and (_agent_cycle_counter % 5) != 0:
            return

        summary = await orchestrator.run_once()
        if summary.get("scout_alerts", 0) > 0:
            mode_label = "FAST" if is_fast_mode else "NORMAL"
            await _broadcast(context.bot, (
                f"🤖 Agent [{mode_label}]: {summary['scout_alerts']} alerts, "
                f"{summary['bets_placed']} bets, {summary['bets_skipped']} skipped"
            ))
    except Exception:
        import traceback
        traceback.print_exc()


async def daily_performance_report(context: ContextTypes.DEFAULT_TYPE):
    """Send daily performance report at 22:00."""
    try:
        monitor = PerformanceMonitor()
        report = monitor.generate_report()
        await _broadcast(context.bot, report)
    except Exception:
        pass


# Global app reference for signal handler
_global_app = None


async def shutdown():
    """Clean shutdown of the bot"""
    global _global_app
    if _global_app:
        await _global_app.stop()
        print("Bot cleanly stopped")


async def error_handler(update, context):
    import traceback
    print(f"ERROR in handler: {context.error}")
    traceback.print_exc()

def main():
    global _global_app

    # Singleton guard: kill any stale bot process before starting
    _kill_stale_bot()
    _write_pid()
    atexit.register(_remove_pid)

    app = build_app()
    _global_app = app

    # Add error handler
    app.add_error_handler(error_handler)

    print(f"Starting polling (pid={os.getpid()})...", flush=True)
    
    app.job_queue.run_daily(scheduled_fetch, time=dtime(hour=7, minute=0), name="fetch_0700", data={"push": True})
    app.job_queue.run_daily(scheduled_fetch, time=dtime(hour=13, minute=0), name="fetch_1300", data={"push": False})
    app.job_queue.run_daily(morning_briefing, time=dtime(hour=6, minute=30), name="morning_briefing")

    # frequent grading sweep
    app.job_queue.run_repeating(scheduled_grading, interval=1800, first=300, name="auto_grading")

    # regular learning checks + API health + weekly retrain
    app.job_queue.run_daily(learning_status_push, time=dtime(hour=20, minute=0), name="learning_daily")
    app.job_queue.run_daily(api_health_push, time=dtime(hour=20, minute=5), name="api_health_daily")
    app.job_queue.run_daily(weekly_retrain, time=dtime(hour=3, minute=15), days=(6,), name="retrain_weekly")

    # Agent orchestrator: polls every 60s, but internally skips to ~5min
    # unless events are kicking off within the next hour (adaptive fast mode)
    app.job_queue.run_repeating(agent_cycle, interval=60, first=60, name="agent_cycle")

    # Daily performance report + self-evaluation at 22:00
    app.job_queue.run_daily(daily_performance_report, time=dtime(hour=22, minute=0), name="daily_report")

    app.run_polling(close_loop=False)


if __name__ == "__main__":
    main()
