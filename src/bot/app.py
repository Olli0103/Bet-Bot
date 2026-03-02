from datetime import time as dtime
import signal
import asyncio

from telegram.ext import Application, CommandHandler, MessageHandler, ContextTypes, filters, ApplicationHandlerStop

from src.bot.handlers import (
    start,
    menu_value_bets,
    balance,
    combo_suggestions,
    settings_menu,
    help_menu,
    refresh_data,
    push_daily_signals,
)
from src.core.api_health import format_api_health_report, run_api_health_check
from src.core.autograding import run_auto_grading
from src.core.live_feed import fetch_and_build_signals
from src.core.learning_monitor import learning_health
from src.core.ml_trainer import auto_train_all_models, auto_train_model
from src.core.settings import settings
from src.data.models import Base
from src.data.postgres import engine


def build_app() -> Application:
    Base.metadata.create_all(bind=engine)

    app = Application.builder().token(settings.telegram_bot_token).build()

    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("status", balance))
    app.add_handler(MessageHandler(filters.Regex("^Heutige Value Bets$"), menu_value_bets))
    app.add_handler(MessageHandler(filters.Regex("(?i)^kombi[\s\-–—_]*vorschläge$"), combo_suggestions))
    app.add_handler(MessageHandler(filters.Regex("^Daten aktualisieren$"), refresh_data))
    app.add_handler(MessageHandler(filters.Regex("^Kontostand$"), balance))
    app.add_handler(MessageHandler(filters.Regex("^Einstellungen$"), settings_menu))
    app.add_handler(MessageHandler(filters.Regex("^Hilfe$"), help_menu))

    return app


async def scheduled_fetch(context: ContextTypes.DEFAULT_TYPE):
    try:
        items = await __import__('asyncio').to_thread(fetch_and_build_signals, 1000)
        chat_id = settings.telegram_chat_id
        if chat_id:
            await context.bot.send_message(chat_id=chat_id, text=f"📡 Datenupdate fertig: {len(items)} Signals")
            push = bool((context.job.data or {}).get("push", False)) if context.job else False
            if push:
                await push_daily_signals(context.bot, str(chat_id))
    except Exception:
        pass


async def scheduled_grading(context: ContextTypes.DEFAULT_TYPE):
    try:
        settled = await run_auto_grading()
        if settled > 0 and settings.telegram_chat_id:
            await context.bot.send_message(chat_id=settings.telegram_chat_id, text=f"✅ Auto-Grading: {settled} Wetten bewertet")
    except Exception:
        pass


async def morning_briefing(context: ContextTypes.DEFAULT_TYPE):
    chat_id = settings.telegram_chat_id
    if not chat_id:
        return
    await context.bot.send_message(chat_id=chat_id, text="🌅 Morning Briefing bereit. Tippe auf Heutige Value Bets.")


async def learning_status_push(context: ContextTypes.DEFAULT_TYPE):
    try:
        h = learning_health()
        if settings.telegram_chat_id:
            await context.bot.send_message(
                chat_id=settings.telegram_chat_id,
                text=(
                    f"🧠 Learning Check\n"
                    f"Total: {h['total']} | Settled: {h['settled']} | Open: {h['open']}\n"
                    f"W/L: {h['wins']}/{h['losses']} | Hit: {h['hit_rate_pct']}% | PnL: {h['pnl']:.2f} EUR"
                ),
            )
    except Exception:
        pass


async def weekly_retrain(context: ContextTypes.DEFAULT_TYPE):
    try:
        msg = await __import__('asyncio').to_thread(auto_train_all_models, 100)
        if settings.telegram_chat_id:
            await context.bot.send_message(chat_id=settings.telegram_chat_id, text=f"📈 Weekly Retrain: {msg}")
    except Exception:
        pass


async def api_health_push(context: ContextTypes.DEFAULT_TYPE):
    try:
        result = await run_api_health_check()
        if settings.telegram_chat_id:
            await context.bot.send_message(chat_id=settings.telegram_chat_id, text=format_api_health_report(result))
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
    app = build_app()
    _global_app = app
    
    # Add error handler
    app.add_error_handler(error_handler)

    print("Starting polling...", flush=True)
    
    app.job_queue.run_daily(scheduled_fetch, time=dtime(hour=7, minute=0), name="fetch_0700", data={"push": True})
    app.job_queue.run_daily(scheduled_fetch, time=dtime(hour=13, minute=0), name="fetch_1300", data={"push": False})
    app.job_queue.run_daily(morning_briefing, time=dtime(hour=6, minute=30), name="morning_briefing")

    # frequent grading sweep
    app.job_queue.run_repeating(scheduled_grading, interval=1800, first=300, name="auto_grading")

    # regular learning checks + API health + weekly retrain
    app.job_queue.run_daily(learning_status_push, time=dtime(hour=20, minute=0), name="learning_daily")
    app.job_queue.run_daily(api_health_push, time=dtime(hour=20, minute=5), name="api_health_daily")
    app.job_queue.run_daily(weekly_retrain, time=dtime(hour=3, minute=15), days=(6,), name="retrain_weekly")

    app.run_polling(close_loop=False)


if __name__ == "__main__":
    main()
