"""Entry point: python -m src.bot

Without arguments, runs the monolithic bot (backward compatible).
Use subcommands for split architecture:
    python -m src.bot.core_worker
    python -m src.bot.telegram_worker
"""
from src.bot.app import main

main()
