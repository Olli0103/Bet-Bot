#!/usr/bin/env python3
import atexit
import fcntl
from pathlib import Path

# Load .env EARLY — before any module imports that read env vars (SSL, etc.)
from dotenv import load_dotenv
load_dotenv(override=True)

from src.bot.app import main


LOCK_PATH = Path(__file__).resolve().parents[1] / "run" / "run_bot.lock"
LOCK_PATH.parent.mkdir(parents=True, exist_ok=True)
_lock_file = open(LOCK_PATH, "w")
try:
    fcntl.flock(_lock_file.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
except BlockingIOError:
    raise SystemExit("run_bot already running (lock held)")


def _cleanup():
    try:
        fcntl.flock(_lock_file.fileno(), fcntl.LOCK_UN)
    except Exception:
        pass


atexit.register(_cleanup)


if __name__ == "__main__":
    main()
