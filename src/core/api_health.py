from __future__ import annotations

from datetime import datetime, timezone

from src.core.settings import settings
from src.data.redis_cache import cache
from src.integrations.apisports_fetcher import APISportsFetcher
from src.integrations.news_fetcher import NewsFetcher
from src.integrations.odds_fetcher import OddsFetcher


def _ts() -> str:
    return datetime.now(timezone.utc).isoformat()


def _ok(name: str):
    cache.set_json(f"health:{name}", {"ok": True, "last_success": _ts()}, ttl_seconds=7 * 24 * 3600)


def _fail(name: str, error: str):
    prev = cache.get_json(f"health:{name}") or {}
    fails = int(prev.get("fails", 0)) + 1
    payload = {
        "ok": False,
        "fails": fails,
        "last_error": error[:220],
        "last_check": _ts(),
        "last_success": prev.get("last_success", ""),
    }
    cache.set_json(f"health:{name}", payload, ttl_seconds=7 * 24 * 3600)


async def run_api_health_check() -> dict:
    out = {}

    # Odds API
    try:
        if not settings.odds_api_key:
            raise RuntimeError("missing ODDS_API_KEY")
        odds = OddsFetcher()
        sports = await odds.get_sports_async(ttl_seconds=30)
        n = len(sports) if isinstance(sports, list) else 0
        out["oddsapi"] = {"ok": True, "sports": n}
        _ok("oddsapi")
    except Exception as e:
        out["oddsapi"] = {"ok": False, "error": str(e)}
        _fail("oddsapi", str(e))

    # API-Sports
    try:
        if not settings.apisports_api_key:
            raise RuntimeError("missing APISPORTS_API_KEY")
        api = APISportsFetcher()
        today = datetime.now(timezone.utc).date().isoformat()
        payload = await api.get("fixtures", params={"date": today})
        rows = payload.get("response") if isinstance(payload, dict) else []
        out["apisports"] = {"ok": True, "rows": len(rows or [])}
        _ok("apisports")
    except Exception as e:
        out["apisports"] = {"ok": False, "error": str(e)}
        _fail("apisports", str(e))

    # NewsAPI
    try:
        if not settings.newsapi_key:
            raise RuntimeError("missing NEWSAPI_KEY")
        news = NewsFetcher()
        payload = await news.get("top-headlines", params={"q": "sports", "language": "en", "pageSize": 1, "apiKey": settings.newsapi_key})
        total = int(payload.get("totalResults", 0)) if isinstance(payload, dict) else 0
        out["newsapi"] = {"ok": True, "totalResults": total}
        _ok("newsapi")
    except Exception as e:
        out["newsapi"] = {"ok": False, "error": str(e)}
        _fail("newsapi", str(e))

    return out


def format_api_health_report(result: dict) -> str:
    def line(name: str, d: dict) -> str:
        if d.get("ok"):
            extra = ", ".join([f"{k}={v}" for k, v in d.items() if k != "ok"])
            return f"✅ {name}: OK" + (f" ({extra})" if extra else "")
        return f"❌ {name}: {d.get('error', 'unknown error')}"

    return "\n".join([
        "🩺 API Health Check",
        line("OddsAPI", result.get("oddsapi", {})),
        line("API-Sports", result.get("apisports", {})),
        line("NewsAPI", result.get("newsapi", {})),
    ])
