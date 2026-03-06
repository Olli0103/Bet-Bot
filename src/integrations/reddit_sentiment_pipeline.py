from __future__ import annotations

import hashlib
import logging
from typing import Any, Dict, List, Tuple

import feedparser

from src.config.sentiment_sources import SENTIMENT_SOURCES
from src.data.redis_cache import cache
from src.integrations.ollama_sentiment import OllamaSentimentClient

log = logging.getLogger(__name__)

_GUID_TTL = 14 * 24 * 3600
_HEADERS_TTL = 7 * 24 * 3600


def _guid_key(guid: str) -> str:
    h = hashlib.md5((guid or "").encode("utf-8")).hexdigest()[:24]
    return f"reddit:rss:guid:{h}"


def _headers_key(feed_id: str) -> str:
    h = hashlib.md5((feed_id or "").encode("utf-8")).hexdigest()[:24]
    return f"reddit:rss:headers:{h}"


def _get_feed_headers(feed_id: str) -> Tuple[str | None, str | None]:
    data = cache.get_json(_headers_key(feed_id)) or {}
    return data.get("etag"), data.get("modified")


def _save_feed_headers(feed_id: str, etag: str | None, modified: str | None) -> None:
    cache.set_json(_headers_key(feed_id), {"etag": etag, "modified": modified}, ttl_seconds=_HEADERS_TTL)


def _is_guid_processed(guid: str) -> bool:
    return cache.get(_guid_key(guid)) is not None


def _mark_guid_processed(guid: str) -> None:
    cache.set(_guid_key(guid), "1", ttl_seconds=_GUID_TTL)


def _norm_label(label: str) -> float:
    l = (label or "").lower().strip()
    if l == "positive":
        return 1.0
    if l == "negative":
        return -1.0
    return 0.0


def fetch_feed_smart(url: str, feed_id: str) -> Tuple[List[dict], int]:
    """Fetch feed with conditional requests (ETag/Last-Modified), return deltas + status."""
    etag, modified = _get_feed_headers(feed_id)
    feed = feedparser.parse(url, etag=etag, modified=modified)

    status = int(getattr(feed, "status", 200) or 200)
    if status == 304:
        log.info("reddit_rss_304: %s", url)
        return [], 304

    new_etag = getattr(feed, "etag", etag)
    new_modified = getattr(feed, "modified", modified)
    _save_feed_headers(feed_id, new_etag, new_modified)

    out: List[dict] = []
    for e in feed.entries:
        guid = e.get("id") or e.get("link") or ""
        if not guid or _is_guid_processed(guid):
            continue
        _mark_guid_processed(guid)
        out.append(dict(e))

    log.info("reddit_rss_200: %s new=%d", url, len(out))
    return out, status


def run_sentiment_pipeline(max_items_per_run: int = 30) -> Dict[str, Any]:
    """Run tiered RSS pipeline and return weighted sentiment/public-hype aggregates."""
    nlp = OllamaSentimentClient()

    tiers = ["core", "fact_only", "high_noise"]
    processed = 0
    sentiment_delta = 0.0
    public_hype_index = 0.0

    detail: Dict[str, Any] = {
        "processed": 0,
        "tier_counts": {"core": 0, "fact_only": 0, "high_noise": 0},
        "weighted": {"sentiment_delta": 0.0, "public_hype_index": 0.0},
        "feeds": {"http_200": 0, "http_304": 0, "other": 0, "delta_items": 0},
    }

    for tier in tiers:
        if processed >= max_items_per_run:
            break

        cfg = SENTIMENT_SOURCES.get(tier, {})
        weight = float(cfg.get("default_weight", 0.0))
        required_keywords = [str(k).lower() for k in (cfg.get("required_keywords") or [])]

        for feed_cfg in cfg.get("feeds", []):
            if processed >= max_items_per_run:
                break

            url = str(feed_cfg.get("url") or "").strip()
            if not url:
                continue

            deltas, http_status = fetch_feed_smart(url, feed_id=url)
            if http_status == 304:
                detail["feeds"]["http_304"] += 1
            elif http_status == 200:
                detail["feeds"]["http_200"] += 1
            else:
                detail["feeds"]["other"] += 1
            detail["feeds"]["delta_items"] += len(deltas)

            for item in deltas:
                if processed >= max_items_per_run:
                    break

                title = str(item.get("title") or "")
                summary = str(item.get("summary") or "")
                content = f"{title}\n{summary}".strip()
                if not content:
                    continue

                if tier == "fact_only":
                    low = content.lower()
                    if required_keywords and not any(k in low for k in required_keywords):
                        continue

                try:
                    res = nlp.analyze(text=content, context=f"Reddit tier={tier}")
                    raw_score = _norm_label(res.label) * float(res.confidence)
                except Exception as exc:
                    log.warning("reddit_sentiment_llm_error: %s", exc)
                    raw_score = 0.0

                if tier in ("core", "fact_only"):
                    sentiment_delta += raw_score * weight
                else:
                    public_hype_index += raw_score * weight

                detail["tier_counts"][tier] += 1
                processed += 1

    detail["processed"] = processed
    detail["weighted"]["sentiment_delta"] = round(sentiment_delta, 6)
    detail["weighted"]["public_hype_index"] = round(public_hype_index, 6)

    log.info(
        "reddit_pipeline_done: processed=%d sentiment_delta=%.4f public_hype_index=%.4f",
        processed,
        sentiment_delta,
        public_hype_index,
    )
    return detail
