from typing import Any, Dict, Optional

import asyncio
import concurrent.futures

import httpx
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type


class APIFetchError(Exception):
    pass


def _safe_sync_run(coro, timeout: float = 30):
    """Run a coroutine synchronously, safely handling nested event loops.

    Always wraps the coroutine in ``asyncio.wait_for(...)`` to ensure it is
    properly awaited and bounded by a timeout — this prevents the
    ``RuntimeWarning: coroutine was never awaited`` leak that occurred when
    the thread pool or loop lifecycle discarded an un-awaited coroutine.

    If called from within a running event loop (e.g. the Telegram bot's loop),
    spawns a new thread with its own event loop to avoid the
    ``asyncio.run() cannot be called from a running event loop`` error.
    """
    async def _with_timeout():
        return await asyncio.wait_for(coro, timeout=timeout)

    try:
        asyncio.get_running_loop()
        # Already inside an event loop — offload to a thread with its own loop
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
            return pool.submit(asyncio.run, _with_timeout()).result(timeout=timeout + 5)
    except RuntimeError:
        # No running loop — safe to use asyncio.run() directly
        return asyncio.run(_with_timeout())


class AsyncBaseFetcher:
    def __init__(self, base_url: str, headers: Optional[Dict[str, str]] = None, timeout: int = 20):
        self.base_url = base_url.rstrip("/")
        self.headers = headers or {}
        self.timeout = timeout
        self.client = httpx.AsyncClient(base_url=self.base_url, headers=self.headers, timeout=self.timeout)

    async def close(self) -> None:
        """Close the underlying HTTP client to free resources."""
        await self.client.aclose()

    @retry(
        retry=retry_if_exception_type((httpx.HTTPError, APIFetchError)),
        wait=wait_exponential(multiplier=1, min=1, max=30),
        stop=stop_after_attempt(6),
        reraise=True,
    )
    async def get(self, path: str, params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        response = await self.client.get(path.lstrip('/'), params=params)
        if response.status_code in (408, 429, 500, 502, 503, 504):
            raise APIFetchError(f"Transient API error {response.status_code} for {path}")
        response.raise_for_status()
        return response.json()
