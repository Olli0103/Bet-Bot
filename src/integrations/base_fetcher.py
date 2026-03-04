from typing import Any, Dict, Optional

import asyncio
import concurrent.futures
import logging
import os
import ssl

import httpx
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

log = logging.getLogger(__name__)


class APIFetchError(Exception):
    pass


def _insecure_ssl_fallback() -> bool:
    """Check if insecure SSL fallback is enabled via environment."""
    return os.getenv("INSECURE_SSL_FALLBACK", "false").lower() in ("true", "1", "yes")


def build_ssl_context() -> ssl.SSLContext:
    """Build a unified SSL context for all HTTP clients.

    By default uses secure settings. When INSECURE_SSL_FALLBACK=true,
    disables certificate verification (emergency fallback only).
    """
    if _insecure_ssl_fallback():
        ctx = ssl.create_default_context()
        ctx.check_hostname = False
        ctx.verify_mode = ssl.CERT_NONE
        log.warning("SSL: insecure fallback mode active (INSECURE_SSL_FALLBACK=true)")
        return ctx
    return ssl.create_default_context()


def build_httpx_ssl_verify():
    """Return the appropriate verify parameter for httpx clients."""
    if _insecure_ssl_fallback():
        return False
    return True


def _safe_sync_run(coro, timeout: float = 30):
    """Run a coroutine synchronously, safely handling nested event loops.

    Always creates a fresh event loop in a dedicated thread to avoid
    'Event loop is closed' errors from reusing dead loops. The coroutine
    is bounded by a timeout to prevent hangs.
    """
    async def _with_timeout():
        return await asyncio.wait_for(coro, timeout=timeout)

    def _run_in_new_loop():
        """Run the coroutine in a brand-new event loop (thread-safe)."""
        loop = asyncio.new_event_loop()
        try:
            return loop.run_until_complete(_with_timeout())
        finally:
            # Gracefully shutdown any remaining tasks
            try:
                pending = asyncio.all_tasks(loop)
                for task in pending:
                    task.cancel()
                if pending:
                    loop.run_until_complete(asyncio.gather(*pending, return_exceptions=True))
            except Exception:
                pass
            loop.close()

    try:
        asyncio.get_running_loop()
        # Already inside an event loop — offload to a thread with its own loop
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
            future = pool.submit(_run_in_new_loop)
            return future.result(timeout=timeout + 5)
    except RuntimeError:
        # No running loop — run directly in a new loop
        return _run_in_new_loop()


class AsyncBaseFetcher:
    """Base HTTP fetcher using httpx with unified SSL handling.

    Each ``get()`` call creates a fresh ``httpx.AsyncClient`` to avoid
    event-loop lifecycle issues (the client is tied to the loop that
    created it — reusing a client across loops causes 'Event loop is
    closed' errors).
    """

    def __init__(self, base_url: str, headers: Optional[Dict[str, str]] = None, timeout: int = 20):
        self.base_url = base_url.rstrip("/")
        self.headers = headers or {}
        self.timeout = timeout
        self._verify = build_httpx_ssl_verify()

    def _make_client(self) -> httpx.AsyncClient:
        """Create a fresh httpx.AsyncClient bound to the current event loop."""
        return httpx.AsyncClient(
            base_url=self.base_url,
            headers=self.headers,
            timeout=self.timeout,
            verify=self._verify,
        )

    async def close(self) -> None:
        """No-op for backward compatibility — clients are now per-request."""
        pass

    @retry(
        retry=retry_if_exception_type((httpx.HTTPError, APIFetchError)),
        wait=wait_exponential(multiplier=1, min=1, max=30),
        stop=stop_after_attempt(6),
        reraise=True,
    )
    async def get(self, path: str, params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        async with self._make_client() as client:
            response = await client.get(path.lstrip('/'), params=params)
            if response.status_code in (408, 429, 500, 502, 503, 504):
                raise APIFetchError(f"Transient API error {response.status_code} for {path}")
            response.raise_for_status()
            return response.json()
