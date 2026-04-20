"""
dep_extractor/analysis/url_validator.py
=========================================
Async concurrent URL validation for online dependencies.

Educational: The monolith validated URLs one-by-one in a synchronous loop.
With 20+ git/http dependencies this caused 30+ seconds of dead wall time
(each request blocks until it either succeeds or hits the timeout).

This module uses Python's asyncio + httpx to fire ALL requests concurrently.
A semaphore caps max concurrent connections to avoid overwhelming GitHub's
rate limiter. Total time is now bounded by the SLOWEST single URL, not the
SUM of all timeouts.

Diagram:
    Sync (old):  [URL1─5s] [URL2─5s] [URL3─3s] ... = 13+ seconds total
    Async (new): [URL1─5s]
                 [URL2─5s] ← all running simultaneously
                 [URL3─3s]
                 max = 5 seconds
"""

from __future__ import annotations

import asyncio
import logging
import time
from urllib.parse import urlparse

try:
    import httpx
    _HTTPX_AVAILABLE = True
except ImportError:
    httpx = None  # type: ignore[assignment]
    _HTTPX_AVAILABLE = False

from dep_extractor.models import UrlValidationResult

logger = logging.getLogger(__name__)

_USER_AGENT = "Mozilla/5.0 (ComfyUI-Dependency-Auditor/2.0)"


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def validate_urls_sync(
    urls: list[str],
    timeout_seconds: int = 5,
    max_concurrent: int = 10,
) -> list[UrlValidationResult]:
    """
    Synchronous entry point for the async URL validator.

    This wrapper exists so that the CLI (which runs in a non-async context)
    can call the async validator without managing an event loop itself.

    Args:
        urls:            List of raw requirement URLs to validate.
        timeout_seconds: Per-request timeout in seconds.
        max_concurrent:  Maximum simultaneous HTTP connections.

    Returns:
        List of UrlValidationResult, in the same order as `urls`.
    """
    if not urls:
        return []

    if not _HTTPX_AVAILABLE:
        logger.warning(
            "httpx not installed — URL validation disabled. "
            "Install with: uv pip install httpx"
        )
        return [
            UrlValidationResult(url=u, is_valid=False, error="httpx not installed")
            for u in urls
        ]

    return asyncio.run(
        validate_urls_async(urls, timeout_seconds, max_concurrent)
    )


async def validate_urls_async(
    urls: list[str],
    timeout_seconds: int = 5,
    max_concurrent: int = 10,
) -> list[UrlValidationResult]:
    """
    Validate all URLs concurrently using asyncio + httpx.

    Implementation details:
      - A single shared AsyncClient is reused across all requests (connection pooling).
      - An asyncio.Semaphore caps concurrency to avoid GitHub API rate limiting.
      - Results are gathered in original URL order (asyncio.gather preserves order).
      - HEAD requests are tried first; GET is used as fallback for servers that
        return 405 (Method Not Allowed) to HEAD.

    Args:
        urls:            Raw requirement URL strings (may include git+, #egg= etc.)
        timeout_seconds: Per-request timeout in seconds.
        max_concurrent:  Maximum simultaneous open connections.

    Returns:
        List of UrlValidationResult in the same order as input `urls`.
    """
    semaphore = asyncio.Semaphore(max_concurrent)
    timeout = httpx.Timeout(timeout=float(timeout_seconds))

    async with httpx.AsyncClient(
        timeout=timeout,
        follow_redirects=True,
        headers={"User-Agent": _USER_AGENT},
    ) as client:
        tasks = [
            _validate_single_url(client, semaphore, url, timeout_seconds)
            for url in urls
        ]
        return await asyncio.gather(*tasks)


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

async def _validate_single_url(
    client: "httpx.AsyncClient",
    semaphore: asyncio.Semaphore,
    url: str,
    timeout: int,
) -> UrlValidationResult:
    """
    Validate a single URL, respecting the concurrency semaphore.

    Strategy:
        1. Strip `git+` prefix and `#egg=...` markers — these confuse HTTP clients.
        2. Attempt a HEAD request (lightweight: no body download).
        3. If 405/501, fall back to a GET (some servers reject HEAD).
        4. Record latency for diagnostics.

    Args:
        client:    Shared httpx.AsyncClient instance.
        semaphore: Controls maximum concurrency.
        url:       Raw URL from requirements.txt.
        timeout:   Per-request timeout seconds.

    Returns:
        UrlValidationResult for this URL.
    """
    clean_url = _strip_url_markers(url)

    # Fast-fail for malformed URLs (no scheme = not a network resource)
    parsed = urlparse(clean_url)
    if parsed.scheme not in ("http", "https"):
        return UrlValidationResult(
            url=url,
            is_valid=False,
            error=f"Unsupported scheme: {parsed.scheme!r}",
        )

    async with semaphore:
        start = time.monotonic()
        try:
            response = await client.head(clean_url)
            elapsed_ms = (time.monotonic() - start) * 1000

            # 405 Method Not Allowed → retry with GET
            if response.status_code == 405:
                start = time.monotonic()
                response = await client.get(clean_url)
                elapsed_ms = (time.monotonic() - start) * 1000
                method = "GET"
            else:
                method = "HEAD"

            is_valid = response.status_code < 400
            error = None if is_valid else f"HTTP {response.status_code}"

            return UrlValidationResult(
                url=url,
                is_valid=is_valid,
                error=error,
                latency_ms=round(elapsed_ms, 1),
                method_used=method,
            )

        except httpx.TimeoutException:
            elapsed_ms = (time.monotonic() - start) * 1000
            return UrlValidationResult(
                url=url,
                is_valid=False,
                error=f"Timeout after {timeout}s",
                latency_ms=round(elapsed_ms, 1),
            )
        except Exception as exc:
            elapsed_ms = (time.monotonic() - start) * 1000
            return UrlValidationResult(
                url=url,
                is_valid=False,
                error=str(exc),
                latency_ms=round(elapsed_ms, 1),
            )


def _strip_url_markers(url: str) -> str:
    """
    Prepares a raw requirement URL for HTTP probing by stripping markers.

    Strips:
      - `git+` prefix (git+https://... → https://...)
      - `#egg=name` fragment
      - `@branch` pinned branch reference
      - `.git` suffix (GitHub accepts both with and without)

    Examples:
        git+https://github.com/org/repo.git@main#egg=pkg
        → https://github.com/org/repo.git
    """
    clean = url.strip()
    if clean.startswith("git+"):
        clean = clean[4:]
    # Strip @branch pins (before fragment)
    if "#" in clean:
        clean = clean[: clean.index("#")]
    if "@" in clean:
        # Keep the domain+path, strip the ref
        # e.g. https://github.com/org/repo@main → https://github.com/org/repo
        clean = clean[: clean.rindex("@")]
    return clean
