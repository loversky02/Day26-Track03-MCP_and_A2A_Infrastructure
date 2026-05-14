"""Bonus Challenge 3: Retry Logic cho A2A Delegation.

Demonstrate exponential backoff retry khi gọi A2A agent thất bại.
"""

import asyncio
import os
import sys
import time
import logging

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from dotenv import load_dotenv
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception,
    before_sleep_log,
    after_log,
)

import httpx

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("retry_demo")


# ── Strategy 1: tenacity decorator (clean, production-ready) ──────────────

def is_server_error(exception: BaseException) -> bool:
    """Chỉ retry trên lỗi server (5xx) và network errors."""
    if isinstance(exception, httpx.HTTPStatusError):
        return exception.response.status_code >= 500
    if isinstance(exception, (httpx.ConnectError, httpx.TimeoutException,
                               httpx.RemoteProtocolError, httpx.NetworkError)):
        return True
    return False


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=10),
    retry=retry_if_exception(is_server_error),
    before_sleep=before_sleep_log(logger, logging.WARNING),
    after=after_log(logger, logging.INFO),
)
async def delegate_with_retry(endpoint: str) -> str:
    """Gọi agent card endpoint với retry tự động.

    Args:
        endpoint: Base URL của target agent.

    Returns:
        Text content từ agent card.

    Raises:
        httpx.HTTPStatusError: Trên lỗi 4xx (không retry).
        httpx.ConnectError: Sau 3 lần retry thất bại.
    """
    async with httpx.AsyncClient(timeout=10.0) as client:
        url = f"{endpoint}/.well-known/agent.json"
        resp = await client.get(url)
        resp.raise_for_status()
        return resp.text


# ── Strategy 2: Manual retry loop (không cần thư viện ngoài) ───────────────

async def delegate_with_manual_retry(
    endpoint: str,
    max_retries: int = 3,
    base_delay: float = 2.0,
) -> str:
    """Retry thủ công với exponential backoff — không cần tenacity.

    Tốt cho môi trường hạn chế dependency.
    """
    last_exc: Exception | None = None

    for attempt in range(max_retries + 1):
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                url = f"{endpoint}/.well-known/agent.json"
                resp = await client.get(url)
                resp.raise_for_status()
                return resp.text
        except httpx.HTTPStatusError as e:
            if e.response.status_code < 500:
                raise  # 4xx: không retry
            last_exc = e
        except (httpx.ConnectError, httpx.TimeoutException,
                httpx.RemoteProtocolError) as e:
            last_exc = e

        if attempt < max_retries:
            delay = base_delay * (2 ** attempt)
            logger.warning(
                "Attempt %d/%d failed for %s: %s. Retrying in %.1fs...",
                attempt + 1, max_retries + 1, endpoint, last_exc, delay,
            )
            await asyncio.sleep(delay)

    raise last_exc  # type: ignore[misc]


# ── Strategy 3: Wrapper cho A2A delegate function ─────────────────────────

async def delegate_with_retry_wrapper(
    delegate_fn,
    endpoint: str,
    question: str,
    context_id: str,
    trace_id: str,
    depth: int,
    max_retries: int = 3,
) -> str:
    """Wrap A2A delegate() với retry logic.

    Dùng cho production code: gọi delegate_fn với từng tham số,
    tự động retry trên transient failures.

    Args:
        delegate_fn: Hàm delegate gốc (từ common.a2a_client).
        endpoint: URL của target agent.
        question: Câu hỏi.
        context_id: A2A context ID.
        trace_id: Trace ID.
        depth: Độ sâu delegation.
        max_retries: Số lần retry tối đa (default 3).

    Returns:
        Kết quả từ agent, hoặc error message nếu tất cả retry đều fail.
    """
    last_error: str = ""

    for attempt in range(max_retries + 1):
        try:
            return await delegate_fn(
                endpoint=endpoint,
                question=question,
                context_id=context_id,
                trace_id=trace_id,
                depth=depth,
            )
        except httpx.HTTPStatusError as e:
            if e.response.status_code < 500:
                raise  # Client error — không retry
            last_error = f"HTTP {e.response.status_code}"
        except (httpx.ConnectError, httpx.TimeoutException,
                httpx.RemoteProtocolError) as e:
            last_error = str(e)
        except Exception as e:
            last_error = str(e)

        if attempt < max_retries:
            delay = 2.0 * (2 ** attempt)
            logger.warning(
                "Delegate retry %d/%d (depth=%d, %s): %s — waiting %.1fs",
                attempt + 1, max_retries, depth, endpoint, last_error, delay,
            )
            await asyncio.sleep(delay)

    return f"[Retry exhausted after {max_retries + 1} attempts] {last_error}"


# ── Demo ──────────────────────────────────────────────────────────────────

async def main():
    load_dotenv()

    print("=" * 70)
    print("BONUS 3: RETRY LOGIC DEMO")
    print("=" * 70)

    # Test 1: tenacity-based retry (sẽ fail vì không có server ở port này)
    print("\n--- Test 1: tenacity @retry decorator ---")
    bad_endpoint = "http://localhost:19999"

    start = time.time()
    try:
        result = await delegate_with_retry(bad_endpoint)
        print(f"OK: {result[:100]}")
    except Exception as e:
        elapsed = time.time() - start
        print(f"FAIL sau {elapsed:.1f}s: {type(e).__name__}: {e}")
        print("  (3 attempts với exponential backoff: 2s + 4s = ~6s total)")

    # Test 2: Manual retry
    print("\n--- Test 2: Manual retry loop ---")
    start = time.time()
    try:
        result = await delegate_with_manual_retry(bad_endpoint)
        print(f"OK: {result[:100]}")
    except Exception as e:
        elapsed = time.time() - start
        print(f"FAIL sau {elapsed:.1f}s: {type(e).__name__}: {e}")

    # Test 3: Production wrapper pattern
    print("\n--- Test 3: Delegate wrapper pattern ---")
    print("Pattern demo (no actual call):")
    print("""
    # Trong customer_agent/graph.py, thay vì:
    result = await delegate(endpoint, question, context_id, trace_id, depth)

    # Dùng:
    from bonus.retry_logic import delegate_with_retry_wrapper
    result = await delegate_with_retry_wrapper(
        delegate, endpoint, question, context_id, trace_id, depth,
        max_retries=3,
    )
    """)

    print("=" * 70)
    print("KEY TAKEAWAY:")
    print("  - Chỉ retry trên lỗi 5xx và network errors (không retry 4xx)")
    print("  - Exponential backoff: 2s → 4s → 8s (min=2, max=10)")
    print("  - Max 3 attempts (1 gốc + 2 retry)")
    print("  - tenacity decorator cho code sạch; manual loop cho zero-dependency")
    print("=" * 70)


if __name__ == "__main__":
    asyncio.run(main())
