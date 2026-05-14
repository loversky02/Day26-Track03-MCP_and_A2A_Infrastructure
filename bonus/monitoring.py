"""Bonus Challenge 4: Monitoring & Observability.

Cung cấp:
1. Custom @trace decorator — đo thời gian thực thi và log tự động
2. LangSmith integration — tracing & observability cho LangGraph
3. AgentMetrics — thu thập performance metrics cơ bản
"""

from __future__ import annotations

import asyncio
import functools
import inspect
import logging
import os
import sys
import time
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from typing import Callable

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from dotenv import load_dotenv

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("monitoring")


# ── Strategy 1: Custom @trace decorator ───────────────────────────────────

def trace(func: Callable | None = None, *, name: str | None = None):
    """Decorator đo thời gian thực thi và log kết quả.

    Dùng cho cả sync và async functions.

    Usage:
        @trace
        async def my_agent(state): ...

        @trace(name="custom_span_name")
        def my_tool(args): ...
    """
    def decorator(fn: Callable):
        span_name = name or fn.__name__

        @functools.wraps(fn)
        async def async_wrapper(*args, **kwargs):
            start = time.perf_counter()
            logger.info("[%s] START", span_name)
            try:
                result = await fn(*args, **kwargs)
                elapsed = time.perf_counter() - start
                logger.info("[%s] OK  (%.2fs)", span_name, elapsed)
                return result
            except Exception as e:
                elapsed = time.perf_counter() - start
                logger.error("[%s] FAIL (%.2fs): %s", span_name, elapsed, e)
                raise

        @functools.wraps(fn)
        def sync_wrapper(*args, **kwargs):
            start = time.perf_counter()
            logger.info("[%s] START", span_name)
            try:
                result = fn(*args, **kwargs)
                elapsed = time.perf_counter() - start
                logger.info("[%s] OK  (%.2fs)", span_name, elapsed)
                return result
            except Exception as e:
                elapsed = time.perf_counter() - start
                logger.error("[%s] FAIL (%.2fs): %s", span_name, elapsed, e)
                raise

        if inspect.iscoroutinefunction(fn):
            return async_wrapper
        return sync_wrapper

    if func is not None:
        return decorator(func)
    return decorator


# ── Strategy 2: AgentMetrics collector ────────────────────────────────────

@dataclass
class AgentMetrics:
    """Thu thập performance metrics cho từng agent invocation."""

    agent_name: str
    total_calls: int = 0
    total_time: float = 0.0
    min_time: float = float("inf")
    max_time: float = 0.0
    error_count: int = 0
    last_error: str | None = None
    timings: list[float] = field(default_factory=list)

    @property
    def avg_time(self) -> float:
        return self.total_time / self.total_calls if self.total_calls else 0.0

    @property
    def error_rate(self) -> float:
        return self.error_count / self.total_calls if self.total_calls else 0.0

    @property
    def p95_time(self) -> float:
        if not self.timings:
            return 0.0
        sorted_times = sorted(self.timings)
        idx = int(len(sorted_times) * 0.95)
        return sorted_times[min(idx, len(sorted_times) - 1)]

    def record(self, elapsed: float, error: str | None = None):
        self.total_calls += 1
        self.total_time += elapsed
        self.min_time = min(self.min_time, elapsed)
        self.max_time = max(self.max_time, elapsed)
        self.timings.append(elapsed)
        if error:
            self.error_count += 1
            self.last_error = error

    def summary(self) -> str:
        lines = [
            f"--- {self.agent_name} Metrics ---",
            f"  Calls:      {self.total_calls}",
            f"  Avg time:   {self.avg_time:.3f}s",
            f"  P95 time:   {self.p95_time:.3f}s",
            f"  Min/Max:    {self.min_time:.3f}s / {self.max_time:.3f}s",
            f"  Errors:     {self.error_count} ({self.error_rate:.1%})",
        ]
        if self.last_error:
            lines.append(f"  Last error: {self.last_error[:100]}")
        return "\n".join(lines)


# Global metrics registry
_metrics_registry: dict[str, AgentMetrics] = {}


def get_metrics(agent_name: str) -> AgentMetrics:
    """Lấy hoặc tạo metrics collector cho agent."""
    if agent_name not in _metrics_registry:
        _metrics_registry[agent_name] = AgentMetrics(agent_name=agent_name)
    return _metrics_registry[agent_name]


def print_all_metrics():
    """In tất cả metrics đã thu thập."""
    if not _metrics_registry:
        logger.info("No metrics collected yet.")
        return
    for name, m in sorted(_metrics_registry.items()):
        print(m.summary())


# ── Strategy 3: Async context manager cho tracing ─────────────────────────

@asynccontextmanager
async def traced_span(span_name: str, metrics_agent: str | None = None):
    """Context manager đo thời gian 1 span với optional metrics recording.

    Usage:
        async with traced_span("law_agent.invoke", metrics_agent="law_agent"):
            result = await llm.invoke(prompt)
    """
    start = time.perf_counter()
    error: str | None = None
    try:
        logger.info("[%s] START", span_name)
        yield
    except Exception as e:
        error = str(e)
        logger.error("[%s] FAIL: %s", span_name, e)
        raise
    finally:
        elapsed = time.perf_counter() - start
        status = "FAIL" if error else "OK"
        logger.info("[%s] %s  (%.3fs)", span_name, status, elapsed)
        if metrics_agent:
            get_metrics(metrics_agent).record(elapsed, error)


# ── Strategy 4: LangSmith Integration ─────────────────────────────────────

def setup_langsmith():
    """Cấu hình LangSmith tracing (set env vars trước khi import langchain).

    Environment variables cần thiết:
        LANGCHAIN_TRACING_V2=true
        LANGCHAIN_API_KEY=ls_...
        LANGCHAIN_PROJECT=legal-multiagent
    """
    project = os.getenv("LANGCHAIN_PROJECT", "legal-multiagent")
    api_key = os.getenv("LANGCHAIN_API_KEY", "")
    tracing_enabled = os.getenv("LANGCHAIN_TRACING_V2", "").lower() == "true"

    if not tracing_enabled or not api_key:
        logger.info(
            "LangSmith not configured. Set LANGCHAIN_TRACING_V2=true, "
            "LANGCHAIN_API_KEY=ls_..., LANGCHAIN_PROJECT=legal-multiagent"
        )
        return False

    logger.info("LangSmith tracing enabled for project: %s", project)
    logger.info("Dashboard: https://smith.langchain.com")
    return True


# ── Demo ──────────────────────────────────────────────────────────────────

@trace(name="demo_law_agent")
async def demo_law_agent(question: str) -> str:
    """Simulate law agent processing."""
    await asyncio.sleep(0.5)  # Simulate LLM call
    return f"[Phân tích pháp lý cho: {question[:50]}...]"


@trace(name="demo_tax_agent")
async def demo_tax_agent(question: str) -> str:
    """Simulate tax agent processing."""
    await asyncio.sleep(0.3)
    return "[Phân tích thuế...]"


@trace(name="demo_failing_agent")
async def demo_failing_agent() -> str:
    """Simulate agent that fails sometimes."""
    await asyncio.sleep(0.2)
    raise ConnectionError("Registry unavailable — simulated failure")


async def main():
    load_dotenv()

    print("=" * 70)
    print("BONUS 4: MONITORING & OBSERVABILITY DEMO")
    print("=" * 70)

    # Demo 1: @trace decorator
    print("\n--- Demo 1: @trace decorator ---")
    result = await demo_law_agent("Công ty bị rò rỉ dữ liệu thì sao?")
    print(f"  Result: {result}")

    await demo_tax_agent("Hậu quả thuế khi bị phạt GDPR?")

    try:
        await demo_failing_agent()
    except ConnectionError:
        print("  (Caught expected error)")

    # Demo 2: traced_span context manager + metrics
    print("\n--- Demo 2: traced_span + AgentMetrics ---")
    for i in range(5):
        async with traced_span(f"law_agent.call_{i}", metrics_agent="law_agent"):
            await asyncio.sleep(0.1 + (i * 0.05))  # Simulate varied latency

    try:
        async with traced_span("tax_agent.call_err", metrics_agent="tax_agent"):
            await asyncio.sleep(0.1)
            raise ValueError("Tool invocation failed")
    except ValueError:
        pass

    # Demo 3: Print metrics
    print("\n--- Demo 3: Agent Performance Report ---")
    print_all_metrics()

    # Demo 4: LangSmith setup
    print("\n--- Demo 4: LangSmith Setup ---")
    configured = setup_langsmith()
    if not configured:
        print("""
    Để bật LangSmith tracing:
        export LANGCHAIN_TRACING_V2=true
        export LANGCHAIN_API_KEY=ls_...
        export LANGCHAIN_PROJECT=legal-multiagent

    Sau đó tất cả LangChain/LangGraph calls sẽ tự động được trace.
    Dashboard: https://smith.langchain.com
        """)

    print("=" * 70)
    print("KEY TAKEAWAY:")
    print("  - @trace decorator: auto-measure mọi async/sync function")
    print("  - traced_span context manager: measure specific code blocks")
    print("  - AgentMetrics: collect per-agent stats (avg, p95, error rate)")
    print("  - LangSmith: full tracing dashboard cho LangGraph workflows")
    print("=" * 70)


if __name__ == "__main__":
    asyncio.run(main())
