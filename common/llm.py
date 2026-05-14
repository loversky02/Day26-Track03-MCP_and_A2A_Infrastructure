"""Shared LLM factory for all agents.

Supports any OpenAI-compatible API provider via environment variables.
Auto-detects: DeepSeek, OpenAI, Anthropic (via langchain-anthropic), OpenRouter.

Priority: ANTHROPIC_API_KEY > OPENAI_API_KEY > LLM_API_KEY > OPENROUTER_API_KEY
"""

import os

from langchain_openai import ChatOpenAI


def get_llm():
    """Return a ChatOpenAI client auto-detecting the configured provider."""

    # --- Anthropic direct (requires langchain-anthropic) ---
    anthropic_key = os.getenv("ANTHROPIC_API_KEY")
    if anthropic_key and anthropic_key not in ("", "your_key_here"):
        try:
            from langchain_anthropic import ChatAnthropic
            return ChatAnthropic(
                model=os.getenv("ANTHROPIC_MODEL", "claude-sonnet-4-5"),
                api_key=anthropic_key,
            )
        except ImportError:
            pass  # Fall through to OpenAI-compatible

    # --- OpenAI direct ---
    openai_key = os.getenv("OPENAI_API_KEY")
    if openai_key and openai_key not in ("", "your_key_here"):
        return ChatOpenAI(
            model=os.getenv("OPENAI_MODEL", "gpt-4o"),
            api_key=openai_key,
        )

    # --- Generic OpenAI-compatible (DeepSeek, Z.AI, MiniMax, MIMO, etc.) ---
    llm_key = os.getenv("LLM_API_KEY")
    if llm_key and llm_key not in ("", "your_key_here"):
        return ChatOpenAI(
            model=os.getenv("LLM_MODEL", "deepseek-chat"),
            api_key=llm_key,
            base_url=os.getenv("LLM_BASE_URL", "https://api.deepseek.com/v1"),
        )

    # --- OpenRouter (fallback) ---
    router_key = os.getenv("OPENROUTER_API_KEY")
    if router_key and router_key not in ("", "your_key_here"):
        return ChatOpenAI(
            model=os.getenv("OPENROUTER_MODEL", "anthropic/claude-sonnet-4-5"),
            api_key=router_key,
            base_url="https://openrouter.ai/api/v1",
        )

    raise RuntimeError(
        "No API key found. Set one of: LLM_API_KEY, OPENAI_API_KEY, "
        "ANTHROPIC_API_KEY, or OPENROUTER_API_KEY in .env"
    )
