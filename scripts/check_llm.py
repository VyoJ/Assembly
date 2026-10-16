"""Smoke-test the configured LLM endpoint (Token Factory by default) before a real run.

    uv run python scripts/check_llm.py

Checks: the model is listed, plain text, structured output (LLM_STRUCTURED_METHOD) and one tool call,
all through agent.llm so retries / concurrency / max_tokens behave exactly as in the pipeline.
"""

import asyncio
import sys

import httpx
from pydantic import BaseModel

from agent import llm
from core.config import settings


class Echo(BaseModel):
    city: str
    population_millions: float


def get_weather(city: str) -> str:
    """Return the current weather for a city."""
    return f"Sunny, 24C in {city}"


def check_models() -> None:
    base = settings.llm_base_url.rstrip("/")
    headers = {"Authorization": f"Bearer {settings.llm_api_key}"}
    r = httpx.get(f"{base}/models", headers=headers, timeout=30)
    r.raise_for_status()
    ids = [m["id"] for m in r.json().get("data", [])]
    print(f"models ({len(ids)}): {', '.join(ids)}")
    if settings.llm_model not in ids:
        sys.exit(f"LLM_MODEL={settings.llm_model!r} is not served; pick one of the above")
    try:
        u = httpx.get(f"{base}/usage", headers=headers, timeout=30).json()
        print(f"usage: rpm_limit={u.get('rpm_limit')} daily ${u.get('daily_cost_used_usd')} used of "
              f"${u.get('daily_cost_limit_usd')} (resets {u.get('daily_reset_at')})")
    except Exception as e:  # noqa: BLE001 - usage is informational only
        print(f"usage: unavailable ({e})")


async def main() -> None:
    if settings.llm_mock:
        sys.exit("LLM_MOCK=1: nothing to check")
    print(f"endpoint={settings.llm_base_url} model={settings.llm_model} structured={settings.llm_structured_method}")
    check_models()

    out = await llm.text(llm.messages("You are terse.", "Say hello in one sentence."), mock=lambda: "")
    print(f"text: {out!r}")
    assert out.strip(), "empty content (raise LLM_MAX_TOKENS or lower LLM_REASONING_EFFORT)"

    obj = await llm.structured(
        Echo, llm.messages("Answer with the requested fields.", "Largest city in Japan and its population?"),
        mock=lambda: Echo(city="", population_millions=0),
    )
    print(f"structured: {obj!r}")

    from langchain_core.tools import StructuredTool

    calls = []
    out = await llm.text(
        llm.messages("Use the tool, then answer in one sentence.", "What's the weather in Lisbon?"),
        mock=lambda: "", tools=[StructuredTool.from_function(get_weather)], max_tool_calls=2,
        on_tool_call=lambda name, args: calls.append((name, args)),
    )
    print(f"tool calls: {calls}\ntool answer: {out!r}")
    assert calls, "model did not call the tool"
    print("OK")


if __name__ == "__main__":
    asyncio.run(main())
