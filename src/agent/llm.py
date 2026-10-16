"""LLM access. Every call takes a `mock` callable used when LLM_MOCK=1, so the graph runs offline.

Default backend: AMD Token Factory (OpenAI-compatible). The whole team shares one key, so calls are capped by a
process-wide semaphore and 429s / 5xx are retried with backoff by the OpenAI client.
"""

import asyncio
import json
from collections.abc import Callable
from typing import Any

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import BaseTool
from langchain_openai import ChatOpenAI
from pydantic import BaseModel

from core.config import settings

_slots: asyncio.Semaphore | None = None


def _slot() -> asyncio.Semaphore:
    global _slots
    if _slots is None:
        _slots = asyncio.Semaphore(settings.llm_max_concurrency)
    return _slots


def get_llm() -> ChatOpenAI:
    kw = {"reasoning_effort": settings.llm_reasoning_effort} if settings.llm_reasoning_effort else {}
    return ChatOpenAI(
        base_url=settings.llm_base_url,
        model=settings.llm_model,
        api_key=settings.llm_api_key,
        temperature=settings.llm_temperature,
        max_tokens=settings.llm_max_tokens,
        max_retries=settings.llm_max_retries,
        timeout=settings.llm_timeout_s,
        **kw,
    )


async def _invoke(runnable, msgs: list[BaseMessage]):
    async with _slot():
        return await runnable.ainvoke(msgs)


def text_of(msg: AIMessage) -> str:
    if isinstance(msg.content, str):
        return msg.content
    return "".join(b.get("text", "") if isinstance(b, dict) else str(b) for b in msg.content)


def messages(system: str, user: str) -> list[BaseMessage]:
    return [SystemMessage(system), HumanMessage(user)]


async def structured[T: BaseModel](schema: type[T], msgs: list[BaseMessage], mock: Callable[[], T]) -> T:
    if settings.llm_mock:
        return mock()
    llm = get_llm().with_structured_output(schema, method=settings.llm_structured_method)
    return await _invoke(llm, msgs)


async def text(
    msgs: list[BaseMessage],
    mock: Callable[[], str],
    tools: list[BaseTool] | None = None,
    max_tool_calls: int | None = None,
    on_tool_call: Callable[[str, dict], Any] | None = None,
) -> str:
    """Plain-text completion, optionally with a bounded number of read-only tool calls first."""
    if settings.llm_mock:
        return mock()
    llm = get_llm()
    if not tools:
        return text_of(await _invoke(llm, msgs))

    by_name = {t.name: t for t in tools}
    budget = settings.max_tool_calls if max_tool_calls is None else max_tool_calls
    bound = llm.bind_tools(tools)
    msgs = list(msgs)
    used = 0
    while True:
        ai = await _invoke(bound, msgs)
        msgs.append(ai)
        if not ai.tool_calls:
            return text_of(ai)
        for call in ai.tool_calls:
            if used >= budget:
                out = "Tool budget exhausted. Answer now in the required format."
            else:
                used += 1
                if on_tool_call:
                    on_tool_call(call["name"], call["args"])
                try:
                    out = by_name[call["name"]].invoke(call["args"])
                except Exception as e:  # noqa: BLE001 - tool errors go back to the model, not up the stack
                    out = f"error: {e}"
            msgs.append(ToolMessage(content=str(out)[:8000], tool_call_id=call["id"]))
        if used >= budget:
            msgs.append(HumanMessage("You have used all tool calls. Give your final answer now, in the required format."))
            return text_of(await _invoke(llm, msgs))


def dumps(obj: Any) -> str:
    return json.dumps(obj, indent=2, default=str)
