"""LLM access. Every call takes a `mock` callable used when LLM_MOCK=1, so the graph runs offline.

Default backend: AMD Token Factory (OpenAI-compatible). The whole team shares one key, so calls are capped by a
process-wide semaphore and 429s / 5xx are retried with backoff by the OpenAI client.
"""

import asyncio
import json
import re
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


# Some models (e.g. DeepSeek) write their native tool-call syntax or reasoning tags into plain text when no tool
# is bound, or after the tool budget runs out. Such a reply is not an answer.
_TOOL_MARKUP = re.compile(r"<[｜|]DSML[｜|]|<tool_call>|<\|tool_call|<function_calls>|<invoke name=", re.IGNORECASE)


def clean_reply(reply: str) -> tuple[str, bool]:
    """-> (reply without leaked reasoning, whether it still contains fake tool-call markup)."""
    if "</think>" in reply:
        reply = reply.rsplit("</think>", 1)[1]
    return reply.strip(), bool(_TOOL_MARKUP.search(reply))


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
    final_instruction: str = "Give your final answer now, in the format this step asks for.",
    tool_limits: dict[str, int] | None = None,
    trace: Callable[[dict], Any] | None = None,
) -> str:
    """Plain-text completion, optionally with a bounded number of tool calls first.

    final_instruction: sent when the tool budget runs out; name the format THIS step expects (notes, code...).
    tool_limits: per-tool caps inside the overall budget, e.g. {"run_python": 1}.
    trace: called once with {"reply", "tool_calls"} for debugging (e.g. appended to llm_calls.jsonl).
    """
    if settings.llm_mock:
        return mock()
    llm = get_llm()
    calls: list[dict] = []

    async def done(reply: str, history: list[BaseMessage]) -> str:
        reply, fake_tools = clean_reply(reply)
        if fake_tools or not reply:  # one retry without tools, telling the model to answer directly
            why = "Tool calls are not available now." if fake_tools else "Your reply was empty."
            retry = text_of(await _invoke(llm, [*history, AIMessage(reply or "(empty)"),
                                                HumanMessage(f"{why} {final_instruction}")]))
            reply, _ = clean_reply(retry)
            calls.append({"tool": "(retry)", "args": {"reason": why}})
        if trace:
            trace({"reply": reply, "tool_calls": calls})
        return reply

    if not tools:
        return await done(text_of(await _invoke(llm, msgs)), msgs)

    by_name = {t.name: t for t in tools}
    budget = settings.max_tool_calls if max_tool_calls is None else max_tool_calls
    limits = tool_limits or {}
    per_tool: dict[str, int] = {}
    bound = llm.bind_tools(tools)
    msgs = list(msgs)
    used = 0
    while True:
        ai = await _invoke(bound, msgs)
        msgs.append(ai)
        if not ai.tool_calls:
            return await done(text_of(ai), msgs[:-1])
        for call in ai.tool_calls:
            name = call["name"]
            if used >= budget:
                out = f"Tool budget exhausted. {final_instruction}"
            elif per_tool.get(name, 0) >= limits.get(name, budget):
                used += 1  # blocked calls still count, so a model can't loop on a capped tool
                out = f"{name} limit for this step reached ({limits[name]}). Use what you already know."
            else:
                used += 1
                per_tool[name] = per_tool.get(name, 0) + 1
                calls.append({"tool": name, "args": call["args"]})
                if on_tool_call:
                    on_tool_call(name, call["args"])
                try:
                    out = by_name[name].invoke(call["args"])
                except Exception as e:  # noqa: BLE001 - tool errors go back to the model, not up the stack
                    out = f"error: {e}"
            msgs.append(ToolMessage(content=str(out)[:8000], tool_call_id=call["id"]))
        if used >= budget:
            msgs.append(HumanMessage(f"You have used all tool calls. {final_instruction}"))
            return await done(text_of(await _invoke(llm, msgs)), msgs)


async def followup(msgs: list[BaseMessage], previous: str, instruction: str, mock: Callable[[], str]) -> str:
    """One more turn without tools: the model's previous reply plus a correction (e.g. 'notes only')."""
    if settings.llm_mock:
        return mock()
    reply = text_of(await _invoke(get_llm(), [*msgs, AIMessage(previous), HumanMessage(instruction)]))
    return clean_reply(reply)[0]


def dumps(obj: Any) -> str:
    return json.dumps(obj, indent=2, default=str)
