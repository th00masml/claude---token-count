"""Common model interface.

Backends implement one method, ``chat`` (a single model turn). ``generate`` and
``generate_with_tools`` are built on top of it, so caching and budget control can wrap
``chat`` once and cover every strategy.
"""

from __future__ import annotations

import json
import re
from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import asdict, dataclass, field, replace
from typing import Any


@dataclass
class Message:
    role: str  # "user" | "assistant" | "tool"
    content: str
    tool_call_id: str | None = None
    tool_calls: list[dict] | None = None  # assistant only: [{"id", "name", "arguments"}]


@dataclass
class GenParams:
    temperature: float = 0.0
    top_p: float | None = None
    max_tokens: int = 2048
    stop: list[str] | None = None
    seed: int | None = None  # also distinguishes samples in the cache key (S5)
    extra: dict[str, Any] = field(default_factory=dict)  # backend-specific (top_k, ...)

    def with_(self, **kw) -> "GenParams":
        return replace(self, **kw)


@dataclass
class ToolSpec:
    name: str
    description: str
    parameters: dict  # JSON schema


@dataclass
class Generation:
    text: str
    input_tokens: int = 0
    output_tokens: int = 0
    latency_s: float = 0.0
    tool_calls: list[dict] = field(default_factory=list)
    stop_reason: str | None = None
    cached: bool = False
    n_calls: int = 1
    cached_calls: int = 0

    def to_dict(self) -> dict:
        return asdict(self)


ToolHandler = Callable[[str, dict], str]

TEXT_TOOL_INSTRUCTIONS = """
You can inspect the database before answering. To run an exploratory query, reply with
nothing but one query inside <explore></explore> tags, for example
<explore>SELECT DISTINCT status FROM orders</explore>
You will receive the result (at most 20 rows). You may do this at most {n} times.
When you are ready, reply with the final SQL query in a ```sql code block and no <explore> tag."""

_EXPLORE_RE = re.compile(r"<explore>(.*?)</explore>", re.IGNORECASE | re.DOTALL)


class ModelError(RuntimeError):
    """A backend call failed for a reason that retrying will not fix (bad request, context overflow)."""


class Model(ABC):
    """A chat model behind some endpoint.

    name: config key; adapter: optional LoRA adapter (vLLM); native_tools: whether the
    backend supports function calling (otherwise a text protocol is used for S3).
    """

    name: str = "model"
    adapter: str | None = None
    native_tools: bool = True
    backend: str = "unknown"

    @abstractmethod
    def chat(self, messages: list[Message], system: str | None = None,
             params: GenParams | None = None, tools: list[ToolSpec] | None = None) -> Generation:
        """One model turn."""

    def generate(self, messages: list[Message], system: str | None = None,
                 params: GenParams | None = None) -> Generation:
        return self.chat(messages, system, params)

    def generate_with_tools(self, messages: list[Message], tools: list[ToolSpec],
                            handler: ToolHandler, system: str | None = None,
                            params: GenParams | None = None, max_tool_calls: int = 2,
                            ) -> tuple[Generation, list[Message]]:
        """Tool loop. At most ``max_tool_calls`` tool executions; further calls get an
        error observation and one extra turn to answer. Returns the final generation
        (tokens, latency and call counts summed over turns) and the transcript."""
        if not self.native_tools:
            return self._text_tool_loop(messages, tools, handler, system, params, max_tool_calls)
        transcript = list(messages)
        total = Generation(text="", n_calls=0)
        used = 0
        finished = False
        for _ in range(max_tool_calls + 2):
            g = self.chat(transcript, system, params, tools)
            _accumulate(total, g)
            if not g.tool_calls:
                finished = True
                break
            transcript.append(Message("assistant", g.text, tool_calls=g.tool_calls))
            total.tool_calls.extend(g.tool_calls)
            for tc in g.tool_calls:
                if used < max_tool_calls:
                    obs = handler(tc["name"], tc.get("arguments") or {})
                    used += 1
                    if used == max_tool_calls:
                        obs += "\n(No more exploratory queries are allowed. Give the final SQL query now.)"
                else:
                    obs = "Query limit reached. Give the final SQL query now."
                transcript.append(Message("tool", obs, tool_call_id=tc["id"]))
        # keep the last text even if that turn also asked for a tool: it may hold the final SQL
        total.text = g.text
        total.stop_reason = g.stop_reason
        if finished:
            transcript.append(Message("assistant", g.text))
        return total, transcript

    def _text_tool_loop(self, messages, tools, handler, system, params, max_tool_calls):
        sys = (system or "") + TEXT_TOOL_INSTRUCTIONS.format(n=max_tool_calls)
        tool_name = tools[0].name if tools else "run_sql"
        transcript = list(messages)
        total = Generation(text="", n_calls=0)
        used = 0
        finished = False
        for _ in range(max_tool_calls + 2):
            g = self.chat(transcript, sys, params)
            _accumulate(total, g)
            m = _EXPLORE_RE.search(g.text or "")
            if not m:
                finished = True
                break
            query = m.group(1).strip()
            transcript.append(Message("assistant", g.text))
            if used < max_tool_calls:
                obs = handler(tool_name, {"query": query})
                total.tool_calls.append({"id": f"text-{used}", "name": tool_name, "arguments": {"query": query}})
                used += 1
                if used == max_tool_calls:
                    obs += "\n(No more exploratory queries are allowed. Give the final SQL query now.)"
            else:
                obs = "Query limit reached. Give the final SQL query now."
            transcript.append(Message("user", f"Result of the exploratory query:\n{obs}"))
        total.text = g.text if finished else ""
        total.stop_reason = g.stop_reason
        if finished:
            transcript.append(Message("assistant", g.text))
        return total, transcript


def _accumulate(total: Generation, g: Generation) -> None:
    total.input_tokens += g.input_tokens
    total.output_tokens += g.output_tokens
    total.latency_s += g.latency_s
    total.n_calls += g.n_calls
    total.cached_calls += g.cached_calls + (1 if g.cached else 0)


def messages_to_dicts(messages: list[Message]) -> list[dict]:
    return [{k: v for k, v in asdict(m).items() if v is not None} for m in messages]


def parse_tool_arguments(raw: Any) -> dict:
    if isinstance(raw, dict):
        return raw
    try:
        out = json.loads(raw or "{}")
        return out if isinstance(out, dict) else {"query": str(out)}
    except json.JSONDecodeError:
        return {"query": str(raw)}
