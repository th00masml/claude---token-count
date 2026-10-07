"""Wrapper adding the disk cache (and, for Bedrock, the budget) around any Model."""

from __future__ import annotations

from t2sbench.models.base import GenParams, Generation, Message, Model, ToolSpec
from t2sbench.models.budget import Budget
from t2sbench.models.cache import DiskCache, cache_key


def estimate_tokens(messages: list[Message], system: str | None) -> int:
    chars = sum(len(m.content or "") for m in messages) + len(system or "")
    return int(chars / 3.0) + 50  # conservative (over-estimates) for the budget check


class CachedModel(Model):
    def __init__(self, inner: Model, cache: DiskCache, budget: Budget | None = None,
                 identity: str | None = None):
        self.inner = inner
        self.cache = cache
        self.budget = budget if inner.backend == "bedrock" else None
        self.name = inner.name
        self.adapter = inner.adapter
        self.native_tools = inner.native_tools
        self.backend = inner.backend
        # identity = what really answers (resolved Bedrock id / HF repo + precision)
        self.identity = identity or getattr(inner, "model_id", None) or getattr(inner, "served_name", inner.name)

    def chat(self, messages, system=None, params: GenParams | None = None,
             tools: list[ToolSpec] | None = None) -> Generation:
        p = params or GenParams()
        key = cache_key(self.identity, self.adapter, {"messages": messages, "system": system, "tools": tools}, p)
        hit = self.cache.get(key)
        if hit is not None:
            g = Generation(**hit)
            g.cached = True
            return g
        if self.budget is not None:
            self.budget.check(self.name, estimate_tokens(messages, system), p.max_tokens)
        g = self.inner.chat(messages, system, p, tools)
        if self.budget is not None:
            self.budget.record(self.name, g.input_tokens, g.output_tokens)
        self.cache.put(key, g.to_dict())
        return g

    def count_tokens(self, messages, system=None):
        fn = getattr(self.inner, "count_tokens", None)
        return fn(messages, system) if fn else None
