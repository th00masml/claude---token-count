"""OpenAI-compatible chat endpoint (local vLLM)."""

from __future__ import annotations

import json
import time

from t2sbench.models.base import (GenParams, Generation, Message, Model, ModelError, ToolSpec,
                                  parse_tool_arguments)


class OpenAICompatModel(Model):
    backend = "vllm"

    def __init__(self, name: str, base_url: str, served_name: str | None = None,
                 adapter: str | None = None, client=None, native_tools: bool = True,
                 timeout_s: float = 900):
        self.name = name
        self.base_url = base_url
        self.served_name = served_name or name
        self.adapter = adapter
        self.native_tools = native_tools
        if client is None:
            from openai import OpenAI

            client = OpenAI(base_url=base_url, api_key="EMPTY", timeout=timeout_s, max_retries=3)
        self.client = client

    @staticmethod
    def _convert(messages: list[Message], system: str | None) -> list[dict]:
        out = [{"role": "system", "content": system}] if system else []
        for m in messages:
            if m.role == "tool":
                out.append({"role": "tool", "tool_call_id": m.tool_call_id, "content": m.content})
            elif m.role == "assistant" and m.tool_calls:
                out.append({"role": "assistant", "content": m.content or None, "tool_calls": [
                    {"id": tc["id"], "type": "function",
                     "function": {"name": tc["name"], "arguments": json.dumps(tc.get("arguments") or {})}}
                    for tc in m.tool_calls]})
            else:
                out.append({"role": m.role, "content": m.content})
        return out

    def chat(self, messages, system=None, params=None, tools: list[ToolSpec] | None = None) -> Generation:
        p = params or GenParams()
        kw = {"model": self.adapter or self.served_name, "messages": self._convert(messages, system),
              "temperature": p.temperature, "max_tokens": p.max_tokens}
        if p.top_p is not None:
            kw["top_p"] = p.top_p
        if p.stop:
            kw["stop"] = p.stop
        if p.seed is not None:
            kw["seed"] = p.seed
        if p.extra:
            kw["extra_body"] = dict(p.extra)
        if tools:
            kw["tools"] = [{"type": "function", "function": {
                "name": t.name, "description": t.description, "parameters": t.parameters}} for t in tools]
            kw["tool_choice"] = "auto"
        t0 = time.perf_counter()
        try:
            resp = self.client.chat.completions.create(**kw)
        except Exception as e:
            status = getattr(e, "status_code", None)
            if status == 400:  # e.g. prompt longer than max-model-len
                raise ModelError(str(e)) from e
            raise
        latency = time.perf_counter() - t0
        choice = resp.choices[0]
        msg = choice.message
        calls = [{"id": tc.id, "name": tc.function.name, "arguments": parse_tool_arguments(tc.function.arguments)}
                 for tc in (msg.tool_calls or [])]
        usage = resp.usage
        return Generation(text=msg.content or "", input_tokens=int(getattr(usage, "prompt_tokens", 0) or 0),
                          output_tokens=int(getattr(usage, "completion_tokens", 0) or 0),
                          latency_s=latency, tool_calls=calls, stop_reason=choice.finish_reason)

    def count_tokens(self, messages: list[Message], system: str | None = None) -> int | None:
        """Prompt length via vLLM's /tokenize (chat template applied). None if unavailable."""
        import httpx

        root = self.base_url.rstrip("/").removesuffix("/v1")
        try:
            r = httpx.post(f"{root}/tokenize", timeout=60, json={
                "model": self.adapter or self.served_name, "messages": self._convert(messages, system),
                "add_generation_prompt": True})
            r.raise_for_status()
            return int(r.json()["count"])
        except (httpx.HTTPError, KeyError, ValueError):
            return None
