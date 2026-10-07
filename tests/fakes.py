"""Scripted stand-ins for real models and API clients."""

from __future__ import annotations

from types import SimpleNamespace

from t2sbench.models.base import Generation, Model


class FakeModel(Model):
    """responder(messages, system, params, tools) -> str | Generation"""

    def __init__(self, responder, name="fake", backend="vllm", native_tools=True):
        self.responder = responder
        self.name = name
        self.backend = backend
        self.native_tools = native_tools
        self.calls = []

    def chat(self, messages, system=None, params=None, tools=None):
        self.calls.append({"messages": list(messages), "system": system, "params": params, "tools": tools})
        out = self.responder(messages, system, params, tools)
        if isinstance(out, Generation):
            return out
        return Generation(text=out, input_tokens=100, output_tokens=20, latency_s=0.01)


def sql_reply(sql: str) -> str:
    return f"Here you go:\n```sql\n{sql}\n```"


class FakeBedrockClient:
    def __init__(self, responses):
        self.responses = list(responses)
        self.requests = []

    def converse(self, **req):
        self.requests.append(req)
        r = self.responses.pop(0)
        if isinstance(r, Exception):
            raise r
        return r


def fake_openai_client(text="```sql\nSELECT 1\n```", tool_calls=None):
    calls = []

    def create(**kw):
        calls.append(kw)
        msg = SimpleNamespace(content=text, tool_calls=tool_calls)
        return SimpleNamespace(choices=[SimpleNamespace(message=msg, finish_reason="stop")],
                               usage=SimpleNamespace(prompt_tokens=11, completion_tokens=7))

    client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
    return client, calls
