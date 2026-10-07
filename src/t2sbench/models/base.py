"""Common model interface. Concrete clients (Bedrock Converse, OpenAI-compatible vLLM)
arrive in stage 2; strategies only depend on this interface."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class Message:
    role: str  # "user" | "assistant" | "tool"
    content: str
    tool_call_id: str | None = None
    tool_calls: list[dict] | None = None


@dataclass
class GenParams:
    temperature: float = 0.0
    top_p: float | None = None
    max_tokens: int = 1024
    stop: list[str] | None = None
    seed: int | None = None
    extra: dict[str, Any] = field(default_factory=dict)  # model-card specific knobs


@dataclass
class ToolSpec:
    name: str
    description: str
    parameters: dict  # JSON schema


@dataclass
class Generation:
    text: str
    input_tokens: int
    output_tokens: int
    latency_s: float
    tool_calls: list[dict] = field(default_factory=list)  # [{"id", "name", "arguments"}]
    stop_reason: str | None = None
    cached: bool = False
    raw: dict | None = None

    def to_dict(self) -> dict:
        return asdict(self)


ToolHandler = Callable[[str, dict], str]


class Model(ABC):
    """A chat model behind some endpoint. ``name`` is the config key, ``adapter`` an
    optional LoRA adapter name (vLLM ``--enable-lora``)."""

    name: str
    adapter: str | None = None

    @abstractmethod
    def generate(self, messages: list[Message], system: str | None = None,
                 params: GenParams | None = None) -> Generation: ...

    @abstractmethod
    def generate_with_tools(self, messages: list[Message], tools: list[ToolSpec],
                            handler: ToolHandler, system: str | None = None,
                            params: GenParams | None = None, max_tool_calls: int = 2,
                            ) -> tuple[Generation, list[Message]]:
        """Run a tool loop: call the model, execute requested tools via ``handler``,
        feed observations back, stop when the model answers or the budget is spent.
        Returns the final generation (tokens/latency summed over turns) and the transcript."""
