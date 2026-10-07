"""Strategy interface and shared helpers. Each strategy S0-S5 is its own class."""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path

from t2sbench.models.base import GenParams, Message, Model, messages_to_dicts
from t2sbench.prompts import PromptTemplate


@dataclass
class Question:
    id: str
    dataset: str  # bird_ev | bird_noev | syn_pl | syn_en | syn_val ...
    db_id: str
    db_path: Path
    question: str
    gold_sql: str
    evidence: str | None = None  # set only for the "with evidence" BIRD variant
    lang: str = "en"
    difficulty: str | None = None
    template_id: str | None = None


@dataclass
class StrategyOutput:
    sql: str | None  # None = no answer
    prompt: list[dict]  # full prompt(s) sent (system + messages), for the results parquet
    responses: list[str]
    input_tokens: int = 0
    output_tokens: int = 0
    model_time_s: float = 0.0
    db_time_s: float = 0.0  # exploration / repair / voting executions (not the final scoring)
    n_calls: int = 0
    cached_calls: int = 0
    meta: dict = field(default_factory=dict)


@dataclass
class Context:
    """What a strategy needs besides the question: schema text and prompt template."""
    template: PromptTemplate
    params: GenParams
    schema_s0: str
    schema_s2: str
    fewshot: str | None = None


class Strategy(ABC):
    name: str = "S?"
    uses_values: bool = True

    @abstractmethod
    def run(self, model: Model, q: Question, ctx: Context) -> StrategyOutput: ...

    # helpers ---------------------------------------------------------------
    def build(self, q: Question, ctx: Context, schema: str, examples: str | None = None):
        system, user = ctx.template.render(schema, q.question, q.evidence, examples)
        return system, [Message("user", user)]


def record_prompt(system: str | None, messages: list[Message]) -> list[dict]:
    return ([{"role": "system", "content": system}] if system else []) + messages_to_dicts(messages)


class Timer:
    def __init__(self):
        self.total = 0.0

    def __enter__(self):
        self._t0 = time.perf_counter()
        return self

    def __exit__(self, *exc):
        self.total += time.perf_counter() - self._t0
