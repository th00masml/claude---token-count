"""Strategy interface. S0-S5 implementations arrive in stage 2 (S0, S2) and stage 4 (S1, S3-S5)."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path

from t2sbench.models.base import Model


@dataclass
class Question:
    id: str
    db_id: str
    db_path: Path
    question: str
    gold_sql: str
    evidence: str | None = None  # BIRD only; None = "without evidence" variant
    lang: str = "en"
    dataset: str = "bird"


@dataclass
class StrategyOutput:
    sql: str | None  # None = no answer
    prompt: list[dict]  # full prompt(s) sent, for the results parquet
    responses: list[str]
    input_tokens: int = 0
    output_tokens: int = 0
    model_time_s: float = 0.0
    db_time_s: float = 0.0  # time spent in exploration / repair executions, not the final eval
    n_calls: int = 0
    meta: dict = field(default_factory=dict)


class Strategy(ABC):
    name: str

    @abstractmethod
    def run(self, model: Model, q: Question) -> StrategyOutput: ...
