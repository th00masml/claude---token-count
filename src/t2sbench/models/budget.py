"""Bedrock cost accounting and the hard budget stop (50 USD by default).

Spend is persisted in results/bedrock_spend.json so the limit holds across runs. Before
each uncached Bedrock call the worst-case cost (estimated prompt + max_tokens output) is
checked; the run stops with BudgetExceeded *before* the limit would be crossed.
"""

from __future__ import annotations

import json
import threading
from pathlib import Path

import yaml


class BudgetExceeded(RuntimeError):
    pass


class UnpricedModel(RuntimeError):
    pass


class Budget:
    def __init__(self, prices_path: str | Path = "config/prices.yaml",
                 spend_path: str | Path = "results/bedrock_spend.json",
                 budget_usd: float | None = None, allow_unpriced: bool = False):
        with open(prices_path, encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
        self.prices = cfg.get("per_million_tokens", {})
        self.budget = float(budget_usd if budget_usd is not None else cfg.get("budget_usd", 50.0))
        self.spend_path = Path(spend_path)
        self.allow_unpriced = allow_unpriced
        self._lock = threading.Lock()
        self.reserved = 0.0  # worst-case cost of calls in flight (concurrent threads)
        self.state = {"total_usd": 0.0, "by_model": {}}
        if self.spend_path.exists():
            self.state = json.loads(self.spend_path.read_text())

    @property
    def spent(self) -> float:
        return float(self.state["total_usd"])

    def price(self, model: str) -> tuple[float, float] | None:
        p = self.prices.get(model)
        if not p or p.get("input") is None or p.get("output") is None:
            return None
        return float(p["input"]), float(p["output"])

    def cost(self, model: str, input_tokens: int, output_tokens: int) -> float | None:
        p = self.price(model)
        if p is None:
            return None
        return (input_tokens * p[0] + output_tokens * p[1]) / 1e6

    def check(self, model: str, est_input_tokens: int, max_output_tokens: int) -> float:
        """Reserve the worst-case cost of the next call; returns the reserved amount, which the
        caller must hand back via record() or release(). Calls in flight on other threads
        count against the budget, so concurrency cannot overshoot it."""
        est = self.cost(model, est_input_tokens, max_output_tokens)
        if est is None:
            if self.allow_unpriced:
                return 0.0
            raise UnpricedModel(f"no price for {model} in config/prices.yaml; fill it in before calling it")
        with self._lock:
            if self.spent + self.reserved + est > self.budget:
                raise BudgetExceeded(
                    f"Bedrock spend {self.spent:.4f} USD (+{self.reserved:.4f} in flight) + next call up to "
                    f"{est:.4f} USD would exceed the {self.budget:.2f} USD budget. Stopped; ask before continuing.")
            self.reserved += est
        return est

    def release(self, reserved: float) -> None:
        with self._lock:
            self.reserved = max(0.0, self.reserved - reserved)

    def record(self, model: str, input_tokens: int, output_tokens: int, reserved: float = 0.0) -> float | None:
        c = self.cost(model, input_tokens, output_tokens)
        with self._lock:
            self.reserved = max(0.0, self.reserved - reserved)
            m = self.state["by_model"].setdefault(model, {"usd": 0.0, "input_tokens": 0, "output_tokens": 0, "calls": 0})
            m["input_tokens"] += input_tokens
            m["output_tokens"] += output_tokens
            m["calls"] += 1
            if c is not None:
                m["usd"] += c
                self.state["total_usd"] += c
            self.spend_path.parent.mkdir(parents=True, exist_ok=True)
            tmp = self.spend_path.with_suffix(".tmp")
            tmp.write_text(json.dumps(self.state, indent=2))
            tmp.replace(self.spend_path)
        return c
