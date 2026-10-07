"""S0 (schema only), S1 (S0 + 3 fixed few-shot examples), S2 (S1 + static values)."""

from __future__ import annotations

from t2sbench.models.base import Model
from t2sbench.sqlutil import extract_sql
from t2sbench.strategies.base import Context, Question, Strategy, StrategyOutput, record_prompt


class _SingleShot(Strategy):
    with_fewshot = False

    def schema(self, ctx: Context) -> str:
        return ctx.schema_s2 if self.uses_values else ctx.schema_s0

    def run(self, model: Model, q: Question, ctx: Context) -> StrategyOutput:
        system, msgs = self.build(q, ctx, self.schema(ctx), ctx.fewshot if self.with_fewshot else None)
        g = model.generate(msgs, system, ctx.params)
        return StrategyOutput(sql=extract_sql(g.text), prompt=record_prompt(system, msgs), responses=[g.text],
                              input_tokens=g.input_tokens, output_tokens=g.output_tokens,
                              model_time_s=g.latency_s, n_calls=1, cached_calls=int(g.cached),
                              meta={"stop_reason": g.stop_reason})


class S0(_SingleShot):
    """Schema (DDL) only, zero-shot."""
    name = "S0"
    uses_values = False


class S1(_SingleShot):
    """S0 + 3 fixed few-shot examples."""
    name = "S1"
    uses_values = False
    with_fewshot = True


class S2(_SingleShot):
    """S1 + static values of low-cardinality text columns (+ code meanings on the synthetic DB)."""
    name = "S2"
    uses_values = True
    with_fewshot = True
