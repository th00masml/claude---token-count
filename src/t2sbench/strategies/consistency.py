"""S5: S2 + self-consistency: 5 samples at temperature 0.7, majority vote over execution results."""

from __future__ import annotations

from collections import Counter

from t2sbench.evaluate import normalize_rows
from t2sbench.executor import execute
from t2sbench.models.base import Model
from t2sbench.sqlutil import extract_sql
from t2sbench.strategies.base import Context, Question, StrategyOutput, Timer, record_prompt
from t2sbench.strategies.basic import S2


def result_signature(rows) -> tuple:
    """Order-insensitive signature of a result (multiset of normalized tuples)."""
    return tuple(sorted(Counter(normalize_rows(rows)).items(), key=repr))


class S5(S2):
    name = "S5"
    n_samples = 5
    temperature = 0.7

    def run(self, model: Model, q: Question, ctx: Context) -> StrategyOutput:
        system, msgs = self.build(q, ctx, ctx.schema_s2, ctx.fewshot)
        db_timer = Timer()
        out = StrategyOutput(sql=None, prompt=record_prompt(system, msgs), responses=[])
        candidates = []  # (sample index, sql, signature or None)
        for i in range(self.n_samples):
            p = ctx.params.with_(temperature=self.temperature, seed=i)
            g = model.generate(msgs, system, p)
            out.responses.append(g.text)
            out.input_tokens += g.input_tokens
            out.output_tokens += g.output_tokens
            out.model_time_s += g.latency_s
            out.n_calls += 1
            out.cached_calls += int(g.cached)
            sql = extract_sql(g.text)
            sig = None
            if sql:
                with db_timer:
                    res = execute(q.db_path, sql)
                if res.ok:
                    sig = result_signature(res.rows)
            candidates.append((i, sql, sig))
        votes = Counter(sig for _, _, sig in candidates if sig is not None)
        if votes:
            best_sig, n = max(votes.items(), key=lambda kv: (kv[1], -min(i for i, _, s in candidates if s == kv[0])))
            out.sql = next(sql for _, sql, s in candidates if s == best_sig)
            out.meta = {"votes": n, "distinct_results": len(votes),
                        "executable_samples": sum(votes.values())}
        else:  # nothing executes: fall back to the first sample that contains SQL
            out.sql = next((sql for _, sql, _ in candidates if sql), None)
            out.meta = {"votes": 0, "distinct_results": 0, "executable_samples": 0}
        out.db_time_s = db_timer.total
        return out
