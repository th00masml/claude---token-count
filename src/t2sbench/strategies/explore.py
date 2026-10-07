"""S3: S2 + dynamic exploration (up to 2 read-only SELECTs with LIMIT 20 via a tool)."""

from __future__ import annotations

from t2sbench.executor import execute
from t2sbench.models.base import Model, ToolSpec, messages_to_dicts
from t2sbench.sqlutil import extract_sql
from t2sbench.strategies.base import Context, Question, Strategy, StrategyOutput, Timer
from t2sbench.strategies.basic import S2
from t2sbench.validator import validate_sql

EXPLORE_TOOL = ToolSpec(
    name="run_sql",
    description=("Run one read-only SQLite SELECT query on the database and see at most 20 rows. "
                 "Use it to inspect values or check a join before writing the final query."),
    parameters={"type": "object", "properties": {"query": {"type": "string", "description": "a single SELECT query"}},
                "required": ["query"]},
)
EXPLORE_SYSTEM_SUFFIX = ("\nYou may call the run_sql tool at most {n} times to inspect the data before you answer. "
                         "Finish with the final query in a ```sql code block.")


def format_rows(columns: list[str], rows: list[tuple], max_rows: int = 20, max_cell: int = 100) -> str:
    def cell(v):
        s = "NULL" if v is None else str(v)
        return s if len(s) <= max_cell else s[: max_cell - 3] + "..."

    lines = [" | ".join(columns)]
    lines += [" | ".join(cell(v) for v in r) for r in rows[:max_rows]]
    if not rows:
        lines.append("(no rows)")
    return "\n".join(lines)


def run_limited(db_path, query: str, limit: int = 20) -> tuple[str, float]:
    """Validate, then run ``query`` wrapped in LIMIT. Returns (observation text, seconds)."""
    v = validate_sql(query)
    if not v.ok:
        return f"Error: {v.error}", 0.0
    wrapped = f"SELECT * FROM ({query.strip().rstrip(';')}) LIMIT {limit}"
    res = execute(db_path, wrapped, max_rows=limit)
    if not res.ok:
        return f"Error: {res.error}", res.elapsed_s
    return format_rows(res.columns, res.rows, limit), res.elapsed_s


class S3(S2):
    name = "S3"
    max_tool_calls = 2

    def run(self, model: Model, q: Question, ctx: Context) -> StrategyOutput:
        system, msgs = self.build(q, ctx, ctx.schema_s2, ctx.fewshot)
        system = (system or "") + EXPLORE_SYSTEM_SUFFIX.format(n=self.max_tool_calls)
        db_timer = Timer()
        queries: list[str] = []

        def handler(_name: str, args: dict) -> str:
            query = str(args.get("query", ""))
            queries.append(query)
            with db_timer:
                obs, _ = run_limited(q.db_path, query)
            return obs

        g, transcript = model.generate_with_tools(msgs, [EXPLORE_TOOL], handler, system, ctx.params,
                                                  max_tool_calls=self.max_tool_calls)
        return StrategyOutput(
            sql=extract_sql(g.text), prompt=[{"role": "system", "content": system}] + messages_to_dicts(transcript),
            responses=[m.content for m in transcript if m.role == "assistant"],
            input_tokens=g.input_tokens, output_tokens=g.output_tokens, model_time_s=g.latency_s,
            db_time_s=db_timer.total, n_calls=g.n_calls, cached_calls=g.cached_calls,
            meta={"explorations": len(queries), "exploration_queries": queries,
                  "tool_mode": "native" if model.native_tools else "text"})
