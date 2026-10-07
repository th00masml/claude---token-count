"""S4: S2 + repair loop: parse / validation / execution errors go back to the model, max 2 times."""

from __future__ import annotations

from t2sbench.executor import execute
from t2sbench.models.base import Message, Model
from t2sbench.sqlutil import extract_sql
from t2sbench.strategies.base import Context, Question, StrategyOutput, Timer, record_prompt
from t2sbench.strategies.basic import S2

REPAIR_MSG = ("The query failed with this error:\n{error}\n\n"
              "Fix the query. Reply with the corrected SQLite query in a ```sql code block.")
NO_SQL_MSG = "I could not find an SQL query in your reply. Reply with the SQLite query in a ```sql code block."


class S4(S2):
    name = "S4"
    max_repairs = 2

    def run(self, model: Model, q: Question, ctx: Context) -> StrategyOutput:
        system, msgs = self.build(q, ctx, ctx.schema_s2, ctx.fewshot)
        db_timer = Timer()
        out = StrategyOutput(sql=None, prompt=[], responses=[])
        errors = []
        for attempt in range(self.max_repairs + 1):
            g = model.generate(msgs, system, ctx.params)
            out.responses.append(g.text)
            out.input_tokens += g.input_tokens
            out.output_tokens += g.output_tokens
            out.model_time_s += g.latency_s
            out.n_calls += 1
            out.cached_calls += int(g.cached)
            sql = extract_sql(g.text)
            out.sql = sql
            if sql is None:
                err = "no SQL in reply"
                feedback = NO_SQL_MSG
            else:
                with db_timer:
                    res = execute(q.db_path, sql)
                if res.ok:
                    break
                err = res.error
                feedback = REPAIR_MSG.format(error=res.error)
            errors.append(err)
            if attempt == self.max_repairs:
                break
            msgs = msgs + [Message("assistant", g.text), Message("user", feedback)]
        out.prompt = record_prompt(system, msgs)
        out.db_time_s = db_timer.total
        out.meta = {"repair_rounds": attempt, "errors": errors}
        return out
