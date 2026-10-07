"""Stage 5: SQRL in its native protocol.

The model thinks, then emits actions after </think>: <sql>...</sql> to explore (the result
comes back in the card's <observation> format) or <answer>...</answer> with the final SQL.
We parse the raw response (no vLLM reasoning parser) and look only at the text after the
last </think>. Up to 5 explorations, each validated and run read-only, observation trimmed
to 20 rows; no <answer> after 5 rounds = no answer. SQRL-single disables exploration: the
first turn must contain <answer>.
"""

from __future__ import annotations

import csv
import io
import json
import re
from dataclasses import dataclass, field
from pathlib import Path

import yaml

from t2sbench.executor import execute
from t2sbench.models.base import GenParams, Message, Model, ModelError
from t2sbench.prompts import PLACEHOLDER
from t2sbench.sqlutil import extract_sql, strip_think
from t2sbench.strategies.base import Question, StrategyOutput, Timer, record_prompt
from t2sbench.validator import validate_sql

_SQL = re.compile(r"<sql>(.*?)</sql>", re.IGNORECASE | re.DOTALL)
_ANSWER = re.compile(r"<answer>(.*?)</answer>", re.IGNORECASE | re.DOTALL)


class SqrlConfigError(RuntimeError):
    pass


@dataclass
class SqrlProtocol:
    system: str
    user: str
    observation: str
    card_revision: str | None
    evidence_format: str | None
    single_mode_instruction: str | None
    history: str
    result_format: str
    sampling: dict
    max_model_len: int
    max_explorations: int
    max_rows: int
    source: str | None = None
    extra: dict = field(default_factory=dict)

    @classmethod
    def load(cls, path: str | Path = "config/prompts/sqrl.yaml", require_complete: bool = True) -> "SqrlProtocol":
        raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
        missing = [k for k in ("system", "user", "observation") if not raw.get(k) or PLACEHOLDER in str(raw[k])]
        if require_complete and (missing or not raw.get("card_revision")):
            raise SqrlConfigError(
                f"config/prompts/sqrl.yaml is incomplete (placeholders in {missing or '-'}, "
                f"card_revision={raw.get('card_revision')}). Copy the prompt, user template and "
                "<observation> format verbatim from the model card first.")
        keys = cls.__dataclass_fields__
        return cls(**{k: raw.get(k) for k in keys if k in raw and k != "extra"})

    def params(self) -> GenParams:
        s = self.sampling
        return GenParams(temperature=s["temperature"], top_p=s.get("top_p"), max_tokens=s["max_tokens"])

    def render_user(self, schema: str, q: Question, single: bool) -> str:
        question = q.question
        evidence = q.evidence or ""
        if q.evidence and self.evidence_format:
            question = self.evidence_format.format(question=q.question, evidence=q.evidence)
        text = self.user.format(schema=schema, question=question, evidence=evidence)
        if single and self.single_mode_instruction:
            text += "\n" + self.single_mode_instruction
        return text

    def render_rows(self, columns: list[str], rows: list[tuple]) -> str:
        rows = rows[: self.max_rows]
        if self.result_format == "json":
            return json.dumps([dict(zip(columns, r)) for r in rows], ensure_ascii=False, default=str)
        if self.result_format == "csv":
            buf = io.StringIO()
            w = csv.writer(buf)
            w.writerow(columns)
            w.writerows(rows)
            return buf.getvalue().strip()
        head = "| " + " | ".join(columns) + " |"
        sep = "|" + "---|" * len(columns)
        body = ["| " + " | ".join("NULL" if v is None else str(v) for v in r) + " |" for r in rows]
        return "\n".join([head, sep, *body]) if columns else "(no columns)"


def parse_action(text: str) -> tuple[str, str | None]:
    """('answer', sql) | ('sql', query) | ('none', None), from the text after the last </think>."""
    tail = strip_think(text)
    a = _ANSWER.findall(tail)
    if a:
        inner = a[-1].strip()
        return "answer", extract_sql(inner) or inner.strip().rstrip(";") or None
    s = _SQL.findall(tail)
    if s:
        return "sql", s[-1].strip()
    return "none", None


def observe(proto: SqrlProtocol, db_path: Path, query: str) -> tuple[str, float]:
    v = validate_sql(query)
    if not v.ok:
        return proto.observation.format(result=f"Error: {v.error}"), 0.0
    res = execute(db_path, query, max_rows=proto.max_rows)
    if not res.ok:
        return proto.observation.format(result=f"Error: {res.error}"), res.elapsed_s
    return proto.observation.format(result=proto.render_rows(res.columns, res.rows)), res.elapsed_s


class SqrlAgent:
    """name: 'SQRL-agentic' or 'SQRL-single'."""

    def __init__(self, proto: SqrlProtocol, agentic: bool):
        self.proto = proto
        self.agentic = agentic
        self.name = "SQRL-agentic" if agentic else "SQRL-single"

    def run(self, model: Model, q: Question, schema: str) -> StrategyOutput:
        p = self.proto
        msgs = [Message("user", p.render_user(schema, q, single=not self.agentic))]
        params = p.params()
        db_timer = Timer()
        out = StrategyOutput(sql=None, prompt=[], responses=[])
        explorations = 0
        max_prompt = 0
        overflow = False
        rounds = p.max_explorations + 1 if self.agentic else 1
        for _ in range(rounds):
            # vLLM rejects prompt + max_tokens > max-model-len, so shrink max_tokens to what fits
            counter = getattr(model, "count_tokens", None)
            n_prompt = counter(msgs, p.system) if counter else None
            call_params = params
            if n_prompt is not None:
                max_prompt = max(max_prompt, n_prompt)
                room = p.max_model_len - n_prompt - 16
                if room < 256:
                    overflow = True
                    out.meta["model_error"] = f"prompt of {n_prompt} tokens leaves no room to answer"
                    break
                call_params = params.with_(max_tokens=min(params.max_tokens, room))
            try:
                g = model.generate(msgs, p.system, call_params)
            except ModelError as e:  # prompt + observations no longer fit max-model-len
                overflow = True
                out.meta["model_error"] = str(e)[:500]
                break
            out.responses.append(g.text)
            out.input_tokens += g.input_tokens
            out.output_tokens += g.output_tokens
            out.model_time_s += g.latency_s
            out.n_calls += 1
            out.cached_calls += int(g.cached)
            max_prompt = max(max_prompt, g.input_tokens)
            kind, payload = parse_action(g.text)
            if kind == "answer":
                out.sql = payload
                break
            if kind == "sql" and self.agentic and explorations < p.max_explorations:
                explorations += 1
                with db_timer:
                    obs, _ = observe(p, q.db_path, payload)
                kept = strip_think(g.text) if p.history == "strip_think" else g.text
                msgs = msgs + [Message("assistant", kept.strip()), Message("user", obs)]
                continue
            break  # no action, exploration in single mode, or out of rounds -> no answer
        out.prompt = record_prompt(p.system, msgs)
        out.db_time_s = db_timer.total
        out.meta.update({
            "explorations": explorations, "max_prompt_tokens": max_prompt,
            "context_over_90pct": overflow or max_prompt > 0.9 * p.max_model_len,
            "card_revision": p.card_revision,
        })
        return out


class SqrlStrategyAdapter:
    """Lets bench.run_config drive SqrlAgent like a normal strategy (schema = DDL only)."""

    def __init__(self, agent: SqrlAgent):
        self.agent = agent
        self.name = agent.name

    def run(self, model, q, ctx):
        return self.agent.run(model, q, ctx.schema_s0)
