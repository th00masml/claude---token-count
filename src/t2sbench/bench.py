"""Benchmark runner: one configuration = model x strategy x dataset (x adapter).

Results go to results/<stage>/<model>[@<adapter>]__<strategy>__<dataset>.parquet, one row
per question with the full prompt, responses, SQL, error, timings and tokens. Re-running
a configuration only processes questions missing from its parquet file (and replays
cached model calls), so interrupted runs resume.
"""

from __future__ import annotations

import json
import logging
import re
import threading
import time
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from t2sbench.datasets.synthetic.generator import load_codes
from t2sbench.evaluate import execution_match
from t2sbench.models.base import Model, ModelError
from t2sbench.models.budget import BudgetExceeded, UnpricedModel
from t2sbench.prompts import default_params, fewshot_text, template_for
from t2sbench.schema import schema_text
from t2sbench.strategies import get_strategy
from t2sbench.strategies.base import Context, Question
from t2sbench.validator import validate_sql

log = logging.getLogger(__name__)

DATASETS = ("bird_ev", "bird_noev", "syn_pl", "syn_en")
SYN_DIR = Path("data/synthetic")
BIRD_SAMPLE = Path("data/bird_dev_sample150.json")
BIRD_ROOT = Path("data/bird")


# --------------------------------------------------------------------------- datasets

def load_dataset(name: str, syn_dir: Path = SYN_DIR, bird_sample: Path = BIRD_SAMPLE,
                 bird_root: Path = BIRD_ROOT) -> list[Question]:
    if name.startswith("bird"):
        from t2sbench.datasets.bird import db_path, locate

        if not bird_sample.exists():
            raise FileNotFoundError(f"{bird_sample} missing: run `t2sbench fetch-bird dev` and `t2sbench sample-bird`")
        _, dbdir = locate("dev", bird_root)
        payload = json.loads(bird_sample.read_text())
        with_ev = name == "bird_ev"
        return [Question(id=q["id"], dataset=name, db_id=q["db_id"], db_path=db_path(dbdir, q["db_id"]),
                         question=q["question"], gold_sql=q["sql"],
                         evidence=(q.get("evidence") or None) if with_ev else None,
                         lang="en", difficulty=q.get("difficulty"))
                for q in payload["questions"]]
    from t2sbench.datasets.synthetic.build import build_all, read_jsonl

    db = syn_dir / "prod.sqlite"
    if not db.exists():
        build_all(syn_dir)
    split, lang = {"syn_pl": ("test", "pl"), "syn_en": ("test", "en"),
                   "syn_val": ("val", None), "syn_train": ("train", None)}[name]
    recs = read_jsonl(syn_dir / f"{split}.jsonl")
    return [Question(id=r["id"], dataset=name, db_id=r["db_id"], db_path=db, question=r["question"],
                     gold_sql=r["sql"], lang=r["lang"], template_id=r["template_id"])
            for r in recs if lang is None or r["lang"] == lang]


# --------------------------------------------------------------------------- runner

@dataclass
class RunConfig:
    stage: str
    model: str
    strategy: str
    dataset: str
    adapter: str | None = None
    qids: list[str] | None = None
    concurrency: int = 4
    allow_unverified_prompts: bool = False

    @property
    def key(self) -> str:
        m = self.model + (f"@{self.adapter}" if self.adapter else "")
        return f"{m}__{self.strategy}__{self.dataset}"


def make_strategy(name: str):
    if name.startswith("SQRL"):
        from t2sbench.sqrl import SqrlAgent, SqrlProtocol, SqrlStrategyAdapter

        return SqrlStrategyAdapter(SqrlAgent(SqrlProtocol.load(), agentic=name == "SQRL-agentic"))
    return get_strategy(name)


def result_path(results_dir: Path, cfg: RunConfig) -> Path:
    return Path(results_dir) / cfg.stage / f"{cfg.key}.parquet"


class ConcurrencyGuard:
    """Watches the vLLM log; on preemption / KV-cache pressure lowers concurrency 4 -> 2 (logged)."""

    PATTERN = re.compile(r"preempt|No available memory for the cache blocks|CUDA out of memory|"
                         r"KV cache .*(insufficient|not enough)", re.IGNORECASE)

    def __init__(self, limit: int, vllm_log: Path | None, events_log: Path = Path("logs/concurrency.log"),
                 floor: int = 2):
        self.limit = limit
        self.vllm_log = vllm_log
        self.events_log = events_log
        self.floor = floor
        self._pos = vllm_log.stat().st_size if vllm_log and vllm_log.exists() else 0
        self.events: list[str] = []

    def poll(self, context: str) -> int:
        if not self.vllm_log or not self.vllm_log.exists() or self.limit <= self.floor:
            return self.limit
        with open(self.vllm_log, encoding="utf-8", errors="replace") as f:
            f.seek(self._pos)
            chunk = f.read()
            self._pos = f.tell()
        m = self.PATTERN.search(chunk)
        if m:
            old, self.limit = self.limit, self.floor
            msg = (f"{datetime.now(timezone.utc).isoformat()} {context}: concurrency {old} -> {self.limit} "
                   f"after vLLM log line matching '{m.group(0)}'")
            self.events.append(msg)
            log.warning(msg)
            self.events_log.parent.mkdir(parents=True, exist_ok=True)
            with open(self.events_log, "a", encoding="utf-8") as f:
                f.write(msg + "\n")
        return self.limit


class ContextFactory:
    def __init__(self, model_spec: dict, dataset: str, allow_unverified: bool):
        self.template = template_for(model_spec, allow_unverified)
        self.params = default_params(self.template)
        self.dataset = dataset
        self.codes = load_codes() if dataset.startswith("syn") else None
        self.fewshot = fewshot_text("bird_ev" if dataset == "bird_ev" else dataset)
        self._schemas: dict[str, tuple[str, str]] = {}
        self._lock = threading.Lock()

    def __call__(self, q: Question) -> Context:
        with self._lock:
            if q.db_id not in self._schemas:
                self._schemas[q.db_id] = (schema_text(q.db_path, False),
                                          schema_text(q.db_path, True, self.codes))
        s0, s2 = self._schemas[q.db_id]
        return Context(self.template, self.params, s0, s2, self.fewshot)


def _precision(spec: dict) -> str:
    if spec["backend"] == "bedrock":
        return "provider"
    if spec.get("quantization"):
        return f"{spec['quantization']} ({spec.get('checkpoint') or 'online'})"
    return spec.get("dtype", "auto")


def run_config(cfg: RunConfig, model: Model, model_spec: dict, results_dir: Path = Path("results"),
               vllm_log: Path | None = None, budget=None, questions: list[Question] | None = None,
               max_failure_rate: float = 0.2) -> pd.DataFrame:
    out_path = result_path(results_dir, cfg)
    questions = questions if questions is not None else load_dataset(cfg.dataset)
    if cfg.qids is not None:
        wanted = set(cfg.qids)
        questions = [q for q in questions if q.id in wanted]
    done = pd.read_parquet(out_path) if out_path.exists() else pd.DataFrame()
    done_ids = set(done["qid"]) if len(done) else set()
    todo = [q for q in questions if q.id not in done_ids]
    log.info("%s: %d questions, %d already done", cfg.key, len(questions), len(questions) - len(todo))
    if not todo:
        return done

    strategy = make_strategy(cfg.strategy)
    ctx_factory = ContextFactory(model_spec, cfg.dataset, cfg.allow_unverified_prompts)
    guard = ConcurrencyGuard(cfg.concurrency, vllm_log)
    gold_cache: dict = {}
    rows: list[dict] = []
    failures: list[str] = []
    fatal: list[BaseException] = []

    def one(q: Question) -> dict | None:
        """Never raises: anything unexpected (schema build, scoring, ...) is a transient failure."""
        try:
            return _one(q)
        except (BudgetExceeded, UnpricedModel) as e:
            fatal.append(e)
        except Exception as e:
            log.exception("question %s failed", q.id)
            failures.append(f"{q.id}: {e}")
        return None

    def _one(q: Question) -> dict | None:
        if fatal:
            return None
        ctx = ctx_factory(q)
        t0 = time.perf_counter()
        try:
            so = strategy.run(model, q, ctx)
            model_error = None
        except (BudgetExceeded, UnpricedModel) as e:
            fatal.append(e)
            return None
        except ModelError as e:  # unrecoverable for this question -> counts as no answer
            so, model_error = None, str(e)
        except Exception as e:  # transient / unexpected: not written, retried on resume
            log.exception("question %s failed", q.id)
            failures.append(f"{q.id}: {e}")
            return None
        ex = execution_match(q.db_path, so.sql if so else None, q.gold_sql, gold_cache=gold_cache)
        wall = time.perf_counter() - t0
        cost = budget.cost(cfg.model, so.input_tokens, so.output_tokens) if (budget and so) else None
        return {
            "qid": q.id, "stage": cfg.stage, "model": cfg.model, "adapter": cfg.adapter,
            "strategy": cfg.strategy, "dataset": cfg.dataset, "backend": model_spec["backend"],
            "precision": _precision(model_spec), "max_model_len": model_spec.get("max_model_len"),
            "quantized": bool(model_spec.get("quantization")),
            "prompt_template": ctx.template.name, "prompt_verified": ctx.template.verified,
            "db_id": q.db_id, "lang": q.lang, "difficulty": q.difficulty, "template_id": q.template_id,
            "question": q.question, "evidence": q.evidence, "gold_sql": q.gold_sql,
            "prompt": json.dumps(so.prompt if so else [], ensure_ascii=False),
            "responses": json.dumps(so.responses if so else [], ensure_ascii=False),
            "pred_sql": so.sql if so else None,
            "correct": ex.correct,
            "syntax_ok": bool(so and so.sql and validate_sql(so.sql).ok),
            "no_answer": not (so and so.sql),
            "exec_ok": ex.pred_ok,
            "error": model_error or ex.pred_error, "error_kind": "model_error" if model_error else ex.pred_error_kind,
            "gold_error": ex.gold_error, "ordered": ex.ordered,
            "model_time_s": so.model_time_s if so else 0.0,
            "db_time_s": (so.db_time_s if so else 0.0) + ex.pred_db_time_s,
            "total_time_s": wall,
            "input_tokens": so.input_tokens if so else 0, "output_tokens": so.output_tokens if so else 0,
            "n_calls": so.n_calls if so else 0, "cached_calls": so.cached_calls if so else 0,
            "cost_usd": cost,
            "meta": json.dumps({**(so.meta if so else {}), "concurrency": guard.limit,
                                "concurrency_events": guard.events}, ensure_ascii=False, default=str),
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

    def flush():
        df = pd.concat([done, pd.DataFrame(rows)], ignore_index=True) if rows else done
        out_path.parent.mkdir(parents=True, exist_ok=True)
        tmp = out_path.with_suffix(".tmp")
        df.to_parquet(tmp, index=False)
        tmp.replace(out_path)
        return df

    pending = list(todo)
    running = set()
    flushed = 0
    try:
        with ThreadPoolExecutor(max_workers=cfg.concurrency) as pool:
            while (pending or running) and not fatal:
                limit = guard.poll(cfg.key)
                while pending and len(running) < limit:
                    running.add(pool.submit(one, pending.pop(0)))
                finished, running = wait(running, return_when=FIRST_COMPLETED)
                for f in finished:
                    r = f.result()
                    if r is not None:
                        rows.append(r)
                if len(rows) - flushed >= 10:
                    flush()
                    flushed = len(rows)
            wait(running)
            for f in running:
                r = f.result()
                if r is not None:
                    rows.append(r)
    finally:
        df = flush()  # whatever finished is kept, even on KeyboardInterrupt
    if fatal:
        raise fatal[0]
    if failures:
        log.warning("%s: %d questions failed with exceptions (will be retried on resume)", cfg.key, len(failures))
        if len(failures) > max_failure_rate * len(todo):
            raise RuntimeError(f"{cfg.key}: {len(failures)}/{len(todo)} questions failed: {failures[:3]}")
    return df


def load_results(results_dir: Path = Path("results"), stage: str | None = None) -> pd.DataFrame:
    pattern = f"{stage}/*.parquet" if stage else "*/*.parquet"
    files = sorted(Path(results_dir).glob(pattern))
    if not files:
        return pd.DataFrame()
    return pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
