import json

import pandas as pd
import pytest

from fakes import FakeModel, sql_reply
from t2sbench.bench import ConcurrencyGuard, RunConfig, load_dataset, result_path, run_config
from t2sbench.models.budget import BudgetExceeded

SPEC = {"name": "fake", "backend": "vllm", "dtype": "bfloat16", "max_model_len": 16384}


def gold_answering_model(questions):
    by_q = {q.question: q.gold_sql for q in questions}

    def respond(messages, system, params, tools):
        text = messages[0].content
        for qtext, sql in by_q.items():
            if qtext in text:
                return sql_reply(sql)
        return "no idea"
    return respond


def test_run_config_writes_parquet_and_resumes(synth, tmp_path):
    qs = load_dataset("syn_en", syn_dir=synth[0])[:6]
    model = FakeModel(gold_answering_model(qs[:4]))  # last two get no answer
    cfg = RunConfig("stage3", "fake", "S2", "syn_en", concurrency=2)
    df = run_config(cfg, model, SPEC, tmp_path, questions=qs[:3])
    assert len(df) == 3 and df.correct.all() and len(model.calls) == 3
    df = run_config(cfg, model, SPEC, tmp_path, questions=qs)
    assert len(df) == 6 and len(model.calls) == 6  # only the 3 missing were run
    assert df.correct.sum() == 4 and df.no_answer.sum() == 2
    saved = pd.read_parquet(result_path(tmp_path, cfg))
    row = saved.iloc[0]
    assert json.loads(row.prompt)[0]["role"] == "system" and row.precision == "bfloat16"
    for col in ("pred_sql", "gold_sql", "error", "model_time_s", "db_time_s", "total_time_s",
                "input_tokens", "output_tokens", "syntax_ok", "exec_ok"):
        assert col in saved.columns
    run_config(cfg, model, SPEC, tmp_path, questions=qs)
    assert len(model.calls) == 6


def test_qids_filter(synth, tmp_path):
    qs = load_dataset("syn_pl", syn_dir=synth[0])
    model = FakeModel(lambda *a: sql_reply("SELECT 1"))
    cfg = RunConfig("smoke", "fake", "S0", "syn_pl", qids=[qs[0].id, qs[5].id])
    df = run_config(cfg, model, SPEC, tmp_path, questions=qs)
    assert sorted(df.qid) == sorted([qs[0].id, qs[5].id])


def test_budget_stop_propagates(synth, tmp_path):
    qs = load_dataset("syn_pl", syn_dir=synth[0])[:3]

    def respond(*a):
        raise BudgetExceeded("stop")

    with pytest.raises(BudgetExceeded):
        run_config(RunConfig("stage3", "fake", "S0", "syn_pl"), FakeModel(respond), SPEC, tmp_path, questions=qs)


def test_transient_errors_not_recorded(synth, tmp_path):
    qs = load_dataset("syn_pl", syn_dir=synth[0])[:10]
    n = {"i": 0}

    def respond(*a):
        n["i"] += 1
        if n["i"] == 2:
            raise ConnectionError("network")
        return sql_reply("SELECT 1")

    df = run_config(RunConfig("stage3", "fake", "S0", "syn_pl", concurrency=1), FakeModel(respond), SPEC, tmp_path,
                    questions=qs)
    assert len(df) == 9  # the failed question will be retried on the next run


def test_concurrency_guard(tmp_path):
    log = tmp_path / "vllm.log"
    log.write_text("INFO startup\n")
    g = ConcurrencyGuard(4, log, events_log=tmp_path / "events.log")
    assert g.poll("x") == 4
    with open(log, "a") as f:
        f.write("WARNING Sequence group 12 is preempted by PreemptionMode.RECOMPUTE\n")
    assert g.poll("x") == 2 and "4 -> 2" in (tmp_path / "events.log").read_text()
