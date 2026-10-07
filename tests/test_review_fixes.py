"""Regression tests for bugs found in code review (each test names the bug)."""

import json
import os
import stat
import subprocess
import threading
import time
from pathlib import Path

import pytest

from fakes import FakeModel, sql_reply
from t2sbench.bench import RunConfig, load_dataset, run_config
from t2sbench.evaluate import execution_match
from t2sbench.models.base import Generation, Message, ToolSpec
from t2sbench.models.budget import Budget, BudgetExceeded
from t2sbench.models.cache import DiskCache
from t2sbench.models.cached import CachedModel

SPEC = {"name": "fake", "backend": "vllm"}


def test_discovery_prefers_on_demand_base_id(monkeypatch, tmp_path):
    """list_foundation_models also lists provisioned-only ':300k' variants that sort last."""
    from t2sbench.models import discovery

    summaries = [
        {"modelId": "amazon.nova-lite-v1:0", "outputModalities": ["TEXT"], "inferenceTypesSupported": ["ON_DEMAND"]},
        {"modelId": "amazon.nova-lite-v1:0:300k", "outputModalities": ["TEXT"], "inferenceTypesSupported": ["PROVISIONED"]},
    ]

    class Client:
        def list_foundation_models(self):
            return {"modelSummaries": summaries}

        def list_inference_profiles(self, **kw):
            return {"inferenceProfileSummaries": []}

    import boto3

    monkeypatch.setattr(boto3, "client", lambda *a, **k: Client())
    cfg = {"bedrock": {"region": "us-east-1"},
           "models": [{"name": "nova-lite", "backend": "bedrock", "match": r"amazon\.nova-lite"}]}
    (a,) = discovery.discover_bedrock(cfg)
    assert a.available and a.resolved_id == "amazon.nova-lite-v1:0"


def test_budget_reserves_in_flight_calls(tmp_path):
    """check() must count calls still in flight on other threads."""
    p = tmp_path / "prices.yaml"
    p.write_text("per_million_tokens:\n  m: {input: 0.0, output: 1000000.0}\nbudget_usd: 2.5\n")
    b = Budget(p, tmp_path / "s.json")
    b.check("m", 0, 1)  # 1 USD worst case
    b.check("m", 0, 1)
    with pytest.raises(BudgetExceeded):
        b.check("m", 0, 1)  # 2 in flight + 1 > 2.5
    b.record("m", 0, 0, reserved=1.0)  # first call finished costing nothing
    b.check("m", 0, 1)


def test_budget_released_when_call_fails(tmp_path):
    p = tmp_path / "prices.yaml"
    p.write_text("per_million_tokens:\n  m: {input: 0.0, output: 1000000.0}\nbudget_usd: 1.5\n")
    b = Budget(p, tmp_path / "s.json")

    def boom(*a):
        raise ConnectionError("x")

    m = CachedModel(FakeModel(boom, name="m", backend="bedrock"), DiskCache(tmp_path / "c"), b)
    from t2sbench.models.base import GenParams

    for _ in range(3):
        with pytest.raises(ConnectionError):
            m.generate([Message("user", "q")], params=GenParams(max_tokens=1))
    assert b.reserved == 0.0


def test_truncated_results_are_not_compared_as_complete(synth):
    db = synth[0] / "prod.sqlite"
    gold_1000 = "SELECT REJ_ID FROM PROD_REJ ORDER BY REJ_ID LIMIT 1000"
    pred_more = "SELECT REJ_ID FROM PROD_REJ ORDER BY REJ_ID"
    assert not execution_match(db, pred_more, gold_1000).correct
    # both over the cap: compared on the full results
    assert execution_match(db, "SELECT REJ_ID FROM PROD_REJ", "SELECT r.REJ_ID FROM PROD_REJ r").correct
    assert not execution_match(db, "SELECT REJ_ID FROM PROD_REJ WHERE REJ_ID > 1",
                               "SELECT REJ_ID FROM PROD_REJ").correct


def test_runner_survives_non_model_exceptions(synth, tmp_path, monkeypatch):
    """An exception in scoring must not abort the run or lose finished rows."""
    qs = load_dataset("syn_pl", syn_dir=synth[0])[:12]
    import t2sbench.bench as bench

    real = bench.execution_match

    def flaky(db, pred, gold, **kw):
        if gold == qs[5].gold_sql:
            raise RuntimeError("scoring blew up")
        return real(db, pred, gold, **kw)

    monkeypatch.setattr(bench, "execution_match", flaky)
    df = run_config(RunConfig("s", "fake", "S0", "syn_pl", concurrency=3),
                    FakeModel(lambda *a: sql_reply("SELECT 1")), SPEC, tmp_path, questions=qs)
    assert len(df) == 11 and qs[5].id not in set(df.qid)


def test_native_tool_loop_keeps_sql_from_last_tool_turn():
    def respond(messages, system, params, tools):
        return Generation("```sql\nSELECT 9\n```", tool_calls=[{"id": "x", "name": "run_sql", "arguments": {}}])

    g, _ = FakeModel(respond).generate_with_tools([Message("user", "q")], [ToolSpec("run_sql", "", {})],
                                                  lambda n, a: "rows", max_tool_calls=2)
    assert "SELECT 9" in g.text


def test_s3_text_mode_has_no_tool_instructions(synth):
    from t2sbench.bench import ContextFactory
    from t2sbench.strategies import get_strategy

    q = load_dataset("syn_en", syn_dir=synth[0])[0]
    ctx = ContextFactory(SPEC, "syn_en", False)(q)
    m = FakeModel(lambda *a: sql_reply("SELECT 1"), native_tools=False)
    get_strategy("S3").run(m, q, ctx)
    assert "run_sql tool" not in m.calls[0]["system"] and "<explore>" in m.calls[0]["system"]


def test_stage3_skips_unpriced_bedrock_models(tmp_path, monkeypatch):
    from t2sbench import stages

    prices = tmp_path / "prices.yaml"
    prices.write_text("per_million_tokens:\n  priced: {input: 1, output: 1}\nbudget_usd: 50\n")

    class Reg:
        def spec(self, m):
            return {"backend": "bedrock"}

    monkeypatch.setattr(stages, "SKIP_LOG", tmp_path / "skipped.jsonl")
    out = stages.runnable(Reg(), Budget(prices, tmp_path / "s.json"), ["priced", "unpriced"], "stage3")
    assert out == ["priced"] and "unpriced" in (tmp_path / "skipped.jsonl").read_text()


def test_serve_sh_exit_code_survives_broken_nvidia_smi(tmp_path):
    """nvidia-smi printing an error on stdout used to abort the trap under set -u (exit 1)."""
    repo = Path(__file__).resolve().parents[1]
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    smi = bin_dir / "nvidia-smi"
    smi.write_text("#!/bin/sh\necho 'No devices were found'\nexit 6\n")
    smi.chmod(smi.stat().st_mode | stat.S_IEXEC)
    env = {**os.environ, "PATH": f"{bin_dir}:{os.environ['PATH']}", "VLLM_BIN": str(repo / "tests" / "fake_vllm.py"),
           "FAKE_VLLM_FAIL": "1", "PORT": "1", "HEALTH_TIMEOUT": "20",
           "T2SBENCH_CMD": f"{os.sys.executable} -m t2sbench.run"}
    for d in ("config", "scripts"):
        (tmp_path / d).symlink_to(repo / d)
    r = subprocess.run([str(tmp_path / "scripts" / "serve.sh"), "qwen2.5-coder-7b", "--", "true"],
                       cwd=tmp_path, env=env, capture_output=True, text=True, timeout=120)
    assert r.returncode == 3, r.stderr[-2000:]
