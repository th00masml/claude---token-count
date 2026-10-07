"""End-to-end through the real path: serve.sh -> (fake) vLLM HTTP server -> OpenAI client ->
runner -> parquet, plus the stages.run_model subprocess path and LoRA adapter routing."""

import json
import os
import shutil
import socket
import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest

from t2sbench.bench import load_dataset

REPO = Path(__file__).resolve().parents[1]
pytestmark = pytest.mark.skipif(shutil.which("curl") is None, reason="serve.sh needs curl")


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@pytest.fixture()
def workdir(tmp_path, synth):
    """A scratch checkout: config/scripts symlinked, synthetic data from the fixture, own results/."""
    for d in ("config", "scripts"):
        (tmp_path / d).symlink_to(REPO / d)
    (tmp_path / "data").mkdir()
    (tmp_path / "data" / "synthetic").symlink_to(synth[0])
    (tmp_path / "results").mkdir()
    (tmp_path / "results" / "model_availability.json").write_text(json.dumps([
        {"name": "qwen2.5-coder-7b", "backend": "vllm", "role": "candidate", "available": True,
         "resolved_id": "Qwen/Qwen2.5-Coder-7B-Instruct", "reason": "test"}]))
    qs = load_dataset("syn_en", syn_dir=synth[0])
    good = qs[:6]  # the fake server knows the gold SQL of the first 6 questions only
    (tmp_path / "answers.json").write_text(json.dumps({q.question: q.gold_sql for q in good}))
    (tmp_path / "qids.json").write_text(json.dumps({"syn_en": [q.id for q in qs[:10]]}))
    env = {**os.environ, "VLLM_BIN": str(REPO / "tests" / "fake_vllm.py"), "PORT": str(free_port()),
           "HEALTH_TIMEOUT": "30", "T2SBENCH_CMD": f"{sys.executable} -m t2sbench.run",
           "FAKE_VLLM_ANSWERS": str(tmp_path / "answers.json"), "GPU_FREE_MB": "999999"}
    env.pop("T2S_VLLM_BASE_URL", None)
    return tmp_path, env


def serve(workdir, env, *inner, extra=()):
    cmd = [str(workdir / "scripts" / "serve.sh"), "qwen2.5-coder-7b", *extra, "--", *inner]
    return subprocess.run(cmd, cwd=workdir, env=env, capture_output=True, text=True, timeout=300)


def t2s(*a):
    return [sys.executable, "-m", "t2sbench.run", *a]


def port_closed(port: str) -> bool:
    with socket.socket() as s:
        return s.connect_ex(("127.0.0.1", int(port))) != 0


def test_serve_run_and_stop(workdir):
    wd, env = workdir
    env["FAKE_VLLM_EXPLORE"] = "1"
    env["FAKE_VLLM_PREEMPT_AFTER"] = "3"
    r = serve(wd, env, *t2s("run", "--stage", "itest", "--model", "qwen2.5-coder-7b", "--strategy", "S0",
                            "--strategy", "S3", "--dataset", "syn_en", "--qids-file", str(wd / "qids.json")))
    assert r.returncode == 0, r.stdout[-3000:] + r.stderr[-3000:]
    s0 = pd.read_parquet(wd / "results/itest/qwen2.5-coder-7b__S0__syn_en.parquet")
    s3 = pd.read_parquet(wd / "results/itest/qwen2.5-coder-7b__S3__syn_en.parquet")
    assert len(s0) == 10 and s0.correct.sum() == 6
    assert len(s3) == 10 and s3.correct.sum() == 6
    meta = s3["meta"].map(json.loads)
    assert (meta.map(lambda m: m["explorations"]) == 1).all() and (s3.n_calls == 2).all()
    assert (s0.precision == "bfloat16").all() and (s0.input_tokens > 0).all()
    # preemption in the server log lowered concurrency 4 -> 2 and was logged
    assert "4 -> 2" in (wd / "logs" / "concurrency.log").read_text()
    assert "fake vllm starting qwen2.5-coder-7b" in (wd / "logs" / "vllm_qwen2.5-coder-7b.log").read_text()
    assert port_closed(env["PORT"]), "server still running after serve.sh exited"


def test_serve_startup_failure_exit_code(workdir):
    wd, env = workdir
    env["FAKE_VLLM_FAIL"] = "1"
    r = serve(wd, env, "true")
    assert r.returncode == 3, r.stderr
    assert "died during startup" in r.stderr


def test_stages_run_model_with_lora_adapters(workdir, monkeypatch):
    wd, env = workdir
    from t2sbench import stages
    from t2sbench.models.budget import Budget
    from t2sbench.models.registry import Registry

    for k, v in env.items():
        monkeypatch.setenv(k, v)
    monkeypatch.chdir(wd)
    adapter = wd / "adapters" / "qwen2.5-coder-7b-L2-50" / "final"
    adapter.mkdir(parents=True)
    reg = Registry(budget=Budget())
    ok = stages.run_model(reg, Budget(), "itest6", "qwen2.5-coder-7b", ("S2",), ("syn_en",),
                          adapters=(None, "qwen2.5-coder-7b-L2-50"), qids_file=wd / "qids.json",
                          loras={"qwen2.5-coder-7b-L2-50": str(adapter)})
    assert ok
    base = pd.read_parquet(wd / "results/itest6/qwen2.5-coder-7b__S2__syn_en.parquet")
    lora = pd.read_parquet(wd / "results/itest6/qwen2.5-coder-7b@qwen2.5-coder-7b-L2-50__S2__syn_en.parquet")
    assert len(base) == len(lora) == 10
    # the fake server echoes the model name it was asked for: requests really went to the adapter
    assert lora.responses.str.contains(r"\[qwen2.5-coder-7b-L2-50\]").all()
    assert base.responses.str.contains(r"\[qwen2.5-coder-7b\]").all()
    # a server that cannot start makes run_model skip the model and log it
    monkeypatch.setenv("FAKE_VLLM_FAIL", "1")
    assert stages.run_model(reg, Budget(), "itest6", "qwen2.5-coder-7b", ("S2",), ("syn_en",)) is False
    assert "serve.sh exit 3" in (wd / "logs" / "skipped.jsonl").read_text()
