import json
from pathlib import Path

import pandas as pd
import pytest
import yaml

from fakes import FakeModel
from t2sbench.bench import load_dataset
from t2sbench.models.base import Generation
from t2sbench.sqrl import SqrlAgent, SqrlConfigError, SqrlProtocol, parse_action
from t2sbench.stages import StageStop, sqrl_checks


@pytest.fixture()
def proto(tmp_path):
    raw = yaml.safe_load(Path("config/prompts/sqrl.yaml").read_text())
    raw.update(system="SYS", user="Schema:\n{schema}\nQuestion: {question}", observation="<observation>\n{result}\n</observation>",
               card_revision="test")
    p = tmp_path / "sqrl.yaml"
    p.write_text(yaml.safe_dump(raw))
    return SqrlProtocol.load(p)


def test_repo_config_refuses_placeholders():
    with pytest.raises(SqrlConfigError):
        SqrlProtocol.load("config/prompts/sqrl.yaml")


def test_parse_action_only_after_think():
    assert parse_action("<think><answer>SELECT 0</answer></think><sql>SELECT 1</sql>") == ("sql", "SELECT 1")
    assert parse_action("<think>x</think>\n<answer>\n```sql\nSELECT 2\n```\n</answer>") == ("answer", "SELECT 2")
    assert parse_action("<think>only thinking</think>") == ("none", None)
    assert parse_action("<sql>SELECT 1</sql><answer>SELECT 3</answer>")[0] == "answer"


def _q(synth):
    return load_dataset("syn_en", syn_dir=synth[0])[0]


def test_agentic_explores_then_answers(proto, synth):
    replies = iter(["<think>a</think><sql>SELECT * FROM PROD_REJ</sql>",
                    "<think>b</think><sql>SELECT LIN_KOD FROM LIN</sql>",
                    "<think>c</think><answer>SELECT COUNT(*) FROM LIN</answer>"])
    m = FakeModel(lambda *a: next(replies))
    out = SqrlAgent(proto, agentic=True).run(m, _q(synth), "DDL")
    assert out.sql == "SELECT COUNT(*) FROM LIN" and out.meta["explorations"] == 2
    obs = m.calls[1]["messages"][-1].content
    assert obs.startswith("<observation>") and obs.count("\n|") == 22  # header + separator + 20 rows
    assert m.calls[1]["messages"][1].content == "<sql>SELECT * FROM PROD_REJ</sql>"  # think stripped
    assert m.calls[0]["system"] == "SYS" and m.calls[0]["params"].temperature == 0.7


def test_no_answer_after_five_explorations(proto, synth):
    m = FakeModel(lambda *a: "<think>x</think><sql>SELECT 1</sql>")
    out = SqrlAgent(proto, agentic=True).run(m, _q(synth), "DDL")
    assert out.sql is None and out.meta["explorations"] == 5 and out.n_calls == 6


def test_single_mode_rejects_exploration(proto, synth):
    m = FakeModel(lambda *a: "<think>x</think><sql>SELECT 1</sql>")
    out = SqrlAgent(proto, agentic=False).run(m, _q(synth), "DDL")
    assert out.sql is None and out.n_calls == 1
    m = FakeModel(lambda *a: "<think>x</think><answer>SELECT 1</answer>")
    assert SqrlAgent(proto, agentic=False).run(m, _q(synth), "DDL").sql == "SELECT 1"


def test_context_flag(proto, synth):
    m = FakeModel(lambda *a: Generation("<answer>SELECT 1</answer>", input_tokens=15000, output_tokens=5))
    out = SqrlAgent(proto, agentic=True).run(m, _q(synth), "DDL")
    assert out.meta["context_over_90pct"] is True


def _df(n, correct, flagged=0):
    return pd.DataFrame({"qid": [f"q{i}" for i in range(n)], "model": "sqrl-9b", "strategy": "SQRL-agentic",
                         "dataset": "bird_ev", "correct": [i < correct for i in range(n)],
                         "meta": [json.dumps({"context_over_90pct": i < flagged, "max_prompt_tokens": 1}) for i in range(n)],
                         "question": "q", "evidence": "e", "gold_sql": "g", "pred_sql": "p",
                         "responses": json.dumps(["r"]), "prompt": "[]"})


def test_harness_check(tmp_path):
    sqrl_checks(_df(100, 69), "sqrl-9b", "SQRL-agentic", "bird_ev", tmp_path)
    with pytest.raises(StageStop, match="harness"):
        sqrl_checks(_df(100, 50), "sqrl-9b", "SQRL-agentic", "bird_ev", tmp_path)
    text = (tmp_path / "harness_check_transcripts.md").read_text()
    assert text.count("\n## q") == 10


def test_context_overflow_stop(tmp_path):
    sqrl_checks(_df(100, 70, flagged=5), "sqrl-4b", "SQRL-agentic", "syn_en", tmp_path)
    with pytest.raises(StageStop, match="90%"):
        sqrl_checks(_df(100, 70, flagged=6), "sqrl-4b", "SQRL-agentic", "syn_en", tmp_path)


def test_max_tokens_shrinks_to_fit_context(proto, synth):
    m = FakeModel(lambda *a: "<answer>SELECT 1</answer>")
    m.count_tokens = lambda msgs, system: 12000
    out = SqrlAgent(proto, agentic=True).run(m, _q(synth), "DDL")
    assert m.calls[0]["params"].max_tokens == 16384 - 12000 - 16
    assert out.sql == "SELECT 1" and out.meta["context_over_90pct"] is False
    m2 = FakeModel(lambda *a: "<answer>SELECT 1</answer>")
    m2.count_tokens = lambda msgs, system: 16300
    out2 = SqrlAgent(proto, agentic=True).run(m2, _q(synth), "DDL")
    assert out2.sql is None and out2.meta["context_over_90pct"] is True and m2.calls == []
