import pytest

from fakes import FakeModel, sql_reply
from t2sbench.bench import ContextFactory, load_dataset
from t2sbench.datasets.synthetic.generator import load_codes
from t2sbench.models.base import Generation
from t2sbench.schema import ddl, render_values, value_lists
from t2sbench.sqlutil import extract_sql
from t2sbench.strategies import get_strategy
from t2sbench.strategies.explore import run_limited


@pytest.mark.parametrize("text,expected", [
    ("```sql\nSELECT 1;\n```", "SELECT 1"),
    ("first ```sql\nSELECT 1\n``` then ```sql\nSELECT 2\n```", "SELECT 2"),
    ("<think>```sql\nSELECT bad\n```</think> final: ```sql\nSELECT good\n```", "SELECT good"),
    ("<answer>SELECT a FROM t</answer>", "SELECT a FROM t"),
    ("The query is: SELECT a FROM t WHERE b = 1;\n\nThis returns a.", "SELECT a FROM t WHERE b = 1"),
    ("```\nWITH x AS (SELECT 1) SELECT * FROM x\n```", "WITH x AS (SELECT 1) SELECT * FROM x"),
    ("I cannot answer.", None), ("", None), (None, None),
])
def test_extract_sql(text, expected):
    assert extract_sql(text) == expected


@pytest.fixture()
def syn_questions(synth, monkeypatch):
    out, _ = synth
    return load_dataset("syn_pl", syn_dir=out)[:3], out


def test_schema_values(synth, tmp_path):
    db = synth[0] / "prod.sqlite"
    assert "CREATE TABLE ZLEC_POZ" in ddl(db)
    vals = value_lists(db, cache_dir=tmp_path)
    assert vals["ZLEC.ZLEC_STAT"] == ["A", "N", "R", "W", "Z"]
    assert "ZLEC.ZLEC_NR" not in vals  # 420 distinct values
    assert "MCH.ROK_PROD" not in vals  # INTEGER column
    text = render_values(vals, load_codes())
    assert "'R' = in progress (pl: w realizacji)" in text and "'L01'" in text


def test_s0_s1_s2_prompts(syn_questions):
    qs, _ = syn_questions
    ctxf = ContextFactory({"name": "fake", "backend": "vllm"}, "syn_pl", False)
    ctx = ctxf(qs[0])
    prompts = {}
    for name in ("S0", "S1", "S2"):
        m = FakeModel(lambda *a: sql_reply("SELECT 1"))
        out = get_strategy(name).run(m, qs[0], ctx)
        assert out.sql == "SELECT 1" and out.n_calls == 1
        prompts[name] = m.calls[0]["messages"][0].content
    assert "Column values" not in prompts["S0"] and "Example 1" not in prompts["S0"]
    assert "Example 1" in prompts["S1"] and "Column values" not in prompts["S1"]
    assert "Example 1" in prompts["S2"] and "'Z' = completed / closed" in prompts["S2"]
    assert qs[0].question in prompts["S0"]


def test_s3_exploration(syn_questions):
    qs, _ = syn_questions
    ctx = ContextFactory({"name": "fake", "backend": "vllm"}, "syn_pl", False)(qs[0])
    state = {"n": 0}

    def respond(messages, system, params, tools):
        state["n"] += 1
        if state["n"] == 1:
            return Generation("", 10, 1, 0.1, tool_calls=[{"id": "1", "name": "run_sql",
                                                         "arguments": {"query": "SELECT * FROM PROD_REJ"}}])
        assert messages[-1].role == "tool"
        assert len(messages[-1].content.splitlines()) <= 21 + 1  # header + 20 rows (+ limit note)
        return Generation(sql_reply("SELECT COUNT(*) FROM PROD_REJ"), 10, 1, 0.1)

    out = get_strategy("S3").run(FakeModel(respond), qs[0], ctx)
    assert out.sql == "SELECT COUNT(*) FROM PROD_REJ" and out.meta["explorations"] == 1 and out.n_calls == 2


def test_exploration_query_is_validated(synth):
    db = synth[0] / "prod.sqlite"
    obs, _ = run_limited(db, "DELETE FROM LIN")
    assert obs.startswith("Error") and "Delete" in obs
    obs, _ = run_limited(db, "SELECT LIN_KOD FROM LIN ORDER BY 1")
    assert obs.splitlines()[0] == "LIN_KOD" and "L01" in obs


def test_s4_repairs(syn_questions):
    qs, _ = syn_questions
    ctx = ContextFactory({"name": "fake", "backend": "vllm"}, "syn_pl", False)(qs[0])
    replies = iter([sql_reply("SELECT nope FROM LIN"), sql_reply("SELECT LIN_KOD FROM LIN")])
    m = FakeModel(lambda *a: next(replies))
    out = get_strategy("S4").run(m, qs[0], ctx)
    assert out.sql == "SELECT LIN_KOD FROM LIN" and out.n_calls == 2 and out.meta["repair_rounds"] == 1
    assert "no such column" in m.calls[1]["messages"][-1].content


def test_s4_stops_after_two_repairs(syn_questions):
    qs, _ = syn_questions
    ctx = ContextFactory({"name": "fake", "backend": "vllm"}, "syn_pl", False)(qs[0])
    m = FakeModel(lambda *a: sql_reply("SELECT nope FROM LIN"))
    out = get_strategy("S4").run(m, qs[0], ctx)
    assert out.n_calls == 3 and out.sql == "SELECT nope FROM LIN"


def test_s5_majority_vote(syn_questions):
    qs, _ = syn_questions
    ctx = ContextFactory({"name": "fake", "backend": "vllm"}, "syn_pl", False)(qs[0])
    answers = iter([sql_reply("SELECT 1"), sql_reply("SELECT 2"), sql_reply("SELECT 1 + 1"),
                    sql_reply("SELECT broken FROM"), sql_reply("SELECT 2.0")])
    m = FakeModel(lambda *a: next(answers))
    out = get_strategy("S5").run(m, qs[0], ctx)
    assert out.sql == "SELECT 2"  # 2, 1+1 and 2.0 give the same result
    assert out.meta["votes"] == 3 and out.n_calls == 5
    assert [c["params"].temperature for c in m.calls] == [0.7] * 5
    assert len({c["params"].seed for c in m.calls}) == 5
