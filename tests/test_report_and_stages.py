import json

import pandas as pd
import pytest

from fakes import FakeModel, sql_reply
from t2sbench.bench import RunConfig, load_dataset, run_config
from t2sbench.models.budget import Budget
from t2sbench.models.registry import Registry
from t2sbench.report import aggregate, build_report, categorize_error
from t2sbench.stages import estimate_full_run, top_models

BEDROCK = {"name": "bedrock-a", "backend": "bedrock"}
LOCAL = {"name": "local-b", "backend": "vllm", "quantization": "awq", "checkpoint": "x/y-AWQ", "max_model_len": 16384}


def accuracy_model(qs, k):
    """Answers the first k questions (by order) with the gold SQL, the rest wrongly."""
    good = {q.question: q.gold_sql for q in qs[:k]}

    def respond(messages, system, params, tools):
        text = messages[0].content
        for qt, sql in good.items():
            if qt in text:
                return sql_reply(sql)
        return sql_reply("SELECT -1")  # no gold answer is -1
    return respond


@pytest.fixture(scope="module")
def fake_results(tmp_path_factory, synth):
    root = tmp_path_factory.mktemp("bench")
    res = root / "results"
    syn = synth[0]
    for ds in ("syn_pl", "syn_en"):
        qs = load_dataset(ds, syn_dir=syn)
        for spec, acc in ((BEDROCK, {"S0": 10, "S2": 30}), (LOCAL, {"S0": 8, "S2": 25})):
            for st, k in acc.items():
                run_config(RunConfig("stage3", spec["name"], st, ds), FakeModel(accuracy_model(qs, k), backend=spec["backend"]),
                           spec, res, questions=qs)
        for ad, k in (("", 25), ("local-b-L2-50", 28), ("local-b-L2-400", 36), ("local-b-L1", 26)):
            run_config(RunConfig("stage6", "local-b", "S2", ds, ad or None), FakeModel(accuracy_model(qs, k)), LOCAL, res,
                       questions=qs)
        # SQRL rows: reuse the runner with S2 and relabel (the native agent is tested separately)
        for st, k, expl in (("SQRL-agentic", 33, 2), ("SQRL-single", 27, 0)):
            cfg = RunConfig("stage5", "sqrl-9b", "S2", ds)
            df = run_config(cfg, FakeModel(accuracy_model(qs, k)), {"name": "sqrl-9b", "backend": "vllm", "dtype": "bfloat16"},
                            res, questions=qs)
            p = res / "stage5" / f"sqrl-9b__S2__{ds}.parquet"
            df = df.assign(strategy=st, meta=json.dumps({"explorations": expl}))
            df.to_parquet(res / "stage5" / f"sqrl-9b__{st}__{ds}.parquet")
            p.unlink()
    adapters = root / "adapters" / "local-b-L2-400"
    adapters.mkdir(parents=True)
    (adapters / "train_meta.json").write_text(json.dumps({
        "name": "local-b-L2-400", "train_examples": 400, "dropped_too_long": 0, "best_checkpoint": "checkpoint-25",
        "best_val_ex": 0.8, "train_seconds": 1800, "adapter_size_bytes": 80 * 2**20}))
    return root


def test_report_end_to_end(fake_results):
    root = fake_results
    out = build_report(root / "results", root / "report", root / "adapters")
    text = out.read_text()
    for section in ("## Results per configuration", "## Effect of S2", "## Polish vs English", "## SQRL",
                    "## LoRA", "## Error analysis", "## Recommendation"):
        assert section in text, section
    for png in ("accuracy_vs_time.png", "lora_learning_curve.png", "sqrl_explorations.png"):
        assert (root / "report" / png).stat().st_size > 1000
    assert "lower bound" in text and "out of the training distribution" in text
    assert "30.0 min" in text and "80.0 MiB" in text
    # numbers in the report match the parquet data
    agg = pd.read_csv(root / "report" / "configs.csv")
    r = agg[(agg.model == "bedrock-a") & (agg.strategy == "S2") & (agg.dataset == "syn_pl")].iloc[0]
    assert r.ex == pytest.approx(30 / 40) and f"{100 * r.ex:.1f}%" in text
    ea = pd.read_csv(root / "report" / "error_analysis.csv")
    # best config = local-b@local-b-L2-400 S2 (36/40 on both sets) -> only 8 errors exist to sample
    assert "local-b@local-b-L2-400 S2" in text
    assert len(ea) == 8 and set(ea.auto_category) <= {"wrong tables", "wrong join", "wrong filter value",
                                                       "wrong aggregation", "dialect", "other"}


def test_manual_error_labels_survive_rebuild(fake_results):
    root = fake_results
    build_report(root / "results", root / "report", root / "adapters")
    csv = root / "report" / "error_analysis.csv"
    ea = pd.read_csv(csv, dtype={"manual_category": str})
    ea.loc[0, "manual_category"] = "dialect"
    ea.to_csv(csv, index=False)
    build_report(root / "results", root / "report", root / "adapters")
    assert pd.read_csv(csv).loc[0, "manual_category"] == "dialect"


def test_aggregate_ci_and_cost(fake_results):
    from t2sbench.bench import load_results

    df = load_results(fake_results / "results", "stage3")
    agg = aggregate(df)
    r = agg[(agg.model == "local-b") & (agg.strategy == "S0") & (agg.dataset == "syn_en")].iloc[0]
    assert r.n == 40 and r.ex == pytest.approx(0.2) and r.ci_lo < 0.2 < r.ci_hi
    assert r.quantized and r.precision.startswith("awq")


def test_top_models(fake_results):
    assert top_models(2, fake_results / "results") == ["bedrock-a", "local-b"]


def test_categorize_error():
    gold = "SELECT COUNT(*) FROM a JOIN b ON a.id = b.a_id WHERE b.s = 'R'"
    assert categorize_error(None, gold, "no answer", "no_answer") == "other"
    assert categorize_error("SELECT x FROM a", gold, "no such column: x", "execution") == "wrong tables"
    assert categorize_error("SELECT COUNT(*) FROM a JOIN c ON a.id = c.a_id WHERE c.s = 'R'", gold, None, None) == "wrong tables"
    assert categorize_error("SELECT COUNT(*) FROM a JOIN b ON a.id = b.id WHERE b.s = 'R'", gold, None, None) == "wrong join"
    assert categorize_error("SELECT COUNT(*) FROM a JOIN b ON a.id = b.a_id WHERE b.s = 'Z'", gold, None, None) == "wrong filter value"
    assert categorize_error("SELECT SUM(a.id) FROM a JOIN b ON a.id = b.a_id WHERE b.s = 'R'", gold, None, None) == "wrong aggregation"
    assert categorize_error("SELECT DATEDIFF(x) FROM a", gold, "no such function: DATEDIFF", "execution") == "dialect"


def test_estimate_full_run(tmp_path, synth, monkeypatch):
    res = tmp_path / "results"
    qs = load_dataset("syn_pl", syn_dir=synth[0])[:4]
    for spec in (BEDROCK, LOCAL):
        for st in ("S0", "S2"):
            run_config(RunConfig("smoke", spec["name"], st, "syn_pl"), FakeModel(lambda *a: sql_reply("SELECT 1"),
                       backend=spec["backend"]), spec, res, questions=qs)
    cfg = tmp_path / "models.yaml"
    cfg.write_text("bedrock: {region: us-east-1, max_concurrency: 4}\nvllm: {base_url: x, concurrency: 4}\nmodels:\n"
                   "  - {name: bedrock-a, backend: bedrock, match: x, role: candidate}\n"
                   "  - {name: bedrock-c, backend: bedrock, match: y, role: candidate}\n"
                   "  - {name: local-b, backend: vllm, hf_repo: x/y, role: candidate}\n")
    av = tmp_path / "av.json"
    av.write_text(json.dumps([{"name": n, "available": True, "resolved_id": n, "reason": "ok", "backend": b, "role": "candidate"}
                              for n, b in (("bedrock-a", "bedrock"), ("bedrock-c", "bedrock"), ("local-b", "vllm"))]))
    prices = tmp_path / "prices.yaml"
    prices.write_text("per_million_tokens:\n  bedrock-a: {input: 1.0, output: 2.0}\n  bedrock-c: {input: 2.0, output: 4.0}\nbudget_usd: 50\n")
    budget = Budget(prices, tmp_path / "spend.json")
    reg = Registry(cfg, av, tmp_path / "cache", budget)
    monkeypatch.setattr("t2sbench.stages.load_dataset",
                        lambda ds: load_dataset(ds, syn_dir=synth[0]) if ds.startswith("syn") else (_ for _ in ()).throw(FileNotFoundError(ds)))
    s = estimate_full_run(reg, budget, out=tmp_path / "est.json", results_dir=res)
    # 2 datasets x 40 questions x 2 strategies x (100 in, 20 out) tokens
    assert s["stage3_cost_usd"] == pytest.approx(2 * 40 * 2 * (100 * 1 + 20 * 2) / 1e6 + 2 * 40 * 2 * (100 * 2 + 20 * 4) / 1e6)
    est = json.loads((tmp_path / "est.json").read_text())
    assert {r["basis"] for r in est["configs"] if r["model"] == "bedrock-c"} == {"backend mean"}
    assert s["within_budget"]
