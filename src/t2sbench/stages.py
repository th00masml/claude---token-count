"""Stage orchestration. Each stage function runs its configurations and stops for review.

Bedrock models run in-process. vLLM models run in a child process wrapped by
scripts/serve.sh (one model on the GPU at a time); if the server cannot start (e.g. no
memory for the KV cache), the model is skipped and the reason logged in logs/skipped.jsonl.
"""

from __future__ import annotations

import json
import logging
import os
import shlex
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from t2sbench.bench import DATASETS, RunConfig, load_dataset, load_results, run_config
from t2sbench.datasets.bird import stratified_sample
from t2sbench.models.budget import Budget
from t2sbench.models.registry import Registry

log = logging.getLogger(__name__)
RESULTS = Path("results")
SKIP_LOG = Path("logs/skipped.jsonl")
QID_DIR = RESULTS / "qids"

STAGE3_STRATEGIES = ("S0", "S2")
STAGE4_STRATEGIES = ("S1", "S3", "S4", "S5")
SQRL_MODELS = ("sqrl-9b", "sqrl-4b")  # 9b first: its BIRD-with-evidence run is the harness check
SQRL_VARIANTS = ("SQRL-agentic", "SQRL-single")
LORA_DATASETS = ("bird_noev", "syn_pl", "syn_en")
LORA_CURVE = (50, 100, 200, 400)

# Rough cost multipliers of stage-4 strategies relative to S2, used ONLY for the pre-run
# estimate (S3: up to 3 turns with growing context, S4: ~1.3 calls, S5: 5 samples).
STAGE4_MULTIPLIER = {"S1": 1.0, "S3": 2.5, "S4": 1.3, "S5": 5.0}


class StageStop(RuntimeError):
    """A check failed and the run must stop for a human decision."""


def log_skip(model: str, reason: str, stage: str) -> None:
    SKIP_LOG.parent.mkdir(parents=True, exist_ok=True)
    with open(SKIP_LOG, "a", encoding="utf-8") as f:
        f.write(json.dumps({"ts": datetime.now(timezone.utc).isoformat(), "stage": stage, "model": model,
                            "reason": reason}) + "\n")
    log.warning("skip %s in %s: %s", model, stage, reason)


def _write_qids(name: str, qids: dict[str, list[str]]) -> Path:
    QID_DIR.mkdir(parents=True, exist_ok=True)
    p = QID_DIR / f"{name}.json"
    p.write_text(json.dumps(qids, indent=1))
    return p


def run_model(registry: Registry, budget: Budget, stage: str, model: str, strategies, datasets,
              adapters=(None,), qids_file: Path | None = None, allow_unverified: bool = False,
              loras: dict[str, str] | None = None) -> bool:
    """Run all (strategy x dataset x adapter) configs for one model. Returns False if skipped."""
    spec = registry.spec(model)
    if spec["backend"] == "bedrock":
        qids = json.loads(qids_file.read_text()) if qids_file else {}
        m = registry.build(model)
        for ad in adapters:
            for ds in datasets:
                for st in strategies:
                    cfg = RunConfig(stage, model, st, ds, ad, qids.get(ds), registry.cfg["bedrock"].get("max_concurrency", 4),
                                    allow_unverified)
                    run_config(cfg, m, spec, RESULTS, budget=budget)
        return True
    cmd = ["scripts/serve.sh", model]
    for name, path in (loras or {}).items():
        cmd += ["--lora", f"{name}={path}"]
    cmd += ["--", *shlex.split(os.environ.get("T2SBENCH_CMD", "uv run t2sbench")), "run", "--stage", stage, "--model", model]
    for st in strategies:
        cmd += ["--strategy", st]
    for ds in datasets:
        cmd += ["--dataset", ds]
    for ad in adapters:
        cmd += ["--adapter", ad or "base"]
    if qids_file:
        cmd += ["--qids-file", str(qids_file)]
    if allow_unverified:
        cmd += ["--allow-unverified-prompts"]
    log.info("running %s", " ".join(cmd))
    rc = subprocess.run(cmd).returncode
    if rc in (3, 4):
        log_skip(model, f"vLLM server failed to start (serve.sh exit {rc}), see logs/vllm_{model}.log", stage)
        return False
    if rc != 0:
        raise StageStop(f"{model}: run exited with code {rc}; see the log above")
    return True


# --------------------------------------------------------------------------- stage 2

def smoke_qids(n_bird: int = 10, n_syn_pairs: int = 5, seed: int = 42) -> dict[str, list[str]]:
    bird = load_dataset("bird_ev")
    picked = stratified_sample([{"question_id": i, "difficulty": q.difficulty, "id": q.id}
                                for i, q in enumerate(bird)], n=n_bird, seed=seed)
    bird_ids = [p["id"] for p in picked]
    syn = load_dataset("syn_pl")
    rng = np.random.default_rng(seed)
    pairs = sorted({q.id.rsplit("-", 1)[0] for q in syn})
    chosen = sorted(rng.choice(pairs, size=n_syn_pairs, replace=False).tolist())
    return {"bird_ev": bird_ids, "bird_noev": bird_ids,
            "syn_pl": [f"{p}-pl" for p in chosen], "syn_en": [f"{p}-en" for p in chosen]}


def smoke(registry: Registry, budget: Budget, allow_unverified: bool = False,
          bedrock_models: list[str] | None = None, local_model: str | None = None) -> dict:
    """Stage 2: 20 questions (10 BIRD in both evidence variants + 5 PL/EN synthetic pairs),
    2 Bedrock models and 1 local model, S0 and S2. Then the full-run estimate."""
    if bedrock_models is None:
        avail = registry.available(("candidate",), "bedrock")
        priced = [m for m in avail if budget.price(m)]
        bedrock_models = sorted(priced, key=lambda m: sum(budget.price(m)))[:2]
    if local_model is None:
        local = registry.available(("candidate",), "vllm")
        local_model = "qwen2.5-coder-7b" if "qwen2.5-coder-7b" in local else (local[0] if local else None)
    qfile = _write_qids("smoke", smoke_qids())
    for m in bedrock_models + ([local_model] if local_model else []):
        run_model(registry, budget, "smoke", m, STAGE3_STRATEGIES, DATASETS, qids_file=qfile,
                  allow_unverified=allow_unverified)
    return estimate_full_run(registry, budget)


def estimate_full_run(registry: Registry, budget: Budget, out: Path = RESULTS / "smoke_estimate.json",
                      results_dir: Path = RESULTS) -> dict:
    """Extrapolate tokens, cost and time of stages 3 and 4 from the smoke results.

    Per (backend, strategy, dataset family) we take the mean input/output tokens and the
    mean per-question wall time measured in the smoke test. Models not in the smoke test
    get the backend mean (prompts are identical across models; only tokenizers and answer
    length differ). Stage-4 numbers use STAGE4_MULTIPLIER and assume the top 3 are the 3
    most expensive Bedrock candidates (upper bound)."""
    df = load_results(results_dir, "smoke")
    if df.empty:
        raise StageStop("no smoke results")
    n_questions = {}
    for ds in DATASETS:
        try:
            n_questions[ds] = len(load_dataset(ds))
        except FileNotFoundError as e:
            log.warning("estimate: %s not available (%s), left out", ds, e)
    fam = lambda ds: "bird" if ds.startswith("bird") else "syn"  # noqa: E731
    df["family"] = df["dataset"].map(fam)
    per_model = df.groupby(["model", "strategy", "family"]).agg(
        in_tok=("input_tokens", "mean"), out_tok=("output_tokens", "mean"), wall=("total_time_s", "mean")).reset_index()
    per_backend = df.groupby(["backend", "strategy", "family"]).agg(
        in_tok=("input_tokens", "mean"), out_tok=("output_tokens", "mean"), wall=("total_time_s", "mean")).reset_index()

    def lookup(model, backend, strategy, family):
        r = per_model[(per_model.model == model) & (per_model.strategy == strategy) & (per_model.family == family)]
        src = "measured"
        if r.empty:
            r = per_backend[(per_backend.backend == backend) & (per_backend.strategy == strategy) & (per_backend.family == family)]
            src = "backend mean"
        if r.empty:
            return None
        return float(r.in_tok.iloc[0]), float(r.out_tok.iloc[0]), float(r.wall.iloc[0]), src

    rows = []
    models = registry.available(("candidate", "reference"))
    for m in models:
        spec = registry.spec(m)
        strategies = spec.get("strategies") or STAGE3_STRATEGIES
        for st in strategies:
            for ds in n_questions:
                est = lookup(m, spec["backend"], st if st in ("S0", "S2") else "S2", fam(ds))
                if est is None:
                    continue
                in_tok, out_tok, wall, src = est
                n = n_questions[ds]
                cost = budget.cost(m, int(in_tok * n), int(out_tok * n)) if spec["backend"] == "bedrock" else 0.0
                conc = registry.cfg["bedrock"].get("max_concurrency", 4) if spec["backend"] == "bedrock" \
                    else registry.cfg["vllm"].get("concurrency", 4)
                rows.append({"stage": "stage3", "model": m, "backend": spec["backend"], "strategy": st, "dataset": ds,
                             "n": n, "input_tokens": in_tok * n, "output_tokens": out_tok * n,
                             "cost_usd": cost, "hours": wall * n / conc / 3600, "basis": src})
    est = pd.DataFrame(rows)
    s3 = est[est.stage == "stage3"]
    bedrock_s2 = s3[(s3.backend == "bedrock") & (s3.strategy == "S2")]
    top_cost = bedrock_s2.groupby("model").cost_usd.sum().sort_values(ascending=False).head(3)
    s4_cost = float(top_cost.sum() * sum(STAGE4_MULTIPLIER.values())) if len(top_cost) else 0.0
    summary = {
        "stage3_cost_usd": float(s3.cost_usd.sum(skipna=True)),
        "stage3_cost_unpriced_models": sorted(set(s3[s3.cost_usd.isna()].model)),
        "stage3_hours_bedrock": float(s3[s3.backend == "bedrock"].hours.sum()),
        "stage3_hours_local_gpu": float(s3[s3.backend == "vllm"].hours.sum()),
        "stage4_cost_usd_upper_bound": s4_cost,
        "stage4_multipliers": STAGE4_MULTIPLIER,
        "spent_so_far_usd": budget.spent,
        "budget_usd": budget.budget,
    }
    summary["projected_total_usd"] = summary["spent_so_far_usd"] + summary["stage3_cost_usd"] + s4_cost
    summary["within_budget"] = summary["projected_total_usd"] <= budget.budget
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"summary": summary, "configs": rows}, indent=1, default=float))
    est.to_csv(out.with_suffix(".csv"), index=False)
    return summary


# --------------------------------------------------------------------------- stage 3 / 4

def runnable(registry: Registry, budget: Budget, models: list[str], stage: str) -> list[str]:
    """Drop Bedrock models without a price (the budget could not track them) with a log entry."""
    out = []
    for m in models:
        if registry.spec(m)["backend"] == "bedrock" and budget.price(m) is None and not budget.allow_unpriced:
            log_skip(m, "no price in config/prices.yaml (budget cannot track it)", stage)
            continue
        out.append(m)
    return out


def stage3(registry: Registry, budget: Budget, allow_unverified: bool = False) -> None:
    for m in runnable(registry, budget, registry.available(("candidate", "reference")), "stage3"):
        spec = registry.spec(m)
        strategies = spec.get("strategies") or STAGE3_STRATEGIES
        run_model(registry, budget, "stage3", m, strategies, DATASETS, allow_unverified=allow_unverified)


def top_models(n: int = 3, results_dir: Path = RESULTS) -> list[str]:
    """Rank stage-3 candidates by S2 execution accuracy averaged over the 4 test sets
    (each set weighted equally). The reference model and SQRL are excluded."""
    df = load_results(results_dir, "stage3")
    df = df[(df.strategy == "S2") & (df.adapter.isna())]
    ref = {"reference-large"}
    acc = df[~df.model.isin(ref)].groupby(["model", "dataset"]).correct.mean().unstack()
    acc = acc.dropna(subset=[c for c in DATASETS if c in acc.columns])
    ranking = acc.mean(axis=1).sort_values(ascending=False)
    top = ranking.head(n).index.tolist()
    (results_dir / "stage3_ranking.json").write_text(json.dumps(
        {"ranking": ranking.round(4).to_dict(), "top": top}, indent=1))
    return top


def stage4(registry: Registry, budget: Budget, allow_unverified: bool = False) -> list[str]:
    top = top_models()
    for m in runnable(registry, budget, top, "stage4"):
        run_model(registry, budget, "stage4", m, STAGE4_STRATEGIES, DATASETS, allow_unverified=allow_unverified)
    return top


# --------------------------------------------------------------------------- stage 5 (SQRL)

def sqrl_checks(df: pd.DataFrame, model: str, strategy: str, dataset: str, out_dir: Path = RESULTS / "stage5") -> None:
    """Called after every SQRL configuration (inside the serving process)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    meta = df["meta"].map(json.loads)
    flags = df[meta.map(lambda m: bool(m.get("context_over_90pct")))]
    if len(flags):
        flags[["qid", "model", "strategy", "dataset"]].assign(
            max_prompt_tokens=meta[flags.index].map(lambda m: m.get("max_prompt_tokens"))
        ).to_csv(out_dir / f"context_flags__{model}__{strategy}__{dataset}.csv", index=False)
    share = len(flags) / max(1, len(df))
    if share > 0.05:
        raise StageStop(f"{model} {strategy} {dataset}: {len(flags)}/{len(df)} questions ({share:.1%}) had a prompt "
                        "over 90% of the context window (> 5%). Stopped; please decide how to proceed.")
    if model == "sqrl-9b" and strategy == "SQRL-agentic" and dataset == "bird_ev":
        ex = float(df.correct.mean())
        if ex < 0.60:
            sample = df.sample(n=min(10, len(df)), random_state=42)
            lines = [f"# Harness check failed: SQRL-9B-agentic on BIRD dev with evidence, EX = {ex:.3f} (< 0.60)\n"]
            for _, r in sample.iterrows():
                lines.append(f"## {r.qid} (correct={r.correct})\n\n**Question:** {r.question}\n\n"
                             f"**Evidence:** {r.evidence}\n\n**Gold:** `{r.gold_sql}`\n\n**Pred:** `{r.pred_sql}`\n")
                for i, resp in enumerate(json.loads(r.responses)):
                    lines.append(f"### turn {i + 1}\n\n```\n{resp}\n```\n")
                lines.append(f"### full prompt\n\n```json\n{r.prompt}\n```\n")
            (out_dir / "harness_check_transcripts.md").write_text("\n".join(lines), encoding="utf-8")
            raise StageStop(f"harness check failed: SQRL-9B-agentic EX {ex:.3f} < 0.60 on BIRD with evidence; "
                            "10 transcripts in results/stage5/harness_check_transcripts.md")


def stage5(registry: Registry, budget: Budget) -> None:
    from t2sbench.sqrl import SqrlProtocol

    SqrlProtocol.load()  # fail fast if the card prompt has not been copied in
    for m in SQRL_MODELS:
        if not registry.is_available(m):
            log_skip(m, "not available (discover-models)", "stage5")
            continue
        # order matters: (agentic, bird_ev) first so the harness check runs before anything else
        run_model(registry, budget, "stage5", m, SQRL_VARIANTS, ("bird_ev", "bird_noev", "syn_en", "syn_pl"))


# --------------------------------------------------------------------------- stage 6 (LoRA)

def lora_bases(registry: Registry) -> list[str]:
    """Qwen2.5-Coder-7B + the best open 14B model among the stage-3 top 3 (falls back to the
    best-ranked open 14B overall, which the report then states)."""
    ranking_file = RESULTS / "stage3_ranking.json"
    top = json.loads(ranking_file.read_text())["top"] if ranking_file.exists() else top_models()
    ranking = json.loads(ranking_file.read_text())["ranking"]
    open14 = [m for m in ranking if registry.spec(m)["backend"] == "vllm" and registry.spec(m).get("size_b") == 14]
    pick = next((m for m in top if m in open14), open14[0] if open14 else None)
    return ["qwen2.5-coder-7b"] + ([pick] if pick else [])


def adapter_name(base: str, variant: str, n: int | None = None) -> str:
    return f"{base}-{variant}" + (f"-{n}" if n else "")


def stage6_eval(registry: Registry, budget: Budget, adapters_dir: Path = Path("adapters"),
                allow_unverified: bool = False) -> None:
    for base in lora_bases(registry):
        loras = {}
        for name in [adapter_name(base, "L1"), adapter_name(base, "L3")] + \
                    [adapter_name(base, "L2", n) for n in LORA_CURVE]:
            path = adapters_dir / name / "final"
            if path.exists():
                loras[name] = str(path.resolve())
            else:
                log_skip(f"{base}:{name}", f"adapter missing at {path}", "stage6")
        main = [None] + [a for a in (adapter_name(base, "L1"), adapter_name(base, "L2", 400),
                                     adapter_name(base, "L3")) if a in loras]
        curve = [a for a in (adapter_name(base, "L2", n) for n in LORA_CURVE[:-1]) if a in loras]
        run_model(registry, budget, "stage6", base, ("S2",), LORA_DATASETS, adapters=main,
                  allow_unverified=allow_unverified, loras=loras)
        if curve:
            run_model(registry, budget, "stage6", base, ("S2",), ("syn_pl", "syn_en"), adapters=curve,
                      allow_unverified=allow_unverified, loras=loras)
