"""Command line entry point: `uv run t2sbench --help`."""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path

import typer

app = typer.Typer(no_args_is_help=True, add_completion=False)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")


def _budget(allow_unpriced: bool = False):
    from t2sbench.models.budget import Budget

    return Budget(allow_unpriced=allow_unpriced)


def _registry(budget=None):
    from t2sbench.models.registry import Registry

    return Registry(budget=budget)


# --------------------------------------------------------------------------- data

@app.command("build-synthetic")
def build_synthetic(out_dir: Path = Path("data/synthetic"), seed: int = 42) -> None:
    """Generate the synthetic production DB and its test / train / val question sets."""
    from t2sbench.datasets.synthetic.build import build_all

    m = build_all(out_dir, seed)
    typer.echo(json.dumps({"rows": m["row_counts"],
                           "questions": {k: v["n"] for k, v in m["questions"].items()}}, indent=2))


@app.command("fetch-bird")
def fetch_bird(split: str = typer.Argument("dev", help="dev or train"),
               root: Path = Path("data/bird")) -> None:
    """Download and extract BIRD dev or train (public data)."""
    from t2sbench.datasets.bird import fetch

    typer.echo(fetch(split, root))


@app.command("sample-bird")
def sample_bird(root: Path = Path("data/bird"), n: int = 150, seed: int = 42,
                out: Path = Path("data/bird_dev_sample150.json")) -> None:
    """Stratified (by difficulty) sample of BIRD dev."""
    from t2sbench.datasets.bird import build_dev_sample

    p = build_dev_sample(root, n, seed, out)
    typer.echo(json.dumps({k: p[k] for k in ("n", "pool_size", "by_difficulty")}
                          | {"excluded": len(p["excluded"])}, indent=2))


# --------------------------------------------------------------------------- models

@app.command("discover-models")
def discover_models(config: Path = Path("config/models.yaml"),
                    out: Path = Path("results/model_availability.json")) -> None:
    """Check which configured Bedrock / HF models are available. Never fails on a missing model."""
    from t2sbench.models.discovery import discover

    for a in discover(config, out):
        typer.echo(f"{'OK  ' if a.available else 'SKIP'} {a.name:24s} {a.resolved_id or '-':50s} {a.reason}")


@app.command("fetch-card")
def fetch_card(model: str, out_dir: Path = Path("config/model_cards")) -> None:
    """Download a model card (README.md) at the current commit and print the commit sha.
    Use it to copy official prompts verbatim into config/prompts/*.yaml."""
    import httpx

    from t2sbench.models.discovery import load_models_config

    spec = {m["name"]: m for m in load_models_config()["models"]}[model]
    repo = spec["hf_repo"]
    info = httpx.get(f"https://huggingface.co/api/models/{repo}", timeout=30).raise_for_status().json()
    sha = info["sha"]
    readme = httpx.get(f"https://huggingface.co/{repo}/raw/{sha}/README.md", timeout=30).raise_for_status().text
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{model}@{sha[:12]}.md"
    path.write_text(f"<!-- {repo} README.md at commit {sha} -->\n" + readme, encoding="utf-8")
    typer.echo(f"{repo} commit {sha}\nsaved to {path}")


@app.command("vllm-args")
def vllm_args_cmd(model: str, port: int = 8000,
                  lora: list[str] = typer.Option([], help="NAME=PATH, repeatable")) -> None:
    """Print the vLLM arguments for a model, one per line (used by scripts/serve.sh)."""
    from t2sbench.models.discovery import load_models_config
    from t2sbench.serving import vllm_args

    spec = {m["name"]: m for m in load_models_config()["models"]}[model]
    loras = dict(x.split("=", 1) for x in lora)
    typer.echo("\n".join(vllm_args(spec, port, loras or None)))


# --------------------------------------------------------------------------- running

@app.command("run")
def run_cmd(stage: str = typer.Option(...), model: str = typer.Option(...),
            strategy: list[str] = typer.Option(..., help="S0..S5, SQRL-agentic, SQRL-single; repeatable"),
            dataset: list[str] = typer.Option(..., help="bird_ev, bird_noev, syn_pl, syn_en; repeatable"),
            adapter: list[str] = typer.Option(["base"], help="LoRA adapter name or 'base'; repeatable"),
            qids_file: Path | None = typer.Option(None, help="JSON {dataset: [qid, ...]} to restrict questions"),
            concurrency: int | None = None, allow_unverified_prompts: bool = False,
            allow_unpriced: bool = False) -> None:
    """Run configurations for ONE model (for vLLM models the server must be up; see serve.sh)."""
    from t2sbench.bench import RunConfig, run_config
    from t2sbench.stages import sqrl_checks

    budget = _budget(allow_unpriced)
    reg = _registry(budget)
    spec = reg.spec(model)
    qids = json.loads(qids_file.read_text()) if qids_file else {}
    conc = concurrency or (reg.cfg["bedrock"] if spec["backend"] == "bedrock" else reg.cfg["vllm"]).get(
        "max_concurrency" if spec["backend"] == "bedrock" else "concurrency", 4)
    vllm_log = Path(os.environ["T2S_VLLM_LOG"]) if os.environ.get("T2S_VLLM_LOG") else None
    for ad in adapter:
        ad = None if ad == "base" else ad
        m = reg.build(model, adapter=ad)
        for ds in dataset:
            for st in strategy:
                cfg = RunConfig(stage, model, st, ds, ad, qids.get(ds), conc, allow_unverified_prompts)
                df = run_config(cfg, m, spec, Path("results"), vllm_log=vllm_log, budget=budget)
                typer.echo(f"{cfg.key}: EX {df.correct.mean():.3f} on {len(df)} questions "
                           f"(Bedrock spend so far {budget.spent:.4f} USD)")
                if st.startswith("SQRL"):
                    sqrl_checks(df, model, st, ds)


@app.command("stage")
def stage_cmd(number: str = typer.Argument(..., help="2 | estimate | 3 | 4 | 5 | 6-train | 6-eval | 7"),
              allow_unverified_prompts: bool = False, allow_unpriced: bool = False) -> None:
    """Run one benchmark stage end to end, then stop for review."""
    from t2sbench import stages

    budget = _budget(allow_unpriced)
    reg = _registry(budget)
    if number == "2":
        typer.echo(json.dumps(stages.smoke(reg, budget, allow_unverified_prompts), indent=2))
        typer.echo("Smoke test done. Review results/smoke_estimate.json before running stage 3.")
    elif number == "estimate":
        typer.echo(json.dumps(stages.estimate_full_run(reg, budget), indent=2))
    elif number == "3":
        stages.stage3(reg, budget, allow_unverified_prompts)
        typer.echo(json.dumps({"top3": stages.top_models()}, indent=2))
    elif number == "4":
        typer.echo(json.dumps({"stage4_models": stages.stage4(reg, budget, allow_unverified_prompts)}))
    elif number == "5":
        stages.stage5(reg, budget)
    elif number == "6-train":
        from t2sbench.train_lora import run_variant

        for base in stages.lora_bases(reg):
            run_variant(base, "L1", allow_unverified=allow_unverified_prompts)
            for n in stages.LORA_CURVE:
                run_variant(base, "L2", n, allow_unverified=allow_unverified_prompts)
            run_variant(base, "L3", allow_unverified=allow_unverified_prompts)
    elif number == "6-eval":
        stages.stage6_eval(reg, budget, allow_unverified=allow_unverified_prompts)
    elif number == "7":
        from t2sbench.report import build_report

        typer.echo(build_report())
    else:
        raise typer.BadParameter(number)


@app.command("train-lora")
def train_lora_cmd(base: str, variant: str = typer.Argument(..., help="L1 | L2 | L3"),
                   n: int | None = typer.Option(None, help="L2: number of pairs (50/100/200/400)"),
                   allow_unverified_prompts: bool = False) -> None:
    """Train one LoRA adapter (QLoRA 4-bit) on the GPU."""
    from t2sbench.train_lora import run_variant

    typer.echo(json.dumps(run_variant(base, variant, n, allow_unverified=allow_unverified_prompts), indent=1))


@app.command("report")
def report_cmd() -> None:
    """Build report/REPORT.md and charts from results/."""
    from t2sbench.report import build_report

    typer.echo(build_report())


@app.command("check-sql")
def check_sql(db: Path, sql: str) -> None:
    """Validate and run one query read-only (debug helper)."""
    from t2sbench.executor import execute

    r = execute(db, sql)
    typer.echo(json.dumps({"ok": r.ok, "error": r.error, "columns": r.columns,
                           "rows": r.rows[:20], "truncated": r.truncated}, default=str, indent=1))


if __name__ == "__main__":
    app()
