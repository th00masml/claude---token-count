"""Command line entry point: `uv run t2sbench --help`."""

from __future__ import annotations

import json
import logging
from pathlib import Path

import typer

app = typer.Typer(no_args_is_help=True, add_completion=False)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")


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


@app.command("discover-models")
def discover_models(config: Path = Path("config/models.yaml"),
                    out: Path = Path("results/model_availability.json")) -> None:
    """Check which configured Bedrock / HF models are available. Never fails on a missing model."""
    from t2sbench.models.discovery import discover

    for a in discover(config, out):
        typer.echo(f"{'OK  ' if a.available else 'SKIP'} {a.name:24s} {a.resolved_id or '-':50s} {a.reason}")


@app.command("check-sql")
def check_sql(db: Path, sql: str) -> None:
    """Validate and run one query read-only (debug helper)."""
    from t2sbench.executor import execute

    r = execute(db, sql)
    typer.echo(json.dumps({"ok": r.ok, "error": r.error, "columns": r.columns,
                           "rows": r.rows[:20], "truncated": r.truncated}, default=str, indent=1))


if __name__ == "__main__":
    app()
