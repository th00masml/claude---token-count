"""Stage 7: report/REPORT.md + PNG charts, computed only from files in results/ and adapters/.

Every number in the report comes from the per-question parquet files (or train_meta.json
for LoRA training time and adapter size). Sections without data are marked as such.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import sqlglot  # noqa: E402
from sqlglot import exp  # noqa: E402

from t2sbench.bench import DATASETS, load_results  # noqa: E402
from t2sbench.evaluate import bootstrap_ci, mcnemar  # noqa: E402

log = logging.getLogger(__name__)
CFG_KEYS = ["model", "adapter", "strategy", "dataset"]
DATASET_LABEL = {"bird_ev": "BIRD dev, with evidence", "bird_noev": "BIRD dev, no evidence",
                 "syn_pl": "synthetic, Polish", "syn_en": "synthetic, English"}
ERROR_CATEGORIES = ["wrong tables", "wrong join", "wrong filter value", "wrong aggregation", "dialect", "other"]


# --------------------------------------------------------------------------- helpers

def pct(x) -> str:
    return "n/a" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{100 * x:.1f}%"


def secs(x) -> str:
    return "n/a" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x:.2f} s"


def usd(x) -> str:
    return "n/a" if x is None or (isinstance(x, float) and np.isnan(x)) else f"${x:.4f}"


def md_table(df: pd.DataFrame) -> str:
    if df.empty:
        return "_no data_\n"
    cols = list(df.columns)
    lines = ["| " + " | ".join(map(str, cols)) + " |", "|" + "---|" * len(cols)]
    for _, r in df.iterrows():
        lines.append("| " + " | ".join("" if pd.isna(v) else str(v) for v in r.tolist()) + " |")
    return "\n".join(lines) + "\n"


def cfg_label(r) -> str:
    a = f"@{r['adapter']}" if isinstance(r.get("adapter"), str) and r["adapter"] else ""
    return f"{r['model']}{a} {r['strategy']}"


def pair_key(qid: str) -> str:
    return qid.rsplit("-", 1)[0] if qid.endswith(("-pl", "-en")) else qid


# --------------------------------------------------------------------------- aggregation

def aggregate(df: pd.DataFrame, n_boot: int = 1000) -> pd.DataFrame:
    rows = []
    df = df.assign(adapter=df["adapter"].fillna(""))
    for key, g in df.groupby(["stage"] + CFG_KEYS, dropna=False):
        mean, lo, hi = bootstrap_ci(g["correct"].tolist(), n_boot=n_boot)
        n = len(g)
        cost = g["cost_usd"].astype(float)
        rows.append({
            "stage": key[0], "model": key[1], "adapter": key[2], "strategy": key[3], "dataset": key[4],
            "backend": g["backend"].iloc[0], "precision": g["precision"].iloc[0],
            "max_model_len": g["max_model_len"].iloc[0], "quantized": bool(g["quantized"].iloc[0]),
            "prompt_verified": bool(g["prompt_verified"].all()), "n": n,
            "ex": mean, "ci_lo": lo, "ci_hi": hi,
            "syntax_ok": g["syntax_ok"].mean(), "no_answer": g["no_answer"].mean(),
            "time_median": g["total_time_s"].median(), "time_p90": g["total_time_s"].quantile(0.9),
            "model_time_median": g["model_time_s"].median(), "model_time_p90": g["model_time_s"].quantile(0.9),
            "db_time_median": g["db_time_s"].median(), "db_time_p90": g["db_time_s"].quantile(0.9),
            "in_tokens_mean": g["input_tokens"].mean(), "out_tokens_mean": g["output_tokens"].mean(),
            "cost_per_100": (cost.sum() / n * 100) if cost.notna().all() and n else np.nan,
        })
    return pd.DataFrame(rows)


def paired(df: pd.DataFrame, a: dict, b: dict, key=lambda q: q) -> tuple[pd.Series, pd.Series]:
    """Correctness vectors of two configurations aligned on question (key maps qid -> pairing id)."""
    def sel(c):
        m = np.ones(len(df), dtype=bool)
        for k, v in c.items():
            m &= (df[k].fillna("") == (v or "")).to_numpy() if k == "adapter" else (df[k] == v).to_numpy()
        s = df[m]
        return s.assign(_k=s["qid"].map(key)).drop_duplicates("_k").set_index("_k")["correct"]
    x, y = sel(a), sel(b)
    common = x.index.intersection(y.index)
    return x.loc[common], y.loc[common]


def mcnemar_row(df, a: dict, b: dict, label_a: str, label_b: str, key=lambda q: q) -> dict | None:
    x, y = paired(df, a, b, key)
    if len(x) == 0:
        return None
    r = mcnemar(x.to_numpy(), y.to_numpy())
    return {"A": label_a, "B": label_b, "n": len(x), "EX A": pct(x.mean()), "EX B": pct(y.mean()),
            "B - A (pp)": f"{100 * (y.mean() - x.mean()):+.1f}", "A only": r.b, "B only": r.c,
            "McNemar p": f"{r.p_value:.4f}", "test": r.method}


# --------------------------------------------------------------------------- error analysis

def _tables(t):
    return {x.name.lower() for x in t.find_all(exp.Table)}


def _joins(t):
    out = set()
    for j in t.find_all(exp.Join):
        on = j.args.get("on")
        if on is not None:
            out.add(frozenset(c.name.lower() for c in on.find_all(exp.Column)))
    return out


def _filter_literals(t):
    out = set()
    for w in t.find_all(exp.Where, exp.Having):
        out |= {str(lit.this).lower() for lit in w.find_all(exp.Literal)}
    return out


def _aggregation(t):
    aggs = sorted(type(n).__name__ for n in t.find_all(exp.AggFunc))
    groups = sorted(c.name.lower() for g in t.find_all(exp.Group) for c in g.find_all(exp.Column))
    return aggs, groups


def categorize_error(pred_sql: str | None, gold_sql: str, error: str | None, error_kind: str | None) -> str:
    """Heuristic first guess; report/error_analysis.csv has a manual_category column that overrides it."""
    if not pred_sql:
        return "other"
    err = (error or "").lower()
    if error_kind in ("execution", "validation"):
        if "no such table" in err or "no such column" in err:
            return "wrong tables"
        if "no such function" in err or "syntax" in err or "near" in err or "parse error" in err:
            return "dialect"
        return "other"
    try:
        p = sqlglot.parse_one(pred_sql, read="sqlite")
        g = sqlglot.parse_one(gold_sql, read="sqlite")
    except Exception:
        return "dialect"
    if _tables(p) != _tables(g):
        return "wrong tables"
    if _joins(p) != _joins(g):
        return "wrong join"
    if _filter_literals(p) != _filter_literals(g):
        return "wrong filter value"
    if _aggregation(p) != _aggregation(g):
        return "wrong aggregation"
    return "other"


def best_config(agg: pd.DataFrame) -> pd.Series | None:
    """Configuration with the highest mean EX over the four test sets (must have all four)."""
    main = agg[agg.stage.isin(["stage3", "stage4", "stage5", "stage6"])]
    if main.empty:
        return None
    wide = main.groupby(["model", "adapter", "strategy"]).agg(
        n_sets=("dataset", "nunique"), ex=("ex", "mean")).reset_index()
    full = wide[wide.n_sets == len(DATASETS)]
    pool = full if len(full) else wide
    return pool.sort_values("ex", ascending=False).iloc[0]


def error_analysis(df: pd.DataFrame, best: pd.Series, out_csv: Path, n: int = 20, seed: int = 42) -> pd.DataFrame:
    sel = df[(df.model == best.model) & (df.adapter.fillna("") == best.adapter) & (df.strategy == best.strategy)
             & (~df.correct) & df.stage.isin(["stage3", "stage4", "stage5", "stage6"])]
    if sel.empty:
        return pd.DataFrame()
    sample = sel.sample(n=min(n, len(sel)), random_state=seed)
    out = sample[["qid", "dataset", "question", "gold_sql", "pred_sql", "error"]].copy()
    out["auto_category"] = [categorize_error(r.pred_sql, r.gold_sql, r.error, r.error_kind) for r in sample.itertuples()]
    out["manual_category"] = ""
    if out_csv.exists():  # keep manual labels from a previous run
        prev = pd.read_csv(out_csv).fillna("")
        manual = dict(zip(prev["qid"] + "|" + prev["dataset"], prev.get("manual_category", "")))
        out["manual_category"] = [manual.get(f"{q}|{d}", "") for q, d in zip(out.qid, out.dataset)]
    out.to_csv(out_csv, index=False)
    out["category"] = [m if m in ERROR_CATEGORIES else a for m, a in zip(out.manual_category, out.auto_category)]
    return out


# --------------------------------------------------------------------------- plots

def plot_accuracy_vs_time(agg: pd.DataFrame, path: Path) -> None:
    main = agg[agg.stage.isin(["stage3", "stage4", "stage5", "stage6"])]
    main = main[~((main.stage == "stage6") & (main.adapter == ""))]  # L0 duplicates the stage-3 point
    fig, axes = plt.subplots(len(DATASETS), 2, figsize=(13, 4.2 * len(DATASETS)), squeeze=False)
    for i, ds in enumerate(DATASETS):
        for j, backend in enumerate(["bedrock", "vllm"]):
            ax = axes[i][j]
            d = main[(main.dataset == ds) & (main.backend == backend)]
            ax.set_title(f"{DATASET_LABEL[ds]} | {'Bedrock' if backend == 'bedrock' else 'local RTX 4090'}")
            ax.set_xlabel("median time per question [s]")
            ax.set_ylabel("execution accuracy")
            if d.empty:
                ax.text(0.5, 0.5, "no data", ha="center", va="center", transform=ax.transAxes)
                continue
            sqrl = d.strategy.str.startswith("SQRL")
            ax.errorbar(d[~sqrl].time_median, d[~sqrl].ex,
                        yerr=[d[~sqrl].ex - d[~sqrl].ci_lo, d[~sqrl].ci_hi - d[~sqrl].ex],
                        fmt="o", color="#4C72B0", alpha=0.75, label="configuration")
            if sqrl.any():
                ax.scatter(d[sqrl].time_median, d[sqrl].ex, marker="*", s=260, color="#C44E52", zorder=5, label="SQRL")
            if backend == "bedrock":
                b = d.sort_values("ex", ascending=False).iloc[0]
                ax.scatter([b.time_median], [b.ex], s=320, facecolors="none", edgecolors="#DD8452", linewidths=2.5,
                           zorder=6, label="best Bedrock")
            for _, r in d.iterrows():
                ax.annotate(cfg_label(r), (r.time_median, r.ex), fontsize=7, xytext=(3, 3), textcoords="offset points")
            ax.set_ylim(0, 1)
            ax.legend(fontsize=8, loc="lower right")
    fig.suptitle("Accuracy vs. response time (Bedrock and local times are not comparable)", fontsize=13)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def plot_learning_curve(df_lora: pd.DataFrame, path: Path) -> bool:
    d = df_lora[df_lora.adapter.str.contains(r"-L2-\d+$", regex=True, na=False)].copy()
    if d.empty:
        return False
    d["n_pairs"] = d.adapter.str.extract(r"-L2-(\d+)$")[0].astype(int)
    d["base"] = d["model"]
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for (base, ds), g in d.groupby(["base", "dataset"]):
        g = g.sort_values("n_pairs")
        ax.errorbar(g.n_pairs, g.ex, yerr=[g.ex - g.ci_lo, g.ci_hi - g.ex], marker="o", capsize=3, label=f"{base} {ds}")
    l0 = df_lora[df_lora.adapter == ""]
    for (base, ds), g in l0[l0.dataset.isin(["syn_pl", "syn_en"])].groupby(["model", "dataset"]):
        ax.axhline(g.ex.iloc[0], ls=":", lw=1, color="grey")
        ax.annotate(f"L0 {base} {ds}", (d.n_pairs.min(), g.ex.iloc[0]), fontsize=7)
    ax.set_xscale("log", base=2)
    ax.set_xticks(sorted(d.n_pairs.unique()))
    ax.set_xticklabels(sorted(d.n_pairs.unique()))
    ax.set_xlabel("domain training pairs (L2)")
    ax.set_ylabel("execution accuracy (S2)")
    ax.set_ylim(0, 1)
    ax.legend(fontsize=8)
    ax.set_title("L2 learning curve on the synthetic test set")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return True


def plot_sqrl_explorations(df5: pd.DataFrame, path: Path) -> bool:
    d = df5[df5.strategy == "SQRL-agentic"]
    if d.empty:
        return False
    d = d.assign(explorations=d["meta"].map(lambda m: json.loads(m).get("explorations", 0)))
    fig, ax = plt.subplots(figsize=(7, 4))
    for (model, ds), g in d.groupby(["model", "dataset"]):
        counts = g.explorations.value_counts().reindex(range(0, 6), fill_value=0)
        ax.plot(counts.index, counts.values / len(g), marker="o", label=f"{model} {ds}")
    ax.set_xlabel("exploratory queries per question")
    ax.set_ylabel("share of questions")
    ax.legend(fontsize=8)
    ax.set_title("SQRL-agentic: number of explorations")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return True


# --------------------------------------------------------------------------- report

def _main_table(agg: pd.DataFrame, backend: str) -> pd.DataFrame:
    d = agg[(agg.backend == backend) & agg.stage.isin(["stage3", "stage4", "stage5", "stage6"])].sort_values(
        ["dataset", "ex"], ascending=[True, False])
    return pd.DataFrame({
        "stage": d.stage, "model": d.model, "adapter": d.adapter, "strategy": d.strategy, "dataset": d.dataset,
        "n": d.n, "EX": d.ex.map(pct), "95% CI": [f"[{pct(a)}, {pct(b)}]" for a, b in zip(d.ci_lo, d.ci_hi)],
        "valid SQL": d.syntax_ok.map(pct), "no answer": d.no_answer.map(pct),
        "median time": d.time_median.map(secs), "p90 time": d.time_p90.map(secs),
        "median model / db": [f"{secs(a)} / {secs(b)}" for a, b in zip(d.model_time_median, d.db_time_median)],
        "tokens in/out": [f"{a:.0f} / {b:.0f}" for a, b in zip(d.in_tokens_mean, d.out_tokens_mean)],
        "cost / 100 q": d.cost_per_100.map(usd) if backend == "bedrock" else "local",
    })


def build_report(results_dir: Path = Path("results"), report_dir: Path = Path("report"),
                 adapters_dir: Path = Path("adapters"), config_path: Path = Path("config/models.yaml")) -> Path:
    from t2sbench.models.discovery import load_models_config

    report_dir.mkdir(parents=True, exist_ok=True)
    df = load_results(results_dir)
    if df.empty:
        raise RuntimeError(f"no results under {results_dir}")
    df = df[df.stage != "smoke"].copy()
    df["adapter"] = df["adapter"].fillna("")
    agg = aggregate(df)
    agg.to_csv(report_dir / "configs.csv", index=False)
    cfg = load_models_config(config_path)
    S: list[str] = ["# Text-to-SQL benchmark: small models, strategies and LoRA\n",
                    "All numbers below are computed from `results/*/*.parquet` (and `adapters/*/train_meta.json`) "
                    "by `t2sbench report`. Per-configuration aggregates are in `report/configs.csv`.\n"]

    # ---- setup
    S.append("## Setup\n")
    avail_file = results_dir / "model_availability.json"
    if avail_file.exists():
        av = pd.DataFrame(json.loads(avail_file.read_text()))
        S.append("Model availability (from `discover-models`):\n\n" +
                 md_table(av[["name", "backend", "role", "available", "resolved_id", "reason"]]))
    skip_file = Path("logs/skipped.jsonl")
    if skip_file.exists():
        sk = pd.read_json(skip_file, lines=True)
        S.append("\nModels or adapters skipped during runs:\n\n" + md_table(sk[["stage", "model", "reason"]]))
    open_models = agg[agg.backend == "vllm"].drop_duplicates("model")
    if len(open_models):
        S.append("\nOpen models on the RTX 4090. Results of quantized models are a **lower bound** of what the "
                 "model can do at full precision.\n\n" + md_table(pd.DataFrame({
                     "model": open_models.model, "precision": open_models.precision,
                     "max-model-len": open_models.max_model_len.map(lambda v: "" if pd.isna(v) else int(v)),
                     "quantized (lower bound)": open_models.quantized.map({True: "yes", False: "no"})})))
    conc = Path("logs/concurrency.log")
    if conc.exists() and conc.read_text().strip():
        S.append("\nConcurrency reductions (vLLM preemption / KV-cache pressure):\n\n```\n" + conc.read_text() + "```\n")
    unverified = agg[~agg.prompt_verified]
    if len(unverified):
        S.append(f"\n**Warning:** {unverified.model.nunique()} model(s) ran with a prompt template not yet verified "
                 f"against the model card: {', '.join(sorted(unverified.model.unique()))}.\n")

    # ---- main tables
    S.append("\n## Results per configuration\n")
    S.append("Times measured on the local GPU and on Bedrock are reported separately and are not comparable.\n")
    S.append("\n### Bedrock\n\n" + md_table(_main_table(agg, "bedrock")))
    S.append("\n### Local (vLLM, RTX 4090)\n\n" + md_table(_main_table(agg, "vllm")))
    plot_accuracy_vs_time(agg, report_dir / "accuracy_vs_time.png")
    S.append("\n![accuracy vs time](accuracy_vs_time.png)\n")

    # ---- evidence
    S.append("\n## Effect of `evidence` on BIRD\n")
    rows = []
    for (m, a, s), _ in agg[agg.dataset == "bird_ev"].groupby(["model", "adapter", "strategy"]):
        r = mcnemar_row(df, {"model": m, "adapter": a, "strategy": s, "dataset": "bird_noev"},
                        {"model": m, "adapter": a, "strategy": s, "dataset": "bird_ev"},
                        f"{m} {s} no evidence", f"{m} {s} with evidence")
        if r:
            rows.append(r)
    S.append(md_table(pd.DataFrame(rows)))

    # ---- S2 on synthetic
    S.append("\n## Effect of S2 (static values and code meanings) on the synthetic DB\n")
    rows = []
    for ds in ("syn_pl", "syn_en"):
        for m in sorted(agg[(agg.dataset == ds) & (agg.strategy == "S0")].model.unique()):
            r = mcnemar_row(df, {"model": m, "adapter": "", "strategy": "S0", "dataset": ds},
                            {"model": m, "adapter": "", "strategy": "S2", "dataset": ds}, f"{m} S0 {ds}", f"{m} S2 {ds}")
            if r:
                rows.append(r)
    S.append(md_table(pd.DataFrame(rows)))

    # ---- PL vs EN
    S.append("\n## Polish vs English questions on the synthetic DB\n")
    S.append("Pairs share the same gold SQL; the test pairs a question with its translation.\n\n")
    rows = []
    for (m, a, s), _ in agg[agg.dataset == "syn_pl"].groupby(["model", "adapter", "strategy"]):
        r = mcnemar_row(df, {"model": m, "adapter": a, "strategy": s, "dataset": "syn_en"},
                        {"model": m, "adapter": a, "strategy": s, "dataset": "syn_pl"},
                        f"{m}{'@' + a if a else ''} {s} EN", f"{m}{'@' + a if a else ''} {s} PL", key=pair_key)
        if r:
            rows.append(r)
    S.append(md_table(pd.DataFrame(rows)))

    # ---- SQRL
    S.append("\n## SQRL (native protocol)\n")
    df5 = df[df.stage == "stage5"]
    if df5.empty:
        S.append("_no stage-5 results_\n")
    else:
        S.append("Polish questions are **out of the training distribution** (SQRL was trained on English questions "
                 "and SQLite); they are listed separately.\n\n")
        rows, expl_rows, time_rows = [], [], []
        for m in sorted(df5.model.unique()):
            for ds in DATASETS:
                r = mcnemar_row(df5, {"model": m, "strategy": "SQRL-single", "dataset": ds},
                                {"model": m, "strategy": "SQRL-agentic", "dataset": ds},
                                f"{m} single {ds}", f"{m} agentic {ds}")
                if r:
                    r = {"dataset": ds + (" (out of distribution)" if ds == "syn_pl" else ""), **r}
                    r["fixed by agentic"] = r.pop("B only")
                    r["broken by agentic"] = r.pop("A only")
                    rows.append(r)
                d = df5[(df5.model == m) & (df5.dataset == ds) & (df5.strategy == "SQRL-agentic")]
                if len(d):
                    e = d["meta"].map(lambda x: json.loads(x).get("explorations", 0))
                    dist = e.value_counts().reindex(range(6), fill_value=0)
                    expl_rows.append({"model": m, "dataset": ds,
                                      **{f"{k} expl.": int(v) for k, v in dist.items()},
                                      "mean expl. (correct)": f"{e[d.correct].mean():.2f}" if d.correct.any() else "n/a",
                                      "mean expl. (wrong)": f"{e[~d.correct].mean():.2f}" if (~d.correct).any() else "n/a"})
                for st in ("SQRL-agentic", "SQRL-single"):
                    d = df5[(df5.model == m) & (df5.dataset == ds) & (df5.strategy == st)]
                    if len(d):
                        time_rows.append({"model": m, "variant": st, "dataset": ds, "EX": pct(d.correct.mean()),
                                          "median time": secs(d.total_time_s.median()),
                                          "p90 time": secs(d.total_time_s.quantile(0.9))})
        S.append("Agentic vs single, paired on the same questions:\n\n" + md_table(pd.DataFrame(rows)))
        S.append("\nExplorations per question (agentic):\n\n" + md_table(pd.DataFrame(expl_rows)))
        S.append("\nTimes:\n\n" + md_table(pd.DataFrame(time_rows)))
        flags = sorted((results_dir / "stage5").glob("context_flags__*.csv"))
        if flags:
            fl = pd.concat([pd.read_csv(f) for f in flags])
            S.append(f"\nQuestions whose prompt exceeded 90% of the context window: {len(fl)} "
                     "(listed in `results/stage5/context_flags__*.csv`).\n")
        if plot_sqrl_explorations(df5, report_dir / "sqrl_explorations.png"):
            S.append("\n![SQRL explorations](sqrl_explorations.png)\n")

    # ---- LoRA
    S.append("\n## LoRA\n")
    lora_agg = agg[agg.stage == "stage6"]
    if lora_agg.empty:
        S.append("_no stage-6 results_\n")
    else:
        t = lora_agg.copy()
        t["variant"] = [a[len(m) + 1:] if a else "L0 (base)" for a, m in zip(t.adapter, t.model)]
        S.append("EX in S2 (BIRD dev without evidence and the synthetic set, PL and EN separately):\n\n" + md_table(
            pd.DataFrame({"base": t.model, "variant": t.variant, "dataset": t.dataset, "n": t.n, "EX": t.ex.map(pct),
                          "95% CI": [f"[{pct(a)}, {pct(b)}]" for a, b in zip(t.ci_lo, t.ci_hi)],
                          "median time": t.time_median.map(secs)}).sort_values(
                ["base", "dataset", "variant"], key=lambda c: c.str.replace(r"-(\d+)$", lambda m: f"-{int(m.group(1)):04d}", regex=True)
                if c.name == "variant" else c)))
        if plot_learning_curve(lora_agg, report_dir / "lora_learning_curve.png"):
            S.append("\n![L2 learning curve](lora_learning_curve.png)\n")
        rows = []
        for base in sorted(lora_agg.model.unique()):
            cand = lora_agg[(lora_agg.model == base) & (lora_agg.adapter != "") &
                            ~lora_agg.adapter.str.contains(r"-L2-(?:50|100|200)$", regex=True)]
            if cand.empty:
                continue
            best_ad = cand.groupby("adapter").ex.mean().idxmax()
            for ds in sorted(cand.dataset.unique()):
                r = mcnemar_row(df[df.stage == "stage6"], {"model": base, "adapter": "", "strategy": "S2", "dataset": ds},
                                {"model": base, "adapter": best_ad, "strategy": "S2", "dataset": ds},
                                f"{base} L0 {ds}", f"{best_ad} {ds}")
                if r:
                    rows.append(r)
        S.append("\nL0 vs the best LoRA variant (best = highest mean EX over the evaluated sets):\n\n" + md_table(pd.DataFrame(rows)))
        sql_tuned = agg[(agg.model.str.contains("omnisql|arctic|sqrl")) &
                        (agg.strategy.isin(["S2", "SQRL-agentic"])) & agg.dataset.isin(["bird_noev", "syn_pl", "syn_en"])]
        S.append("\nSQL-specialised models on the same sets (OmniSQL / Arctic in S2, SQRL agentic):\n\n" + md_table(
            pd.DataFrame({"model": sql_tuned.model, "strategy": sql_tuned.strategy, "dataset": sql_tuned.dataset,
                          "EX": sql_tuned.ex.map(pct), "precision": sql_tuned.precision})))
        metas = [json.loads(p.read_text()) for p in sorted(adapters_dir.glob("*/train_meta.json"))]
        if metas:
            S.append("\nTraining cost:\n\n" + md_table(pd.DataFrame([{
                "adapter": m["name"], "train examples": m["train_examples"], "dropped (>4096 tok)": m["dropped_too_long"],
                "best checkpoint": m["best_checkpoint"], "val EX": pct(m["best_val_ex"]),
                "train time": f"{m['train_seconds'] / 60:.1f} min",
                "adapter size": f"{m['adapter_size_bytes'] / 2**20:.1f} MiB"} for m in metas])))

    # ---- error analysis
    S.append("\n## Error analysis of the best configuration\n")
    best = best_config(agg)
    if best is None:
        S.append("_no data_\n")
    else:
        ea = error_analysis(df, best, report_dir / "error_analysis.csv")
        S.append(f"Best configuration by mean EX over the test sets: **{best.model}"
                 f"{'@' + best.adapter if best.adapter else ''} {best.strategy}** ({pct(best.ex)}). "
                 f"{len(ea)} random errors (seed 42). Categories come from a heuristic comparison of predicted and "
                 "gold SQL unless `manual_category` is filled in `report/error_analysis.csv` (rerun the report "
                 "after labelling).\n\n")
        if len(ea):
            counts = ea.category.value_counts().reindex(ERROR_CATEGORIES, fill_value=0)
            S.append(md_table(pd.DataFrame({"category": counts.index, "errors": counts.values})))
            S.append("\n<details><summary>the sampled errors</summary>\n\n" + md_table(
                ea[["qid", "dataset", "category", "question"]]) + "\n</details>\n")

    # ---- recommendation draft
    S.append("\n## Recommendation (draft generated from the numbers; review before sharing)\n")
    S.append(recommendation(agg, df, best))
    out = report_dir / "REPORT.md"
    out.write_text("\n".join(S), encoding="utf-8")
    return out


def recommendation(agg: pd.DataFrame, df: pd.DataFrame, best) -> str:
    main = agg[agg.stage.isin(["stage3", "stage4", "stage5", "stage6"])]
    if main.empty:
        return "_no data_\n"
    syn = main[main.dataset.isin(["syn_pl", "syn_en"])]
    lines = []
    target = syn.groupby(["model", "adapter", "strategy", "backend"]).agg(
        ex=("ex", "mean"), t=("time_median", "mean"), cost=("cost_per_100", "mean"), n=("dataset", "nunique")).reset_index()
    target = target[target.n == 2].sort_values("ex", ascending=False)
    if len(target):
        t = target.iloc[0]
        lines.append(f"- Closest to the target use (synthetic DB with codes, PL+EN mean): **{t.model}"
                     f"{'@' + t.adapter if t.adapter else ''} {t.strategy}** with EX {pct(t.ex)}, median time "
                     f"{secs(t.t)} ({'Bedrock' if t.backend == 'bedrock' else 'local GPU'})"
                     + (f", {usd(t.cost)} per 100 questions." if t.backend == "bedrock" else "."))
        for backend, label in (("bedrock", "Bedrock"), ("vllm", "local")):
            b = target[target.backend == backend]
            if len(b):
                r = b.iloc[0]
                lines.append(f"- Best {label} option on that set: {r.model}{'@' + r.adapter if r.adapter else ''} "
                             f"{r.strategy}, EX {pct(r.ex)}, median time {secs(r.t)}.")
    for ds in ("syn_pl", "syn_en"):
        s3 = main[(main.stage == "stage3") & (main.dataset == ds)]
        both = set(s3[s3.strategy == "S0"].model) & set(s3[s3.strategy == "S2"].model)
        s0 = s3[(s3.strategy == "S0") & s3.model.isin(both)].ex.mean()
        s2 = s3[(s3.strategy == "S2") & s3.model.isin(both)].ex.mean()
        if not np.isnan(s0) and not np.isnan(s2):
            lines.append(f"- Static values and code meanings (S0 -> S2, mean over the {len(both)} stage-3 models "
                         f"run with both) on {ds}: {pct(s0)} -> {pct(s2)}.")
    lora = agg[agg.stage == "stage6"]
    if len(lora):
        for base in sorted(lora.model.unique()):
            l0 = lora[(lora.model == base) & (lora.adapter == "") & lora.dataset.isin(["syn_pl", "syn_en"])].ex.mean()
            cand = lora[(lora.model == base) & (lora.adapter != "") & lora.dataset.isin(["syn_pl", "syn_en"])]
            if len(cand) and not np.isnan(l0):
                best_ad = cand.groupby("adapter").ex.mean()
                lines.append(f"- LoRA on {base}: S2 base {pct(l0)} vs best adapter {best_ad.idxmax()} "
                             f"{pct(best_ad.max())} on the synthetic set (see the McNemar table above).")
    if best is not None:
        scope = "all four test sets" if best.n_sets == len(DATASETS) else f"the {best.n_sets} test sets it was run on"
        lines.append(f"- Overall best configuration (mean EX over {scope}): {best.model}"
                     f"{'@' + best.adapter if best.adapter else ''} {best.strategy} ({pct(best.ex)}).")
    return "\n".join(lines) + "\n"
