"""Execution accuracy, result comparison and statistics (bootstrap CI, McNemar)."""

from __future__ import annotations

import math
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy import stats

from t2sbench.executor import DEFAULT_MAX_ROWS, DEFAULT_TIMEOUT_S, execute
from t2sbench.validator import has_top_level_order_by

FLOAT_DIGITS = 4
FULL_COMPARE_MAX_ROWS = 200_000  # only when both results exceed the 1000-row cap


def _norm_value(v):
    if isinstance(v, bool):
        v = int(v)
    if isinstance(v, (int, float)):
        f = float(v)
        if math.isnan(f):
            return "NaN"
        r = round(f, FLOAT_DIGITS)
        return 0.0 if r == 0 else r  # fold -0.0 into 0.0
    if isinstance(v, (bytes, bytearray, memoryview)):
        return bytes(v).hex()
    return v


def normalize_rows(rows: Sequence[Sequence]) -> list[tuple]:
    return [tuple(_norm_value(v) for v in row) for row in rows]


def results_match(pred_rows: Sequence[Sequence], gold_rows: Sequence[Sequence], ordered: bool) -> bool:
    """Compare results as a multiset of tuples (or as a list when ``ordered``).

    Numbers are compared as floats rounded to 4 decimals, so 3 == 3.0 and
    0.33333 == 0.333333. Column order matters, column names do not.
    """
    p, g = normalize_rows(pred_rows), normalize_rows(gold_rows)
    if ordered:
        return p == g
    return Counter(p) == Counter(g)


@dataclass
class ExOutcome:
    correct: bool
    pred_ok: bool
    pred_error: str | None
    pred_error_kind: str | None
    gold_error: str | None
    ordered: bool
    pred_truncated: bool
    gold_truncated: bool
    pred_db_time_s: float
    gold_db_time_s: float
    pred_n_rows: int


def execution_match(
    db_path: str | Path,
    pred_sql: str | None,
    gold_sql: str,
    timeout_s: float = DEFAULT_TIMEOUT_S,
    max_rows: int = DEFAULT_MAX_ROWS,
    gold_cache: dict | None = None,
) -> ExOutcome:
    ordered = has_top_level_order_by(gold_sql)
    key = (str(db_path), gold_sql)
    gold = gold_cache.get(key) if gold_cache is not None else None
    if gold is None:
        gold = execute(db_path, gold_sql, timeout_s=timeout_s, max_rows=max_rows)
        if gold_cache is not None:
            gold_cache[key] = gold
    if pred_sql is None or not pred_sql.strip():
        return ExOutcome(False, False, "no answer", "no_answer", gold.error, ordered, False,
                         gold.truncated, 0.0, gold.elapsed_s, 0)
    pred = execute(db_path, pred_sql, timeout_s=timeout_s, max_rows=max_rows)
    # A capped result is not the full result: if only one side hit the cap they differ; if both
    # did, compare the complete results (gold SQL is trusted, pred already ran once).
    if gold.ok and pred.ok and (pred.truncated or gold.truncated):
        if pred.truncated != gold.truncated:
            correct = False
        else:
            g_full = execute(db_path, gold_sql, timeout_s=timeout_s, max_rows=FULL_COMPARE_MAX_ROWS)
            p_full = execute(db_path, pred_sql, timeout_s=timeout_s, max_rows=FULL_COMPARE_MAX_ROWS)
            correct = bool(g_full.ok and p_full.ok and not g_full.truncated and not p_full.truncated
                           and results_match(p_full.rows, g_full.rows, ordered))
        return ExOutcome(correct, pred.ok, pred.error, pred.error_kind, gold.error, ordered,
                         pred.truncated, gold.truncated, pred.elapsed_s, gold.elapsed_s, len(pred.rows))
    correct = bool(gold.ok and pred.ok and results_match(pred.rows, gold.rows, ordered))
    return ExOutcome(correct, pred.ok, pred.error, pred.error_kind, gold.error, ordered,
                     pred.truncated, gold.truncated, pred.elapsed_s, gold.elapsed_s, len(pred.rows))


# --------------------------------------------------------------------------- stats

def bootstrap_ci(correct: Sequence[bool], n_boot: int = 1000, alpha: float = 0.05,
                 seed: int = 42) -> tuple[float, float, float]:
    """Mean and percentile bootstrap (1 - alpha) CI of a 0/1 vector."""
    x = np.asarray(correct, dtype=float)
    if x.size == 0:
        return (float("nan"),) * 3
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, x.size, size=(n_boot, x.size))
    means = x[idx].mean(axis=1)
    lo, hi = np.quantile(means, [alpha / 2, 1 - alpha / 2])
    return float(x.mean()), float(lo), float(hi)


@dataclass
class McNemarResult:
    b: int  # A correct, B wrong
    c: int  # A wrong, B correct
    statistic: float
    p_value: float
    method: str


def mcnemar(a: Sequence[bool], b: Sequence[bool]) -> McNemarResult:
    """Paired McNemar test. Exact binomial when b + c < 25, else chi-square with continuity correction."""
    a = np.asarray(a, dtype=bool)
    bb = np.asarray(b, dtype=bool)
    if a.shape != bb.shape:
        raise ValueError("paired vectors must have the same length")
    n01 = int(np.sum(a & ~bb))
    n10 = int(np.sum(~a & bb))
    n = n01 + n10
    if n == 0:
        return McNemarResult(n01, n10, 0.0, 1.0, "exact")
    if n < 25:
        p = stats.binomtest(min(n01, n10), n, 0.5).pvalue
        return McNemarResult(n01, n10, float(min(n01, n10)), float(min(1.0, p)), "exact")
    chi2 = (abs(n01 - n10) - 1) ** 2 / n
    return McNemarResult(n01, n10, float(chi2), float(stats.chi2.sf(chi2, 1)), "chi2_cc")
