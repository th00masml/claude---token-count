import numpy as np

from t2sbench.evaluate import bootstrap_ci, execution_match, mcnemar, results_match


def test_multiset_ignores_order():
    assert results_match([(1, "a"), (2, "b")], [(2, "b"), (1, "a")], ordered=False)


def test_multiset_counts_duplicates():
    assert not results_match([(1,), (1,)], [(1,)], ordered=False)


def test_order_matters_when_ordered():
    assert not results_match([(1,), (2,)], [(2,), (1,)], ordered=True)
    assert results_match([(2,), (1,)], [(2,), (1,)], ordered=True)


def test_numeric_rounding_and_types():
    assert results_match([(3,)], [(3.0,)], ordered=False)
    assert results_match([(1 / 3,)], [(0.33333333,)], ordered=False)
    assert not results_match([(0.1234,)], [(0.1235,)], ordered=False)
    assert results_match([(True,)], [(1,)], ordered=False)
    assert results_match([(-0.0,)], [(0.0,)], ordered=False)


def test_column_order_matters_and_none():
    assert not results_match([(1, "a")], [("a", 1)], ordered=False)
    assert results_match([(None, 1)], [(None, 1.0)], ordered=False)
    assert not results_match([], [(None,)], ordered=False)


def test_execution_match_on_synthetic(synth):
    out, _ = synth
    db = out / "prod.sqlite"
    gold = "SELECT LIN_KOD FROM LIN WHERE LIN_STAT = 'A' ORDER BY LIN_KOD"
    assert execution_match(db, "SELECT LIN_KOD FROM LIN WHERE LIN_STAT='A' ORDER BY 1", gold).correct
    wrong_order = execution_match(db, "SELECT LIN_KOD FROM LIN WHERE LIN_STAT='A' ORDER BY 1 DESC", gold)
    assert wrong_order.ordered and not wrong_order.correct
    none = execution_match(db, None, gold)
    assert not none.correct and none.pred_error_kind == "no_answer"
    bad = execution_match(db, "DROP TABLE LIN", gold)
    assert not bad.correct and bad.pred_error_kind == "validation"


def test_bootstrap_ci():
    x = [True] * 70 + [False] * 30
    mean, lo, hi = bootstrap_ci(x, n_boot=1000, seed=42)
    assert mean == 0.7 and lo < 0.7 < hi and 0.55 < lo and hi < 0.85
    assert bootstrap_ci(x, seed=42) == bootstrap_ci(x, seed=42)


def test_mcnemar():
    a = np.array([True] * 30 + [False] * 10 + [True] * 2 + [False] * 8)
    b = np.array([True] * 30 + [True] * 10 + [False] * 2 + [False] * 8)
    r = mcnemar(a, b)
    assert (r.b, r.c) == (2, 10) and r.method == "exact" and r.p_value < 0.05
    same = mcnemar(a, a)
    assert same.p_value == 1.0
