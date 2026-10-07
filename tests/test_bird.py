import pytest

from t2sbench.datasets.bird import assert_train_dev_disjoint, stratified_sample


def _fake(n_simple=925, n_mod=465, n_chal=144):
    qs, i = [], 0
    for d, n in (("simple", n_simple), ("moderate", n_mod), ("challenging", n_chal)):
        for _ in range(n):
            qs.append({"question_id": i, "difficulty": d, "db_id": f"db{i % 11}", "question": f"q{i}"})
            i += 1
    return qs


def test_stratified_sample_is_proportional_and_deterministic():
    qs = _fake()
    s = stratified_sample(qs, n=150, seed=42)
    assert len(s) == 150 and len({q["question_id"] for q in s}) == 150
    by = {d: sum(q["difficulty"] == d for q in s) for d in ("simple", "moderate", "challenging")}
    assert by == {"simple": 90, "moderate": 46, "challenging": 14}
    assert s == stratified_sample(qs, n=150, seed=42)
    assert s != stratified_sample(qs, n=150, seed=7)


def test_train_dev_disjoint():
    dev = [{"db_id": "a", "question": "How many?"}]
    assert_train_dev_disjoint([{"db_id": "b", "question": "Other"}], dev)
    with pytest.raises(ValueError):
        assert_train_dev_disjoint([{"db_id": "a", "question": "x"}], dev)
    with pytest.raises(ValueError):
        assert_train_dev_disjoint([{"db_id": "b", "question": "how many?"}], dev)
