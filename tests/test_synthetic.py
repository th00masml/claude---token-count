import sqlite3
from pathlib import Path

import pytest
import sqlglot
from sqlglot import exp

from t2sbench.datasets.synthetic.build import read_jsonl
from t2sbench.datasets.synthetic.generator import load_codes
from t2sbench.datasets.synthetic.templates import TEMPLATES
from t2sbench.executor import execute

REPO_DATA = Path(__file__).resolve().parents[1] / "data" / "synthetic"


def _load(out):
    return {s: read_jsonl(out / f"{s}.jsonl") for s in ("test", "train", "val")}


def _skeleton(sql: str) -> str:
    tree = sqlglot.parse_one(sql, read="sqlite")
    return tree.transform(lambda n: exp.Placeholder() if isinstance(n, exp.Literal) else n).sql()


# ------------------------------------------------------------------ database

def test_table_count_and_size(synth):
    _, m = synth
    counts = m["row_counts"]
    assert 12 <= len(counts) <= 15
    assert sum(counts.values()) > 5000
    for big in ("PROD_REJ", "PRZEST", "BRAKI", "ZLEC_POZ"):
        assert counts[big] >= 1000, big


def test_legacy_naming(synth):
    out, _ = synth
    con = sqlite3.connect(out / "prod.sqlite")
    tables = [r[0] for r in con.execute("SELECT name FROM sqlite_master WHERE type='table'")]
    assert "ZLEC_POZ" in tables and "MCH_STAT" in tables
    assert all(t == t.upper() for t in tables)


def test_codes_match_codes_yaml(synth):
    """Every status/code column holds only values documented in codes.yaml."""
    out, _ = synth
    con = sqlite3.connect(out / "prod.sqlite")
    codes = load_codes()
    for col, mapping in codes.items():
        table, column = col.split(".")
        values = {str(r[0]) for r in con.execute(f"SELECT DISTINCT {column} FROM {table}")}
        assert values <= set(mapping), f"{col}: undocumented {values - set(mapping)}"
        if "STAT" in column or column in ("KAT", "PLAN_FL"):
            assert len(values) >= 2, f"{col} has a single value"


def test_single_letter_status_codes(synth):
    codes = load_codes()
    for col in ("ZLEC.ZLEC_STAT", "ZLEC_POZ.POZ_STAT", "MCH_STAT.STAT", "BRAKI.BR_STAT",
                "OPER.OPER_STAT", "PRZEST.PLAN_FL", "LIN.LIN_STAT"):
        assert all(len(k) == 1 for k in codes[col]), col


def test_foreign_keys_hold(synth):
    out, _ = synth
    con = sqlite3.connect(out / "prod.sqlite")
    assert con.execute("PRAGMA foreign_key_check").fetchall() == []


def test_internal_consistency(synth):
    out, _ = synth
    con = sqlite3.connect(out / "prod.sqlite")
    # ILOSC_BR equals non-rejected scrap reports
    bad = con.execute("""
        SELECT COUNT(*) FROM PROD_REJ r
        WHERE r.ILOSC_BR <> COALESCE((SELECT SUM(b.ILOSC) FROM BRAKI b
                                      WHERE b.REJ_ID = r.REJ_ID AND b.BR_STAT <> 'O'), 0)""").fetchone()[0]
    assert bad == 0
    # downtime duration matches its timestamps
    bad = con.execute("""SELECT COUNT(*) FROM PRZEST
        WHERE CAST((julianday(TS_DO) - julianday(TS_OD)) * 1440 + 0.5 AS INTEGER) <> CZAS_MIN""").fetchone()[0]
    assert bad == 0
    # planned flag follows the reason category
    bad = con.execute("""SELECT COUNT(*) FROM PRZEST s JOIN PRZYCZ p ON p.PRZYCZ_KOD = s.PRZYCZ_KOD
        WHERE (p.KAT = 'P') <> (s.PLAN_FL = 'T')""").fetchone()[0]
    assert bad == 0


def test_generator_is_deterministic(synth, tmp_path):
    from t2sbench.datasets.synthetic.build import build_all

    out, m = synth
    m2 = build_all(tmp_path, seed=42)
    assert m2 == m
    assert _load(tmp_path) == _load(out)


@pytest.mark.skipif(not (REPO_DATA / "test.jsonl").exists(), reason="no committed question files")
def test_committed_files_match_generator(synth):
    out, _ = synth
    assert _load(REPO_DATA) == _load(out)


# ------------------------------------------------------------------ questions

def test_split_sizes(synth):
    sets = _load(synth[0])
    test, train, val = sets["test"], sets["train"], sets["val"]
    assert len(test) == 80
    assert sum(r["lang"] == "pl" for r in test) == 40 and sum(r["lang"] == "en" for r in test) == 40
    assert len(train) == 400
    assert sum(r["lang"] == "pl" for r in train) == 200
    assert len(val) == 40 and sum(r["lang"] == "pl" for r in val) == 20


def test_pl_en_pairs_share_sql(synth):
    test = _load(synth[0])["test"]
    pairs = {}
    for r in test:
        pairs.setdefault(r["pair_id"], {})[r["lang"]] = r
    assert len(pairs) == 40
    for p in pairs.values():
        assert p["pl"]["sql"] == p["en"]["sql"]
        assert p["pl"]["template_id"] == p["en"]["template_id"]
        assert p["pl"]["question"] != p["en"]["question"]


def test_every_gold_sql_executes_and_is_non_empty(synth):
    out, _ = synth
    db = out / "prod.sqlite"
    for split, recs in _load(out).items():
        for r in recs:
            res = execute(db, r["sql"])
            assert res.ok, (r["id"], res.error)
            assert res.rows, (r["id"], "empty result")
            assert not all(v is None for v in res.rows[0]), (r["id"], "NULL result")
            assert not res.truncated, r["id"]


def test_no_test_template_in_training_pool(synth):
    sets = _load(synth[0])
    test_t = {r["template_id"] for r in sets["test"]}
    pool = sets["train"] + sets["val"]
    assert test_t.isdisjoint({r["template_id"] for r in pool})
    assert test_t == {t.id for t in TEMPLATES if t.split == "test"}
    # stronger: no SQL skeleton (literals masked) and no question text is shared
    test_skel = {_skeleton(r["sql"]) for r in sets["test"]}
    assert test_skel.isdisjoint({_skeleton(r["sql"]) for r in pool})
    assert {r["question"] for r in sets["test"]}.isdisjoint({r["question"] for r in pool})


def test_train_and_val_are_disjoint_instances(synth):
    sets = _load(synth[0])
    assert {r["sql"] for r in sets["train"]}.isdisjoint({r["sql"] for r in sets["val"]})
    assert len({r["sql"] for r in sets["train"]}) == 400
    ids = [r["id"] for s in sets.values() for r in s]
    assert len(ids) == len(set(ids))
