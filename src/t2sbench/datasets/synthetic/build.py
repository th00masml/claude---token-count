"""Build the synthetic DB and its question sets.

Outputs (in ``out_dir``):
  prod.sqlite        the database (not committed, rebuilt deterministically)
  test.jsonl         40 questions x 2 languages (same SQL for the PL/EN pair)
  train.jsonl        400 pairs (200 PL + 200 EN) from train-only templates
  val.jsonl          40 pairs (20 PL + 20 EN), same templates as train, disjoint instances
  manifest.json      seed, row counts, per-template counts
"""

from __future__ import annotations

import json
import random
import sqlite3
from collections import Counter
from pathlib import Path

from t2sbench.datasets.synthetic.generator import generate
from t2sbench.datasets.synthetic.templates import TEMPLATES, Ctx, Template
from t2sbench.executor import execute

DB_ID = "synthetic_prod"
N_TEST_PER_TEMPLATE = 4
N_TRAIN = 400
N_VAL = 40
MAX_TRIES = 300


class TemplateExhausted(RuntimeError):
    pass


def _result_is_informative(rows: list[tuple], allow_zero: bool) -> bool:
    if not rows:
        return False
    if len(rows) == 1 and all(v is None or (not allow_zero and v == 0) for v in rows[0]):
        return False
    return True


def _order_is_deterministic(db: Path, sql: str, key: int, k: int) -> bool:
    probe = sql.replace(f"LIMIT {k}", f"LIMIT {k + 1}")
    res = execute(db, probe)
    if not res.ok or len(res.rows) < k:
        return False
    keys = [r[key] for r in res.rows]
    return all(keys[i] > keys[i + 1] for i in range(len(keys) - 1))


def instantiate(t: Template, ctx: Ctx, db: Path, rng: random.Random, seen_sql: set[str]) -> dict:
    """Draw slots until the gold SQL is new, executes and returns an informative result."""
    for _ in range(MAX_TRIES):
        slots = t.sample(ctx, rng)
        sql = t.sql.format(**slots)
        if sql in seen_sql:
            continue
        res = execute(db, sql)
        if not res.ok:
            raise RuntimeError(f"gold SQL of {t.id} failed: {res.error}\n{sql}")
        if not _result_is_informative(res.rows, t.allow_zero):
            continue
        if t.order_check and not _order_is_deterministic(db, sql, *t.order_check):
            continue
        variant = rng.randrange(len(t.pl))
        seen_sql.add(sql)
        return {
            "template_id": t.id,
            "intent": t.intent,
            "variant": variant,
            "sql": sql,
            "pl": t.pl[variant].format(**slots),
            "en": t.en[variant].format(**slots),
            "n_rows": len(res.rows),
        }
    raise TemplateExhausted(t.id)


def _record(inst: dict, qid: str, split: str, lang: str, pair_id: str | None = None) -> dict:
    return {
        "id": qid,
        "pair_id": pair_id,
        "split": split,
        "lang": lang,
        "db_id": DB_ID,
        "template_id": inst["template_id"],
        "intent": inst["intent"],
        "variant": inst["variant"],
        "question": inst[lang],
        "sql": inst["sql"],
    }


def build_questions(db: Path, seed: int = 42) -> dict[str, list[dict]]:
    rng = random.Random(seed)
    con = sqlite3.connect(f"file:{db.resolve()}?mode=ro", uri=True)
    ctx = Ctx(con)
    seen: set[str] = set()

    test: list[dict] = []
    for t in [t for t in TEMPLATES if t.split == "test"]:
        for i in range(N_TEST_PER_TEMPLATE):
            inst = instantiate(t, ctx, db, rng, seen)
            pair = f"syn-test-{t.id}-{i}"
            test.append(_record(inst, f"{pair}-pl", "test", "pl", pair))
            test.append(_record(inst, f"{pair}-en", "test", "en", pair))

    # Train pool: round-robin over train templates until N_TRAIN + N_VAL unique instances.
    train_t = [t for t in TEMPLATES if t.split == "train"]
    pool: list[dict] = []
    active = list(train_t)
    while len(pool) < N_TRAIN + N_VAL and active:
        for t in list(active):
            if len(pool) >= N_TRAIN + N_VAL:
                break
            try:
                pool.append(instantiate(t, ctx, db, rng, seen))
            except TemplateExhausted:
                active.remove(t)
    con.close()
    if len(pool) < N_TRAIN + N_VAL:
        raise RuntimeError(f"train pool too small: {len(pool)}")

    # Each pool instance is rendered in exactly one language. Languages alternate within
    # each template (so PL and EN cover every template equally), then the pool is shuffled.
    by_t: dict[str, int] = {}
    langs = []
    for inst in pool:
        k = by_t.get(inst["template_id"], rng.randrange(2))
        langs.append("pl" if k % 2 == 0 else "en")
        by_t[inst["template_id"]] = k + 1
    # templates with an odd instance count leave a surplus; flip one instance per template
    # (within templates where that language leads) until the pool is exactly half/half
    half = len(pool) // 2
    for extra in ("pl", "en"):
        other = "en" if extra == "pl" else "pl"
        for j in reversed(range(len(pool))):
            if langs.count(extra) <= half:
                break
            t = pool[j]["template_id"]
            idx = [i for i in range(len(pool)) if pool[i]["template_id"] == t]
            n_extra = sum(langs[i] == extra for i in idx)
            if langs[j] == extra and n_extra > len(idx) - n_extra:
                langs[j] = other
    order = list(range(len(pool)))
    rng.shuffle(order)
    records = [_record(pool[j], f"syn-pool-{i:04d}-{langs[j]}", "train", langs[j])
               for i, j in enumerate(order)]
    pl = [r for r in records if r["lang"] == "pl"]
    en = [r for r in records if r["lang"] == "en"]
    val = pl[: N_VAL // 2] + en[: N_VAL // 2]
    train = pl[N_VAL // 2:] + en[N_VAL // 2:]
    for r in val:
        r["split"] = "val"
        r["id"] = r["id"].replace("syn-pool", "syn-val")
    for r in train:
        r["id"] = r["id"].replace("syn-pool", "syn-train")
    train.sort(key=lambda r: r["id"])
    val.sort(key=lambda r: r["id"])
    return {"test": test, "train": train, "val": val}


def write_jsonl(path: Path, records: list[dict]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def read_jsonl(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def build_all(out_dir: str | Path = "data/synthetic", seed: int = 42) -> dict:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    db = out / "prod.sqlite"
    counts = generate(db, seed=seed)
    sets = build_questions(db, seed=seed)
    for name, recs in sets.items():
        write_jsonl(out / f"{name}.jsonl", recs)
    manifest = {
        "seed": seed,
        "db": db.name,
        "row_counts": counts,
        "questions": {
            name: {
                "n": len(recs),
                "by_lang": dict(Counter(r["lang"] for r in recs)),
                "by_template": dict(sorted(Counter(r["template_id"] for r in recs).items())),
            }
            for name, recs in sets.items()
        },
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")
    return manifest
