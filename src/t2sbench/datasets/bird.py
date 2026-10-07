"""BIRD (public) download, loading and the stratified 150-question dev sample."""

from __future__ import annotations

import hashlib
import json
import logging
import random
import shutil
import urllib.request
import zipfile
from collections import Counter
from pathlib import Path

from t2sbench.executor import execute

log = logging.getLogger(__name__)

BIRD_URLS = {
    "dev": "https://bird-bench.oss-cn-beijing.aliyuncs.com/dev.zip",
    "train": "https://bird-bench.oss-cn-beijing.aliyuncs.com/train.zip",
}
DIFFICULTIES = ("simple", "moderate", "challenging")


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _extract_nested(zip_path: Path, dest: Path) -> None:
    with zipfile.ZipFile(zip_path) as z:
        z.extractall(dest)
    # BIRD ships databases as a zip inside the zip (dev_databases.zip / train_databases.zip)
    for inner in list(dest.rglob("*_databases.zip")):
        with zipfile.ZipFile(inner) as z:
            z.extractall(inner.parent)
        inner.unlink()


def fetch(split: str, root: str | Path = "data/bird", url: str | None = None) -> Path:
    """Download and extract BIRD ``split`` ("dev" or "train"). Returns the split dir."""
    root = Path(root)
    target = root / split
    if (target / ".complete").exists():
        log.info("BIRD %s already present in %s", split, target)
        return target
    target.mkdir(parents=True, exist_ok=True)
    url = url or BIRD_URLS[split]
    zip_path = root / f"{split}.zip"
    if not zip_path.exists():
        log.info("downloading %s", url)
        tmp = zip_path.with_suffix(".part")
        with urllib.request.urlopen(url) as r, open(tmp, "wb") as f:  # noqa: S310 (fixed https URL)
            shutil.copyfileobj(r, f, length=1 << 20)
        tmp.rename(zip_path)
    _extract_nested(zip_path, target)
    manifest = {"split": split, "url": url, "zip_sha256": _sha256(zip_path)}
    (target / "MANIFEST.json").write_text(json.dumps(manifest, indent=2) + "\n")
    (target / ".complete").touch()
    return target


def locate(split: str, root: str | Path = "data/bird") -> tuple[Path, Path]:
    """Return (questions json, databases dir) for an extracted split, whatever the archive layout."""
    base = Path(root) / split
    qfile = next(iter(sorted(base.rglob(f"{split}.json"))), None)
    dbdir = next(iter(sorted(p for p in base.rglob(f"{split}_databases") if p.is_dir())), None)
    if qfile is None or dbdir is None:
        raise FileNotFoundError(f"BIRD {split} not found under {base}; run `t2sbench fetch-bird {split}`")
    # databases may be nested one more level (train_databases/train_databases/<db_id>)
    if not any(dbdir.glob("*/*.sqlite")) and (dbdir / dbdir.name).is_dir():
        dbdir = dbdir / dbdir.name
    return qfile, dbdir


def db_path(dbdir: Path, db_id: str) -> Path:
    return dbdir / db_id / f"{db_id}.sqlite"


def load_questions(split: str, root: str | Path = "data/bird") -> tuple[list[dict], Path]:
    qfile, dbdir = locate(split, root)
    with open(qfile, encoding="utf-8") as f:
        data = json.load(f)
    for i, q in enumerate(data):
        q.setdefault("question_id", i)
    return data, dbdir


def stratified_sample(questions: list[dict], n: int = 150, seed: int = 42,
                      key: str = "difficulty") -> list[dict]:
    """Proportional stratified sample by ``key`` (largest-remainder allocation)."""
    groups: dict[str, list[dict]] = {}
    for q in questions:
        groups.setdefault(q[key], []).append(q)
    total = len(questions)
    quotas = {g: n * len(v) / total for g, v in groups.items()}
    alloc = {g: int(x) for g, x in quotas.items()}
    for g in sorted(groups, key=lambda g: (-(quotas[g] - alloc[g]), g))[: n - sum(alloc.values())]:
        alloc[g] += 1
    rng = random.Random(seed)
    out = []
    for g in sorted(groups):
        members = sorted(groups[g], key=lambda q: q["question_id"])
        out.extend(rng.sample(members, alloc[g]))
    return sorted(out, key=lambda q: q["question_id"])


def build_dev_sample(root: str | Path = "data/bird", n: int = 150, seed: int = 42,
                     out: str | Path = "data/bird_dev_sample150.json") -> dict:
    """Sample BIRD dev. Questions whose gold SQL fails, times out or returns >1000 rows are
    removed from the pool *before* sampling (they cannot be scored under our executor
    limits); how many were removed is stored in the output file."""
    questions, dbdir = load_questions("dev", root)
    eligible, excluded = [], []
    for q in questions:
        res = execute(db_path(dbdir, q["db_id"]), q["SQL"])
        if res.ok and not res.truncated:
            eligible.append(q)
        else:
            excluded.append({"question_id": q["question_id"], "db_id": q["db_id"],
                             "reason": res.error or "more than 1000 rows"})
    sample = stratified_sample(eligible, n=n, seed=seed)
    payload = {
        "seed": seed,
        "n": len(sample),
        "pool_size": len(questions),
        "excluded": excluded,
        "by_difficulty": dict(Counter(q["difficulty"] for q in sample)),
        "questions": [
            {"id": f"bird-dev-{q['question_id']}", "question_id": q["question_id"],
             "db_id": q["db_id"], "difficulty": q["difficulty"], "question": q["question"],
             "evidence": q.get("evidence", ""), "sql": q["SQL"]}
            for q in sample
        ],
    }
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    Path(out).write_text(json.dumps(payload, indent=1, ensure_ascii=False) + "\n")
    return payload


def assert_train_dev_disjoint(train: list[dict], dev: list[dict]) -> None:
    """LoRA training must not see BIRD dev databases or questions."""
    dev_dbs = {q["db_id"] for q in dev}
    dev_q = {q["question"].strip().lower() for q in dev}
    bad_db = {q["db_id"] for q in train} & dev_dbs
    bad_q = [q["question"] for q in train if q["question"].strip().lower() in dev_q]
    if bad_db or bad_q:
        raise ValueError(f"BIRD train overlaps dev: dbs={sorted(bad_db)[:5]} questions={bad_q[:3]}")
