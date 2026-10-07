"""Schema text for prompts: DDL (S0) and DDL + low-cardinality text values (S2).

S2 lists, for every text column with COUNT(DISTINCT) <= 50, all its values. Columns
whose values are longer than 80 characters are left out (long free text is not useful as
a filter hint and inflates the prompt). For the synthetic DB the meaning of each code from
codes.yaml is appended. Value lists are cached per database file.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from t2sbench.executor import connect_readonly

TEXT_TYPES = ("CHAR", "TEXT", "CLOB", "STRING", "VARCHAR")
MAX_DISTINCT = 50
MAX_VALUE_LEN = 80


def _q(ident: str) -> str:
    return '"' + ident.replace('"', '""') + '"'


def ddl(db_path: str | Path) -> str:
    con = connect_readonly(db_path)
    try:
        rows = con.execute(
            "SELECT sql FROM sqlite_master WHERE type = 'table' AND name NOT LIKE 'sqlite_%' "
            "AND sql IS NOT NULL ORDER BY rowid").fetchall()
    finally:
        con.close()
    return "\n\n".join(r[0].strip().rstrip(";") + ";" for r in rows)


def _cache_file(db_path: Path, cache_dir: Path) -> Path:
    st = db_path.stat()
    h = hashlib.sha256(f"{db_path.resolve()}|{st.st_size}|{st.st_mtime_ns}".encode()).hexdigest()[:16]
    return cache_dir / f"{db_path.stem}-{h}.json"


def value_lists(db_path: str | Path, cache_dir: str | Path = "cache/schema",
                max_distinct: int = MAX_DISTINCT) -> dict[str, list[str]]:
    """{"TABLE.COLUMN": [values...]} for text columns with <= max_distinct distinct values."""
    db_path = Path(db_path)
    cf = _cache_file(db_path, Path(cache_dir))
    if cf.exists():
        return json.loads(cf.read_text())
    con = connect_readonly(db_path)
    out: dict[str, list[str]] = {}
    try:
        tables = [r[0] for r in con.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%' ORDER BY rowid")]
        for t in tables:
            for _, col, ctype, *_ in con.execute(f"PRAGMA table_info({_q(t)})").fetchall():
                ctype = (ctype or "").upper()
                if ctype and not any(x in ctype for x in TEXT_TYPES):
                    continue
                n = con.execute(
                    f"SELECT COUNT(*) FROM (SELECT DISTINCT {_q(col)} FROM {_q(t)} "
                    f"WHERE {_q(col)} IS NOT NULL LIMIT {max_distinct + 1})").fetchone()[0]
                if n == 0 or n > max_distinct:
                    continue
                vals = [r[0] for r in con.execute(
                    f"SELECT DISTINCT {_q(col)} FROM {_q(t)} WHERE {_q(col)} IS NOT NULL ORDER BY 1")]
                if not all(isinstance(v, str) for v in vals):  # untyped column holding numbers
                    continue
                if any(len(v) > MAX_VALUE_LEN for v in vals):
                    continue
                out[f"{t}.{col}"] = vals
    finally:
        con.close()
    cf.parent.mkdir(parents=True, exist_ok=True)
    cf.write_text(json.dumps(out, ensure_ascii=False))
    return out


def render_values(values: dict[str, list[str]], codes: dict | None = None) -> str:
    lines = [f"Column values (all values of text columns with at most {MAX_DISTINCT} distinct values):"]
    for col, vals in values.items():
        meanings = (codes or {}).get(col)
        if meanings:
            parts = []
            for v in vals:
                m = meanings.get(v)
                parts.append(f"'{v}' = {m['en']} (pl: {m['pl']})" if m else f"'{v}'")
            lines.append(f"- {col}: " + "; ".join(parts))
        else:
            lines.append(f"- {col}: " + ", ".join("'" + v.replace("'", "''") + "'" for v in vals))
    return "\n".join(lines)


def schema_text(db_path: str | Path, with_values: bool, codes: dict | None = None,
                cache_dir: str | Path = "cache/schema") -> str:
    text = ddl(db_path)
    if with_values:
        text += "\n\n" + render_values(value_lists(db_path, cache_dir), codes)
    return text
