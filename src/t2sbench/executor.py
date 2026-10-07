"""Read-only, time- and row-limited SQL execution against SQLite."""

from __future__ import annotations

import sqlite3
import time
from dataclasses import dataclass, field
from pathlib import Path

from t2sbench.validator import validate_sql

DEFAULT_TIMEOUT_S = 10.0
DEFAULT_MAX_ROWS = 1000


@dataclass
class ExecResult:
    ok: bool
    rows: list[tuple] = field(default_factory=list)
    columns: list[str] = field(default_factory=list)
    error: str | None = None
    error_kind: str | None = None  # "validation" | "timeout" | "execution"
    truncated: bool = False
    elapsed_s: float = 0.0


def connect_readonly(db_path: str | Path) -> sqlite3.Connection:
    path = Path(db_path).resolve()
    if not path.exists():
        raise FileNotFoundError(path)
    con = sqlite3.connect(f"file:{path}?mode=ro", uri=True, check_same_thread=False)
    con.execute("PRAGMA query_only = ON")
    return con


def execute(
    db_path: str | Path,
    sql: str,
    timeout_s: float = DEFAULT_TIMEOUT_S,
    max_rows: int = DEFAULT_MAX_ROWS,
    validate: bool = True,
) -> ExecResult:
    """Validate, then run ``sql`` read-only. Never raises for query problems."""
    t0 = time.perf_counter()
    if validate:
        v = validate_sql(sql)
        if not v.ok:
            return ExecResult(False, error=v.error, error_kind="validation",
                              elapsed_s=time.perf_counter() - t0)
    con = connect_readonly(db_path)
    deadline = time.perf_counter() + timeout_s
    timed_out = False

    def _progress() -> int:
        nonlocal timed_out
        if time.perf_counter() > deadline:
            timed_out = True
            return 1  # non-zero aborts the statement
        return 0

    con.set_progress_handler(_progress, 10_000)
    try:
        cur = con.execute(sql)
        rows = cur.fetchmany(max_rows + 1)
        columns = [d[0] for d in (cur.description or [])]
        truncated = len(rows) > max_rows
        return ExecResult(True, rows=rows[:max_rows], columns=columns, truncated=truncated,
                          elapsed_s=time.perf_counter() - t0)
    except sqlite3.Error as e:
        kind = "timeout" if timed_out else "execution"
        msg = f"timeout after {timeout_s:g}s" if timed_out else f"{type(e).__name__}: {e}"
        return ExecResult(False, error=msg, error_kind=kind, elapsed_s=time.perf_counter() - t0)
    finally:
        con.close()
