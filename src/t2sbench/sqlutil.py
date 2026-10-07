"""Extracting SQL from model output and result signatures for voting."""

from __future__ import annotations

import re

_FENCE = re.compile(r"```[ \t]*(?:(?:sqlite|sql)\b)?[ \t]*\n?(.*?)```", re.IGNORECASE | re.DOTALL)
_ANSWER = re.compile(r"<answer>(.*?)</answer>", re.IGNORECASE | re.DOTALL)
_BARE = re.compile(r"\b((?:WITH|SELECT)\b.*)", re.IGNORECASE | re.DOTALL)


def strip_think(text: str) -> str:
    """Everything after the last </think> (reasoning models)."""
    if text and "</think>" in text:
        return text.rsplit("</think>", 1)[1]
    return text or ""


def extract_sql(text: str | None) -> str | None:
    """Last fenced SQL block; else the content of <answer>; else the first bare SELECT/WITH."""
    if not text:
        return None
    body = strip_think(text)
    blocks = [b.strip() for b in _FENCE.findall(body) if b.strip()]
    if blocks:
        return _clean(blocks[-1])
    ans = _ANSWER.findall(body)
    if ans:
        inner = ans[-1]
        blocks = [b.strip() for b in _FENCE.findall(inner) if b.strip()]
        return _clean(blocks[-1] if blocks else inner)
    m = _BARE.search(body)
    if m:
        sql = re.split(r"\n\s*\n", m.group(1), maxsplit=1)[0]  # stop at the first blank line
        sql = _clean(sql)
        # a bare match must at least parse as SQL; otherwise it is prose ("select the right table")
        from t2sbench.validator import validate_sql

        return sql if sql and validate_sql(sql).ok else None
    return None


def _clean(sql: str) -> str | None:
    sql = sql.strip()
    sql = re.sub(r"^\s*sql\s*\n", "", sql, flags=re.IGNORECASE)
    sql = sql.strip().rstrip(";").strip()
    return sql or None
