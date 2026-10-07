"""Extracting SQL from model output and result signatures for voting."""

from __future__ import annotations

import re

_FENCE = re.compile(r"```(?:sql|sqlite)?\s*\n?(.*?)```", re.IGNORECASE | re.DOTALL)
_ANSWER = re.compile(r"<answer>(.*?)</answer>", re.IGNORECASE | re.DOTALL)
_BARE = re.compile(r"\b(WITH|SELECT)\b.*", re.IGNORECASE | re.DOTALL)


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
        sql = m.group(0)
        sql = re.split(r"\n\s*\n", sql, maxsplit=1)[0]  # stop at the first blank line
        return _clean(sql)
    return None


def _clean(sql: str) -> str | None:
    sql = sql.strip()
    sql = re.sub(r"^\s*sql\s*\n", "", sql, flags=re.IGNORECASE)
    sql = sql.strip().rstrip(";").strip()
    return sql or None
