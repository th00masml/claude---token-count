"""Static SQL validation with sqlglot, applied before *every* execution.

Rules: exactly one statement, the statement is a query (SELECT / WITH ... SELECT /
set operation), and no DDL, DML, PRAGMA, ATTACH, transaction control or
SELECT ... INTO anywhere in the tree.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import sqlglot
from sqlglot import exp
from sqlglot.errors import ParseError, TokenError

_FORBIDDEN_NAMES = [
    "Insert", "Update", "Delete", "Create", "Drop", "Alter", "Command", "Pragma",
    "Attach", "Detach", "Merge", "TruncateTable", "Transaction", "Commit", "Rollback",
    "Into", "Use", "Set", "Analyze", "Copy", "LoadData", "Grant", "Revoke",
]
FORBIDDEN_NODES = tuple(getattr(exp, n) for n in _FORBIDDEN_NAMES if hasattr(exp, n))
FORBIDDEN_FUNCTIONS = {"load_extension", "readfile", "writefile", "edit", "fts3_tokenizer"}


@dataclass(frozen=True)
class ValidationResult:
    ok: bool
    error: str | None = None
    expression: exp.Expression | None = None


def validate_sql(sql: str, dialect: str = "sqlite") -> ValidationResult:
    if sql is None or not sql.strip():
        return ValidationResult(False, "empty query")
    # sqlglot logs a warning and falls back to exp.Command on unsupported syntax;
    # we reject Command anyway, so silence the noise.
    logger = logging.getLogger("sqlglot")
    level = logger.level
    logger.setLevel(logging.ERROR)
    try:
        statements = [s for s in sqlglot.parse(sql, read=dialect) if s is not None]
    except (ParseError, TokenError) as e:
        return ValidationResult(False, f"parse error: {str(e).splitlines()[0][:300]}")
    finally:
        logger.setLevel(level)
    if len(statements) != 1:
        return ValidationResult(False, f"expected exactly one statement, got {len(statements)}")
    tree = statements[0]
    if not isinstance(tree, exp.Query):
        return ValidationResult(False, f"only SELECT/WITH queries are allowed, got {type(tree).__name__}")
    for node in tree.walk():
        if isinstance(node, FORBIDDEN_NODES):
            return ValidationResult(False, f"forbidden construct: {type(node).__name__}")
        if isinstance(node, (exp.Anonymous, exp.Func)):
            name = (node.name if isinstance(node, exp.Anonymous) else node.sql_name()).lower()
            if name in FORBIDDEN_FUNCTIONS:
                return ValidationResult(False, f"forbidden function: {name}")
    return ValidationResult(True, None, tree)


def has_top_level_order_by(sql: str, dialect: str = "sqlite") -> bool:
    """True if the outermost query has ORDER BY (result order is then significant)."""
    res = validate_sql(sql, dialect)
    if not res.ok:
        return False
    return res.expression.args.get("order") is not None
