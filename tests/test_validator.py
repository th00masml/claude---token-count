import pytest

from t2sbench.validator import has_top_level_order_by, validate_sql

ALLOWED = [
    "SELECT 1",
    "select * from LIN;",
    "SELECT a FROM t WHERE b IN (SELECT b FROM u)",
    "WITH x AS (SELECT 1 AS a) SELECT a FROM x",
    "WITH RECURSIVE c(n) AS (SELECT 1 UNION ALL SELECT n + 1 FROM c WHERE n < 5) SELECT n FROM c",
    "SELECT a FROM t UNION SELECT a FROM u",
    "SELECT a FROM t EXCEPT SELECT a FROM u",
    "-- comment\nSELECT `weird col` FROM \"T\" LIMIT 20",
    "SELECT CAST(x AS REAL) / 2, strftime('%Y', d) FROM t",
]

REJECTED = [
    ("", "empty"),
    ("   ", "empty"),
    ("SELECT 1; SELECT 2", "exactly one"),
    ("DELETE FROM t", "Delete"),
    ("UPDATE t SET a = 1", "Update"),
    ("INSERT INTO t VALUES (1)", "Insert"),
    ("REPLACE INTO t VALUES (1)", "Command"),
    ("DROP TABLE t", "Drop"),
    ("CREATE TABLE x (a INT)", "Create"),
    ("CREATE TEMP VIEW v AS SELECT 1", "Create"),
    ("ALTER TABLE t ADD COLUMN b", ""),  # parsed as Alter or Command depending on sqlglot
    ("PRAGMA query_only = OFF", "Pragma"),
    ("ATTACH DATABASE 'x.db' AS y", "Attach"),
    ("VACUUM", "Command"),
    ("BEGIN", "Transaction"),
    ("WITH a AS (SELECT 1) DELETE FROM t", "Delete"),
    ("SELECT * INTO t2 FROM t", "Into"),
    ("SELECT load_extension('evil')", "load_extension"),
    ("SELECT 1; DROP TABLE t", "exactly one"),
    ("SELEC 1 FROM", "parse"),
]


@pytest.mark.parametrize("sql", ALLOWED)
def test_allowed(sql):
    res = validate_sql(sql)
    assert res.ok, res.error


@pytest.mark.parametrize("sql,fragment", REJECTED)
def test_rejected(sql, fragment):
    res = validate_sql(sql)
    assert not res.ok
    assert fragment.lower() in res.error.lower()


def test_top_level_order_by():
    assert has_top_level_order_by("SELECT a FROM t ORDER BY a DESC LIMIT 3")
    assert not has_top_level_order_by("SELECT a FROM (SELECT a FROM t ORDER BY a) AS s")
    assert not has_top_level_order_by("SELECT COUNT(*) FROM t")
    assert has_top_level_order_by("WITH x AS (SELECT 1 AS a) SELECT a FROM x ORDER BY a")
