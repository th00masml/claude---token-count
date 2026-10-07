import sqlite3

import pytest

from t2sbench.executor import connect_readonly, execute


@pytest.fixture()
def db(tmp_path):
    p = tmp_path / "t.sqlite"
    con = sqlite3.connect(p)
    con.execute("CREATE TABLE t (a INTEGER, b TEXT)")
    con.executemany("INSERT INTO t VALUES (?, ?)", [(i, f"x{i}") for i in range(2500)])
    con.commit()
    con.close()
    return p


def test_select(db):
    r = execute(db, "SELECT COUNT(*) FROM t")
    assert r.ok and r.rows == [(2500,)] and r.columns == ["COUNT(*)"]


def test_row_cap(db):
    r = execute(db, "SELECT a FROM t")
    assert r.ok and len(r.rows) == 1000 and r.truncated


def test_write_rejected_by_validator(db):
    r = execute(db, "DELETE FROM t")
    assert not r.ok and r.error_kind == "validation"


def test_readonly_even_without_validator(db):
    # defence in depth: mode=ro + query_only must stop writes the validator never saw
    r = execute(db, "DELETE FROM t", validate=False)
    assert not r.ok and r.error_kind == "execution"
    assert execute(db, "SELECT COUNT(*) FROM t").rows == [(2500,)]
    con = connect_readonly(db)
    with pytest.raises(sqlite3.OperationalError):
        con.execute("PRAGMA query_only = OFF; ")
        con.execute("INSERT INTO t VALUES (1, 'y')")
    con.close()


def test_timeout(db):
    sql = ("WITH RECURSIVE c(n) AS (SELECT 1 UNION ALL SELECT n + 1 FROM c) "
           "SELECT COUNT(*) FROM c")
    r = execute(db, sql, timeout_s=0.5)
    assert not r.ok and r.error_kind == "timeout"
    assert r.elapsed_s < 3


def test_execution_error(db):
    r = execute(db, "SELECT nope FROM t")
    assert not r.ok and r.error_kind == "execution" and "nope" in r.error
