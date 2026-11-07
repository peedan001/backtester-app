"""Tests for Stage 1 (config.py, db.py, stage1_run.py).

Run either way:
    python tests/test_stage1.py
    python -m pytest -q
"""
import os
import sqlite3
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import config
import db
import stage1_run


# ---- helpers ---------------------------------------------------------------

def new_db():
    con = db.connect(":memory:")
    db.create_schema(con)
    con.execute("INSERT INTO STOCK (ticker, name) VALUES ('AAPL', 'Apple')")
    con.commit()
    return con


def accepts(con, sql, params=()):
    try:
        con.execute(sql, params)
        return True
    except sqlite3.IntegrityError:
        return False


def rejects(con, sql, params=()):
    return not accepts(con, sql, params)


PRICE = "INSERT INTO PRICE (ticker, date, adj_close) VALUES (?, ?, ?)"
EPS = "INSERT INTO FUNDAMENTAL (ticker, period_end, available_date, eps) VALUES (?, ?, ?, ?)"
RUN = "INSERT INTO RUN (top_n, value_weight, cost_bps) VALUES (?, ?, ?)"


# ---- schema shape ----------------------------------------------------------

def test_all_tables_exist():
    con = new_db()
    assert sorted(db.table_names(con)) == sorted(config.TABLES)


def test_foreign_keys_are_on():
    assert db.foreign_keys_on(new_db())


def test_create_schema_twice_is_safe():
    con = new_db()
    db.create_schema(con)
    db.create_schema(con)
    assert sorted(db.table_names(con)) == sorted(config.TABLES)
    assert db.foreign_keys_on(con)


def test_composite_primary_keys():
    con = new_db()
    expected = {
        "PRICE": ["ticker", "date"],
        "FUNDAMENTAL": ["ticker", "period_end"],
        "SIGNAL": ["ticker", "rebalance_date"],
        "HOLDING": ["run_id", "rebalance_date", "ticker"],
        "RESULT": ["run_id", "date"],
    }
    for table, want in expected.items():
        cols = db.columns(con, table)
        got = [c[0] for c in sorted((c for c in cols if c[3] > 0), key=lambda c: c[3])]
        assert got == want, (table, got)


def test_config_is_consistent():
    assert config.START_DATE < config.END_DATE
    assert 0 <= config.MOMENTUM_SKIP_MONTHS < config.MOMENTUM_LOOKBACK_MONTHS
    assert config.MIN_TOP_N <= config.DEFAULT_TOP_N <= config.MAX_TOP_N
    assert 0 <= config.DEFAULT_VALUE_WEIGHT <= 1
    assert 0 <= config.DEFAULT_COST_BPS <= config.MAX_COST_BPS
    assert config.EPS_LAG_DAYS > 0


# ---- PRICE rules -----------------------------------------------------------

def test_good_price_accepted():
    assert accepts(new_db(), PRICE, ("AAPL", "2020-01-02", 74.0))


def test_duplicate_price_rejected():
    con = new_db()
    assert accepts(con, PRICE, ("AAPL", "2020-01-02", 74.0))
    assert rejects(con, PRICE, ("AAPL", "2020-01-02", 75.0))


def test_same_date_different_ticker_is_fine():
    con = new_db()
    con.execute("INSERT INTO STOCK (ticker, name) VALUES ('MSFT', 'Microsoft')")
    assert accepts(con, PRICE, ("AAPL", "2020-01-02", 74.0))
    assert accepts(con, PRICE, ("MSFT", "2020-01-02", 160.0))


def test_same_ticker_different_dates_is_fine():
    # A price history is many dates for one stock; the key must allow that.
    con = new_db()
    assert accepts(con, PRICE, ("AAPL", "2020-01-02", 74.0))
    assert accepts(con, PRICE, ("AAPL", "2020-01-03", 73.0))
    assert accepts(con, PRICE, ("AAPL", "2020-01-06", 74.5))
    assert con.execute("SELECT COUNT(*) FROM PRICE WHERE ticker='AAPL'").fetchone()[0] == 3


def test_same_quarter_different_ticker_and_other_quarters_fine():
    con = new_db()
    con.execute("INSERT INTO STOCK (ticker, name) VALUES ('MSFT', 'Microsoft')")
    assert accepts(con, EPS, ("AAPL", "2020-03-31", "2020-05-15", 0.65))
    assert accepts(con, EPS, ("AAPL", "2020-06-30", "2020-08-14", 0.65))
    assert accepts(con, EPS, ("MSFT", "2020-03-31", "2020-05-01", 1.40))


def test_unknown_ticker_rejected():
    assert rejects(new_db(), PRICE, ("ZZZZ", "2020-01-02", 10.0))


def test_zero_and_negative_price_rejected():
    con = new_db()
    assert rejects(con, PRICE, ("AAPL", "2020-01-02", 0.0))
    assert rejects(con, PRICE, ("AAPL", "2020-01-02", -3.0))


def test_tiny_positive_price_accepted():
    assert accepts(new_db(), PRICE, ("AAPL", "2020-01-02", 0.0001))


def test_null_price_rejected():
    assert rejects(new_db(), PRICE, ("AAPL", "2020-01-02", None))


def test_text_and_infinite_prices_rejected():
    # SQLite is loosely typed: without typeof() and a finite bound, these slip in.
    con = new_db()
    for bad in ["N/A", "abc", "", "74.0 USD", float("inf"), float("-inf"), 1e15]:
        assert rejects(con, PRICE, ("AAPL", "2020-01-02", bad)), bad
    assert con.execute("SELECT COUNT(*) FROM PRICE").fetchone()[0] == 0


def test_integer_price_accepted():
    assert accepts(new_db(), PRICE, ("AAPL", "2020-01-02", 74))


def test_text_and_infinite_eps_rejected():
    con = new_db()
    for bad in ["N/A", "abc", float("inf"), float("-inf"), 1e15, -1e15]:
        assert rejects(con, EPS, ("AAPL", "2020-03-31", "2020-05-15", bad)), bad


def test_text_and_infinite_signal_values_rejected():
    con = new_db()
    for bad in ["N/A", float("inf")]:
        assert rejects(con, SIG, ("AAPL", "2020-01-31", bad, 0.1, 0.5, 0.5)), bad
        assert rejects(con, SIG, ("AAPL", "2020-01-31", 20.0, bad, 0.5, 0.5)), bad


def test_bad_dates_rejected():
    con = new_db()
    for bad in ["2020-13-45", "2020-02-30", "2020/01/02", "02-01-2020",
                "20200102", "", "not a date", "2020-1-2"]:
        assert rejects(con, PRICE, ("AAPL", bad, 74.0)), bad


def test_null_date_rejected():
    assert rejects(new_db(), PRICE, ("AAPL", None, 74.0))


def test_leap_day_accepted_and_non_leap_rejected():
    con = new_db()
    assert accepts(con, PRICE, ("AAPL", "2020-02-29", 74.0))
    assert rejects(con, PRICE, ("AAPL", "2021-02-29", 74.0))


def test_cannot_delete_stock_that_has_prices():
    con = new_db()
    con.execute(PRICE, ("AAPL", "2020-01-02", 74.0))
    assert rejects(con, "DELETE FROM STOCK WHERE ticker = 'AAPL'")


def test_flag_never_delete():
    con = new_db()
    con.execute(PRICE, ("AAPL", "2020-01-02", 74.0))
    con.execute("UPDATE PRICE SET flagged = 1, flag_reason = 'jump > 50%' WHERE ticker = 'AAPL'")
    row = con.execute("SELECT flagged, flag_reason FROM PRICE").fetchone()
    assert row == (1, "jump > 50%")
    assert con.execute("SELECT COUNT(*) FROM PRICE").fetchone()[0] == 1
    assert rejects(con, "UPDATE PRICE SET flagged = 2")


def test_stock_flags_must_be_0_or_1():
    con = new_db()
    assert rejects(con, "INSERT INTO STOCK (ticker, name, in_sp500) VALUES ('X', 'x', 2)")
    assert rejects(con, "INSERT INTO STOCK (ticker, name, is_benchmark) VALUES ('Y', 'y', -1)")


# ---- FUNDAMENTAL rules (anti look-ahead) ----------------------------------

def test_eps_available_before_period_end_rejected():
    assert rejects(new_db(), EPS, ("AAPL", "2020-03-31", "2020-03-30", 0.65))


def test_eps_available_after_period_end_accepted():
    assert accepts(new_db(), EPS, ("AAPL", "2020-03-31", "2020-05-15", 0.65))


def test_eps_available_same_day_accepted():
    assert accepts(new_db(), EPS, ("AAPL", "2020-03-31", "2020-03-31", 0.65))


def test_negative_eps_is_valid_data():
    # Loss-making firms are real; excluding them is a SIGNAL rule (Stage 4), not a data rule.
    assert accepts(new_db(), EPS, ("AAPL", "2020-03-31", "2020-05-15", -1.20))


def test_duplicate_quarter_rejected():
    con = new_db()
    assert accepts(con, EPS, ("AAPL", "2020-03-31", "2020-05-15", 0.65))
    assert rejects(con, EPS, ("AAPL", "2020-03-31", "2020-05-20", 0.70))


def test_eps_unknown_ticker_and_bad_dates_rejected():
    con = new_db()
    assert rejects(con, EPS, ("ZZZZ", "2020-03-31", "2020-05-15", 0.65))
    assert rejects(con, EPS, ("AAPL", "2020-03-99", "2020-05-15", 0.65))
    assert rejects(con, EPS, ("AAPL", "2020-03-31", "2020/05/15", 0.65))


# ---- SIGNAL rules ----------------------------------------------------------

SIG = ("INSERT INTO SIGNAL (ticker, rebalance_date, pe, momentum, pe_rank, momentum_rank) "
       "VALUES (?, ?, ?, ?, ?, ?)")


def test_signal_ranks_between_0_and_1():
    con = new_db()
    assert accepts(con, SIG, ("AAPL", "2020-01-31", 20.0, 0.1, 0.0, 1.0))
    assert rejects(con, SIG, ("AAPL", "2020-02-29", 20.0, 0.1, 1.01, 0.5))
    assert rejects(con, SIG, ("AAPL", "2020-03-31", 20.0, 0.1, 0.5, -0.01))


def test_signal_allows_missing_values():
    # Missing is not zero: a stock with no usable P/E has NULL, not 0.
    assert accepts(new_db(), SIG, ("AAPL", "2020-01-31", None, None, None, None))


# ---- RUN rules -------------------------------------------------------------

def test_run_boundaries():
    con = new_db()
    assert accepts(con, RUN, (config.MIN_TOP_N, 0.0, 0))
    assert accepts(con, RUN, (config.MAX_TOP_N, 1.0, config.MAX_COST_BPS))
    assert rejects(con, RUN, (config.MIN_TOP_N - 1, 0.5, 10))
    assert rejects(con, RUN, (config.MAX_TOP_N + 1, 0.5, 10))
    assert rejects(con, RUN, (20, -0.01, 10))
    assert rejects(con, RUN, (20, 1.01, 10))
    assert rejects(con, RUN, (20, 0.5, -0.01))
    assert rejects(con, RUN, (20, 0.5, config.MAX_COST_BPS + 1))


def test_run_pe_mode_only_lagged_or_today():
    con = new_db()
    assert accepts(con, "INSERT INTO RUN (top_n, value_weight, cost_bps, pe_mode) VALUES (20, .5, 10, 'today')")
    assert accepts(con, "INSERT INTO RUN (top_n, value_weight, cost_bps, pe_mode) VALUES (20, .5, 10, 'lagged')")
    assert rejects(con, "INSERT INTO RUN (top_n, value_weight, cost_bps, pe_mode) VALUES (20, .5, 10, 'magic')")
    # default is the honest one
    con.execute(RUN, (20, 0.5, 10))
    assert con.execute("SELECT pe_mode FROM RUN ORDER BY run_id DESC LIMIT 1").fetchone()[0] == "lagged"


# ---- HOLDING and RESULT rules ---------------------------------------------

HOLD = "INSERT INTO HOLDING (run_id, rebalance_date, ticker, weight) VALUES (?, ?, ?, ?)"
RES = ("INSERT INTO RESULT (run_id, date, gross_return, cost, net_return, benchmark_return) "
       "VALUES (?, ?, ?, ?, ?, ?)")


def test_holding_rules():
    con = new_db()
    con.execute(RUN, (20, 0.5, 10))
    assert accepts(con, HOLD, (1, "2020-01-31", "AAPL", 0.05))
    assert rejects(con, HOLD, (1, "2020-01-31", "AAPL", 0.05))      # duplicate
    assert rejects(con, HOLD, (1, "2020-02-29", "AAPL", 0.0))       # weight must be > 0
    assert rejects(con, HOLD, (1, "2020-02-29", "AAPL", 1.01))      # weight must be <= 1
    assert rejects(con, HOLD, (99, "2020-02-29", "AAPL", 0.05))     # unknown run
    assert rejects(con, HOLD, (1, "2020-02-29", "ZZZZ", 0.05))      # unknown ticker


def test_result_rules():
    con = new_db()
    con.execute(RUN, (20, 0.5, 10))
    assert accepts(con, RES, (1, "2020-01-31", 0.02, 0.001, 0.019, 0.01))
    assert rejects(con, RES, (1, "2020-01-31", 0.02, 0.001, 0.019, 0.01))   # duplicate day
    assert rejects(con, RES, (1, "2020-02-29", 0.02, -0.001, 0.021, 0.01))  # negative cost
    assert rejects(con, RES, (99, "2020-02-29", 0.02, 0.001, 0.019, 0.01))  # unknown run
    assert accepts(con, RES, (1, "2020-03-31", -0.30, 0.0, -0.30, -0.12))   # losses are fine
    assert rejects(con, RES, (1, "2020-04-30", -1.0, 0.0, -1.0, 0.0))       # cannot lose 100% or more
    assert rejects(con, RES, (1, "2020-04-30", -1.5, 0.0, -1.5, 0.0))
    for bad in ["N/A", float("inf")]:
        assert rejects(con, RES, (1, "2020-05-29", bad, 0.0, 0.0, 0.0)), bad
        assert rejects(con, RES, (1, "2020-05-29", 0.0, bad, 0.0, 0.0)), bad
        assert rejects(con, RES, (1, "2020-05-29", 0.0, 0.0, bad, 0.0)), bad
        assert rejects(con, RES, (1, "2020-05-29", 0.0, 0.0, 0.0, bad)), bad


# ---- the demo script agrees with the database ------------------------------

def test_every_demo_case_behaves_as_designed():
    con = stage1_run.fresh_db()
    for case in stage1_run.CASES:
        outcome, error, ok = stage1_run.attempt(con, case)
        assert ok, (case[0], outcome, error)


def test_demo_exit_code_zero():
    assert stage1_run.demo() == 0


def test_try_mode_runs_one_bad_row():
    assert stage1_run.main(["--try", "8"]) == 0
    assert stage1_run.main(["--try", "1"]) == 1   # row 1 is a GOOD row, so not valid for --try


# ---- plain runner ----------------------------------------------------------

if __name__ == "__main__":
    tests = [(n, f) for n, f in sorted(globals().items()) if n.startswith("test_") and callable(f)]
    failed = 0
    import io
    import contextlib
    for name, fn in tests:
        try:
            with contextlib.redirect_stdout(io.StringIO()):
                fn()
            print(f"PASS  {name}")
        except Exception as e:  # noqa: BLE001
            failed += 1
            print(f"FAIL  {name}: {type(e).__name__}: {e}")
    print(f"\n{len(tests) - failed}/{len(tests)} tests passed")
    sys.exit(1 if failed else 0)
