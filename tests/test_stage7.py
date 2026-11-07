"""Tests for Stage 7: app_logic.py (everything the app computes) and app.py (driven headlessly with Streamlit's AppTest).

Run either way:
    python tests/test_stage7.py
    python -m pytest -q
"""
import contextlib
import io
import math
import os
import sqlite3
import sys
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import app_logic as L
import bias
import config
import db
import engine
import portfolio as pf
import stage3_run

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
_CACHE = {}


def small(**kw):
    """A cached small fake world (30 stocks, 6 failed)."""
    args = dict(edge=0.05, seed=5, reaction=2.0, n_dead=6, n_stocks=30)
    args.update(kw)
    key = tuple(sorted(args.items()))
    if key not in _CACHE:
        _CACHE[key] = L.open_synthetic(**args)
    return _CACHE[key]


def _close_all():
    for w in _CACHE.values():
        w.con.close()
    _CACHE.clear()


try:
    import pytest

    @pytest.fixture(scope="module", autouse=True)
    def _close_at_end():
        yield
        _close_all()
except ImportError:
    pass


def close(a, b, tol=1e-12):
    return abs(a - b) < tol


def raises(exc, fn, *a, **k):
    try:
        fn(*a, **k)
    except exc as e:
        return str(e)
    raise AssertionError(f"{fn.__name__} should have raised {exc.__name__}")


# ---------------------------------------------------------------------------
# opening worlds
# ---------------------------------------------------------------------------

def test_open_synthetic_describes_the_world():
    w = small()
    assert w.synthetic and w.n_stocks == 30 and w.n_dead == 6 and w.first_date == "2013-01-01" and w.last_date == "2025-12-31"
    assert "edge 5%" in w.label and "6 failed" in w.label and "seed 5" in w.label
    assert w.params == {"edge": 0.05, "seed": 5, "reaction": 2.0, "n_dead": 6, "n_stocks": 30, "defects": False}
    assert db.foreign_keys_on(w.con)


def test_the_shared_connection_can_be_used_from_another_thread():
    import threading
    w = small()
    out = []
    t = threading.Thread(target=lambda: out.append(w.con.execute("SELECT COUNT(*) FROM STOCK").fetchone()[0]))
    t.start()
    t.join()
    assert out == [37]                                           # 30 + 6 failed + SPY


def test_open_synthetic_turns_generator_errors_into_app_errors():
    assert "reaction" in raises(L.AppError, L.open_synthetic, reaction=9.0)
    raises(L.AppError, L.open_synthetic, n_dead=99, n_stocks=10)


def test_open_database_refuses_missing_wrong_and_empty_files():
    assert "Type the path" in raises(L.AppError, L.open_database, "  ")
    assert "No file at" in raises(L.AppError, L.open_database, "/nonexistent/x.db")
    with tempfile.TemporaryDirectory() as d:
        junk = os.path.join(d, "junk.db")
        with open(junk, "w") as f:
            f.write("this is not sqlite")
        raises(L.AppError, L.open_database, junk)
        other = os.path.join(d, "other.db")
        con = sqlite3.connect(other)
        con.execute("CREATE TABLE x (a)")
        con.commit()
        con.close()
        assert "not a backtester database" in raises(L.AppError, L.open_database, other)
        empty = os.path.join(d, "empty.db")
        con = db.connect(empty)
        db.create_schema(con)
        con.close()
        assert "no prices" in raises(L.AppError, L.open_database, empty)


def test_open_database_reads_a_saved_world_and_never_writes():
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "w.db")
        market, con = stage3_run.build_world(n_stocks=20, seed=3, defects=False, reaction=2.0, n_dead=3)
        out = db.connect(path)
        con.backup(out)
        out.close()
        con.close()
        w = L.open_database(path)
        assert not w.synthetic and w.n_stocks == 20 and w.n_dead == 3 and w.label == "Database file: w.db"
        assert L.run_one(w, top_n=5)["metrics"]["periods"] > 50
        try:
            w.con.execute("INSERT INTO STOCK VALUES ('ZZZ', 'x', 1, 0)")
        except sqlite3.OperationalError:
            pass
        else:
            raise AssertionError("the app must open databases read-only")
        w.con.close()


# ---------------------------------------------------------------------------
# one backtest
# ---------------------------------------------------------------------------

def test_run_one_matches_the_engine_directly_and_the_ledger_agrees():
    w = small()
    r = L.run_one(w, top_n=5, value_weight=0.5, cost_bps=10)
    rows = stage3_run.all_signals(w.con)
    direct = engine.metrics(w.env.run(pf.build_portfolio(rows, 5, 0.5, 1.0), 10))
    assert close(r["metrics"]["net"]["cagr"], direct["net"]["cagr"]) and r["ledger_problems"] == []
    assert r["choices"] == {"top_n": 5, "value_weight": 0.5, "buffer": 1.0, "cost_bps": 10, "delay": 1,
                            "pe_mode": "lagged", "universe": "survivors"}
    assert r["dead_held"] == 0                                   # failed stocks are not in the survivors universe
    assert L.run_one(w, top_n=5, universe="all")["dead_held"] > 0


def test_run_one_choices_change_the_answer_in_the_right_direction():
    w = small()
    free = L.run_one(w, top_n=5, cost_bps=0)["metrics"]
    costly = L.run_one(w, top_n=5, cost_bps=100)["metrics"]
    assert close(free["cost_drag_cagr"], 0.0, 1e-9) and costly["cost_drag_cagr"] > 0.005
    assert L.run_one(w, top_n=5, pe_mode="today")["metrics"]["net"]["cagr"] != L.run_one(w, top_n=5)["metrics"]["net"]["cagr"]
    assert L.run_one(w, top_n=5, buffer=1.5)["metrics"]["avg_traded_fraction"] < L.run_one(w, top_n=5)["metrics"]["avg_traded_fraction"]


def test_the_trade_delay_reaches_the_engine_and_the_ledger():
    w = small()
    now, later = L.run_one(w, top_n=5, delay=0), L.run_one(w, top_n=5, delay=3)
    assert now["periods"][5].date != later["periods"][5].date and now["periods"][5].signal_date == later["periods"][5].signal_date
    assert now["ledger_problems"] == [] and later["ledger_problems"] == []
    assert later["periods"][5].date == w.env.run(later["portfolio"], 10, 3)[5].date


def test_every_refusal_is_an_app_error_with_a_useful_message():
    w = small()
    bad = [dict(top_n=0), dict(top_n=501), dict(value_weight=1.2), dict(buffer=0.5), dict(cost_bps=-1),
           dict(cost_bps=config.MAX_COST_BPS + 1), dict(delay=-1), dict(delay=config.MAX_EXEC_DELAY_DAYS + 1),
           dict(pe_mode="nope"), dict(universe="everyone")]
    for kw in bad:
        raises(L.AppError, L.run_one, w, **kw)
    assert "basis points" in raises(L.AppError, L.run_one, w, cost_bps=600)
    assert "trade delay" in raises(L.AppError, L.run_one, w, delay=9)
    assert "P/E version" in raises(L.AppError, L.run_one, w, pe_mode="x")
    L.check_choices(1, 0.0, 1.0, 0, 0, "lagged", "survivors")          # the edges of every range are allowed
    L.check_choices(500, 1.0, config.MAX_BUFFER, config.MAX_COST_BPS, config.MAX_EXEC_DELAY_DAYS, "today", "all")


def test_curve_rows_compound_each_series():
    r = L.run_one(small(), top_n=5)
    rows = L.curve_rows(r["periods"])
    p = r["periods"]
    assert len(rows) == len(p) and rows[0]["date"] == p[0].date
    assert close(rows[-1]["after costs"], engine.net_wealth_path(p)[-1])
    g = b = 1.0
    for q in p:
        g *= 1 + q.gross
        b *= 1 + q.benchmark
    assert close(rows[-1]["before costs"], g) and close(rows[-1]["SPY"], b)
    assert rows[-1]["before costs"] >= rows[-1]["after costs"]


def test_metric_rows_and_facts():
    m = L.run_one(small(), top_n=5)["metrics"]
    rows = {r["measure"]: r for r in L.metric_rows(m)}
    assert list(rows)[0] == "Total return" and len(rows) == 5
    cagr = rows["Return per year (CAGR)"]
    assert close(cagr["before costs"], m["gross"]["cagr"]) and close(cagr["after costs"], m["net"]["cagr"]) and close(cagr["SPY"], m["benchmark"]["cagr"])
    assert cagr["kind"] == "pct" and rows[next(k for k in rows if k.startswith("Sharpe"))]["kind"] == "num"
    assert "2%" in next(k for k in rows if k.startswith("Sharpe"))
    facts = {n: (v, k) for n, v, k in L.summary_facts(m)}
    assert close(facts["Beats SPY by, per year (after costs)"][0], m["excess_cagr_net"])
    assert close(facts["Cost drag, per year"][0], m["gross"]["cagr"] - m["net"]["cagr"])
    assert facts["Stocks sold because their data ended"][1] == "int"


def test_holding_rows_describe_one_month():
    r = L.run_one(small(), top_n=5)
    dates = L.signal_dates(r)
    assert dates == sorted(dates) and all(r["portfolio"][d] for d in dates)
    d = dates[30]
    rows, facts = L.holding_rows(r, d)
    assert len(rows) == 5 and close(sum(x["target weight"] for x in rows), 1.0)
    assert sorted(x["rank"] for x in rows) == [x["rank"] for x in rows] or len({x["rank"] for x in rows}) == 5
    assert facts["signal_date"] == d and facts["trade_date"] > d and 0 <= facts["traded_fraction"] <= 2 and facts["cost"] >= 0
    p = next(q for q in r["periods"] if q.signal_date == d)
    assert facts["trade_date"] == p.date and close(facts["traded_fraction"], p.traded_fraction)
    assert "not a month-end" in raises(L.AppError, L.holding_rows, r, "2020-01-15")


def test_fmt():
    nan = float("nan")
    assert L.fmt(0.1234, "pct") == "12.3%" and L.fmt(-0.05, "pct") == "-5.0%" and L.fmt(1.5, "num") == "1.50"
    assert L.fmt(3.0, "int") == "3" and L.fmt(nan, "pct") == "n/a" and L.fmt(None, "num") == "n/a"
    assert L.fmt(0.0, "pct") == "0.0%"


# ---------------------------------------------------------------------------
# bias lab
# ---------------------------------------------------------------------------

def test_bias_summary_matches_bias_module():
    w = small()
    s = L.bias_summary(w, top_n=5)
    direct = bias.evaluate_world(w.con, w.env, 5)
    assert s["has_failed_stocks"] and len(s["rows"]) == 4
    assert close(s["inflation_today"], direct["inflation_today"]) and close(s["survivorship"], direct["survivorship"])
    assert close(s["total"], direct["total"]) and close(s["inflation_no_lag"], direct["inflation_no_lag"])
    assert close(s["rows"][0]["per year after costs"], direct["lagged"]["net_cagr"])
    assert s["rows"][0]["failed stocks held"] == 0 and s["rows"][3]["failed stocks held"] > 0
    assert s["rows"][1]["version"].startswith("No filing lag") and s["rows"][2]["version"].startswith("Hindsight")


def test_bias_summary_without_failed_stocks_has_three_rows_and_no_survivorship():
    s = L.bias_summary(small(n_dead=0), top_n=5)
    assert not s["has_failed_stocks"] and len(s["rows"]) == 3 and s["survivorship"] is None and s["total"] is None


def test_seeds_summary_rows_and_refusals():
    w = small(n_stocks=20, n_dead=4)
    assert w.n_stocks == 20 and w.n_dead == 4 and w.params["n_stocks"] == 20
    calls = []
    rows = L.seeds_summary(w, [1, 2], top_n=5, progress=lambda i, n: calls.append((i, n)))
    assert calls == [(1, 2), (2, 2)]
    assert [r["bias"] for r in rows] == ["No filing lag", "Today's EPS (hindsight)", "Survivorship (failed stocks left out)", "All shortcuts together"]
    assert all(r["worlds"] == 2 and r["verdict"] in ("INFLATED", "DEFLATED", "within noise") for r in rows)
    direct = bias.worlds_over_seeds([1, 2], n_stocks=20, n_dead=4, top_n=5)
    assert close(rows[1]["mean per year"], bias.summarize([r["inflation_today"] for r in direct])["mean"])
    assert len(L.seeds_summary(small(n_stocks=20, n_dead=0), [1, 2], top_n=5)) == 2
    assert "at least 2" in raises(L.AppError, L.seeds_summary, w, [1])
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "w.db")
        con = db.connect(path)
        small(n_stocks=20, n_dead=4).con.backup(con)
        con.close()
        real = L.open_database(path)
        assert "needs a fake market" in raises(L.AppError, L.seeds_summary, real, [1, 2])
        real.con.close()


# ---------------------------------------------------------------------------
# failed stocks
# ---------------------------------------------------------------------------

def test_graveyard_rows_and_story():
    w = small()
    table, handle = L.graveyard_rows(w, top_n=5)
    assert [r["ticker"] for r in table] != [] and len(table) == 6
    assert [r["stopped trading"] for r in table] == sorted(r["stopped trading"] for r in table)
    assert all(-1 < r["fell from peak"] < 0 and r["days from peak"] >= 0 for r in table)
    assert set(L.failed_tickers(w)) == {r["ticker"] for r in table} == {f"D00{i}" for i in range(1, 7)}
    story = L.graveyard_story(w, handle, "d001")
    assert story[0].startswith("D001: the last") and len(story) > 3
    assert "not a failed stock" in raises(L.AppError, L.graveyard_story, w, handle, "S001")
    assert "no failed stocks" in raises(L.AppError, L.graveyard_rows, small(n_dead=0))


def test_price_history_leaves_out_flagged_prices_and_ends_at_the_failure():
    w = small()
    h = L.price_history(w, "D001")
    assert h == sorted(h) and h[0][0] == "2013-01-01" and len(h) > 500 and all(p > 0 for _, p in h)
    table, _ = L.graveyard_rows(w, top_n=5)
    assert h[-1][0] == next(r["stopped trading"] for r in table if r["ticker"] == "D001")
    w.con.execute("PRAGMA query_only = OFF")
    assert L.price_history(w, "NOPE") == []


# ---------------------------------------------------------------------------
# the app, driven headlessly
# ---------------------------------------------------------------------------

def app(db_path=None):
    from streamlit.testing.v1 import AppTest
    at = AppTest.from_file(os.path.join(ROOT, "app.py"), default_timeout=240)
    at.run()
    return at


def _stderr_quiet():
    return contextlib.redirect_stderr(io.StringIO())


def test_app_starts_clean_and_asks_for_a_run():
    with _stderr_quiet():
        at = app()
    assert not at.exception and len(at.tabs) == 4
    assert [t.label for t in at.tabs] == ["Backtest", "Bias lab", "Graveyard", "How it works"]
    assert any("Run backtest" in i.value for i in at.info)
    assert "Fake market: edge 5%" in at.sidebar.success[0].value


def test_app_runs_a_backtest_and_shows_the_numbers():
    with _stderr_quiet():
        at = app()
        at.button[0].click().run()
    assert not at.exception and not at.error and not at.warning
    labels = {m.label: m.value for m in at.metric}
    assert set(labels) == {"Return per year (after costs)", "SPY per year", "Sharpe (after costs)", "Worst fall"}
    assert labels["SPY per year"].endswith("%") and any("Engine check passed" in c.value for c in at.caption)
    direct = L.run_one(L.open_synthetic(), top_n=20)["metrics"]
    assert labels["Return per year (after costs)"] == L.fmt(direct["net"]["cagr"], "pct")


def test_app_warns_when_the_run_looks_ahead():
    with _stderr_quiet():
        at = app()
        pe = next(s for s in at.selectbox if s.label == "P/E version")
        pe.select("today")
        at.button[0].click().run()
    assert not at.exception and any("inflated on purpose" in w.value for w in at.warning)


def test_app_database_source_reports_a_missing_file_in_the_sidebar():
    with _stderr_quiet():
        at = app()
        at.sidebar.radio[0].set_value("Database file").run()
        at.sidebar.text_input[0].set_value("/nonexistent/file.db").run()
    assert not at.exception and any("No file at" in e.value for e in at.sidebar.error)


def test_app_shows_an_error_when_the_independent_ledger_disagrees():
    from unittest import mock
    with _stderr_quiet(), mock.patch.object(engine, "check_engine", return_value=["2020-03-02: 1.0 vs 1.1"]):
        at = app()
        next(s for s in at.slider if s.label == "Stocks held (N)").set_value(7)
        at.button[0].click().run()
    assert not at.exception and any("disagrees with the engine" in e.value and "2020-03-02" in e.value for e in at.error)


def test_app_says_so_when_the_world_has_no_failed_stocks():
    with _stderr_quiet():
        at = app()
        at.sidebar.slider[2].set_value(0).run()
    assert not at.exception and any("no failed stocks" in i.value for i in at.info)
    assert not any(s.label == "Follow one failed stock" for s in at.selectbox)


def test_app_bias_lab_runs_the_ladder():
    with _stderr_quiet():
        at = app()
        ladder = next(b for b in at.button if b.label == "Run the bias ladder")
        ladder.click().run()
    assert not at.exception and not at.error
    labels = [m.label for m in at.metric]
    assert "No filing lag adds" in labels and "Today's EPS adds" in labels and "Survivorship adds" in labels
    assert any("ONE world" in c.value for c in at.caption)


def test_app_graveyard_tab_shows_a_story_when_there_are_failed_stocks():
    with _stderr_quiet():
        at = app()
    assert not at.exception and any(s.label == "Follow one failed stock" for s in at.selectbox)
    assert any(c.value.startswith("D001: the last") for c in at.code)


def test_app_how_it_works_tab_shows_the_study_guide_with_all_answers():
    with _stderr_quiet():
        at = app()
    titles = [e.label for e in at.expander]
    assert any("Check questions and answers" in t for t in titles) and len(titles) >= 8
    with open(os.path.join(ROOT, "STUDY_GUIDE.md"), encoding="utf-8") as f:
        guide = f.read()
    assert guide.count("### ") >= 7 and "Why does FUNDAMENTAL have two dates" in guide and "vendor's backtest" in guide


def test_the_docs_mention_every_file_that_matters():
    with open(os.path.join(ROOT, "README.md"), encoding="utf-8") as f:
        readme = f.read()
    for name in ("app.py", "app_logic.py", "engine.py", "bias.py", "STUDY_GUIDE.md", "streamlit run app.py", "stage2_run.py --live"):
        assert name in readme, name
    for name in ("app.py", "app_logic.py", "STUDY_GUIDE.md", "README.md"):
        assert os.path.isfile(os.path.join(ROOT, name))


# ---- plain runner ----------------------------------------------------------

if __name__ == "__main__":
    tests = [(n, f) for n, f in sorted(globals().items()) if n.startswith("test_") and callable(f)]
    failed = 0
    for name, fn in tests:
        try:
            with contextlib.redirect_stdout(io.StringIO()), _stderr_quiet():
                fn()
            print(f"PASS  {name}")
        except Exception as e:  # noqa: BLE001
            failed += 1
            print(f"FAIL  {name}: {type(e).__name__}: {e}")
    _close_all()
    print(f"\n{len(tests) - failed}/{len(tests)} tests passed")
    sys.exit(1 if failed else 0)
