"""Tests for Stage 2 (synthetic.py, collect.py, stage2_run.py).

Run either way:
    python tests/test_stage2.py
    python -m pytest -q
"""
import contextlib
import datetime as dt
import io
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import collect
import config
import db
import stage2_run
import synthetic

_CACHE = {}


def market(edge=0.05, seed=7, defects=True, value_share=0.5):
    """Build (and cache) a market: the full-size ones take about a second."""
    key = (edge, seed, defects, value_share)
    if key not in _CACHE:
        _CACHE[key] = synthetic.make_market(edge=edge, seed=seed, defects=defects, value_share=value_share)
    return _CACHE[key]


def new_con(*tickers):
    con = db.connect(":memory:")
    db.create_schema(con)
    collect.register_stocks(con, [(t, t, 1, 0) for t in tickers])
    return con


def quiet(fn, *a, **k):
    with contextlib.redirect_stdout(io.StringIO()) as buf:
        out = fn(*a, **k)
    return out, buf.getvalue()


# ===========================================================================
# synthetic: calendar and helpers
# ===========================================================================

def test_business_days_are_monday_to_friday():
    days = synthetic.business_days("2013-01-01", "2013-12-31")
    weekdays = {dt.date.fromisoformat(str(d)).weekday() for d in days}
    assert weekdays == {0, 1, 2, 3, 4}
    assert len(days) == 261


def test_business_days_known_dates():
    days = [str(d) for d in synthetic.business_days("2016-11-21", "2016-11-29")]
    assert days == ["2016-11-21", "2016-11-22", "2016-11-23", "2016-11-24", "2016-11-25",
                    "2016-11-28", "2016-11-29"]          # 26 and 27 November 2016 were a weekend


def test_month_end_indices_are_last_business_day_of_each_month():
    days = synthetic.business_days("2020-01-01", "2020-12-31")
    me = synthetic.month_end_indices(days)
    assert len(me) == 12
    ends = [str(days[i]) for i in me]
    assert ends[0] == "2020-01-31" and ends[1] == "2020-02-28" and ends[3] == "2020-04-30"
    assert ends[4] == "2020-05-29"                         # 30-31 May 2020 were a weekend
    assert ends[-1] == "2020-12-31"


def test_month_end_includes_final_day_even_mid_month():
    days = synthetic.business_days("2020-01-01", "2020-02-14")
    me = synthetic.month_end_indices(days)
    assert [str(days[i]) for i in me] == ["2020-01-31", "2020-02-14"]


def test_quarter_ends():
    q = [str(x) for x in synthetic.quarter_ends(2020, "2021-05-01")]
    assert q == ["2020-03-31", "2020-06-30", "2020-09-30", "2020-12-31", "2021-03-31"]


def test_pct_rank_basic_and_nan_and_edge_cases():
    r = synthetic.pct_rank([10, 30, 20, 40])
    assert list(r) == [0.0, 2 / 3, 1 / 3, 1.0]
    r = synthetic.pct_rank([5, np.nan, 1, 9])
    assert r[2] == 0.0 and r[0] == 0.5 and r[3] == 1.0 and np.isnan(r[1])
    assert list(synthetic.pct_rank([7.0])) == [0.5]
    assert np.isnan(synthetic.pct_rank([np.nan, np.nan])).all()
    assert len(synthetic.pct_rank([])) == 0


def test_ttm_eps_uses_only_filed_quarters_and_last_four():
    eps = np.array([[1.0], [2.0], [3.0], [4.0], [5.0]])
    avail = np.array([["2020-02-01"], ["2020-05-01"], ["2020-08-01"], ["2020-11-01"], ["2021-02-01"]],
                     dtype="datetime64[D]")
    d = np.datetime64
    assert np.isnan(synthetic.ttm_eps_at(eps, avail, d("2020-10-31"))[0])       # only 3 filed
    assert synthetic.ttm_eps_at(eps, avail, d("2020-11-01"))[0] == 10.0          # 1+2+3+4, filed ON the day counts
    assert synthetic.ttm_eps_at(eps, avail, d("2021-01-31"))[0] == 10.0          # fifth not filed yet
    assert synthetic.ttm_eps_at(eps, avail, d("2021-02-01"))[0] == 14.0          # 2+3+4+5


# ===========================================================================
# synthetic: the market and its planted truth
# ===========================================================================

def test_market_is_deterministic_for_a_seed():
    a = synthetic.make_market(n_stocks=20, seed=5, defects=False)
    b = synthetic.make_market(n_stocks=20, seed=5, defects=False)
    c = synthetic.make_market(n_stocks=20, seed=6, defects=False)
    assert a["prices"]["S001"] == b["prices"]["S001"]
    assert a["prices"]["S001"] != c["prices"]["S001"]


def test_market_shape_and_benchmark():
    m = market()
    assert len(m["truth"]["tickers"]) == config.SYNTH_N_STOCKS
    assert len(m["stocks"]) == config.SYNTH_N_STOCKS + 1
    bench = [s for s in m["stocks"] if s[3] == 1]
    assert len(bench) == 1 and bench[0][0] == config.BENCHMARK_TICKER and bench[0][2] == 0
    assert config.BENCHMARK_TICKER in m["prices"]
    assert m["truth"]["days"][0] >= config.DATA_START and m["truth"]["days"][-1] <= config.END_DATE


def test_clean_prices_are_positive_finite_and_dates_unique_sorted():
    m = market(defects=False)
    for t, rows in m["prices"].items():
        dates = [d for d, _ in rows]
        assert dates == sorted(set(dates)), t
        assert all(np.isfinite(p) and p > 0 for _, p in rows), t


def test_no_daily_move_trips_the_jump_flag_in_clean_data():
    P = market(defects=False)["truth"]["price_matrix"]
    worst = np.abs(np.diff(np.log(P), axis=0)).max()
    assert worst < config.JUMP_THRESHOLD, worst


def test_clean_eps_filed_after_quarter_end_with_30_to_55_day_lag():
    m = market(defects=False)
    for t in m["truth"]["tickers"][:10]:
        for period_end, avail, _ in m["eps"][t]:
            lag = (dt.date.fromisoformat(avail) - dt.date.fromisoformat(period_end)).days
            assert 30 <= lag <= 55


def test_truth_signals_match_an_independent_recomputation():
    """Recompute P/E, momentum and the ranks from the PUBLIC feeds only, the way a backtester would."""
    m = market(edge=0.05, seed=3, defects=False)
    tr = m["truth"]
    P = tr["price_matrix"]
    days, me = tr["days"], tr["month_end_idx"]
    tickers = tr["tickers"]
    for k in (13, 40, 100, 155):
        d = days[me[k]]
        pe = []
        for i, t in enumerate(tickers):
            filed = sorted((p, a, e) for p, a, e in m["eps"][t] if a <= d)       # as known on d
            if len(filed) < 4:
                pe.append(np.nan)
                continue
            ttm = sum(e for _, _, e in filed[-4:])
            pe.append(P[me[k], i] / ttm if ttm > 0 else np.nan)
        pe = np.array(pe)
        # EPS in the feed is exactly what the generator used; prices in the feed are rounded to 4 decimals
        feed_close = np.array([[p for dd, p in m["prices"][t] if dd == d][0] for t in tickers])
        assert np.allclose(feed_close, P[me[k]], rtol=1e-4), k
        assert np.array_equal(np.isnan(pe), np.isnan(tr["pe"][k])), k
        # same prices, and EPS exactly as the feed shows it: the match must be essentially exact
        assert np.allclose(pe, tr["pe"][k], rtol=1e-9, equal_nan=True), k
        mom = P[me[k - 1]] / P[me[k - 12]] - 1.0
        assert np.allclose(mom, tr["momentum"][k]), k
        # ranks recomputed from the public numbers agree with the stored truth (allowing a tie flip)
        assert np.nanmax(np.abs(tr["value_rank"][k] - (1 - synthetic.pct_rank(pe)))) < 0.03, k
        assert np.nanmax(np.abs(tr["momentum_rank"][k] - synthetic.pct_rank(mom))) < 1e-9, k


def test_planted_drift_is_a_pure_function_of_the_ranks():
    for share in (0.5, 1.0, 0.0):
        m = market(edge=0.08, seed=4, defects=False, value_share=share)
        tr = m["truth"]
        k = 60
        want = (tr["value_edge"] * np.nan_to_num(2 * tr["value_rank"][k] - 1)
                + tr["momentum_edge"] * np.nan_to_num(2 * tr["momentum_rank"][k] - 1))
        assert np.allclose(tr["planted_drift"][k], want)
        assert abs(tr["planted_drift"][k].max()) <= 0.08 + 1e-12


def test_edge_split_between_value_and_momentum():
    tr = market(edge=0.08, seed=4, defects=False, value_share=0.25)["truth"]
    assert abs(tr["value_edge"] - 0.02) < 1e-12 and abs(tr["momentum_edge"] - 0.06) < 1e-12


def test_planted_edge_makes_best_fifth_beat_worst_fifth():
    big = [synthetic.planted_quintile_spread(synthetic.make_market(seed=s, edge=0.15, defects=False))
           for s in range(1, 6)]
    assert np.mean(big) > 0.08 and min(big) > 0.02


def test_null_world_has_no_systematic_edge():
    vals = [synthetic.planted_quintile_spread(synthetic.make_market(seed=s, edge=0.0, defects=False))
            for s in range(1, 9)]
    assert abs(np.mean(vals)) < 0.03      # mean of 8 noisy worlds; one world is +-3%
    assert max(abs(v) for v in vals) < 0.10


def test_null_world_does_not_beat_the_benchmark_by_construction():
    """Volatility drag is corrected: a random stock earns its beta times the market return, on average."""
    diffs = []
    for s in range(1, 7):
        m = synthetic.make_market(seed=s, edge=0.0, defects=False)
        tr = m["truth"]
        P, me = tr["price_matrix"], tr["month_end_idx"]
        spy = np.array([p for _, p in m["prices"][config.BENCHMARK_TICKER]])
        r = P[me[1:]] / P[me[:-1]] - 1
        rs = spy[me[1:]] / spy[me[:-1]] - 1
        diffs.append((r.mean(axis=1) - rs).mean() * 12)
    assert abs(np.mean(diffs)) < 0.01


def test_edge_does_not_inflate_the_overall_market_much():
    a = synthetic.make_market(seed=2, edge=0.0, defects=False)["truth"]["price_matrix"]
    b = synthetic.make_market(seed=2, edge=0.10, defects=False)["truth"]["price_matrix"]
    ga = np.log(a[-1] / a[0]).mean()
    gb = np.log(b[-1] / b[0]).mean()
    assert abs(gb - ga) < 0.5        # over 13 years; a drift that did not average to zero would be far larger


def test_edge_and_argument_validation():
    for kwargs in ({"edge": -0.01}, {"edge": 0.6}, {"value_share": 1.5}, {"value_share": -0.1}, {"n_stocks": 5}):
        try:
            synthetic.make_market(**kwargs)
        except ValueError:
            continue
        raise AssertionError(f"{kwargs} should have raised")


def test_defects_change_only_the_raw_feeds_not_the_truth():
    a, b = market(defects=False), market(defects=True)
    assert np.array_equal(a["truth"]["price_matrix"], b["truth"]["price_matrix"])
    assert a["defects"] == [] and len(b["defects"]) == 8
    assert {d["kind"] for d in b["defects"]} == {
        "spike", "zero_price", "duplicate_rows", "gap", "eps_before_period",
        "text_price", "infinite_price", "bad_date"}


def test_fake_fetchers_follow_the_contract():
    f = synthetic.fake_fetchers(market(defects=False))
    uni = f["universe"]()
    assert len(uni) == config.SYNTH_N_STOCKS and all(len(x) == 2 for x in uni)
    assert config.BENCHMARK_TICKER not in [t for t, _ in uni]
    rows = f["prices"]("S001", "2020-01-01", "2020-01-31")
    assert rows and all("2020-01-01" <= d <= "2020-01-31" for d, _ in rows)
    assert len(f["eps"]("S001")[0]) == 3
    for name, args in (("prices", ("NOPE", "2020-01-01", "2020-02-01")), ("eps", ("NOPE",))):
        try:
            f[name](*args)
        except KeyError:
            continue
        raise AssertionError("unknown ticker should raise KeyError")


# ===========================================================================
# collect: collecting
# ===========================================================================

def feed(rows):
    return lambda ticker, start, end: rows


def test_collect_prices_inserts_and_is_idempotent():
    con = new_con("AAA")
    rows = [("2020-01-02", 10.0), ("2020-01-03", 11.0)]
    r1 = collect.collect_prices(con, feed(rows), ["AAA"], "2020-01-01", "2020-12-31")
    assert (r1.inserted, r1.updated, r1.unchanged, r1.tickers_ok) == (2, 0, 0, 1)
    r2 = collect.collect_prices(con, feed(rows), ["AAA"], "2020-01-01", "2020-12-31")
    assert (r2.inserted, r2.updated, r2.unchanged) == (0, 0, 2)
    assert con.execute("SELECT COUNT(*) FROM PRICE").fetchone()[0] == 2


def test_collect_prices_updates_a_changed_value_without_duplicating():
    con = new_con("AAA")
    collect.collect_prices(con, feed([("2020-01-02", 10.0)]), ["AAA"])
    r = collect.collect_prices(con, feed([("2020-01-02", 10.5)]), ["AAA"])
    assert (r.inserted, r.updated, r.unchanged) == (0, 1, 0)
    assert con.execute("SELECT adj_close FROM PRICE").fetchall() == [(10.5,)]


def test_collect_prices_counts_each_kind_of_rejection():
    con = new_con("AAA")
    rows = [("2020-01-02", 10.0), ("2020-01-03", 0.0), ("2020-01-06", "N/A"), ("2020-01-07", float("inf")),
            ("2020-13-45", 10.0), ("2020-01-08", None), ("2020-01-09", -5.0), ("2020-01-10", "12.5")]
    r = collect.collect_prices(con, feed(rows), ["AAA"])
    assert r.inserted == 2                                           # 10.0 and the numeric string "12.5"
    assert r.rejected["price not a positive finite number"] == 2     # 0.0 and -5.0
    assert r.rejected["not a number"] == 2                           # "N/A" and None
    assert r.rejected["not a finite number"] == 1
    assert r.rejected["invalid date"] == 1
    assert r.rejected_total == 6
    assert con.execute("SELECT adj_close FROM PRICE ORDER BY date").fetchall() == [(10.0,), (12.5,)]


def test_collect_prices_notes_repeats_inside_the_feed():
    con = new_con("AAA")
    rows = [("2020-01-02", 10.0), ("2020-01-03", 11.0), ("2020-01-02", 10.0), ("2020-01-03", 11.0)]
    r = collect.collect_prices(con, feed(rows), ["AAA"])
    assert r.inserted == 2 and r.unchanged == 2 and r.duplicates_in_feed == 2
    assert con.execute("SELECT COUNT(*) FROM PRICE").fetchone()[0] == 2


def test_malformed_rows_are_counted_not_crashes():
    con = new_con("AAA")
    r = collect.collect_prices(con, feed([("2020-01-02", 10.0), ("only-one-field",), 5, ("a", "b", "c")]), ["AAA"])
    assert r.inserted == 1 and r.rejected["malformed row"] == 3


def test_one_failing_ticker_does_not_stop_the_others():
    con = new_con("AAA", "BBB", "CCC")

    def fetch(ticker, start, end):
        if ticker == "BBB":
            raise ConnectionError("hung up")
        return [("2020-01-02", 10.0)]
    r = collect.collect_prices(con, fetch, ["AAA", "BBB", "CCC"])
    assert r.tickers_ok == 2 and list(r.failed) == ["BBB"] and "ConnectionError" in r.failed["BBB"]
    assert con.execute("SELECT COUNT(DISTINCT ticker) FROM PRICE").fetchone()[0] == 2


def test_empty_feed_is_a_failure_not_a_silent_success():
    con = new_con("AAA")
    r = collect.collect_prices(con, feed([]), ["AAA"])
    assert r.tickers_ok == 0 and "empty feed" in r.failed["AAA"]
    r = collect.collect_eps(con, lambda t: [], ["AAA"])
    assert r.tickers_ok == 0 and "empty feed" in r.failed["AAA"]


def test_unregistered_ticker_is_reported():
    con = new_con("AAA")
    r = collect.collect_prices(con, feed([("2020-01-02", 10.0)]), ["ZZZ"])
    assert r.failed == {"ZZZ": "not registered in STOCK"}
    assert con.execute("SELECT COUNT(*) FROM PRICE").fetchone()[0] == 0


def test_progress_callback_called_per_ticker():
    con = new_con("AAA", "BBB")
    seen = []
    collect.collect_prices(con, feed([("2020-01-02", 10.0)]), ["AAA", "BBB"], progress=lambda i, n, t: seen.append((i, n, t)))
    assert seen == [(1, 2, "AAA"), (2, 2, "BBB")]


def test_collect_eps_rules_and_idempotence():
    con = new_con("AAA")
    rows = [("2020-03-31", "2020-05-15", 0.65), ("2020-06-30", "2020-06-01", 0.7),      # filed BEFORE quarter end
            ("2020-09-30", "2020-11-10", "N/A"), ("2020-12-31", "2021-02-14", float("inf")),
            ("2019-12-31", "2020-02-14", -0.3)]                                          # losses are fine
    r = collect.collect_eps(con, lambda t: rows, ["AAA"])
    assert r.inserted == 2
    assert r.rejected["EPS filed before its quarter ended"] == 1
    assert r.rejected["not a number"] == 1 and r.rejected["not a finite number"] == 1
    r2 = collect.collect_eps(con, lambda t: rows, ["AAA"])
    assert (r2.inserted, r2.updated, r2.unchanged) == (0, 0, 2)
    revised = [("2020-03-31", "2020-05-15", 0.70)]
    r3 = collect.collect_eps(con, lambda t: revised, ["AAA"])
    assert r3.updated == 1
    assert con.execute("SELECT eps FROM FUNDAMENTAL WHERE period_end='2020-03-31'").fetchone()[0] == 0.70


def test_register_stocks_updates_without_duplicating():
    con = new_con()
    collect.register_stocks(con, [("AAA", "Old Name", 1, 0)])
    collect.register_stocks(con, [("AAA", "New Name", 1, 0)])
    assert con.execute("SELECT ticker, name FROM STOCK").fetchall() == [("AAA", "New Name")]


def test_report_summary_mentions_failures_and_rejections():
    con = new_con("AAA")
    r = collect.collect_prices(con, feed([("2020-01-02", 0.0)]), ["AAA", "ZZZ"])
    text = r.summary()
    assert "FAILED ZZZ" in text and "price not a positive finite number" in text


# ===========================================================================
# collect: validation
# ===========================================================================

def load(con, rows, ticker="AAA"):
    collect.collect_prices(con, feed(rows), [ticker])


def test_validator_flags_a_spike_in_and_out_and_keeps_every_row():
    con = new_con("AAA")
    load(con, [("2020-01-02", 100.0), ("2020-01-03", 101.0), ("2020-01-06", 1010.0), ("2020-01-07", 102.0)])
    c = collect.validate_prices(con)
    assert c["jump"] == 2
    assert [(r[1]) for r in collect.flagged_rows(con)] == ["2020-01-06", "2020-01-07"]
    assert con.execute("SELECT COUNT(*) FROM PRICE").fetchone()[0] == 4


def test_validator_jump_threshold_boundary():
    con = new_con("AAA")
    ok = 100.0 * float(np.exp(config.JUMP_THRESHOLD * 0.99))
    bad = 100.0 * float(np.exp(config.JUMP_THRESHOLD * 1.01))
    load(con, [("2020-01-02", 100.0), ("2020-01-03", ok)])
    assert collect.validate_prices(con)["jump"] == 0
    con2 = new_con("AAA")
    load(con2, [("2020-01-02", 100.0), ("2020-01-03", bad)])
    assert collect.validate_prices(con2)["jump"] == 1
    con3 = new_con("AAA")                                  # a big FALL is flagged too
    load(con3, [("2020-01-02", 100.0), ("2020-01-03", 100.0 / float(np.exp(config.JUMP_THRESHOLD * 1.01)))])
    assert collect.validate_prices(con3)["jump"] == 1


def test_validator_flags_gap_but_not_a_long_weekend():
    con = new_con("AAA")
    load(con, [("2020-01-02", 100.0), ("2020-01-03", 100.5), ("2020-01-06", 100.7),    # Fri -> Mon: 3 days
               ("2020-01-30", 101.0)])                                                  # 24 days
    c = collect.validate_prices(con)
    assert c["gap"] == 1
    assert [(r[1], r[3]) for r in collect.flagged_rows(con)] == [("2020-01-30", "gap 24 days")]


def test_validator_gap_boundary():
    con = new_con("AAA")
    d0 = dt.date(2020, 1, 1)
    load(con, [(d0.isoformat(), 100.0), ((d0 + dt.timedelta(days=config.MAX_GAP_DAYS)).isoformat(), 100.0)])
    assert collect.validate_prices(con)["gap"] == 0
    con2 = new_con("AAA")
    load(con2, [(d0.isoformat(), 100.0), ((d0 + dt.timedelta(days=config.MAX_GAP_DAYS + 1)).isoformat(), 100.0)])
    assert collect.validate_prices(con2)["gap"] == 1


def test_validator_is_repeatable_and_clears_stale_flags():
    con = new_con("AAA")
    load(con, [("2020-01-02", 100.0), ("2020-01-03", 1000.0)])
    assert collect.validate_prices(con)["jump"] == 1
    assert collect.validate_prices(con)["jump"] == 1                      # same answer twice
    assert len(collect.flagged_rows(con)) == 1
    load(con, [("2020-01-03", 101.0)])                                    # the source corrected the value
    assert collect.validate_prices(con)["jump"] == 0
    assert collect.flagged_rows(con) == []


def test_validator_reports_both_reasons_on_one_row():
    con = new_con("AAA")
    load(con, [("2020-01-02", 100.0), ("2020-02-20", 1000.0)])
    collect.validate_prices(con)
    reason = collect.flagged_rows(con)[0][3]
    assert "gap" in reason and "jump" in reason


def test_validator_keeps_tickers_separate():
    con = new_con("AAA", "BBB")
    load(con, [("2020-01-02", 100.0), ("2020-01-03", 100.5)], "AAA")
    load(con, [("2020-01-02", 5000.0), ("2020-01-03", 5001.0)], "BBB")   # no jump across tickers
    assert collect.validate_prices(con)["jump"] == 0


def test_validator_on_empty_database():
    assert dict(collect.validate_prices(new_con())) == {}


# ===========================================================================
# collect: the end-to-end path on the synthetic market
# ===========================================================================

def test_collect_all_on_defect_market_handles_every_planted_defect():
    m = market(defects=True)
    con = db.connect(":memory:")
    db.create_schema(con)
    res = collect.collect_all(con, synthetic.fake_fetchers(m))
    assert not res["prices"].failed and not res["eps"].failed
    results = stage2_run.verify_defects(con, m, res)
    assert len(results) == 8
    assert all(ok for _, ok, _ in results), [(d["kind"], seen) for d, ok, seen in results if not ok]
    assert res["prices"].rejected_total == 4 and res["eps"].rejected_total == 1
    assert res["prices"].duplicates_in_feed == 5
    assert dict(res["flags"]) == {"gap": 1, "jump": 2}


def test_collect_all_on_clean_market_has_no_rejections_or_flags():
    m = market(defects=False)
    con = db.connect(":memory:")
    db.create_schema(con)
    res = collect.collect_all(con, synthetic.fake_fetchers(m))
    assert res["prices"].rejected_total == 0 and res["eps"].rejected_total == 0
    assert sum(res["flags"].values()) == 0
    n_days = len(m["truth"]["days"])
    assert con.execute("SELECT COUNT(*) FROM PRICE").fetchone()[0] == n_days * (config.SYNTH_N_STOCKS + 1)
    assert con.execute("SELECT COUNT(*) FROM STOCK WHERE is_benchmark = 1").fetchone()[0] == 1


def test_collect_all_limit_and_ticker_selection():
    m = market(defects=False)
    con = db.connect(":memory:")
    db.create_schema(con)
    res = collect.collect_all(con, synthetic.fake_fetchers(m), limit=3)
    assert res["tickers"] == ["S001", "S002", "S003"]
    assert con.execute("SELECT COUNT(DISTINCT ticker) FROM PRICE").fetchone()[0] == 4      # 3 + benchmark


def test_stored_prices_match_the_synthetic_truth():
    m = market(defects=False)
    con = db.connect(":memory:")
    db.create_schema(con)
    collect.collect_all(con, synthetic.fake_fetchers(m), limit=2)
    d, p = con.execute("SELECT date, adj_close FROM PRICE WHERE ticker='S002' ORDER BY date LIMIT 1 OFFSET 500").fetchone()
    j = m["truth"]["days"].index(d)
    assert abs(p - m["truth"]["price_matrix"][j, 1]) < 1e-3


# ===========================================================================
# collect: parsers, CSV, backup, diagnosis
# ===========================================================================

SP500_HTML = """<html><body><table id="other"><tr><td>nope</td></tr></table>
<table class="wikitable sortable" id="constituents"><tbody>
<tr><th>Symbol</th><th>Security</th><th>GICS Sector</th></tr>
<tr><td><a href="x">MMM</a>\n</td><td>3M</td><td>Industrials</td></tr>
<tr><td><a href="y">BRK.B</a></td><td>Berkshire Hathaway</td><td>Financials</td></tr>
<tr><td><a href="z">BF.B</a></td><td>Brown-Forman</td><td>Staples</td></tr>
</tbody></table></body></html>"""


def test_parse_sp500_html():
    assert collect.parse_sp500_html(SP500_HTML) == [("MMM", "3M"), ("BRK-B", "Berkshire Hathaway"), ("BF-B", "Brown-Forman")]


def test_parse_sp500_html_failures_are_explained():
    for html in ("<html></html>", "<table id='constituents'><tr><th>Foo</th></tr></table>",
                 "<table id='constituents'><tr><th>Symbol</th><th>Security</th></tr></table>"):
        try:
            collect.parse_sp500_html(html)
        except ValueError as e:
            assert str(e)
            continue
        raise AssertionError("should have raised")


def _entry(start, end, filed, val, form):
    return {"start": start, "end": end, "filed": filed, "val": val, "form": form}


def edgar_facts(entries, concept="EarningsPerShareDiluted"):
    return {"facts": {"us-gaap": {concept: {"units": {"USD/shares": entries}}}}}


def test_parse_edgar_quarters_derive_q4_and_keep_earliest_filing():
    facts = edgar_facts([
        _entry("2020-01-01", "2020-03-31", "2020-05-05", 1.0, "10-Q"),
        _entry("2020-04-01", "2020-06-30", "2020-08-05", 1.1, "10-Q"),
        _entry("2020-01-01", "2020-06-30", "2020-08-05", 2.1, "10-Q"),       # six-month figure: ignored
        _entry("2020-07-01", "2020-09-30", "2020-11-05", 1.2, "10-Q"),
        _entry("2020-01-01", "2020-12-31", "2021-02-20", 5.0, "10-K"),       # full year: Q4 = 5.0 - 3.3
        _entry("2020-01-01", "2020-03-31", "2021-05-05", 0.9, "10-Q"),       # restated later: must NOT replace
    ])
    rows = collect.parse_edgar_eps(facts)
    assert rows == [("2020-03-31", "2020-05-05", 1.0), ("2020-06-30", "2020-08-05", 1.1),
                    ("2020-09-30", "2020-11-05", 1.2), ("2020-12-31", "2021-02-20", 1.7)]


def test_parse_edgar_no_q4_when_a_quarter_is_missing():
    facts = edgar_facts([
        _entry("2020-01-01", "2020-03-31", "2020-05-05", 1.0, "10-Q"),
        _entry("2020-07-01", "2020-09-30", "2020-11-05", 1.2, "10-Q"),
        _entry("2020-01-01", "2020-12-31", "2021-02-20", 5.0, "10-K"),
    ])
    assert [r[0] for r in collect.parse_edgar_eps(facts)] == ["2020-03-31", "2020-09-30"]


def test_parse_edgar_falls_back_to_basic_eps_and_survives_junk():
    facts = edgar_facts([_entry("2020-01-01", "2020-03-31", "2020-05-05", 1.0, "10-Q"),
                         {"start": "bad", "end": "2020-03-31", "filed": "2020-05-05", "val": 1, "form": "10-Q"},
                         {"val": 3}], concept="EarningsPerShareBasic")
    assert collect.parse_edgar_eps(facts) == [("2020-03-31", "2020-05-05", 1.0)]
    assert collect.parse_edgar_eps({}) == []
    assert collect.parse_edgar_eps({"facts": {"us-gaap": {}}}) == []


def test_parse_edgar_output_is_accepted_by_the_database():
    facts = edgar_facts([_entry("2020-01-01", "2020-03-31", "2020-05-05", 1.0, "10-Q")])
    con = new_con("AAA")
    r = collect.collect_eps(con, lambda t: collect.parse_edgar_eps(facts), ["AAA"])
    assert r.inserted == 1 and r.rejected_total == 0


def test_edgar_fetcher_needs_a_user_agent_and_sends_it():
    old = os.environ.pop(config.EDGAR_USER_AGENT_ENV, None)
    try:
        fetch = collect.make_edgar_eps_fetcher(http_get=lambda url, headers=None: b"{}", pause=0)
        try:
            fetch("AAPL")
        except RuntimeError as e:
            assert config.EDGAR_USER_AGENT_ENV in str(e)
        else:
            raise AssertionError("should have raised without a user agent")
    finally:
        if old is not None:
            os.environ[config.EDGAR_USER_AGENT_ENV] = old
    calls = []

    def fake_get(url, headers=None):
        calls.append((url, headers))
        if url.endswith("company_tickers.json"):
            return b'{"0": {"cik_str": 320193, "ticker": "AAPL", "title": "Apple"}}'
        return b'{"facts": {"us-gaap": {}}}'
    fetch = collect.make_edgar_eps_fetcher(user_agent="Test Person test@example.com", http_get=fake_get, pause=0)
    assert fetch("aapl") == []
    assert all(h["User-Agent"] == "Test Person test@example.com" for _, h in calls)
    assert any("CIK0000320193" in u for u, _ in calls)
    try:
        fetch("NOPE")
    except KeyError:
        pass
    else:
        raise AssertionError("unknown ticker should raise")


class _FakeSeries:
    def __init__(self, items):
        self._items = items

    def dropna(self):
        return self

    def items(self):
        return self._items


class _FakeIndex:
    def __init__(self, s):
        self.s = s

    def strftime(self, fmt):
        return self.s


class _FakeFrame:
    def __init__(self, close):
        self._close = close

    def __len__(self):
        return 1

    def __getitem__(self, key):
        return self._close


def test_yahoo_frame_to_rows_flat_layout():
    frame = _FakeFrame(_FakeSeries([(_FakeIndex("2020-01-02"), 74.5), (_FakeIndex("2020-01-03"), 75.0)]))
    assert collect.yahoo_frame_to_rows(frame, "AAPL") == [("2020-01-02", 74.5), ("2020-01-03", 75.0)]
    assert collect.yahoo_frame_to_rows(None, "AAPL") == []


def _write(path, text, encoding="utf-8"):
    with open(path, "w", encoding=encoding) as fh:
        fh.write(text)


def test_csv_loaders_and_fetchers(tmp_dir=None):
    import tempfile
    with tempfile.TemporaryDirectory() as d:
        u, p, e = (os.path.join(d, n) for n in ("u.csv", "p.csv", "e.csv"))
        _write(u, "Ticker,Name\naapl,Apple\nMSFT,\n,blank\n")
        _write(p, "date,ticker,adj_close\n2020-01-02,aapl,74.5\n2020-01-03,AAPL,N/A\n2020-13-45,AAPL,70\n2019-01-02,AAPL,60\n",
               encoding="utf-8-sig")
        _write(e, "ticker,period_end,eps,available_date\nAAPL,2020-03-31,0.65,2020-05-01\nAAPL,2020-06-30,0.7,\n")
        assert collect.load_universe_csv(u) == [("AAPL", "Apple"), ("MSFT", "MSFT")]
        eps_rows, estimated = collect.load_eps_csv(e)
        assert estimated == 1
        assert eps_rows["AAPL"][0] == ("2020-03-31", "2020-05-01", "0.65")
        assert eps_rows["AAPL"][1][1] == (dt.date(2020, 6, 30) + dt.timedelta(days=config.EPS_LAG_DAYS)).isoformat()
        f = collect.make_csv_fetchers(u, p, e)
        assert [t for t, _ in f["universe"]()] == ["AAPL", "MSFT"]
        got = f["prices"]("AAPL", "2020-01-01", "2020-12-31")
        assert ("2020-01-02", "74.5") in got and ("2020-01-03", "N/A") in got
        assert ("2019-01-02", "60") not in got                   # outside the date range
        assert ("2020-13-45", "70") in got                        # impossible date passes through so it can be COUNTED
        con = new_con("AAPL")
        r = collect.collect_prices(con, f["prices"], ["AAPL"], "2020-01-01", "2020-12-31")
        assert r.inserted == 1 and r.rejected["not a number"] == 1 and r.rejected["invalid date"] == 1
        assert f["prices"]("NOPE", "2020-01-01", "2020-12-31") == []


def test_csv_with_wrong_columns_is_explained():
    import tempfile
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "p.csv")
        _write(p, "day,symbol,close\n2020-01-02,AAPL,74.5\n")
        try:
            collect.load_prices_csv(p)
        except ValueError as e:
            assert "adj_close" in str(e)
        else:
            raise AssertionError("should have raised")


def test_with_backup_falls_back_on_error_and_on_empty():
    calls = []

    def primary_bad(x):
        raise ConnectionError("down")

    def primary_empty(x):
        return []

    def primary_good(x):
        return [x]

    def backup(x):
        calls.append(x)
        return [x * 2]
    for primary, want, used in ((primary_bad, [4], "backup"), (primary_empty, [4], "backup"), (primary_good, [2], "primary")):
        fn = collect.with_backup(primary, backup)
        assert fn(2) == want and fn.used[used] == 1
    assert calls == [2, 2]


def test_run_checks_reports_each_source_alone_with_a_fix():
    def boom():
        raise ConnectionError("no internet")
    good = {"universe": lambda: [("AAPL", "Apple")], "prices": lambda t, s, e: [("2020-01-02", 1.0)],
            "eps": lambda t: [("2020-03-31", "2020-05-01", 1.0)]}
    res = collect.run_checks(good)
    assert [r["ok"] for r in res] == [True, True, True] and all(r["fix"] == "" for r in res)
    bad = dict(good, universe=boom, eps=lambda t: [])
    res = collect.run_checks(bad)
    assert [r["ok"] for r in res] == [False, True, False]
    assert "ConnectionError" in res[0]["message"] and "--universe-csv" in res[0]["fix"]
    assert "returned nothing" in res[2]["message"] and config.EDGAR_USER_AGENT_ENV in res[2]["fix"]


# ===========================================================================
# stage2_run: the walkthrough and the playgrounds
# ===========================================================================

def test_demo_succeeds_with_defects_clean_and_null_world():
    for kwargs in ({}, {"clean": True}, {"edge": 0.0, "clean": True}):
        code, out = quiet(stage2_run.demo, **kwargs)
        assert code == 0, (kwargs, out[-400:])
        assert "Check question" in out


def test_demo_saves_a_database_file():
    import sqlite3
    import tempfile
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "x.db")
        code, _ = quiet(stage2_run.demo, db_path=path, clean=True)
        assert code == 0
        con = sqlite3.connect(path)
        assert con.execute("SELECT COUNT(*) FROM STOCK").fetchone()[0] == config.SYNTH_N_STOCKS + 1


def test_every_try_case_runs_and_behaves():
    cases = stage2_run._try_cases()
    assert len(cases) == 10
    outputs = [stage2_run.run_try_case(c) for c in cases]
    assert "flagged on AAA: [('2020-01-" in outputs[0] and "jump" in outputs[0]
    assert "rejected" in outputs[1] and "BBB 30" in outputs[1]
    assert "not a number" in outputs[2]
    assert "not a finite number" in outputs[3]
    assert "invalid date" in outputs[4]
    assert "repeated inside the feed" in outputs[5] and "AAA 30" in outputs[5]
    assert "gap" in outputs[6]
    assert "EPS filed before its quarter ended" in outputs[7]
    assert "ConnectionError" in outputs[8] and "BBB 30" in outputs[8] and "AAA 0" in outputs[8]
    assert "empty feed" in outputs[9] and "BBB 30" in outputs[9]


def test_try_mode_and_explore_and_check_run():
    code, out = quiet(stage2_run.main, ["--try", "1"])
    assert code == 0 and "10 times too big" in out
    code, out = quiet(stage2_run.main, ["--try", "99"])
    assert code == 1
    code, out = quiet(stage2_run.main, ["--explore", "0.1"])
    assert code == 0 and "planted edge 10.0% a year" in out


def test_explore_zero_edge_message_and_values():
    vals, out = quiet(stage2_run.explore_one, 0.0, seeds=range(1, 4))
    assert len(vals) == 3 and "pure luck" in out


def test_check_mode_exit_codes():
    good = {"universe": lambda: [("AAPL", "Apple")], "prices": lambda t, s, e: [("2020-01-02", 1.0)],
            "eps": lambda t: [("2020-03-31", "2020-05-01", 1.0)]}
    code, out = quiet(stage2_run.check_mode, good)
    assert code == 0 and "All sources work" in out
    code, out = quiet(stage2_run.check_mode, dict(good, prices=lambda t, s, e: 1 / 0))
    assert code == 1 and "FAIL" in out and "fix:" in out


def test_live_mode_runs_end_to_end_with_fake_fetchers():
    import tempfile
    f = synthetic.fake_fetchers(synthetic.make_market(n_stocks=12, seed=2, defects=True))
    with tempfile.TemporaryDirectory() as d:
        code, out = quiet(stage2_run.live_mode, f, os.path.join(d, "live.db"), 5)
    assert code == 0 and "PRICE:" in out and "FUNDAMENTAL:" in out


# ---- plain runner ----------------------------------------------------------

if __name__ == "__main__":
    tests = [(n, f) for n, f in sorted(globals().items()) if n.startswith("test_") and callable(f)]
    failed = 0
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
