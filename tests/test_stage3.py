"""Tests for Stage 3 (align.py, signals.py, stage3_run.py).

Run either way:
    python tests/test_stage3.py
    python -m pytest -q
"""
import contextlib
import datetime as dt
import io
import os
import sys
import tempfile

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import align
import collect
import config
import db
import signals
import stage3_run
from align import Row

_CACHE = {}


def world(edge=0.05, seed=7, defects=True, value_share=0.5, n_stocks=100):
    """(market, connection, rows with ranks), built once per setting."""
    key = (edge, seed, defects, value_share, n_stocks)
    if key not in _CACHE:
        market, con = stage3_run.build_world(edge=edge, seed=seed, defects=defects,
                                             n_stocks=n_stocks, value_share=value_share)
        _CACHE[key] = (market, con, stage3_run.all_signals(con))
    return _CACHE[key]


def quiet(fn, *a, **k):
    with contextlib.redirect_stdout(io.StringIO()) as buf:
        out = fn(*a, **k)
    return out, buf.getvalue()


# ---- small hand-made databases --------------------------------------------

def bare(prices=None, eps=None, benchmark_days=None):
    """prices: {ticker: [(date, price, flag_reason or None)]}, eps: {ticker: [(period_end, available, eps)]}."""
    con = db.connect(":memory:")
    db.create_schema(con)
    tickers = set((prices or {}).keys()) | set((eps or {}).keys())
    collect.register_stocks(con, [(t, t, 1, 0) for t in sorted(tickers)])
    for t, rows in (prices or {}).items():
        for d, p, why in rows:
            con.execute("INSERT INTO PRICE (ticker, date, adj_close, flagged, flag_reason) VALUES (?, ?, ?, ?, ?)",
                        (t, d, p, 0 if why is None else 1, why))
    for t, rows in (eps or {}).items():
        for q, a, e in rows:
            con.execute("INSERT INTO FUNDAMENTAL (ticker, period_end, available_date, eps) VALUES (?, ?, ?, ?)", (t, q, a, e))
    con.commit()
    return con


def add_benchmark(con, first, last):
    collect.register_stocks(con, [(config.BENCHMARK_TICKER, "B", 0, 1)])
    for d in stage3_run._business_days(first, last):
        con.execute("INSERT INTO PRICE (ticker, date, adj_close) VALUES (?, ?, 100)", (config.BENCHMARK_TICKER, d))
    con.commit()


def one(rows, ticker="AAA", date="2019-06-28"):
    return next(r for r in rows if r.ticker == ticker and r.date == date)


def at(date, **kw):
    """Panel row for AAA in the hand-controlled mini world."""
    con = stage3_run.mini_world(**kw)
    return one(align.build_panel(con, dates=[date]), date=date), con


# ===========================================================================
# month-end calendar
# ===========================================================================

def test_month_ends_are_last_trading_day_of_each_month():
    con = stage3_run.mini_world()
    me = align.month_end_dates(con)
    assert me[:4] == ["2018-01-31", "2018-02-28", "2018-03-30", "2018-04-30"]
    assert "2019-06-28" in me and me[-1] == "2019-12-31" and len(me) == 24
    assert me == sorted(set(me))


def test_unfinished_last_month_is_dropped_and_boundary():
    for last, kept_last_month in (("2019-12-31", True), ("2019-12-27", True), ("2019-12-26", False), ("2019-12-17", False)):
        con = db.connect(":memory:")
        db.create_schema(con)
        add_benchmark(con, "2019-10-01", last)
        me = align.month_end_dates(con)
        assert (me[-1] == last) == kept_last_month, (last, me)
        assert me[-1] in (last, "2019-11-29")


def test_exactly_one_benchmark_is_required():
    con = db.connect(":memory:")
    db.create_schema(con)
    for bad in (0, 2):
        con.execute("DELETE FROM STOCK")
        collect.register_stocks(con, [(f"B{i}", "b", 0, 1) for i in range(bad)])
        try:
            align.month_end_dates(con)
        except ValueError as e:
            assert "benchmark" in str(e)
        else:
            raise AssertionError("should have raised")


def test_synthetic_month_ends_match_the_generators():
    market, con, _ = world()
    assert align.month_end_dates(con) == market["truth"]["month_ends"]


# ===========================================================================
# prices as seen on a date
# ===========================================================================

def book(rows, ticker="AAA"):
    return align.PriceBook(bare(prices={ticker: rows}))


def test_quote_exact_date_and_carry_boundary():
    pb = book([("2020-01-02", 10.0, None)])
    q, why = pb.quote("AAA", "2020-01-02")
    assert (q.price, q.price_date, q.carried_days, why) == (10.0, "2020-01-02", 0, None)
    q, _ = pb.quote("AAA", "2020-01-07")                                    # 5 days later: allowed
    assert q.carried_days == 5 and q.price == 10.0
    q, why = pb.quote("AAA", "2020-01-08")                                  # 6 days later: refused
    assert q is None and why == align.REASON_OLD_PRICE
    q, _ = pb.quote("AAA", "2020-01-08", max_carry=6)
    assert q is not None


def test_quote_never_looks_forward_and_handles_nothing_on_file():
    pb = book([("2020-01-10", 10.0, None)])
    q, why = pb.quote("AAA", "2020-01-09")                                  # the price exists, but only LATER
    assert q is None and why == align.REASON_NO_PRICE
    assert pb.quote("ZZZ", "2020-01-10")[1] == align.REASON_NO_PRICE


def test_quote_uses_latest_price_on_or_before():
    pb = book([("2020-01-02", 10.0, None), ("2020-01-03", 11.0, None), ("2020-01-06", 12.0, None)])
    assert pb.quote("AAA", "2020-01-05")[0].price == 11.0
    assert pb.quote("AAA", "2020-01-06")[0].price == 12.0


def test_jump_flag_makes_a_price_unusable_but_gap_flag_does_not():
    pb = book([("2020-01-02", 10.0, None), ("2020-01-03", 100.0, "jump +2.30 log"),
               ("2020-02-10", 10.5, "gap 38 days"), ("2020-02-11", 10.6, "gap 2 days; jump +0.50 log")])
    assert pb.quote("AAA", "2020-01-03") == (None, align.REASON_FLAGGED)
    q, why = pb.quote("AAA", "2020-02-10")
    assert q.price == 10.5 and why is None
    assert pb.quote("AAA", "2020-02-11") == (None, align.REASON_FLAGGED)
    assert pb.quote("AAA", "2020-01-02")[0].price == 10.0


def test_flagged_price_is_not_replaced_by_an_older_good_one():
    pb = book([("2020-01-02", 10.0, None), ("2020-01-03", 100.0, "jump +2.30 log")])
    assert pb.quote("AAA", "2020-01-04") == (None, align.REASON_FLAGGED)     # the latest price is the bad one


# ===========================================================================
# earnings as seen on a date
# ===========================================================================

def eps_book(rows):
    return align.EpsBook(bare(eps={"AAA": rows}))


Q = ["2018-03-31", "2018-06-30", "2018-09-30", "2018-12-31", "2019-03-31"]


def quarters(values, lag=40, qs=Q):
    return [(q, (dt.date.fromisoformat(q) + dt.timedelta(days=lag)).isoformat(), v) for q, v in zip(qs, values)]


def test_ttm_filing_date_boundary():
    rows = quarters([1, 2, 3, 4, 5])                      # the 5th (2019-03-31) is filed 2019-05-10
    eb = eps_book(rows)
    t, why = eb.ttm("AAA", "2019-05-10")                  # filed ON the day: visible
    assert why is None and t.value == 2 + 3 + 4 + 5 and t.latest_filed == "2019-05-10"
    t, why = eb.ttm("AAA", "2019-05-09")                  # the day before: hidden
    assert why is None and t.value == 1 + 2 + 3 + 4 and t.latest_filed == "2019-02-09"


def test_ttm_takes_four_latest_quarters_by_quarter_end_even_if_filed_out_of_order():
    rows = [("2018-03-31", "2018-05-10", 1.0), ("2018-06-30", "2018-08-09", 2.0), ("2018-09-30", "2018-11-08", 3.0),
            ("2018-12-31", "2019-04-20", 4.0),            # filed LATE, after the next quarter
            ("2019-03-31", "2019-04-10", 5.0)]
    eb = eps_book(rows)
    t, why = eb.ttm("AAA", "2019-04-15")                  # Q1'19 visible, Q4'18 not yet: the four latest VISIBLE quarters
    assert t is None and why == align.REASON_GAP_Q        # are Q1'18..Q3'18 + Q1'19, which skip a quarter
    t, why = eb.ttm("AAA", "2019-05-01")                  # both visible
    assert why is None and t.value == 2 + 3 + 4 + 5 and t.quarters[-1] == "2019-03-31"


def test_ttm_needs_four_quarters():
    eb = eps_book(quarters([1, 2, 3]))
    assert eb.ttm("AAA", "2019-03-01") == (None, align.REASON_FEW_Q)
    assert eps_book([]).ttm("AAA", "2019-03-01") == (None, align.REASON_FEW_Q)
    assert eps_book(quarters([1, 2, 3, 4])).ttm("AAA", "2019-02-09")[1] is None      # the 4th is filed on that day
    assert eps_book(quarters([1, 2, 3, 4])).ttm("AAA", "2019-02-08")[1] == align.REASON_FEW_Q


def test_ttm_refuses_non_consecutive_quarters():
    rows = quarters([1, 2, 4, 5], qs=["2018-03-31", "2018-06-30", "2018-12-31", "2019-03-31"])     # 2018-09-30 missing
    t, why = eps_book(rows).ttm("AAA", "2019-06-01")
    assert t is None and why == align.REASON_GAP_Q
    ok = quarters([1, 2, 3, 4])
    assert eps_book(ok).ttm("AAA", "2019-06-01")[1] is None                  # consecutive: span 275 days, allowed


def test_ttm_staleness_boundary():
    rows = [(q, q, 1.0) for q in ["2018-03-31", "2018-06-30", "2018-09-30", "2018-12-31"]]     # filed the same day
    eb = eps_book(rows)
    last = dt.date(2018, 12, 31)
    assert eb.ttm("AAA", (last + dt.timedelta(days=config.MAX_EPS_STALENESS_DAYS)).isoformat())[1] is None
    t, why = eb.ttm("AAA", (last + dt.timedelta(days=config.MAX_EPS_STALENESS_DAYS + 1)).isoformat())
    assert t is None and why == align.REASON_STALE


def test_ttm_loss_and_zero_are_flagged_but_value_is_kept():
    t, why = eps_book(quarters([-1, -1, -1, -1])).ttm("AAA", "2019-06-01")
    assert why == align.REASON_LOSS and t.value == -4
    t, why = eps_book(quarters([1, -1, 1, -1])).ttm("AAA", "2019-06-01")          # sums to exactly zero
    assert why == align.REASON_LOSS and t.value == 0
    t, why = eps_book(quarters([-1, 2, 2, 2])).ttm("AAA", "2019-06-01")           # one bad quarter, positive total
    assert why is None and t.value == 5


def test_eps_book_rows_and_unknown_ticker():
    eb = eps_book(quarters([1, 2, 3, 4, 5]))
    assert len(eb.rows("AAA")) == 5 and eb.rows("ZZZ") == []
    assert eb.ttm("ZZZ", "2019-06-01") == (None, align.REASON_FEW_Q)


# ===========================================================================
# the panel: P/E and momentum
# ===========================================================================

def stored_price(con, ticker, date):
    return con.execute("SELECT adj_close FROM PRICE WHERE ticker=? AND date=?", (ticker, date)).fetchone()[0]


def test_pe_and_momentum_formulas_on_a_hand_checked_case():
    r, con = at("2019-06-28")
    expected_ttm = 1.0 + 0.1 * 5 + 1.0 + 0.1 * 6 + 1.0 + 0.1 * 7 + 1.0 + 0.1 * 8       # quarters 2018-06-30 .. 2019-03-31
    assert abs(r.ttm_eps - expected_ttm) < 1e-9
    assert abs(r.pe - stored_price(con, "AAA", "2019-06-28") / expected_ttm) < 1e-9
    want_mom = stored_price(con, "AAA", "2019-05-31") / stored_price(con, "AAA", "2018-06-29") - 1
    assert abs(r.momentum - want_mom) < 1e-12
    assert r.mom_price_dates == ("2019-05-31", "2018-06-29") and r.eligible and r.reasons == []


def test_momentum_skips_the_latest_month():
    """Changing the month-end price must not change momentum; changing the previous month-end's must."""
    base, _ = at("2019-06-28")
    moved_now, _ = at("2019-06-28", override={"2019-06-28": 90.0})
    assert moved_now.momentum == base.momentum and moved_now.pe != base.pe
    moved_prev, _ = at("2019-06-28", override={"2019-05-31": 70.0})
    assert moved_prev.momentum != base.momentum and moved_prev.pe == base.pe
    moved_old, _ = at("2019-06-28", override={"2018-06-29": 20.0})
    assert moved_old.momentum != base.momentum


def test_history_boundary_at_twelve_months():
    con = stage3_run.mini_world()
    me = align.month_end_dates(con)
    rows = align.build_panel(con, dates=[me[11], me[12]])
    early, later = rows
    assert early.momentum is None and align.REASON_HISTORY in early.reasons and not early.eligible
    assert later.momentum is not None and align.REASON_HISTORY not in later.reasons


def test_each_exclusion_has_the_right_reason():
    base = stage3_run.default_eps()
    cases = [
        (dict(eps_rows=stage3_run.default_eps(negative_from="2018-06-30")), align.REASON_LOSS),
        (dict(eps_rows=[r for r in base if r[0] != "2018-12-31"]), align.REASON_GAP_Q),
        (dict(eps_rows=[r for r in base if r[0] <= "2018-09-30"]), align.REASON_STALE),
        (dict(eps_rows=[r for r in base if "2018-09-30" <= r[0] <= "2019-03-31"]), align.REASON_FEW_Q),
        (dict(drop=tuple(stage3_run._range("2019-06-20", "2019-06-28"))), align.REASON_OLD_PRICE),
        (dict(override={"2019-06-28": 3100.0}, validate=True), align.REASON_FLAGGED),
    ]
    for kwargs, reason in cases:
        r, _ = at("2019-06-28", **kwargs)
        assert r.pe is None and not r.eligible and reason in r.reasons, (reason, r.reasons)


def test_loss_still_gets_a_momentum_and_a_price():
    r, _ = at("2019-06-28", eps_rows=stage3_run.default_eps(negative_from="2018-06-30"))
    assert r.momentum is not None and r.price is not None and r.ttm_eps == -2.0 and r.pe is None


def test_carried_price_is_used_and_counted():
    r, _ = at("2019-06-28", drop=("2019-06-27", "2019-06-28"))
    assert r.carried_days == 2 and r.price_date == "2019-06-26" and r.pe is not None and r.eligible
    assert align.summarize([r])["carried"] == 1


def test_missing_momentum_price_is_refused_with_its_own_reason():
    far_back = stage3_run._range("2018-06-20", "2018-06-29")                  # kills the t-12 price (2018-06-29)
    r, _ = at("2019-06-28", drop=tuple(far_back))
    assert r.momentum is None and r.pe is not None and not r.eligible
    assert any(w.startswith("momentum (t-12)") for w in r.reasons)
    near = stage3_run._range("2019-05-22", "2019-05-31")                      # kills the t-1 price (2019-05-31)
    r, _ = at("2019-06-28", drop=tuple(near))
    assert any(w.startswith("momentum (t-1)") for w in r.reasons)


def test_a_flagged_price_on_a_lookback_date_blocks_momentum():
    r, _ = at("2019-06-28", override={"2019-05-31": 3000.0}, validate=True)
    assert r.momentum is None and any("flagged" in w for w in r.reasons) and r.pe is not None


def test_eps_becomes_visible_only_when_filed():
    late = [(q, "2019-07-10" if q == "2019-03-31" else f, e) for q, f, e in stage3_run.default_eps()]
    con = stage3_run.mini_world(eps_rows=late)
    june, july = align.build_panel(con, dates=["2019-06-28", "2019-07-31"])
    assert abs(june.ttm_eps - (1.4 + 1.5 + 1.6 + 1.7)) < 1e-9          # quarters ended 2018-03-31 .. 2018-12-31
    assert abs(july.ttm_eps - (1.5 + 1.6 + 1.7 + 1.8)) < 1e-9 and july.eps_filed == "2019-07-10"
    assert june.eps_filed < "2019-06-28"


def test_build_panel_validates_dates_and_selects_tickers():
    con = stage3_run.mini_world()
    try:
        align.build_panel(con, dates=["2019-06-27"])               # not a month-end
    except ValueError as e:
        assert "not a month-end" in str(e)
    else:
        raise AssertionError("should have raised")
    collect.register_stocks(con, [("OLD", "Not in the index", 0, 0)])
    assert align.default_tickers(con) == ["AAA"]                    # benchmark and non-members are left out
    assert [r.ticker for r in align.build_panel(con, tickers=["AAA", "OLD"], dates=["2019-06-28"])] == ["AAA", "OLD"]
    rows = align.build_panel(con, dates=["2019-06-28", "2019-05-31"])
    assert [r.date for r in rows] == ["2019-05-31", "2019-06-28"]    # ordered by date


def test_summarize_counts():
    con = stage3_run.mini_world(eps_rows=stage3_run.default_eps(negative_from="2018-06-30"))
    rows = align.build_panel(con)
    s = align.summarize(rows)
    # Losses begin with the quarter ended 2018-06-30. In 2019-01 the trailing four quarters still hold two
    # profitable ones (total +1.7), so exactly that month is eligible; every later month is a loss.
    assert s["rows"] == 24 and s["eligible"] == 1
    assert s["reasons"][align.REASON_HISTORY] == 12 and s["reasons"][align.REASON_LOSS] == 11


# ===========================================================================
# the look-ahead audit and the scramble experiment
# ===========================================================================

def test_audit_catches_a_row_that_used_the_future():
    ok = Row("A", "2020-01-31", price_date="2020-01-31", eps_filed="2020-01-15", mom_price_dates=("2019-12-31", "2019-01-31"))
    assert align.audit_lookahead([ok]) == []
    for field, value in (("price_date", "2020-02-03"), ("eps_filed", "2020-02-01"), ("mom_price_dates", ("2020-02-28", "2019-01-31"))):
        bad = Row("A", "2020-01-31", price_date="2020-01-31", eps_filed="2020-01-15", mom_price_dates=("2019-12-31", "2019-01-31"))
        setattr(bad, field, value)
        assert len(align.audit_lookahead([bad])) >= 1, field


def test_real_panel_passes_the_audit():
    _, _, rows = world()
    assert align.audit_lookahead(rows) == []


def test_scrambling_the_future_changes_nothing_before_the_cutoff():
    _, con, rows = world(n_stocks=20, seed=11, defects=False)
    cutoff = stage3_run.resolve_month_end(con, "2019-06-30")
    for what in (("prices", "eps"), ("prices",), ("eps",)):             # each leak route is tested on its own
        ex = stage3_run.lookahead_experiment(con, cutoff, scramble=what)
        assert ex["changed_up_to_cutoff"] == 0 and ex["rows_up_to_cutoff"] > 1000, what
        assert ex["changed_after_cutoff"] > 0, what                      # the scramble really bit


# ===========================================================================
# ranks and the combined score
# ===========================================================================

def test_pct_rank_cases():
    assert list(signals.pct_rank([10, 30, 20, 40])) == [0.0, 2 / 3, 1 / 3, 1.0]
    r = signals.pct_rank([5, None, 1, float("nan"), 9])
    assert r[2] == 0.0 and r[0] == 0.5 and r[4] == 1.0 and np.isnan(r[1]) and np.isnan(r[3])
    assert list(signals.pct_rank([7.0])) == [0.5]
    assert len(signals.pct_rank([])) == 0 and np.isnan(signals.pct_rank([None, None])).all()
    assert list(signals.pct_rank([3, 3, 3])) == [0.5, 0.5, 0.5]                    # all tied
    r = signals.pct_rank([1, 2, 2, 3])                                              # the tie shares rank 1.5 of 0..3
    assert list(r) == [0.0, 0.5, 0.5, 1.0]


def mk(ticker, date, pe=None, mom=None):
    return Row(ticker, date, pe=pe, momentum=mom)


def test_ranks_run_in_the_right_direction():
    rows = signals.add_ranks([mk("A", "d1", 5, 0.30), mk("B", "d1", 10, 0.10), mk("C", "d1", 20, -0.20)])
    by = {r.ticker: r for r in rows}
    assert by["A"].pe_rank == 1.0 and by["C"].pe_rank == 0.0 and by["B"].pe_rank == 0.5      # lowest P/E = cheapest = 1
    assert by["A"].momentum_rank == 1.0 and by["C"].momentum_rank == 0.0                      # highest momentum = 1


def test_ranks_are_computed_within_each_month_separately():
    rows = signals.add_ranks([mk("A", "d1", 5, 0.1), mk("B", "d1", 10, 0.2),
                              mk("A", "d2", 50, 0.9), mk("B", "d2", 40, 0.8)])
    got = {(r.ticker, r.date): (r.pe_rank, r.momentum_rank) for r in rows}
    assert got[("A", "d1")] == (1.0, 0.0) and got[("B", "d1")] == (0.0, 1.0)
    assert got[("A", "d2")] == (0.0, 1.0) and got[("B", "d2")] == (1.0, 0.0)


def test_each_signal_is_ranked_among_stocks_that_have_it():
    rows = signals.add_ranks([mk("A", "d", 5, 0.1), mk("B", "d", None, 0.2), mk("C", "d", 9, None), mk("D", "d", 7, 0.3)])
    by = {r.ticker: r for r in rows}
    assert by["B"].pe_rank is None and by["C"].momentum_rank is None
    assert (by["A"].pe_rank, by["D"].pe_rank, by["C"].pe_rank) == (1.0, 0.5, 0.0)            # among A, C, D
    assert (by["A"].momentum_rank, by["B"].momentum_rank, by["D"].momentum_rank) == (0.0, 0.5, 1.0)
    assert [by[t].eligible for t in "ABCD"] == [True, False, False, True]


def test_combined_score_weights_and_validation():
    r = Row("A", "d", pe=5, momentum=0.1, pe_rank=0.8, momentum_rank=0.2)
    assert signals.combined_score(r, 1.0) == 0.8 and signals.combined_score(r, 0.0) == 0.2
    assert abs(signals.combined_score(r, 0.5) - 0.5) < 1e-12 and abs(signals.combined_score(r, 0.25) - 0.35) < 1e-12
    for bad in (-0.01, 1.01):
        try:
            signals.combined_score(r, bad)
        except ValueError:
            continue
        raise AssertionError("weight outside 0..1 must raise")
    try:
        signals.combined_score(Row("B", "d"), 0.5)
    except ValueError:
        pass
    else:
        raise AssertionError("a row without ranks has no score")


def test_scores_for_date_order_ties_and_eligibility():
    rows = signals.add_ranks([mk("A", "d", 5, 0.1), mk("B", "d", 10, 0.3), mk("C", "d", 20, 0.2),
                              mk("D", "d", None, 0.9), mk("A", "x", 5, 0.1)])
    value = [r.ticker for _, r in signals.scores_for_date(rows, "d", 1.0)]
    momentum = [r.ticker for _, r in signals.scores_for_date(rows, "d", 0.0)]
    assert value == ["A", "B", "C"] and momentum == ["B", "C", "A"]               # D has no P/E, so it never appears
    tie = signals.add_ranks([mk("Z", "d", 5, 0.5), mk("M", "d", 5, 0.5)])
    assert [r.ticker for _, r in signals.scores_for_date(tie, "d", 0.5)] == ["M", "Z"]     # equal scores: by ticker


# ===========================================================================
# storing signals
# ===========================================================================

def test_store_signals_is_idempotent_and_updates_changes():
    con = stage3_run.mini_world()
    rows = stage3_run.all_signals(con, dates=["2019-05-31", "2019-06-28"])
    assert signals.store_signals(con, rows) == {"inserted": 2, "updated": 0, "unchanged": 0}
    assert signals.store_signals(con, rows) == {"inserted": 0, "updated": 0, "unchanged": 2}
    rows[0].pe_rank = 0.123
    assert signals.store_signals(con, rows) == {"inserted": 0, "updated": 1, "unchanged": 1}
    assert con.execute("SELECT COUNT(*) FROM SIGNAL").fetchone()[0] == 2
    got = signals.load_signals(con, rows[0].date)
    assert got == [("AAA", rows[0].pe, rows[0].momentum, 0.123, rows[0].momentum_rank)]


def test_store_signals_keeps_missing_values_as_null():
    con = stage3_run.mini_world(eps_rows=stage3_run.default_eps(negative_from="2018-06-30"))
    rows = stage3_run.all_signals(con, dates=["2019-06-28"])
    signals.store_signals(con, rows)
    pe, mom, pe_rank, mom_rank = con.execute("SELECT pe, momentum, pe_rank, momentum_rank FROM SIGNAL").fetchone()
    assert pe is None and pe_rank is None and mom is not None and mom_rank is not None


def test_stored_ranks_satisfy_the_schema_on_a_full_panel():
    _, con, rows = world(n_stocks=20, seed=11, defects=False)
    signals.store_signals(con, rows)           # the CHECK constraints (ranks within 0..1) would raise if violated
    assert con.execute("SELECT COUNT(*) FROM SIGNAL").fetchone()[0] == len(rows)


# ===========================================================================
# against the generator's own truth, and the planted edge
# ===========================================================================

def test_clean_market_matches_the_truth_everywhere():
    market, _, rows = world(defects=False)
    check = stage3_run.truth_check(rows, market)
    assert check["differs"] == {} and check["compared"] > 14000
    assert check["worst_pe"] < 1e-3 and check["worst_mom"] < 1e-3


def test_clean_market_ranks_match_the_truth_ranks():
    market, _, rows = world(defects=False)
    tr = market["truth"]
    t_idx = {t: i for i, t in enumerate(tr["tickers"])}
    m_idx = {d: k for k, d in enumerate(tr["month_ends"])}
    worst_pe = worst_mom = 0.0
    for r in rows:
        k, i = m_idx[r.date], t_idx[r.ticker]
        if k < 12:
            continue
        for mine, theirs, which in ((r.pe_rank, tr["value_rank"][k][i], "pe"), (r.momentum_rank, tr["momentum_rank"][k][i], "mom")):
            assert (mine is None) == np.isnan(theirs)
            if mine is not None:
                d = abs(mine - theirs)
                if which == "pe":
                    worst_pe = max(worst_pe, d)
                else:
                    worst_mom = max(worst_mom, d)
    assert worst_pe < 0.03 and worst_mom < 0.03          # at most a swap of near-tied neighbours (1 step = 0.01)


def test_defects_change_exactly_the_three_stocks_they_touch():
    market, _, rows = world(defects=True)
    check = stage3_run.truth_check(rows, market)
    assert set(check["differs"]) == {"S006", "S009", "S020"}
    assert "availability" in check["differs"]["S020"] and "P/E value" in check["differs"]["S006"]
    s = align.summarize(rows)
    assert s["carried"] == 1 and s["reasons"][align.REASON_GAP_Q] == 9 and s["reasons"][align.REASON_STALE] == 1
    assert s["reasons"][align.REASON_OLD_PRICE] == 1 and s["reasons"][align.REASON_LOSS] == 75
    assert s["reasons"][align.REASON_HISTORY] == 12 * 100


def test_signals_recover_a_strong_planted_edge_and_nothing_in_the_null_world():
    def tstats(edge, share):
        _, con, rows = world(edge=edge, value_share=share, defects=False)
        return {w: stage3_run.ic_summary(stage3_run.information_coefficients(con, rows, w))[1] for w in (1.0, 0.0, 0.5)}
    strong_value = tstats(0.15, 1.0)
    assert strong_value[1.0] > 4 and strong_value[0.0] < 2
    strong_mom = tstats(0.15, 0.0)
    assert strong_mom[0.0] > 4 and strong_mom[1.0] < 2
    both = tstats(0.15, 0.5)
    assert both[0.5] > 3
    null = tstats(0.0, 0.5)
    assert all(abs(v) < 3 for v in null.values()), null


def test_ic_summary_and_ic_requirements():
    mean, t = stage3_run.ic_summary([0.1, 0.1, 0.1, 0.1])
    assert abs(mean - 0.1) < 1e-12
    assert np.isnan(stage3_run.ic_summary([0.1])[0])
    assert stage3_run.ic_summary([0.1, 0.1, 0.1])[1] == float("inf")                # no spread at all: no warning, no crash
    assert stage3_run.ic_summary([-0.1, -0.1, -0.1])[1] == float("-inf")
    assert stage3_run.ic_summary([0.0, 0.0, 0.0]) == (0.0, 0.0)
    mean, t = stage3_run.ic_summary([0.05, 0.15, 0.10, 0.20, 0.0])
    assert abs(mean - 0.10) < 1e-12 and t > 2


# ===========================================================================
# the run script
# ===========================================================================

def test_demo_succeeds_defects_clean_and_null():
    for kwargs in ({}, {"clean": True, "edge": 0.0}):
        code, out = quiet(stage3_run.demo, **kwargs)
        assert code == 0, out[-500:]
        assert "No signal looks into the future." in out and "Check question" in out


def test_demo_saves_a_database_that_explain_can_reopen():
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "s.db")
        code, _ = quiet(stage3_run.demo, clean=True, db_path=path)
        assert code == 0
        code, out = quiet(stage3_run.main, ["--explain", "S010", "2019-06-28", "--db", path])
        assert code == 0 and "S010 as of the month-end 2019-06-28" in out
        import sqlite3
        assert sqlite3.connect(path).execute("SELECT COUNT(*) FROM SIGNAL").fetchone()[0] == 15600


def test_every_try_case_runs_and_says_the_right_thing():
    cases = stage3_run.try_cases()
    assert len(cases) == 10
    out = [stage3_run.run_try_case(i + 1)[1] for i in range(10)]
    assert "ELIGIBLE" in out[0] and "EXCLUDED" not in out[0]
    assert "EPS: loss" in out[1]
    assert out[2].count("ELIGIBLE") == 2 and "2019-07-10" in out[2]
    assert "a quarter is missing" in out[3] and "stale" in out[4] and "fewer than 4" in out[5]
    assert "carried 2 day(s)" in out[6] and "too old" in out[7]
    assert "flagged" in out[8] and "less than 12 months" in out[9]


def test_explain_lines_content():
    _, con, _ = world(defects=False)
    text = "\n".join(stage3_run.explain_lines(con, "s010", "2019-06-30"))               # lower case, non-month-end date
    assert "month-end 2019-06-28" in text and "you asked for 2019-06-30" in text
    assert "HIDDEN" in text and "TTM EPS =" in text and "P/E rank" in text and "place" in text
    assert "latest month is skipped" in text
    early = "\n".join(stage3_run.explain_lines(con, "S010", "2013-06-30"))
    assert "less than 12 months" in early and "NOT ELIGIBLE" in early
    assert "No stock called" in stage3_run.explain_lines(con, "NOPE", "2019-06-30")[0]
    assert "before the first month-end" in stage3_run.explain_lines(con, "S010", "2012-01-01")[0]


def test_explain_shows_the_carried_price_and_the_stale_case():
    _, con, _ = world(defects=True)
    text = "\n".join(stage3_run.explain_lines(con, "S006", "2015-08-31"))
    assert "carried forward 3 day(s) from 2015-08-28" in text
    text = "\n".join(stage3_run.explain_lines(con, "S020", "2016-11-30"))
    assert "no usable price" in text and "NOT ELIGIBLE" in text


def test_top_lines_weight_changes_the_list():
    _, con, _ = world(defects=False)
    rows = stage3_run.all_signals(con, dates=["2019-06-28"])
    value_lines, value_names = stage3_run.top_lines(rows, "2019-06-28", 1.0, 10)
    mom_lines, mom_names = stage3_run.top_lines(rows, "2019-06-28", 0.0, 10)
    assert len(value_names) == len(mom_names) == 10 and value_names != mom_names
    assert "weight 1.00 on value" in value_lines[0]
    code, out = quiet(stage3_run.main, ["--top", "2019-06-28", "0.7", "--clean"])
    assert code == 0 and "weight 0.70 on value" in out and out.count("\n") >= 12


def test_main_argument_errors_and_try():
    assert quiet(stage3_run.main, ["--explain", "S010"])[0] == 1
    assert quiet(stage3_run.main, ["--top", "2019-06-28"])[0] == 1
    assert quiet(stage3_run.main, ["--try", "99"])[0] == 1
    code, out = quiet(stage3_run.main, ["--try", "2"])
    assert code == 0 and "loss-making" in out


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
