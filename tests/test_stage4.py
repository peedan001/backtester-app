"""Tests for Stage 4 (portfolio.py, the RUN.buffer column, stage4_run.py).

Run either way:
    python tests/test_stage4.py
    python -m pytest -q
"""
import contextlib
import dataclasses
import io
import os
import sqlite3
import sys
import tempfile
from unittest import mock

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import collect
import config
import db
import portfolio as pf
import signals
import stage3_run
import stage4_run
from align import Row
from portfolio import Position

_CACHE = {}
_OPEN = []          # connections opened by single tests; closed with the cached worlds


def world(edge=0.05, seed=7, defects=True, n_stocks=100):
    key = (edge, seed, defects, n_stocks)
    if key not in _CACHE:
        market, con = stage3_run.build_world(edge=edge, seed=seed, defects=defects, n_stocks=n_stocks)
        _CACHE[key] = (market, con, stage3_run.all_signals(con))
    return _CACHE[key]


def _close_cached_worlds():
    while _OPEN:
        _OPEN.pop().close()
    for _, con, _ in _CACHE.values():
        con.close()
    _CACHE.clear()


try:
    import pytest

    @pytest.fixture(scope="module", autouse=True)
    def _close_worlds_at_end():
        yield
        _close_cached_worlds()
except ImportError:      # the plain runner calls _close_cached_worlds() itself
    pass


def quiet(fn, *a, **k):
    with contextlib.redirect_stdout(io.StringIO()) as buf:
        out = fn(*a, **k)
    return out, buf.getvalue()


# ---- hand-made months -------------------------------------------------------

def month(date, order):
    """Eligible rows for one month. `order` lists tickers best first; both ranks fall evenly, so the combined
    score is the same for any value weight and strictly decreasing along the list."""
    n = len(order)
    return [Row(t, date, pe=1.0, momentum=0.1, pe_rank=1 - (i + 1) / (n + 1), momentum_rank=1 - (i + 1) / (n + 1))
            for i, t in enumerate(order)]


def scored(order, date="d"):
    return signals.scores_for_date(month(date, order), date, 0.5)


def names(positions):
    return [p.ticker for p in positions]


# ===========================================================================
# the RUN.buffer column
# ===========================================================================

def test_run_buffer_default_and_bounds():
    con = db.connect()
    _OPEN.append(con)
    db.create_schema(con)
    con.execute("INSERT INTO RUN (top_n, value_weight, cost_bps) VALUES (20, 0.5, 10)")
    assert con.execute("SELECT buffer FROM RUN").fetchone() == (config.DEFAULT_BUFFER,)
    insert = "INSERT INTO RUN (top_n, value_weight, cost_bps, buffer) VALUES (20, 0.5, 10, ?)"
    for ok in (config.MIN_BUFFER, 1.5, 2, config.MAX_BUFFER):
        con.execute(insert, (ok,))
    for bad in (config.MIN_BUFFER - 0.01, config.MAX_BUFFER + 0.01, 0, -1, "wide", None):
        try:
            con.execute(insert, (bad,))
        except sqlite3.IntegrityError:
            continue
        raise AssertionError(f"buffer {bad!r} should be refused")


def test_schema_from_an_earlier_stage_is_migrated_without_losing_runs():
    old = sqlite3.connect(":memory:")
    _OPEN.append(old)
    old.executescript(db.SCHEMA.replace("    " + db.BUFFER_COLUMN + ",\n", ""))
    assert "buffer" not in [c[0] for c in db.columns(old, "RUN")]
    old.execute("INSERT INTO RUN (top_n, value_weight, cost_bps) VALUES (10, 0.3, 5)")
    old.commit()
    db.create_schema(old)
    db.create_schema(old)                                           # and a second time: safe
    assert [c[0] for c in db.columns(old, "RUN")].count("buffer") == 1
    assert old.execute("SELECT top_n, value_weight, cost_bps, buffer FROM RUN").fetchall() == [(10, 0.3, 5.0, 1.0)]
    try:
        old.execute("INSERT INTO RUN (top_n, value_weight, cost_bps, buffer) VALUES (20, 0.5, 10, 0.5)")
    except sqlite3.IntegrityError:
        pass
    else:
        raise AssertionError("the migrated column must enforce the same bounds")


# ===========================================================================
# parameters and the buffer limit
# ===========================================================================

def test_parameter_validation():
    for kwargs in ({"top_n": 0}, {"top_n": config.MAX_TOP_N + 1}, {"top_n": 2.5}, {"top_n": 5, "value_weight": 1.1},
                   {"top_n": 5, "value_weight": -0.1}, {"top_n": 5, "buffer": 0.99}, {"top_n": 5, "buffer": 5.01}):
        try:
            pf._check_params(**kwargs)
        except ValueError:
            continue
        raise AssertionError(f"{kwargs} should be refused")
    pf._check_params(1, 0.0, 1.0)
    pf._check_params(config.MAX_TOP_N, 1.0, config.MAX_BUFFER)
    try:
        pf.select_holdings(scored("ABC"), set(), 3, buffer=0.5)
    except ValueError:
        pass
    else:
        raise AssertionError("select_holdings must validate its parameters")


def test_buffer_limit_rounds_down_and_survives_floating_point():
    assert pf.buffer_limit(3, 1.5) == 4 and pf.buffer_limit(20, 1.0) == 20 and pf.buffer_limit(10, 2.0) == 20
    assert pf.buffer_limit(20, 1.15) == 23                 # 1.15 * 20 is 22.999999999999996 in floating point
    assert pf.buffer_limit(45, 1.4) == 63 and pf.buffer_limit(50, 2.3) == 115     # 62.99999999999999 and 114.99999999999999 raw
    assert pf.buffer_limit(20, 1.1) == 22 and pf.buffer_limit(1, 1.99) == 1 and pf.buffer_limit(1, 2.0) == 2


# ===========================================================================
# choosing one month's holdings
# ===========================================================================

def test_pure_top_n_equal_weights():
    pos = pf.select_holdings(scored("ABCDEFGH"), set(), 3)
    assert names(pos) == ["A", "B", "C"] and all(abs(p.weight - 1 / 3) < 1e-15 for p in pos)
    assert [p.place for p in pos] == [1, 2, 3] and not any(p.kept for p in pos)
    assert pos[0].score > pos[1].score > pos[2].score


def test_fewer_eligible_than_n_holds_all_of_them_fully_invested():
    pos = pf.select_holdings(scored("AB"), set(), 3)
    assert names(pos) == ["A", "B"] and [p.weight for p in pos] == [0.5, 0.5]
    assert pf.select_holdings(scored("A"), set(), 3)[0].weight == 1.0


def test_exactly_n_eligible_and_nothing_eligible():
    assert names(pf.select_holdings(scored("ABC"), set(), 3)) == ["A", "B", "C"]
    assert pf.select_holdings([], {"A", "B"}, 3) == []


def test_buffer_keeps_a_slipped_stock_and_skips_a_better_new_one():
    pos = pf.select_holdings(scored("DBCAEF"), set("ABC"), 3, buffer=1.5)      # limit 4: A (place 4) stays
    assert sorted(names(pos)) == ["A", "B", "C"] and all(p.kept for p in pos)
    assert "D" not in names(pos)


def test_buffer_boundary_exactly_at_the_limit_and_one_beyond():
    inside = pf.select_holdings(scored("DBCAEF"), set("ABC"), 3, buffer=1.5)      # A at place 4 = limit
    assert "A" in names(inside)
    beyond = pf.select_holdings(scored("DBCEAF"), set("ABC"), 3, buffer=1.5)      # A at place 5
    assert sorted(names(beyond)) == ["B", "C", "D"]
    assert {p.ticker: p.kept for p in beyond} == {"B": True, "C": True, "D": False}


def test_no_buffer_means_exact_top_n_whatever_was_held():
    pos = pf.select_holdings(scored("DBCAEF"), set("ABC"), 3, buffer=1.0)
    assert names(pos) == ["D", "B", "C"]
    assert [p.kept for p in pos] == [False, True, True]


def test_a_stock_that_lost_eligibility_is_dropped_whatever_the_buffer():
    pos = pf.select_holdings(scored("BCDE"), set("ABC"), 3, buffer=2.0)
    assert names(pos) == ["B", "C", "D"] and "A" not in names(pos)


def test_buffer_never_grows_the_portfolio_and_ignores_unknown_previous_names():
    pos = pf.select_holdings(scored("ABCD"), set("ABC"), 3, buffer=2.0)
    assert names(pos) == ["A", "B", "C"]
    pos = pf.select_holdings(scored("ABCD"), {"ZZZ", "A"}, 3, buffer=1.5)
    assert names(pos) == ["A", "B", "C"]
    pos = pf.select_holdings(scored("ABXCDEF"), {"X"}, 1, buffer=3.0)             # one seat, kept stock at place 3 = limit
    assert names(pos) == ["X"]
    pos = pf.select_holdings(scored("ABXCDEF"), {"X"}, 1, buffer=2.0)
    assert names(pos) == ["A"]


def test_never_more_than_top_n_even_if_more_were_held_before():
    pos = pf.select_holdings(scored("ABCDEFG"), set("ABCDE"), 3, buffer=2.0)       # five held before, only three seats
    assert names(pos) == ["A", "B", "C"]


def test_empty_seats_are_filled_best_place_first():
    pos = pf.select_holdings(scored("ABCDEFG"), {"E", "F"}, 4, buffer=2.0)         # E, F kept (limit 8); two seats to fill
    assert sorted(names(pos)) == ["A", "B", "E", "F"]
    assert [p.place for p in pos] == [1, 2, 5, 6]


def test_positions_are_sorted_best_place_first_and_weights_helper():
    pos = pf.select_holdings(scored("ABCDEFG"), {"E", "C"}, 3, buffer=2.0)
    assert [p.place for p in pos] == sorted(p.place for p in pos)
    assert pf.weights(pos) == {p.ticker: p.weight for p in pos}


# ===========================================================================
# many months
# ===========================================================================

def rows_over_months(orders):
    rows = []
    for i, order in enumerate(orders):
        rows += month(f"2020-{i + 1:02d}-28", order)
    return rows


def test_months_run_in_order_and_the_buffer_remembers_last_month():
    rows = rows_over_months(["ABCDEF", "DABCEF", "DEFABC"])
    sticky = pf.build_portfolio(rows, 3, 0.5, 1.5)
    assert [names(sticky[d]) for d in sorted(sticky)] == [["A", "B", "C"], ["A", "B", "C"], ["D", "E", "A"]]
    pure = pf.build_portfolio(rows, 3, 0.5, 1.0)
    assert [names(pure[d]) for d in sorted(pure)] == [["A", "B", "C"], ["D", "A", "B"], ["D", "E", "F"]]


def test_build_portfolio_does_not_depend_on_row_order_and_an_empty_month_resets_memory():
    rows = rows_over_months(["ABCDEF", "", "ABCDEF"])
    a = pf.build_portfolio(rows, 3, 0.5, 1.5)
    b = pf.build_portfolio(list(reversed(rows)), 3, 0.5, 1.5)
    assert {d: names(p) for d, p in a.items()} == {d: names(p) for d, p in b.items()}
    assert a["2020-02-28"] if "2020-02-28" in a else True
    gap = [Row("A", "2020-02-28")]                          # a month in which nothing is eligible
    p = pf.build_portfolio(rows_over_months(["ABCDEF"]) + gap + month("2020-03-28", ["DABCEF"]), 3, 0.5, 1.5)
    assert p["2020-02-28"] == []
    assert not any(pos.kept for pos in p["2020-03-28"])    # nothing is "kept" after a month with no holdings


def test_value_weight_changes_who_is_held():
    rows = [Row("CHEAP", "d", pe=5, momentum=-0.2, pe_rank=1.0, momentum_rank=0.0),
            Row("HOT", "d", pe=40, momentum=0.9, pe_rank=0.0, momentum_rank=1.0),
            Row("MID", "d", pe=15, momentum=0.1, pe_rank=0.6, momentum_rank=0.6)]
    value = pf.build_portfolio(rows, 1, 1.0)["d"]
    momentum = pf.build_portfolio(rows, 1, 0.0)["d"]
    blend = pf.build_portfolio(rows, 1, 0.5)["d"]
    assert names(value) == ["CHEAP"] and names(momentum) == ["HOT"] and names(blend) == ["MID"]


def test_build_portfolio_validates_parameters():
    for args in ((0, 0.5, 1.0), (5, 1.5, 1.0), (5, 0.5, 0.5)):
        try:
            pf.build_portfolio([], *args)
        except ValueError:
            continue
        raise AssertionError(args)


# ===========================================================================
# turnover and statistics
# ===========================================================================

def test_traded_fraction_and_one_way_turnover():
    abc = {"A": 1 / 3, "B": 1 / 3, "C": 1 / 3}
    assert pf.traded_fraction({}, abc) == 1.0                       # building from cash: buy 100%
    assert pf.traded_fraction(abc, {}) == 1.0
    assert pf.traded_fraction(abc, abc) == 0.0
    assert abs(pf.traded_fraction(abc, {"X": 0.5, "Y": 0.5}) - 2.0) < 1e-12        # a complete swap: sell all, buy all
    assert abs(pf.one_way_turnover(abc, {"X": 0.5, "Y": 0.5}) - 1.0) < 1e-12
    four = {"A": 0.25, "B": 0.25, "C": 0.25, "D": 0.25}
    assert abs(pf.traded_fraction(abc, four) - 0.5) < 1e-12
    assert pf.traded_fraction(abc, four) == pf.traded_fraction(four, abc)           # symmetric
    assert abs(pf.traded_fraction({"A": 0.7, "B": 0.3}, {"A": 0.3, "B": 0.7}) - 0.8) < 1e-12


def pos(*tickers, w=None):
    w = w or 1.0 / len(tickers)
    return [Position(t, w, i + 1, 0.5, False) for i, t in enumerate(tickers)]


def test_names_changed():
    assert pf.names_changed(pos("A", "B", "C"), pos("B", "C", "D")) == ({"D"}, {"A"})
    assert pf.names_changed([], pos("A")) == ({"A"}, set())


def test_portfolio_stats_counts_only_steady_months():
    port = {"d1": [], "d2": pos("A", "B"), "d3": pos("A", "C"), "d4": [], "d5": pos("A", "B")}
    s = pf.portfolio_stats(port, 3)
    assert s["months"] == 5 and s["invested_months"] == 3 and s["empty_months"] == 2
    assert s["partial_months"] == 3 and s["first_invested"] == "d2" and s["avg_names"] == 2
    assert s["steady_months"] == 1                                    # only d3: d2 and d5 follow an empty month
    assert s["avg_replaced"] == 1 and s["avg_replaced_fraction"] == 0.5
    assert abs(s["avg_traded_fraction"] - 1.0) < 1e-12 and abs(s["avg_one_way_turnover"] - 0.5) < 1e-12


def test_portfolio_stats_with_nothing_steady_is_nan_not_a_crash():
    s = pf.portfolio_stats({"d1": pos("A"), "d2": []}, 1)
    assert s["steady_months"] == 0 and s["avg_traded_fraction"] != s["avg_traded_fraction"]
    s = pf.portfolio_stats({}, 5)
    assert s["months"] == 0 and s["first_invested"] is None


def test_estimated_cost_drag():
    assert abs(pf.estimated_cost_drag({"avg_traded_fraction": 0.5}, 10) - 0.006) < 1e-12
    assert pf.estimated_cost_drag({"avg_traded_fraction": 0.5}, 0) == 0.0
    assert abs(pf.estimated_cost_drag({"avg_traded_fraction": 0.25}, 20) - 0.006) < 1e-12


# ===========================================================================
# the independent rule checker
# ===========================================================================

def fixture(top_n=3, buffer=1.5):
    rows = rows_over_months(["ABCDEF", "DABCEF", "DEFABC"])
    port = pf.build_portfolio(rows, top_n, 0.5, buffer)
    assert pf.check_invariants(port, rows, top_n, 0.5, buffer) == []
    return rows, port


def corrupt(port, date, fn):
    out = {d: list(p) for d, p in port.items()}
    out[date] = fn(out[date])
    return out


def violations(port, rows, top_n=3, buffer=1.5):
    return pf.check_invariants(port, rows, top_n, 0.5, buffer)


def expect(msgs, word):
    assert any(word in m for m in msgs), (word, msgs)


def test_checker_passes_correct_portfolios_including_partial_and_empty_months():
    rows = rows_over_months(["ABCDEF", "AB", "", "ABCD"])
    rows += [Row("A", "2020-03-28")]
    for buffer in (1.0, 1.5, 3.0):
        port = pf.build_portfolio(rows, 3, 0.5, buffer)
        assert pf.check_invariants(port, rows, 3, 0.5, buffer) == []


def test_checker_catches_each_kind_of_violation():
    rows, port = fixture()
    d2, d3 = "2020-02-28", "2020-03-28"
    expect(violations(corrupt(port, d2, lambda p: p + [p[0]]), rows), "twice")
    expect(violations(corrupt(port, d2, lambda p: p + pos("E")), rows), "more than top_n")
    expect(violations(corrupt(port, d2, lambda p: []), rows), "nothing held")
    expect(violations(corrupt(port, d2, lambda p: [dataclasses.replace(x, weight=x.weight * 0.9) for x in p]), rows), "add up")
    expect(violations(corrupt(port, d2, lambda p: [dataclasses.replace(x, weight=x.weight * 0.999) for x in p]), rows), "add up")
    expect(violations(corrupt(port, d2, lambda p: [dataclasses.replace(p[0], weight=0.5), dataclasses.replace(p[1], weight=0.3),
                                                   dataclasses.replace(p[2], weight=0.2)]), rows), "not equal")
    expect(violations(corrupt(port, d2, lambda p: [dataclasses.replace(p[0], ticker="ZZZ")] + p[1:]), rows), "not eligible")
    expect(violations(corrupt(port, d2, lambda p: [dataclasses.replace(p[0], place=9)] + p[1:]), rows), "wrong place")
    expect(violations(corrupt(port, d2, lambda p: [dataclasses.replace(p[0], score=0.0)] + p[1:]), rows), "wrong place")
    expect(violations(corrupt(port, d2, lambda p: [dataclasses.replace(p[0], kept=not p[0].kept)] + p[1:]), rows), "kept")


def test_checker_catches_a_kept_stock_beyond_the_buffer_and_a_non_top_n_portfolio():
    rows, port = fixture()
    d3 = "2020-03-28"
    # month 3 order is D E F A B C. Pretend we kept C (place 6, limit is 4) and dropped E
    bad = corrupt(port, d3, lambda p: [dataclasses.replace(x, kept=True) if x.ticker == "A" else x for x in p][:2]
                  + [Position("C", 1 / 3, 6, signals.combined_score(rows[-1], 0.5), True)])
    expect(violations(bad, rows), "beyond the buffer limit")
    pure_rows = rows_over_months(["ABCDEF"])
    pure = pf.build_portfolio(pure_rows, 3, 0.5, 1.0)
    swapped = {"2020-01-28": [pure["2020-01-28"][0], pure["2020-01-28"][1],
                              Position("D", 1 / 3, 4, signals.combined_score(pure_rows[3], 0.5), False)]}
    msgs = pf.check_invariants(swapped, pure_rows, 3, 0.5, 1.0)
    expect(msgs, "exactly the top 3")
    expect(msgs, "better-placed")


def test_checker_catches_a_partial_month_that_leaves_eligible_stocks_out():
    rows = rows_over_months(["AB"])
    port = pf.build_portfolio(rows, 3, 0.5, 1.0)
    short = {"2020-01-28": [Position("A", 1.0, 1, port["2020-01-28"][0].score, False)]}
    expect(pf.check_invariants(short, rows, 3, 0.5, 1.0), "not all of them are held")


def test_checker_catches_too_few_holdings_when_many_are_eligible():
    rows = rows_over_months(["ABCDEF"])
    port = pf.build_portfolio(rows, 3, 0.5, 1.0)
    short = {"2020-01-28": [dataclasses.replace(p, weight=0.5) for p in port["2020-01-28"][:2]]}
    expect(pf.check_invariants(short, rows, 3, 0.5, 1.0), "expected exactly top_n")


# ===========================================================================
# storage
# ===========================================================================

def con_with(*tickers):
    con = db.connect()
    _OPEN.append(con)
    db.create_schema(con)
    collect.register_stocks(con, [(t, t, 1, 0) for t in tickers])
    return con


def test_runs_are_not_duplicated_and_differ_by_every_choice():
    con = con_with("A")
    a = pf.get_or_create_run(con, 20, 0.5, 10, 1.0)
    assert pf.get_or_create_run(con, 20, 0.5, 10, 1.0) == a
    others = {pf.get_or_create_run(con, *args) for args in
              ((10, 0.5, 10, 1.0), (20, 0.6, 10, 1.0), (20, 0.5, 15, 1.0), (20, 0.5, 10, 1.5), (20, 0.5, 10, 1.0, "today"))}
    assert a not in others and len(others) == 5
    assert con.execute("SELECT COUNT(*) FROM RUN").fetchone()[0] == 6
    try:
        pf.get_or_create_run(con, 20, 0.5, 10, 0.5)
    except sqlite3.IntegrityError:
        pass
    else:
        raise AssertionError("an invalid buffer must not create a run")


def test_store_holdings_insert_repeat_update_and_remove():
    con = con_with("A", "B", "C", "D")
    run = pf.get_or_create_run(con, 3, 0.5)
    port = {"d1": pos("A", "B"), "d2": pos("A", "B", "C"), "d3": []}
    port = {"2020-01-31": port["d1"], "2020-02-29": port["d2"], "2020-03-31": []}
    assert pf.store_holdings(con, run, port) == {"inserted": 5, "updated": 0, "unchanged": 0, "deleted": 0}
    assert pf.store_holdings(con, run, port) == {"inserted": 0, "updated": 0, "unchanged": 5, "deleted": 0}
    smaller = {"2020-01-31": pos("A"), "2020-02-29": pos("A", "B", "C", "D")}
    got = pf.store_holdings(con, run, smaller)
    # Jan: A's weight 0.5 -> 1.0 (update), B removed (delete). Feb: A, B, C 1/3 -> 0.25 (3 updates), D new (insert).
    # Mar: the old empty month had no rows, so nothing to delete there.
    assert got == {"inserted": 1, "updated": 4, "unchanged": 0, "deleted": 1}
    loaded = pf.load_holdings(con, run)
    assert set(loaded) == {"2020-01-31", "2020-02-29"} and loaded["2020-01-31"] == {"A": 1.0}
    assert all(abs(w - 0.25) < 1e-12 for w in loaded["2020-02-29"].values()) and len(loaded["2020-02-29"]) == 4
    assert con.execute("SELECT COUNT(*) FROM HOLDING").fetchone()[0] == 5


def test_store_holdings_keeps_runs_apart_and_enforces_the_schema():
    con = con_with("A", "B")
    r1, r2 = pf.get_or_create_run(con, 2, 0.5, buffer=1.0), pf.get_or_create_run(con, 2, 0.5, buffer=1.5)
    pf.store_holdings(con, r1, {"2020-01-31": pos("A", "B")})
    pf.store_holdings(con, r2, {"2020-01-31": pos("A")})
    assert pf.load_holdings(con, r1)["2020-01-31"] == {"A": 0.5, "B": 0.5}
    assert pf.load_holdings(con, r2)["2020-01-31"] == {"A": 1.0}
    try:
        pf.store_holdings(con, r1, {"2020-01-31": pos("NOPE")})            # unknown ticker: the foreign key refuses
    except sqlite3.IntegrityError:
        pass
    else:
        raise AssertionError("an unknown ticker must be refused")
    assert pf.load_holdings(con, 999) == {}


# ===========================================================================
# on the synthetic market
# ===========================================================================

def test_pure_top_n_equals_the_best_scores_each_month():
    _, _, rows = world(defects=False)
    port = pf.build_portfolio(rows, 20, 0.5, 1.0)
    by_date = {}
    for r in rows:
        by_date.setdefault(r.date, []).append(r)
    checked = 0
    for date, positions in port.items():
        order = signals.scores_for_date(by_date[date], date, 0.5)
        assert names(positions) == [r.ticker for _, r in order[:20]], date
        checked += 1
    assert checked == 156


def test_invariants_hold_across_many_settings_on_the_defect_market():
    _, _, rows = world(defects=True)
    for n, w, b in [(20, 0.5, 1.0), (20, 0.5, 1.5), (20, 0.0, 2.0), (20, 1.0, 1.25), (10, 0.7, 1.0), (50, 0.5, 1.5), (100, 0.5, 1.0), (1, 0.5, 3.0)]:
        port = pf.build_portfolio(rows, n, w, b)
        assert pf.check_invariants(port, rows, n, w, b) == [], (n, w, b)


def test_a_bigger_buffer_means_less_trading_and_early_months_are_empty():
    _, _, rows = world(defects=True)
    traded = [pf.portfolio_stats(pf.build_portfolio(rows, 20, 0.5, b), 20)["avg_traded_fraction"] for b in (1.0, 1.25, 1.5, 2.0)]
    assert traded[0] > traded[1] > traded[2] > traded[3] > 0
    s = pf.portfolio_stats(pf.build_portfolio(rows, 20, 0.5, 1.0), 20)
    assert s["empty_months"] == 12 and s["first_invested"] == "2014-01-31" and s["invested_months"] == 144
    assert s["partial_months"] == 0 and s["avg_names"] == 20


def test_n_larger_than_the_universe_holds_everything_eligible():
    _, _, rows = world(defects=False)
    port = pf.build_portfolio(rows, 150, 0.5, 1.0)
    for date, positions in port.items():
        n_elig = sum(1 for r in rows if r.date == date and r.eligible)
        assert len(positions) == n_elig
        assert not positions or abs(sum(p.weight for p in positions) - 1) < 1e-9


def test_stored_holdings_round_trip_on_real_signals():
    _, con, rows = world(n_stocks=30, seed=5, defects=False)
    port = pf.build_portfolio(rows, 10, 0.5, 1.5)
    run = pf.get_or_create_run(con, 10, 0.5, buffer=1.5)
    pf.store_holdings(con, run, port)
    loaded = pf.load_holdings(con, run)
    expected = {d: {p.ticker: p.weight for p in ps} for d, ps in port.items() if ps}
    assert loaded == expected


def test_lookahead_verdict_logic():
    assert stage4_run.lookahead_verdict({"changed_up_to_cutoff": 0, "changed_after_cutoff": 5}) == "OK"
    assert stage4_run.lookahead_verdict({"changed_up_to_cutoff": 1, "changed_after_cutoff": 5}) == "LOOK-AHEAD!"
    assert stage4_run.lookahead_verdict({"changed_up_to_cutoff": 0, "changed_after_cutoff": 0}).startswith("INCONCLUSIVE")


def test_scrambling_the_future_changes_no_earlier_portfolio():
    _, con, _ = world(n_stocks=30, seed=5, defects=False)
    cutoff = stage3_run.resolve_month_end(con, "2019-06-30")
    for what in (("prices", "eps"), ("prices",), ("eps",)):
        ex = stage4_run.portfolio_lookahead(con, cutoff, 5, 0.5, 1.5, scramble=what)
        assert stage4_run.lookahead_verdict(ex) == "OK", (what, ex)
        assert ex["months_up_to_cutoff"] > 50


def test_the_price_scramble_changes_ranks_not_just_levels():
    """A uniform multiplier would leave every rank unchanged and prove nothing about portfolios."""
    _, con, rows = world(n_stocks=30, seed=5, defects=False)
    cutoff = stage3_run.resolve_month_end(con, "2019-06-30")
    copy = stage3_run.scrambled_copy(con, cutoff, ("prices",))
    after = stage3_run.all_signals(copy)
    copy.close()
    changed = sum(1 for a, b in zip(rows, after) if a.date > cutoff and (a.pe_rank != b.pe_rank or a.momentum_rank != b.momentum_rank))
    assert changed > 500


# ===========================================================================
# the run script
# ===========================================================================

def test_try_cases_give_the_documented_answers():
    expected = {1: ["A", "B", "C"], 2: ["A", "B"], 3: ["A", "B", "C"], 4: [], 5: ["B", "C", "A"],
                6: ["D", "B", "C"], 7: ["B", "C", "D"], 8: ["A", "B", "C"], 9: ["A"]}
    assert len(stage4_run.try_cases()) == 9
    for n, want in expected.items():
        name, text, lesson = stage4_run.run_try_case(n)
        if want:
            holds = text.split("holds: ")[1]
            got = [part.split(" (")[0] for part in holds.split("), ")]
            assert got == want, (n, text)
        else:
            assert "nothing (all cash)" in text
        assert lesson
    assert "A(0.95) B(0.95)" in stage4_run.run_try_case(9)[1]
    assert "bought" in stage4_run.run_try_case(6)[1] and "kept" in stage4_run.run_try_case(5)[1]


def test_changes_lines_describe_what_was_kept_sold_and_bought():
    _, _, rows = world(defects=False)
    port = pf.build_portfolio(rows, 20, 0.5, 1.5)
    date = "2019-06-28"
    prev = sorted(port)[sorted(port).index(date) - 1]
    text = "\n".join(stage4_run.changes_lines(rows, port, date, 20, 0.5, 1.5))
    now, before = {p.ticker for p in port[date]}, {p.ticker for p in port[prev]}
    assert f"KEPT   ({len(now & before)})" in text
    assert f"SOLD   ({len(before - now)})" in text and f"BOUGHT ({len(now - before)})" in text
    for t in before - now:
        assert f"{t}: was place" in text
    for t in now - before:
        assert f"{t}: place" in text
    assert "within the top 30" in text and "traded" in text
    assert "is not a month-end" in stage4_run.changes_lines(rows, port, "2019-06-27", 20, 0.5, 1.5)[0]
    first = sorted(port)[0]
    assert "Nothing is held" in "\n".join(stage4_run.changes_lines(rows, port, first, 20, 0.5, 1.5))


def test_changes_lines_explain_a_stock_that_lost_eligibility():
    market, con, rows = world(defects=True)
    port = pf.build_portfolio(rows, 20, 0.5, 1.0)
    dates = sorted(port)
    found = None
    for prev, cur in zip(dates, dates[1:]):
        gone = {p.ticker for p in port[prev]} - {p.ticker for p in port[cur]}
        elig = {r.ticker for r in rows if r.date == cur and r.eligible}
        if gone - elig:
            found = (cur, sorted(gone - elig)[0])
            break
    if found:
        text = "\n".join(stage4_run.changes_lines(rows, port, found[0], 20, 0.5, 1.0))
        assert f"{found[1]}: was place" in text and "no longer eligible" in text


def test_month_on_or_before():
    months = ["2020-01-31", "2020-02-28", "2020-03-31"]
    assert stage4_run.month_on_or_before(months, "2020-03-15") == "2020-02-28"
    assert stage4_run.month_on_or_before(months, "2020-03-31") == "2020-03-31"
    assert stage4_run.month_on_or_before(months, "2030-01-01") == "2020-03-31"
    for bad in ("abc", "2020-13-45", "2019-12-31", ""):
        try:
            stage4_run.month_on_or_before(months, bad)
        except ValueError:
            continue
        raise AssertionError(bad)


def test_build_lines_and_build_mode():
    _, _, rows = world(n_stocks=30, seed=5, defects=False)
    lines, bad = stage4_run.build_lines(rows, 10, 0.5, 1.5)
    text = "\n".join(lines)
    assert bad == [] and "all rules hold" in text and "Top 10" in text and "preview of costs" in text
    assert quiet(stage4_run.build_mode, rows, 10, 0.5, 1.5)[0] == 0


def test_interactive_build_mode_accepts_numbers_and_rejects_bad_ones():
    _, _, rows = world(n_stocks=30, seed=5, defects=False)
    inputs = ["abc", "", "", "5", "1.5", "", "10", "0.8", "1.25", "q", "", ""]
    with mock.patch("builtins.input", side_effect=inputs):
        code, out = quiet(stage4_run.build_mode, rows)
    assert code == 0 and "Please type numbers only" in out and "must be between" in out
    assert "Top 10, 0.80 weight on value" in out
    with mock.patch("builtins.input", side_effect=EOFError):
        assert quiet(stage4_run.build_mode, rows)[0] == 0


def test_interactive_changes_mode_quits_on_enter_and_rejects_bad_input():
    _, _, rows = world(n_stocks=30, seed=5, defects=False)
    inputs = ["abc", "1.5", "2019-06-28", "9", "2019-06-28", "1.5", "2012-01-01", "1.5", ""]
    with mock.patch("builtins.input", side_effect=inputs):
        code, out = quiet(stage4_run.changes_mode, rows, 10)
    assert code == 0
    assert out.count("Please type a number") == 1 and "must be between" in out and "before the first month-end" in out
    assert out.count("Month-end 2019-06-28") == 1


def test_main_argument_handling():
    for argv in (["--build", "20", "0.5"], ["--build", "0", "0.5", "1.5"], ["--build", "20", "1.5", "1.0"],
                 ["--build", "20", "0.5", "0.5"], ["--build", "x", "0.5", "1.5"], ["--changes", "2019-06-28"],
                 ["--changes", "notadate", "1.5"], ["--changes", "2019-06-28", "9"], ["--changes", "2012-01-01", "1.5"],
                 ["--try", "99"]):
        assert quiet(stage4_run.main, argv)[0] == 1, argv
    code, out = quiet(stage4_run.main, ["--build", "20", "0.5", "1.5", "--clean"])
    assert code == 0 and "all rules hold" in out
    code, out = quiet(stage4_run.main, ["--changes", "2019-06-30", "1.5", "--clean"])
    assert code == 0 and "Month-end 2019-06-28" in out
    code, out = quiet(stage4_run.main, ["--try", "5"])
    assert code == 0 and "buffer keeps" in out


def test_demo_succeeds_and_saves_a_database_that_can_be_reopened():
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "p.db")
        code, out = quiet(stage4_run.demo, clean=True, db_path=path)
        assert code == 0, out[-600:]
        assert "all rules hold" in out and "INCONCLUSIVE" not in out and "LOOK-AHEAD" not in out and "Check question" in out
        con = sqlite3.connect(path)
        assert con.execute("SELECT COUNT(*) FROM RUN").fetchone()[0] == 2
        assert con.execute("SELECT COUNT(*) FROM HOLDING").fetchone()[0] == 5760
        con.close()
        code, out = quiet(stage4_run.main, ["--build", "20", "0.5", "1.0", "--db", path])
        assert code == 0 and "all rules hold" in out
    code, out = quiet(stage4_run.demo)                                   # with planted defects too
    assert code == 0


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
    _close_cached_worlds()
    print(f"\n{len(tests) - failed}/{len(tests)} tests passed")
    sys.exit(1 if failed else 0)
