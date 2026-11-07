"""Tests for Stage 5 (engine.py, stage5_run.py, the metrics).

Run either way:
    python tests/test_stage5.py
    python -m pytest -q
"""
import contextlib
import dataclasses
import io
import math
import os
import sqlite3
import sys
import tempfile
from unittest import mock

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import config
import db
import engine
import portfolio as pf
import stage3_run
import stage4_run
import stage5_run
from stage5_run import CAL, S1, S2, S3, T1, T2, T3, flat, mini_world, pos

_CACHE = {}
_OPEN = []


def world(edge=0.05, seed=7, defects=True, n_stocks=100):
    key = (edge, seed, defects, n_stocks)
    if key not in _CACHE:
        market, con = stage3_run.build_world(edge=edge, seed=seed, defects=defects, n_stocks=n_stocks)
        _CACHE[key] = (con, stage3_run.all_signals(con), stage5_run.Env(con))
    return _CACHE[key]


def _close_cached_worlds():
    while _OPEN:
        _OPEN.pop().close()
    for con, _, _ in _CACHE.values():
        con.close()
    _CACHE.clear()


try:
    import pytest

    @pytest.fixture(scope="module", autouse=True)
    def _close_worlds_at_end():
        yield
        _close_cached_worlds()
except ImportError:
    pass


def quiet(fn, *a, **k):
    with contextlib.redirect_stdout(io.StringIO()) as buf:
        out = fn(*a, **k)
    return out, buf.getvalue()


def close(a, b, tol=1e-12):
    return abs(a - b) < tol


def mini(prices):
    con, env = mini_world(prices)
    _OPEN.append(con)
    return env


def drift_world():
    """A doubles between the two trades, B does not."""
    return mini({"SPY": flat(100.0), "A": {**flat(100.0), S2: 150.0, T2: 200.0, S3: 200.0, T3: 200.0}, "B": flat(100.0)})


def rows_of(periods):
    return [(p.date, p.gross, p.cost, p.net, p.benchmark) for p in periods]


# ===========================================================================
# parameters, calendar, schedule
# ===========================================================================

def test_parameter_validation():
    for cost, delay in ((-1, 1), (config.MAX_COST_BPS + 1, 1), ("10", 1), (True, 1), (None, 1),
                        (10, -1), (10, config.MAX_EXEC_DELAY_DAYS + 1), (10, 1.5)):
        try:
            engine._check_params(cost, delay)
        except ValueError:
            continue
        raise AssertionError((cost, delay))
    engine._check_params(0, 0)
    engine._check_params(config.MAX_COST_BPS, config.MAX_EXEC_DELAY_DAYS)
    env = mini({"SPY": flat(100.0), "A": flat(100.0)})
    for kwargs in ({"cost_bps": -5}, {"delay": 9}):
        try:
            env.run({S1: pos("A")}, **{"cost_bps": 10, **kwargs})
        except ValueError:
            continue
        raise AssertionError(kwargs)


def test_trading_days_come_from_the_single_benchmark():
    con, env = mini_world({"SPY": flat(100.0), "A": {S1: 1.0}})
    _OPEN.append(con)
    assert env.calendar == CAL and env.bench == "SPY"
    con.execute("UPDATE STOCK SET is_benchmark = 1 WHERE ticker = 'A'")
    try:
        engine.trading_days(con)
    except ValueError:
        pass
    else:
        raise AssertionError("two benchmarks must be refused")
    con.execute("UPDATE STOCK SET is_benchmark = 0")
    try:
        engine.trading_days(con)
    except ValueError:
        pass
    else:
        raise AssertionError("no benchmark must be refused")


def test_execution_date():
    assert engine.execution_date(CAL, S1) == T1 and engine.execution_date(CAL, S1, 0) == S1
    assert engine.execution_date(CAL, S1, 2) == S2 and engine.execution_date(CAL, S3, 1) == T3
    assert engine.execution_date(CAL, T3, 1) is None and engine.execution_date(CAL, S3, 3) is None
    for bad in ("2020-02-01", "1999-01-01", "2030-01-01"):
        try:
            engine.execution_date(CAL, bad)
        except ValueError:
            continue
        raise AssertionError(bad)


def test_schedule_stops_where_the_data_ends_and_keeps_order():
    port = {S3: pos("A"), S1: pos("A"), S2: pos("B")}
    sched = engine.schedule(port, CAL, 1)
    assert [(a, b) for a, b, _ in sched] == [(S1, T1), (S2, T2), (S3, T3)]
    assert [(a, b) for a, b, _ in engine.schedule({**port, T3: pos("A")}, CAL, 1)] == [(S1, T1), (S2, T2), (S3, T3)]
    assert sched[1][2] == {"B": 1.0}


# ===========================================================================
# the simulation, by hand
# ===========================================================================

def test_trade_is_after_the_signal_and_the_jump_between_belongs_to_someone_else():
    prices = {"SPY": flat(100.0), "A": {S1: 100.0, T1: 110.0, S2: 110.0, T2: 110.0, S3: 110.0, T3: 110.0}}
    env = mini(prices)
    port = {S1: pos("A"), S2: pos("A")}
    late = env.run(port, 0, 1)
    assert [p.date for p in late] == [T1, T2] and close(late[1].gross, 0.0)
    same = env.run(port, 0, 0)
    assert [p.date for p in same] == [S1, S2] and close(same[1].gross, 0.10)
    assert all(p.date >= p.signal_date for p in late + same)
    two = env.run(port, 0, 2)
    assert [p.date for p in two] == [S2, S3] and two[0].signal_date == S1 and two[1].signal_date == S2


def test_inception_row_buys_from_cash_and_pays_the_entry_cost():
    env = mini({"SPY": flat(100.0), "A": flat(100.0), "B": flat(100.0)})
    first = env.run({S1: pos("A", "B")}, 25)[0]
    assert first.date == T1 and first.gross == 0.0 and first.benchmark == 0.0
    assert close(first.traded_fraction, 1.0) and close(first.cost, 0.0025) and close(first.net, -0.0025)
    assert first.before == {} and first.n_held == 2 and close(first.after["A"], 0.5 * (1 - 0.0025) / (1 - 0.0025))


def test_drift_changes_the_weights_and_the_trade():
    env = drift_world()
    per = env.run({S1: pos("A", "B"), S2: pos("A", "B")}, 100)
    p = per[1]
    assert close(p.gross, 0.5) and close(p.before["A"], 2 / 3) and close(p.before["B"], 1 / 3)
    assert close(p.traded_fraction, 1 / 3) and close(p.cost, 0.01 / 3)
    assert close(p.net, 1.5 * (1 - 0.01 / 3) - 1)
    assert close(p.after["A"], 0.5) and close(p.after["B"], 0.5)
    assert pf.traded_fraction({"A": .5, "B": .5}, {"A": .5, "B": .5}) == 0       # what Stage 4's preview would have said


def test_full_swap_costs_twice_and_wealth_is_the_product():
    env = mini({"SPY": flat(100.0), **{t: flat(100.0) for t in "ABCD"}})
    per = env.run({S1: pos("A", "B"), S2: pos("C", "D")}, 50)
    assert close(per[1].traded_fraction, 2.0) and close(per[1].cost, 0.01)
    assert close(engine.net_wealth_path(per)[-1], 0.995 * 0.99)


def test_zero_cost_means_net_equals_gross_and_costs_never_help():
    env = drift_world()
    port = {S1: pos("A", "B"), S2: pos("A", "B"), S3: pos("B")}
    free = env.run(port, 0)
    assert all(p.cost == 0 and p.net == p.gross for p in free)
    paid = env.run(port, 100)
    assert all(close(a.gross, b.gross) for a, b in zip(paid, free))              # costs never change what the holdings earned
    assert all(p.net <= p.gross for p in paid)
    assert engine.net_wealth_path(paid)[-1] < engine.net_wealth_path(free)[-1]


def test_an_empty_target_sells_everything_and_cash_earns_nothing():
    prices = {"SPY": {S1: 100.0, T1: 100.0, S2: 100.0, T2: 100.0, S3: 105.0, T3: 110.0},
              "A": {S1: 100.0, T1: 100.0, S2: 105.0, T2: 105.0, S3: 106.0, T3: 106.0}}
    env = mini(prices)
    per = env.run({S1: pos("A"), S2: [], S3: pos("A")}, 10)
    assert [p.n_held for p in per] == [1, 0, 1]
    assert close(per[1].gross, 0.05) and per[1].after == {} and close(per[1].cash, 1.0) and close(per[1].traded_fraction, 1.0)
    assert close(per[2].gross, 0.0) and close(per[2].benchmark, 0.10) and close(per[2].traded_fraction, 1.0)


def test_nothing_is_held_and_no_row_is_made_until_there_is_something_to_buy():
    env = mini({"SPY": flat(100.0), "A": flat(100.0)})
    per = env.run({S1: [], S2: pos("A")}, 10)
    assert [p.date for p in per] == [T2] and per[0].signal_date == S2
    assert env.run({S1: [], S2: []}, 10) == []
    assert env.run({}, 10) == []


def test_a_stock_with_no_price_cannot_be_bought_and_the_rest_are_re_split():
    env = mini({"SPY": flat(100.0), "A": flat(100.0), "B": flat(100.0), "C": {T2: 100.0, S3: 100.0, T3: 100.0}})
    p = env.run({S1: pos("A", "B", "C")}, 0)[0]
    assert p.skipped == ("C",) and set(p.after) == {"A", "B"} and close(p.after["A"], 0.5) and p.n_held == 2


def test_when_every_target_is_unbuyable_we_stay_in_cash_without_cost():
    env = mini({"SPY": flat(100.0), "C": {T2: 100.0, S3: 100.0, T3: 100.0}})
    p = env.run({S1: pos("C")}, 50)[0]
    assert p.skipped == ("C",) and p.after == {} and close(p.cash, 1.0) and p.cost == 0.0 and p.traded_fraction == 0.0


def frozen_prices():
    return {"SPY": flat(100.0), "A": {S1: 100.0, T1: 100.0, S2: 110.0, T2: 120.0, S3: 120.0, T3: 120.0}, "B": {S1: 100.0, T1: 100.0, "2020-04-20": 100.0}}


def test_a_held_stock_with_no_price_is_frozen_not_sold():
    env = mini(frozen_prices())
    per = env.run({S1: pos("A", "B"), S2: pos("A", "B")}, 100)
    p = per[1]
    assert p.unpriced == 1 and p.skipped == () and close(p.gross, 0.10)
    assert close(p.traded_fraction, 0.0) and close(p.cost, 0.0)
    assert close(p.after["A"], 0.6 / 1.1) and close(p.after["B"], 0.5 / 1.1) and p.n_held == 2


def test_a_frozen_stock_that_is_no_longer_targeted_is_still_frozen_and_the_rest_trade():
    env = mini({**frozen_prices(), "C": flat(100.0)})
    per = env.run({S1: pos("A", "B"), S2: pos("C")}, 0)
    p = per[1]
    assert p.unpriced == 1 and set(p.after) == {"B", "C"}
    assert close(p.after["B"], 0.5 / 1.1) and close(p.after["C"], 0.6 / 1.1)       # A (0.6) sold, C (0.6) bought, B untouched
    assert close(p.traded_fraction, 1.2 / 1.1)


def test_benchmark_is_bought_the_same_day_and_pays_nothing():
    env = mini({"SPY": {S1: 100.0, T1: 100.0, S2: 100.0, T2: 110.0, S3: 121.0, T3: 121.0}, "A": flat(100.0)})
    per = env.run({S1: pos("A"), S2: pos("A"), S3: pos("A")}, 500)
    assert per[0].benchmark == 0.0 and close(per[1].benchmark, 0.10) and close(per[2].benchmark, 0.10)


def test_a_signal_date_that_is_not_a_trading_day_is_refused():
    env = mini({"SPY": flat(100.0), "A": flat(100.0)})
    try:
        env.run({"2020-02-01": pos("A")}, 0)
    except ValueError:
        pass
    else:
        raise AssertionError("a signal on a non-trading day must be refused")


# ===========================================================================
# the share ledger and the checker
# ===========================================================================

def hard_cases():
    drift = {"SPY": flat(100.0), "A": {**flat(100.0), S2: 150.0, T2: 200.0, S3: 200.0, T3: 200.0}, "B": flat(100.0)}
    return [
        (drift, {S1: pos("A", "B"), S2: pos("A", "B"), S3: pos("B")}),
        (frozen_prices(), {S1: pos("A", "B"), S2: pos("A"), S3: pos("A", "B")}),
        ({**frozen_prices(), "C": flat(100.0)}, {S1: pos("A", "B"), S2: pos("C"), S3: pos("A", "C")}),
        ({"SPY": flat(100.0), "A": flat(100.0), "C": {T2: 90.0, S3: 90.0, T3: 95.0}}, {S1: pos("A", "C"), S2: [], S3: pos("A", "C")}),
    ]


def test_the_share_ledger_agrees_with_the_engine_on_every_awkward_case():
    for prices, port in hard_cases():
        env = mini(prices)
        for cost in (0, 37, 250):
            per = env.run(port, cost)
            assert engine.check_engine(per, env.replay(port, cost)) == [], (cost, port)


def test_the_checker_catches_each_kind_of_disagreement():
    env = drift_world()
    port = {S1: pos("A", "B"), S2: pos("A", "B"), S3: pos("B")}
    per = env.run(port, 100)
    ledger = env.replay(port, 100)
    assert engine.check_engine(per, ledger) == []
    wrong_net = [dataclasses.replace(per[0]), dataclasses.replace(per[1], net=per[1].net + 0.01), per[2]]
    assert any("engine wealth" in m for m in engine.check_engine(wrong_net, ledger))
    assert any("rows" in m for m in engine.check_engine(per[:2], ledger))
    assert any("dates differ" in m for m in engine.check_engine(per, [(d + "x", w) for d, w in ledger]))
    bad_cost = [dataclasses.replace(per[0], cost=-0.5)] + per[1:]
    assert any("out of range" in m for m in engine.check_engine(bad_cost, ledger))
    bad_net = [dataclasses.replace(per[0], net=per[0].gross + 0.2)] + per[1:]
    assert any("net return is not" in m for m in engine.check_engine(bad_net, ledger))
    bad_after = [per[0], dataclasses.replace(per[1], after={"A": 0.9, "B": 0.5})] + per[2:]
    assert any("add" in m for m in engine.check_engine(bad_after, ledger))
    bad_before = [per[0], dataclasses.replace(per[1], before={"A": 0.9, "B": 0.5})] + per[2:]
    assert any("drifted" in m for m in engine.check_engine(bad_before, ledger))


# ===========================================================================
# metrics, against numbers worked out by hand
# ===========================================================================

def test_growth_cagr_and_drawdown():
    assert close(engine.growth([0.1, -0.1]), 0.99) and engine.growth([]) == 1.0
    assert close(engine.cagr([0.01] * 12, 1.0), 1.01 ** 12 - 1) and close(engine.cagr([0.01] * 12, 2.0), 1.01 ** 6 - 1)
    assert math.isnan(engine.cagr([0.1], 0)) and math.isnan(engine.cagr([0.1], -1))
    assert close(engine.max_drawdown([0.1, -0.5, 0.2]), -0.5)
    assert engine.max_drawdown([0.1, 0.1]) == 0.0 and engine.max_drawdown([]) == 0.0
    assert close(engine.max_drawdown([-0.1]), -0.1)                                  # a fall from the starting 1.0 counts
    assert close(engine.max_drawdown([0.2, -0.1, -0.1, 0.5, -0.3]), min(0.9 * 0.9 - 1 + 0 * 0, 1.2 * 0.9 * 0.9 / 1.2 - 1, 1.2 * 0.9 * 0.9 * 1.5 * 0.7 / (1.2 * 0.9 * 0.9 * 1.5) - 1))


def test_sharpe_by_hand():
    r = [0.01, 0.03]
    assert close(engine.sharpe(r, 0.0), 0.02 / math.sqrt(0.0002) * math.sqrt(12), 1e-9)
    rf = 1.02 ** (1 / 12) - 1
    assert close(engine.sharpe(r, 0.02), (0.02 - rf) / math.sqrt(0.0002) * math.sqrt(12), 1e-9)
    assert math.isnan(engine.sharpe([0.01, 0.01, 0.01])) and math.isnan(engine.sharpe([0.01])) and math.isnan(engine.sharpe([]))
    assert engine.sharpe([0.01, 0.03], 0.0, 4) < engine.sharpe([0.01, 0.03], 0.0, 12)        # more periods a year, bigger scaling
    assert engine.sharpe(r, 0.10) < engine.sharpe(r, 0.0)


def test_beta_and_information_ratio_by_hand():
    bench = [0.01, -0.02, 0.03, 0.0]
    assert close(engine.beta([2 * b for b in bench], bench), 2.0)
    assert close(engine.beta([b + 0.5 for b in bench], bench), 1.0)
    assert math.isnan(engine.beta([0.1, 0.2], [0.05, 0.05])) and math.isnan(engine.beta([0.1], [0.1]))
    assert math.isnan(engine.beta([0.1, 0.2, 0.3], [0.1, 0.2]))
    assert math.isnan(engine.information_ratio([0.02, 0.03], [0.01, 0.02]))               # constant difference: no spread to divide by
    ir = engine.information_ratio([0.03, 0.01], [0.0, 0.0])
    assert close(ir, 0.02 / math.sqrt(0.0002) * math.sqrt(12), 1e-9)


def period(i, gross, net=None, bench=0.0, traded=0.5, cost=0.0):
    net = gross if net is None else net
    return engine.Period(f"2020-{i + 1:02d}-01", f"2019-{i + 1:02d}-28", gross, cost, net, bench, traded, 5)


def test_metrics_on_a_hand_made_series():
    periods = [period(0, 0.0, -0.001, 0.0, 1.0, 0.001)] + [period(i, 0.01, 0.009, 0.005, 0.4 if i < 3 else 0.2, 0.001) for i in range(1, 13)]
    m = engine.metrics(periods)
    assert m["periods"] == 12 and close(m["years"], 1.0)
    assert close(m["gross"]["total_return"], 1.01 ** 12 - 1) and close(m["gross"]["cagr"], 1.01 ** 12 - 1)
    assert close(m["net"]["total_return"], 0.999 * 1.009 ** 12 - 1)                    # the entry cost counts toward total return
    assert close(m["benchmark"]["total_return"], 1.005 ** 12 - 1)
    assert math.isnan(m["gross"]["sharpe"]) and close(m["gross"]["volatility"], 0.0)  # inception row is NOT a return period
    assert close(m["excess_cagr_net"], m["net"]["cagr"] - m["benchmark"]["cagr"])
    assert close(m["excess_cagr_gross"], m["gross"]["cagr"] - m["benchmark"]["cagr"])
    assert close(m["cost_drag_cagr"], m["gross"]["cagr"] - m["net"]["cagr"]) and m["cost_drag_cagr"] > 0
    assert close(m["hit_rate"], 1.0)
    assert close(m["avg_traded_fraction"], (0.4 * 2 + 0.2 * 10) / 12)                   # every row after the inception purchase
    assert close(m["avg_cost_per_month"], 0.001)


def test_metrics_validation_and_small_inputs():
    one = [period(0, 0.0)]
    for args in ((one,), ([period(0, 0.0), period(1, 0.01)], -0.01), ([period(0, 0.0), period(1, 0.01)], config.MAX_RISK_FREE_RATE + 0.01)):
        try:
            engine.metrics(*args)
        except ValueError:
            continue
        raise AssertionError(args)
    m = engine.metrics([period(0, 0.0), period(1, 0.01)])
    assert close(m["avg_traded_fraction"], 0.5)                                         # the one rebalance after the inception purchase is steady
    assert m["hit_rate"] == 1.0 and m["periods"] == 1


def test_average_trading_ignores_the_entry_purchase_and_re_entry_after_cash():
    ps = [period(0, 0.0, traded=1.0), period(1, 0.01, traded=0.3), period(2, 0.01, traded=0.5)]
    ps += [dataclasses.replace(period(3, 0.0, traded=1.0), n_held=0), period(4, 0.0, traded=1.0), period(5, 0.01, traded=0.1)]
    assert close(engine.metrics(ps)["avg_traded_fraction"], (0.3 + 0.5 + 0.1) / 3)         # the sale into cash, the re-entry and the inception are left out
    cash_only = [period(0, 0.0), dataclasses.replace(period(1, 0.0), n_held=0)]
    assert engine.metrics(cash_only)["avg_traded_fraction"] != engine.metrics(cash_only)["avg_traded_fraction"]


def test_metrics_count_skipped_purchases_and_frozen_holdings():
    env = mini({"SPY": flat(100.0), "A": flat(100.0), "B": flat(100.0), "C": {T2: 100.0, S3: 100.0, T3: 100.0}})
    m = engine.metrics(env.run({S1: pos("A", "B", "C"), S2: pos("A", "B")}, 0))
    assert m["skipped_buys"] == 1 and m["unpriced_marks"] == 0
    m = engine.metrics(mini(frozen_prices()).run({S1: pos("A", "B"), S2: pos("A", "B"), S3: pos("A", "B")}, 0))
    assert m["unpriced_marks"] == 2 and m["skipped_buys"] == 0                      # B has no price on the 2nd and 3rd trade days


def test_null_world_verdict():
    mean, se, ok = stage5_run.null_world_verdict([-0.0198, -0.0189, 0.0181, 0.0006])
    assert ok and close(mean, -0.0050, 1e-4) and close(se, 0.0090, 1e-3)
    assert not stage5_run.null_world_verdict([0.05, 0.06, 0.07, 0.05])[2]             # a consistent profit with no edge: alarm
    assert not stage5_run.null_world_verdict([-0.05, -0.06, -0.07, -0.05])[2]
    assert stage5_run.null_world_verdict([0.0, 0.0, 0.0])[2]
    try:
        stage5_run.null_world_verdict([0.01, 0.02])
    except ValueError:
        pass
    else:
        raise AssertionError("two worlds are not enough to judge noise")


def test_metrics_hit_rate_counts_only_strict_wins():
    ps = [period(0, 0.0), period(1, 0.02, bench=0.01), period(2, 0.01, bench=0.01), period(3, 0.0, bench=0.01), period(4, 0.03, bench=0.0)]
    assert close(engine.metrics(ps)["hit_rate"], 0.5)


# ===========================================================================
# storage
# ===========================================================================

def con_with(*tickers):
    con = db.connect()
    _OPEN.append(con)
    db.create_schema(con)
    return con


def test_store_results_insert_repeat_update_and_remove():
    con = con_with()
    run = pf.get_or_create_run(con, 3, 0.5)
    per = [period(0, 0.0, -0.001), period(1, 0.01, 0.009, 0.005), period(2, 0.02, 0.019, 0.01)]
    assert engine.store_results(con, run, per) == {"inserted": 3, "updated": 0, "unchanged": 0, "deleted": 0}
    assert engine.store_results(con, run, per) == {"inserted": 0, "updated": 0, "unchanged": 3, "deleted": 0}
    changed = [per[0], dataclasses.replace(per[1], net=0.0089)]
    assert engine.store_results(con, run, changed) == {"inserted": 0, "updated": 1, "unchanged": 1, "deleted": 1}
    got = engine.load_results(con, run)
    assert [r["date"] for r in got] == [per[0].date, per[1].date] and close(got[1]["net"], 0.0089)
    assert set(got[0]) == {"date", "gross", "cost", "net", "benchmark"}
    other = pf.get_or_create_run(con, 3, 0.5, buffer=1.5)
    assert engine.load_results(con, other) == []
    engine.store_results(con, other, per[:1])
    assert len(engine.load_results(con, run)) == 2 and len(engine.load_results(con, other)) == 1


def test_store_results_obeys_the_schema():
    con = con_with()
    run = pf.get_or_create_run(con, 3, 0.5)
    for bad in (dataclasses.replace(period(1, 0.0), gross=-1.0), dataclasses.replace(period(1, 0.0), net=-1.5),
                dataclasses.replace(period(1, 0.0), cost=-0.1), dataclasses.replace(period(1, 0.0), benchmark=float("inf"))):
        try:
            engine.store_results(con, run, [bad])
        except sqlite3.IntegrityError:
            con.rollback()
            continue
        raise AssertionError(bad)
    try:
        engine.store_results(con, 999, [period(1, 0.0)])                                # no such run
    except sqlite3.IntegrityError:
        con.rollback()
    else:
        raise AssertionError("an unknown run must be refused")
    assert con.execute("SELECT COUNT(*) FROM RESULT").fetchone()[0] == 0


# ===========================================================================
# on the synthetic market
# ===========================================================================

def test_engine_and_ledger_agree_on_the_defect_market_for_many_settings():
    con, rows, env = world(defects=True)
    for n, w, b, cost, delay in [(20, 0.5, 1.0, 10, 1), (20, 0.5, 1.5, 25, 1), (10, 0.0, 2.0, 0, 0), (50, 1.0, 1.25, 100, 2), (100, 0.5, 1.0, 10, 1), (1, 0.5, 3.0, 5, 5)]:
        port = pf.build_portfolio(rows, n, w, b)
        per = env.run(port, cost, delay)
        assert len(per) > 100 and engine.check_engine(per, env.replay(port, cost, delay)) == [], (n, w, b, cost, delay)


def test_rows_start_the_day_after_the_first_signal_and_dates_increase():
    con, rows, env = world(defects=False)
    port = pf.build_portfolio(rows, 20, 0.5, 1.0)
    per = env.run(port, 10)
    first_signal = min(d for d, p in port.items() if p)
    assert per[0].signal_date == first_signal and per[0].date > first_signal
    assert [p.date for p in per] == sorted(p.date for p in per) and len({p.date for p in per}) == len(per)
    assert all(a.signal_date < b.signal_date for a, b in zip(per, per[1:]))
    assert per[0].gross == 0 and per[0].benchmark == 0 and per[0].traded_fraction == 1.0


def test_after_weights_equal_the_target_whenever_every_stock_was_priced():
    con, rows, env = world(defects=False)
    port = pf.build_portfolio(rows, 20, 0.5, 1.5)
    per = env.run(port, 10)
    checked = 0
    for p in per:
        if not p.skipped and not p.unpriced:
            target = {x.ticker: x.weight for x in port[p.signal_date]}
            assert set(p.after) == set(target) and all(close(p.after[t], target[t]) for t in target), p.date
            checked += 1
    assert checked > 100


def test_costs_and_gross_returns():
    con, rows, env = world(defects=True)
    port = pf.build_portfolio(rows, 20, 0.5, 1.0)
    free, paid = env.run(port, 0), env.run(port, 50)
    assert all(abs(a.gross - b.gross) < 1e-12 for a, b in zip(free, paid))
    assert all(p.net <= p.gross + 1e-15 for p in paid)
    mf, mp = engine.metrics(free), engine.metrics(paid)
    assert mf["net"]["cagr"] == mf["gross"]["cagr"] and mp["net"]["cagr"] < mp["gross"]["cagr"]
    cagrs = [engine.metrics(env.run(port, c))["net"]["cagr"] for c in (0, 10, 25, 50, 100)]
    assert all(a > b for a, b in zip(cagrs, cagrs[1:]))


def test_drift_makes_real_trading_higher_than_the_stage_4_preview():
    con, rows, env = world(defects=True)
    for buffer in (1.0, 1.5):
        port = pf.build_portfolio(rows, 20, 0.5, buffer)
        preview = pf.portfolio_stats(port, 20)["avg_traded_fraction"]
        real = engine.metrics(env.run(port, 10))["avg_traded_fraction"]
        assert real > preview * 1.05, (buffer, preview, real)
    assert engine.metrics(env.run(pf.build_portfolio(rows, 20, 0.5, 1.5), 10))["avg_traded_fraction"] < \
        engine.metrics(env.run(pf.build_portfolio(rows, 20, 0.5, 1.0), 10))["avg_traded_fraction"]


def test_a_strong_planted_edge_is_found_and_the_benchmark_is_unaffected_by_the_strategy():
    con, rows, env = world(edge=0.15, defects=False)
    port = pf.build_portfolio(rows, 20, 0.5, 1.0)
    m = engine.metrics(env.run(port, 0))
    assert m["excess_cagr_gross"] > 0.02
    m2 = engine.metrics(env.run(pf.build_portfolio(rows, 5, 0.5, 1.0), 100))
    assert close(m["benchmark"]["cagr"], m2["benchmark"]["cagr"], 1e-9)        # SPY buy-and-hold, whatever we do


def test_no_edge_gives_no_large_false_profit():
    con, rows, env = world(edge=0.0, seed=2, defects=False)
    m = engine.metrics(env.run(pf.build_portfolio(rows, 20, 0.5, 1.0), 0))
    assert abs(m["excess_cagr_gross"]) < 0.04


def test_returns_do_not_use_the_future():
    con, rows, env = world(n_stocks=30, seed=5, defects=False)
    cutoff = stage3_run.resolve_month_end(con, "2019-06-30")
    for what in (("prices", "eps"), ("prices",), ("eps",)):
        ex = stage5_run.returns_lookahead(con, cutoff, 5, 0.5, 1.5, 10, what)
        assert stage4_run.lookahead_verdict(ex) == "OK", (what, ex)
        assert ex["rows_up_to_cutoff"] > 50 and ex["changed_after_cutoff"] > 50


def test_results_on_the_real_pipeline_round_trip_through_the_database():
    con, rows, env = world(n_stocks=30, seed=5, defects=False)
    port = pf.build_portfolio(rows, 10, 0.5, 1.5)
    per = env.run(port, 10)
    run = pf.get_or_create_run(con, 10, 0.5, 10, 1.5)
    engine.store_results(con, run, per)
    got = engine.load_results(con, run)
    assert len(got) == len(per) and all(close(g["net"], p.net) and close(g["gross"], p.gross) and close(g["benchmark"], p.benchmark) for g, p in zip(got, per))
    assert engine.backtest(con, port, 10)[5].net == per[5].net


# ===========================================================================
# the run script
# ===========================================================================

def test_formatting_helpers():
    assert stage5_run.pc(float("nan")).strip() == "n/a" and stage5_run.pc(0.1234).strip() == "12.34%"
    assert stage5_run.num(float("nan")).strip() == "n/a" and stage5_run.num(1.234).strip() == "1.23"


def test_try_cases_give_the_documented_answers():
    cases = stage5_run.try_cases()
    assert len(cases) == 7
    out = {n: stage5_run.run_try_case(n) for n in range(1, 8)}
    assert all(lesson for _, _, lesson in out.values())
    assert "+10.00%" in out[1][1] and "trade 0 day(s)" in out[1][1]
    assert "66.7%" in out[2][1] and "0.333%" in out[2][1] and "would have said 0 trades" in out[2][1]
    assert "traded  2.00" in out[3][1].replace("traded 2.00", "traded  2.00") or "2.00" in out[3][1]
    assert "1.00%" in out[3][1] and "0.98505" in out[3][1]
    assert "cash" in out[4][1] and "+10.00%" in out[4][1]
    assert "skipped: C" in out[5][1] and "50%" in out[5][1]
    assert "FROZEN" in out[6][1] and "+10.0%" in out[6][1]
    assert "problems found: none" in out[7][1]


def test_try_mode_menu_and_direct_pick():
    assert quiet(stage5_run.try_mode, 3)[0] == 0
    assert quiet(stage5_run.try_mode, 99)[0] == 1
    with mock.patch("builtins.input", side_effect=["zzz", "0", "8", "2", ""]):
        code, out = quiet(stage5_run.try_mode)
    assert code == 0 and out.count("Not on the list") == 3 and "[2] Weights drift" in out
    with mock.patch("builtins.input", side_effect=EOFError):
        assert quiet(stage5_run.try_mode)[0] == 0


def test_run_lines_report_and_optimism_warning():
    con, rows, env = world(n_stocks=30, seed=5, defects=False)
    lines, bad = stage5_run.run_lines(rows, env, 10, 0.5, 1.5, 10)
    text = "\n".join(lines)
    assert bad == [] and "agrees on every date" in text and "SPY" in text and "after costs" in text and "Stage 4 preview" in text
    assert "1 trading day(s) after" in text and "fixed risk-free rate of 2.0%" in text
    assert "SAME close" in "\n".join(stage5_run.run_lines(rows, env, 10, 0.5, 1.5, 10, delay=0)[0])
    lines, bad = stage5_run.run_lines([], env, 10, 0.5, 1.5, 10)
    assert bad == ["no result rows"] and "Not enough" in lines[0]
    assert quiet(stage5_run.run_mode, rows, env, 10, 0.5, 1.5, 10)[0] == 0


def test_interactive_run_mode():
    con, rows, env = world(n_stocks=30, seed=5, defects=False)
    inputs = ["abc", "", "", "", "0", "", "", "", "10", "1.5", "", "", "10", "0.8", "1.25", "20", "5", "", "", "", "q", "", "", ""]
    with mock.patch("builtins.input", side_effect=inputs):
        code, out = quiet(stage5_run.run_mode, rows, env)
    assert code == 0 and "numbers only" in out and "top_n" in out or "between" in out
    assert "Top 10, 0.80 weight on value, buffer 1.25, cost 20 bps" in out
    assert "Top 5, 0.80 weight on value, buffer 1.25, cost 20 bps" in out            # blank answers keep the last good values
    with mock.patch("builtins.input", side_effect=["10", "0.5", "1.5", "9999", "q", "", "", ""]):
        out = quiet(stage5_run.run_mode, rows, env)[1]
    assert "cost_bps must be" in out
    with mock.patch("builtins.input", side_effect=EOFError):
        assert quiet(stage5_run.run_mode, rows, env)[0] == 0


def test_trade_lines_follow_one_rebalance():
    con, rows, env = world(defects=False)
    port = pf.build_portfolio(rows, 20, 0.5, 1.5)
    per = env.run(port, 10)
    first = "\n".join(stage5_run.trade_lines(per, per[0].signal_date, 10))
    assert "first purchase" in first and "No return yet" in first and "BUY" in first and "SELL" not in first
    p = next(x for x in per if x.signal_date == "2019-06-28")
    text = "\n".join(stage5_run.trade_lines(per, "2019-06-28", 10))
    assert f"traded {p.traded_fraction:.3f} of the portfolio" in text and f"{p.cost * 100:.4f}%" in text
    assert f"{p.net * 100:+.2f}%" in text and "BUY" in text and "SELL" in text and "more small trades" in text
    assert "not a signal date" in stage5_run.trade_lines(per, "2019-06-27", 10)[0]
    # the biggest trades come first
    trades = [abs(p.after.get(t, 0) - p.before.get(t, 0)) for t in sorted(set(p.after) | set(p.before))]
    shown = [float(line.split()[3].rstrip("%")) for line in text.splitlines() if line.strip().endswith(("BUY", "SELL"))]
    assert shown == sorted(shown, key=abs, reverse=True) and abs(shown[0]) / 100 == max(trades) or abs(abs(shown[0]) / 100 - max(trades)) < 1e-4


def test_trade_mode_direct_and_interactive():
    con, rows, env = world(n_stocks=30, seed=5, defects=False)
    code, out = quiet(stage5_run.trade_mode, rows, env, "2019-06-30", 10, 0.5, 1.5, 10)
    assert code == 0 and "Signal at the close of 2019-06-28" in out
    assert quiet(stage5_run.trade_mode, rows, env, "2010-01-01", 10, 0.5, 1.5, 10)[0] == 1
    with mock.patch("builtins.input", side_effect=["abc", "2010-01-01", "2019-06-28", ""]):
        code, out = quiet(stage5_run.trade_mode, rows, env, None, 10, 0.5, 1.5, 10)
    assert code == 0 and "Please type a date" in out and "before the first" in out and out.count("Signal at the close of") == 1
    with mock.patch("builtins.input", side_effect=EOFError):
        assert quiet(stage5_run.trade_mode, rows, env, None, 10, 0.5, 1.5, 10)[0] == 0
    assert quiet(stage5_run.trade_mode, [], env)[0] == 1


def test_main_argument_handling():
    bad = (["--run", "20", "0.5", "1.5"], ["--run", "0", "0.5", "1.5", "10"], ["--run", "20", "2", "1.5", "10"],
           ["--run", "20", "0.5", "0.5", "10"], ["--run", "20", "0.5", "1.5", "-1"], ["--run", "x", "0.5", "1.5", "10"],
           ["--trade", "2019-06-28", "2019-07-31"], ["--trade", "notadate"], ["--trade", "2010-01-01", "--clean"],
           ["--try", "99"])
    for argv in bad:
        assert quiet(stage5_run.main, argv)[0] == 1, argv
    code, out = quiet(stage5_run.main, ["--run", "20", "0.5", "1.5", "10", "--clean"])
    assert code == 0 and "agrees on every date" in out
    code, out = quiet(stage5_run.main, ["--trade", "2019-06-30", "--clean"])
    assert code == 0 and "Signal at the close of 2019-06-28" in out
    code, out = quiet(stage5_run.main, ["--try", "6"])
    assert code == 0 and "FROZEN" in out


def test_demo_succeeds_and_saves_a_database_that_can_be_reopened():
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "r.db")
        code, out = quiet(stage5_run.demo, clean=True, db_path=path)
        assert code == 0, out[-800:]
        assert "Everything behaved as designed" in out and "no false profit" in out and "FALSE PROFIT" not in out
        assert "INCONCLUSIVE" not in out and "LOOK-AHEAD" not in out and "agrees on every date" in out and "100 stocks" in out
        con = sqlite3.connect(path)
        assert con.execute("SELECT COUNT(*) FROM RUN").fetchone()[0] == 2
        n = con.execute("SELECT COUNT(*) FROM RESULT").fetchone()[0]
        assert n == 286, n
        con.close()
        code, out = quiet(stage5_run.main, ["--run", "20", "0.5", "1.0", "10", "--db", path])
        assert code == 0 and "agrees on every date" in out


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
