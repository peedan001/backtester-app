"""Tests for Stage 6 (bias): the synthetic world's new options, the P/E modes, delisting, bias.py, stage6_run.py.

Run either way:
    python tests/test_stage6.py
    python -m pytest -q
"""
import contextlib
import hashlib
import io
import math
import os
import sqlite3
import sys
import tempfile
from unittest import mock

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import align
import bias
import collect
import config
import db
import engine
import portfolio as pf
import stage3_run
import stage5_run
import stage6_run
import synthetic
from stage5_run import S1, S2, S3, T1, T2, T3, flat, mini_world, pos

_CACHE = {}
_OPEN = []


def world(**kw):
    """A cached synthetic world: (market, con, env)."""
    defaults = dict(edge=0.05, seed=5, defects=False, n_stocks=30, reaction=2.0, n_dead=6)
    defaults.update(kw)
    key = tuple(sorted(defaults.items()))
    if key not in _CACHE:
        market, con = stage3_run.build_world(**defaults)
        _CACHE[key] = (market, con, engine.Env(con))
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
except ImportError:
    pass


def quiet(fn, *a, **k):
    with contextlib.redirect_stdout(io.StringIO()) as buf:
        out = fn(*a, **k)
    return out, buf.getvalue()


def close(a, b, tol=1e-12):
    return abs(a - b) < tol


def fingerprint(m):
    h = hashlib.sha256()
    h.update(repr(sorted((k, v) for k, v in m["prices"].items())).encode())
    h.update(repr(sorted(m["eps"].items())).encode())
    h.update(repr(m["stocks"]).encode())
    h.update(repr(m["defects"]).encode())
    h.update(m["truth"]["price_matrix"].tobytes())
    h.update(np.nan_to_num(m["truth"]["score"]).tobytes())
    return h.hexdigest()[:16]


def hindsight_con():
    con = stage6_run._hindsight_con()
    _OPEN.append(con)
    return con


# ===========================================================================
# the synthetic world: options are off by default
# ===========================================================================

def test_default_markets_are_bit_for_bit_what_earlier_stages_produced():
    """The fingerprints were taken before Stage 6 touched the generator. If these change, Stages 1 to 5 would change."""
    assert fingerprint(synthetic.make_market()) == "8ac2ecf2e40ab0f6"
    assert fingerprint(synthetic.make_market(edge=0.0, defects=False, n_stocks=30, seed=5)) == "42c2fa7c0d907adb"
    assert fingerprint(synthetic.make_market(edge=0.15, value_share=0.3)) == "30ed02f78b6d2d22"
    assert fingerprint(synthetic.make_market(reaction=0.0, n_dead=0)) == "8ac2ecf2e40ab0f6"


def test_new_option_validation():
    for kw in ({"reaction": -0.1}, {"reaction": config.SYNTH_MAX_REACTION + 0.1}, {"n_dead": -1},
               {"n_dead": config.SYNTH_MAX_DEAD + 1}, {"n_dead": 2.5}):
        try:
            synthetic.make_market(n_stocks=10, **kw)
        except ValueError:
            continue
        raise AssertionError(kw)
    synthetic.make_market(n_stocks=10, reaction=config.SYNTH_MAX_REACTION, n_dead=config.SYNTH_MAX_DEAD, defects=False)


def test_graveyard_stocks_stop_trading_and_are_not_in_todays_list():
    m = synthetic.make_market(n_stocks=10, n_dead=5, seed=3, defects=False)
    tr = m["truth"]
    assert tr["dead_tickers"] == [f"D{i}" for i in ("001", "002", "003", "004", "005")] and len(tr["live_tickers"]) == 10
    assert tr["tickers"] == tr["live_tickers"] + tr["dead_tickers"] and tr["n_dead"] == 5
    flags = {t: in_sp for t, _, in_sp, _ in m["stocks"]}
    assert all(flags[t] == 1 for t in tr["live_tickers"]) and all(flags[t] == 0 for t in tr["dead_tickers"])
    assert m["stocks"][-1][3] == 1                                                        # the benchmark is still last
    days = tr["days"]
    for t in tr["dead_tickers"]:
        series = m["prices"][t]
        last = tr["delist_date"][t]
        assert series[-1][0] == last and len(series) == days.index(last) + 1 and series[0][0] == days[0]
        assert all(d <= last for d, _ in series)
        assert all(filed <= last for _, filed, _ in m["eps"][t]) and len(m["eps"][t]) > 20
    assert all(len(m["prices"][t]) == len(days) for t in tr["live_tickers"])


def test_failures_fall_hard_and_are_a_mix_of_abrupt_and_gradual():
    m = synthetic.make_market(n_stocks=10, n_dead=30, seed=4, defects=False)
    tr = m["truth"]
    falls = []
    for t in tr["dead_tickers"]:
        last = np.datetime64(tr["delist_date"][t])
        window = [p for d, p in m["prices"][t] if last - np.datetime64(d) <= np.timedelta64(365, "D")]
        falls.append(window[-1] / max(window))
    assert np.median(falls) < 0.35 and max(falls) < 0.8
    kinds = [v < 100 for v in tr["crash_days"].values()]
    assert 5 < sum(kinds) < 25                                                             # both kinds occur
    assert set(tr["crash_days"].values()) - {config.SYNTH_CRASH_DAYS} <= set(range(config.SYNTH_ABRUPT_DAYS[0], config.SYNTH_ABRUPT_DAYS[1] + 1))
    assert all(config.SYNTH_FIRST_DELIST_MONTH <= 0 or d > tr["days"][600] for d in tr["delist_date"].values())
    assert all(0.85 <= v <= 2.35 for v in tr["crash"].values())


def test_signals_stop_for_failed_stocks_and_defects_go_into_live_stocks_only():
    m = synthetic.make_market(n_stocks=10, n_dead=5, seed=3, defects=False)
    tr = m["truth"]
    dead_cols = [tr["tickers"].index(t) for t in tr["dead_tickers"]]
    for t, j in zip(tr["dead_tickers"], dead_cols):
        after = [k for k, d in enumerate(tr["month_ends"]) if d > tr["delist_date"][t]]
        assert after and np.isnan(tr["pe"][after, j]).all() and np.isnan(tr["momentum"][after, j]).all()
    for seed in range(1, 6):
        defects = synthetic.make_market(n_stocks=20, n_dead=40, seed=seed)["defects"]
        assert len(defects) == 8 and all(d["ticker"].startswith("S") for d in defects), seed


def test_universe_fetcher_hands_over_only_todays_list_but_prices_exist_for_the_dead():
    m = synthetic.make_market(n_stocks=10, n_dead=3, seed=2, defects=False)
    f = synthetic.fake_fetchers(m)
    assert [t for t, _ in f["universe"]()] == m["truth"]["live_tickers"]
    assert f["prices"]("D001", "2013-01-01", "2030-01-01")[-1][0] == m["truth"]["delist_date"]["D001"]
    assert len(f["eps"]("D002")) > 0


def test_planted_quintile_spread_works_with_failed_stocks():
    m = synthetic.make_market(n_stocks=40, n_dead=10, seed=2, edge=0.15, defects=False)
    spread = synthetic.planted_quintile_spread(m)
    assert math.isfinite(spread)


# ---- earnings reaction --------------------------------------------------------

def jump_inputs(eps_last, nq=5, first_eps=1.0, avail=("2019-05-10", "2019-05-12", "2019-05-14")):
    days = np.arange(np.datetime64("2019-01-01"), np.datetime64("2019-12-31"), dtype="datetime64[D]")
    n = len(eps_last)
    eps = np.full((nq, n), first_eps)
    eps[nq - 1] = eps_last
    avail_q = np.full((nq, n), np.datetime64("2018-01-01"))
    avail_q[nq - 1] = np.array(avail[:n], dtype="datetime64[D]")
    return eps, avail_q, days


def test_earnings_jump_is_the_reaction_times_the_change_in_trailing_eps_centred_on_the_filing_day():
    eps, avail, days = jump_inputs([1.0, 2.0, 0.0])
    # trailing EPS 4.0 before; 4.0, 5.0, 3.0 after: changes 0, +25%, -25%; reaction 0.5 -> 0, +0.125, -0.125, then centred
    jump = synthetic._earnings_jumps(eps, avail, days, 0.5)
    raw = np.array([0.0, 0.125, -0.125])
    centre = math.log(np.mean(np.exp(raw)))
    want = raw - centre
    day_of = [int(np.searchsorted(days, a)) for a in avail[4]]
    for i in range(3):
        assert close(jump[day_of[i], i], want[i]), i
    mask = np.zeros_like(jump, dtype=bool)
    mask[day_of, range(3)] = True
    assert not jump[~mask].any()                                                        # nothing happens on any other day
    assert close(sum(np.exp(jump[day_of[i], i]) - 1 for i in range(3)), 0.0, 1e-12)      # average simple jump is zero
    assert jump[day_of[1], 1] > 0 > jump[day_of[2], 2]                                   # good news up, bad news down


def test_earnings_jump_is_clipped_and_uses_a_denominator_floor():
    eps, avail, days = jump_inputs([1.0, 10.0, -10.0])
    jump = synthetic._earnings_jumps(eps, avail, days, 1.0)
    day_of = [int(np.searchsorted(days, a)) for a in avail[4]]
    c = math.log(np.mean(np.exp([0.0, 0.25, -0.25])))
    assert close(jump[day_of[1], 1], 0.25 - c) and close(jump[day_of[2], 2], -0.25 - c)       # huge changes are capped at 0.25
    eps, avail, days = jump_inputs([0.55, 0.05], first_eps=0.05, avail=("2019-05-10", "2019-05-12"))
    jump = synthetic._earnings_jumps(eps, avail, days, 0.1)
    day_of = [int(np.searchsorted(days, a)) for a in avail[4]]
    # stock 0: trailing EPS 0.2 -> 0.7 (+0.5); the floor makes the base 0.5, not 0.2, so change = 1.0 and the jump is 0.1
    assert close(jump[day_of[0], 0] - jump[day_of[1], 1], 0.1)


def test_filings_before_the_first_trading_day_and_in_the_far_future_cause_no_jump():
    eps, avail, days = jump_inputs([2.0, 2.0, 2.0], avail=("2018-12-01", "2019-05-12", "2100-01-01"))
    jump = synthetic._earnings_jumps(eps, avail, days, 1.0)
    assert not jump[:, 0].any() and not jump[:, 2].any() and jump[:, 1].any()


def test_reaction_changes_prices_but_not_the_average_stock_or_the_validator():
    base = synthetic.make_market(n_stocks=60, defects=False, seed=2)
    strong = synthetic.make_market(n_stocks=60, defects=False, seed=2, reaction=config.SYNTH_MAX_REACTION)
    P0, P1 = base["truth"]["price_matrix"], strong["truth"]["price_matrix"]
    assert not np.allclose(P0, P1)
    daily = np.exp(np.log(P1[1:] / P1[:-1])) - 1
    assert daily.max() < config.JUMP_THRESHOLD and daily.min() > -config.JUMP_THRESHOLD     # a real filing is never mistaken for a bad tick
    assert base["truth"]["reaction"] == 0 and strong["truth"]["reaction"] == config.SYNTH_MAX_REACTION
    market, con = stage3_run.build_world(n_stocks=20, defects=False, reaction=config.SYNTH_MAX_REACTION, seed=2)
    _OPEN.append(con)
    assert con.execute("SELECT COUNT(*) FROM PRICE WHERE flagged = 1").fetchone()[0] == 0


def test_build_world_registers_failed_stocks_outside_todays_list():
    market, con, env = world()
    flags = dict(con.execute("SELECT ticker, in_sp500 FROM STOCK WHERE is_benchmark = 0"))
    assert sum(flags.values()) == 30 and len(flags) == 36
    assert align.default_tickers(con) == sorted(market["truth"]["live_tickers"])
    assert align.all_tickers(con) == sorted(market["truth"]["tickers"])
    assert align.tickers_for(con, "survivors") == align.default_tickers(con) and align.tickers_for(con, "all") == align.all_tickers(con)
    try:
        align.tickers_for(con, "nonsense")
    except ValueError:
        pass
    else:
        raise AssertionError("an unknown universe must be refused")
    assert bias.dead_tickers(con) == set(market["truth"]["dead_tickers"])
    assert con.execute("SELECT MAX(date) FROM PRICE WHERE ticker = 'D001'").fetchone()[0] == market["truth"]["delist_date"]["D001"]


# ===========================================================================
# the three P/E modes
# ===========================================================================

def test_eps_modes_by_hand():
    con = stage6_run._eps_con({"X": stage6_run._quarters((2019, 1), (2020, 2), lambda y, q, i: [1.0, 1.0, 1.0, 1.0, 2.0, 3.0][i])}, {})
    _OPEN.append(con)
    book = align.EpsBook(con)
    date = "2020-04-30"
    lagged, why = book.ttm("X", date)
    assert why is None and close(lagged.value, 4.0) and lagged.latest_filed == "2020-02-09"
    no_lag, why = book.ttm("X", date, "no_lag")
    assert why is None and close(no_lag.value, 5.0) and no_lag.latest_filed == "2020-05-10" > date
    today, why = book.ttm("X", date, "today")
    assert why is None and close(today.value, 7.0) and today.latest_filed == "2020-08-09"
    assert book.ttm("X", date, "lagged") == book.ttm("X", date)
    assert book.ttm("X", "2018-01-01", "today")[0].value == 7.0 and book.ttm("X", "2030-01-01", "today")[0].value == 7.0   # 'today' ignores the date
    try:
        book.ttm("X", date, "tomorrow")
    except ValueError:
        pass
    else:
        raise AssertionError("an unknown mode must be refused")


def test_no_lag_uses_a_quarter_from_the_day_it_ends_and_not_a_day_earlier():
    con = stage6_run._eps_con({"X": stage6_run._quarters((2019, 1), (2020, 1), lambda y, q, i: [1.0, 1.0, 1.0, 1.0, 2.0][i])}, {})
    _OPEN.append(con)
    book = align.EpsBook(con)
    assert close(book.ttm("X", "2020-03-31", "no_lag")[0].value, 5.0)                    # the quarter ended that day
    assert close(book.ttm("X", "2020-03-30", "no_lag")[0].value, 4.0)                    # one day earlier it had not
    assert close(book.ttm("X", "2020-05-09", "lagged")[0].value, 4.0) and close(book.ttm("X", "2020-05-10", "lagged")[0].value, 5.0)


def test_today_mode_has_no_staleness_rule_but_the_other_modes_do():
    q = stage6_run._quarters((2016, 1), (2016, 4), lambda y, qq, i: 1.0)
    con = stage6_run._eps_con({"X": q}, {})
    _OPEN.append(con)
    book = align.EpsBook(con)
    ttm, why = book.ttm("X", "2018-06-30")
    assert ttm is None and why == align.REASON_STALE
    assert book.ttm("X", "2018-06-30", "no_lag")[1] == align.REASON_STALE
    assert book.ttm("X", "2018-06-30", "today")[1] is None


def test_panel_modes_and_the_audit():
    con = hindsight_con()
    out = {}
    for mode in config.EPS_MODES:
        rows = {r.ticker: r for r in signals_of(con, mode)}
        out[mode] = rows
        bad = align.audit_lookahead(list(rows.values()))
        assert (len(bad) == 0) == (mode == "lagged"), mode
        assert all(w[2] == "EPS filing" for w in bad)
    assert close(out["lagged"]["X"].pe, 10.0) and close(out["lagged"]["Y"].pe, 10.0)
    assert out["lagged"]["X"].pe_rank == out["lagged"]["Y"].pe_rank == 0.5
    assert close(out["today"]["X"].pe, 10.0) and close(out["today"]["Y"].pe, 40 / 7) and out["today"]["Y"].pe_rank > out["today"]["X"].pe_rank
    try:
        align.build_panel(con, pe_mode="nope")
    except ValueError:
        pass
    else:
        raise AssertionError("an unknown mode must be refused")
    assert align.build_panel(con, tickers=["X"], dates=["2019-12-31"]) [0].pe == out["lagged"]["X"].pe      # lagged is the default


def signals_of(con, mode):
    from signals import add_ranks
    return add_ranks(align.build_panel(con, tickers=["X", "Y"], dates=["2019-12-31"], pe_mode=mode))


# ===========================================================================
# the price book: where does a series end?
# ===========================================================================

def book_con(rows, flagged=()):
    con = db.connect()
    _OPEN.append(con)
    db.create_schema(con)
    collect.register_stocks(con, [("A", "A", 1, 0), (config.BENCHMARK_TICKER, "b", 0, 1)])
    for d, p in rows:
        con.execute("INSERT INTO PRICE (ticker, date, adj_close, flagged, flag_reason) VALUES ('A', ?, ?, ?, ?)",
                    (d, p, int(d in flagged), "jump: test" if d in flagged else None))
    con.commit()
    return align.PriceBook(con)


def test_price_book_last_date_last_price_and_delisting_boundary():
    book = book_con([("2020-01-10", 10.0), ("2020-01-20", 20.0)])
    assert book.last_date("A") == "2020-01-20" and book.last_price("A") == 20.0
    assert book.last_date("ZZZ") is None and book.last_price("ZZZ") is None and not book.delisted("ZZZ", "2020-06-01")
    assert not book.delisted("A", "2020-01-25")                     # exactly 5 days: the carry limit still reaches it
    assert book.delisted("A", "2020-01-26")                         # 6 days: the data has ended
    assert not book.delisted("A", "2020-01-05")                     # a date BEFORE the last price: not a failure
    assert not book.delisted("A", "2020-01-30", max_carry=10) and book.delisted("A", "2020-01-31", max_carry=10)


def test_the_last_price_ignores_a_final_bad_tick():
    book = book_con([("2020-01-10", 10.0), ("2020-01-20", 20.0), ("2020-01-21", 200.0)], flagged={"2020-01-21"})
    assert book.last_price("A") == 20.0 and book.last_date("A") == "2020-01-21"
    assert book_con([("2020-01-10", 5.0)], flagged={"2020-01-10"}).last_price("A") is None


# ===========================================================================
# the engine: a stock whose data ends
# ===========================================================================

def delist_prices():
    return {"SPY": flat(100.0), "B": flat(100.0), "A": {S1: 100.0, T1: 100.0, S2: 90.0, "2020-03-04": 30.0}}


def test_a_stock_whose_data_ends_is_marked_at_its_last_price_and_sold():
    env = mini_env(delist_prices())
    port = {S1: pos("A", "B"), S2: pos("A", "B"), S3: pos("B")}
    per = env.run(port, 10)
    assert per[1].delisted == () and per[1].unpriced == 0 and close(per[1].gross, -0.05)    # still quoted (carried) on March 2
    p = per[2]
    assert p.delisted == ("A",) and p.unpriced == 0 and set(p.after) == {"B"} and close(p.after["B"], 1.0)
    assert close(p.gross, 0.5 * (30 / 90 - 1))                      # marked from the last price we knew (90) to the final one (30)
    assert close(p.cost, 0.001 * p.traded_fraction) and p.traded_fraction > 0.4       # selling A (and rebalancing) is traded value
    assert engine.check_engine(per, env.replay(port, 10)) == []


def test_delisting_with_nothing_left_to_buy_goes_to_cash():
    env = mini_env({"SPY": flat(100.0), "A": {S1: 100.0, T1: 100.0, S2: 100.0, "2020-03-03": 50.0}})
    port = {S1: pos("A"), S2: pos("A"), S3: []}
    per = env.run(port, 0)
    p = per[2]
    assert p.delisted == ("A",) and p.after == {} and close(p.cash, 1.0) and close(p.gross, -0.5) and close(engine.net_wealth_path(per)[-1], 0.5)
    assert engine.check_engine(per, env.replay(port, 0)) == []
    assert engine.metrics(per)["delisted_sales"] == 1


def test_a_gap_that_resumes_is_frozen_not_delisted():
    prices = {"SPY": flat(100.0), "A": flat(100.0), "B": {S1: 100.0, T1: 100.0, S2: 100.0, T2: 100.0, "2020-04-20": 100.0}}
    env = mini_env(prices)
    port = {S1: pos("A", "B"), S2: pos("A", "B"), S3: pos("A", "B")}
    p = env.run(port, 10)[2]
    assert p.unpriced == 1 and p.delisted == () and set(p.after) == {"A", "B"} and p.traded_fraction == 0
    assert engine.check_engine(env.run(port, 10), env.replay(port, 10)) == []


def test_carry_limit_decides_when_a_series_counts_as_ended():
    """A last price 5 days before the trade day is still carried; 6 days and it is a delisting."""
    for last, expect_delisted in (("2020-03-27", False), ("2020-03-26", True)):          # trade day 2020-04-01
        prices = {"SPY": flat(100.0), "A": {S1: 100.0, T1: 100.0, S2: 100.0, T2: 100.0, last: 80.0}}
        per = mini_env(prices).run({S1: pos("A"), S3: pos("A")}, 0)
        assert (per[-1].delisted == ("A",)) == expect_delisted, last
        assert close(per[-1].gross, -0.2)


def test_delisting_marks_to_the_last_unflagged_price():
    con, env = mini_world({"SPY": flat(100.0), "A": {S1: 100.0, T1: 100.0, S2: 100.0, "2020-03-02": 70.0}})
    _OPEN.append(con)
    con.execute("INSERT INTO PRICE (ticker, date, adj_close, flagged, flag_reason) VALUES ('A', '2020-03-03', 700.0, 1, 'jump: spike')")
    con.commit()
    env = engine.Env(con)
    p = env.run({S1: pos("A"), S3: pos("A")}, 0)[-1]
    assert p.delisted == ("A",) and close(p.gross, -0.3)


def mini_env(prices):
    con, env = mini_world(prices)
    _OPEN.append(con)
    return env


def test_the_share_ledger_agrees_on_delisting_cases_at_every_cost():
    cases = [
        (delist_prices(), {S1: pos("A", "B"), S2: pos("A", "B"), S3: pos("B")}),
        (delist_prices(), {S1: pos("A", "B"), S2: pos("A", "B"), S3: []}),
        ({"SPY": flat(100.0), "A": {S1: 100.0, T1: 100.0, S2: 100.0, "2020-03-03": 50.0}, "C": flat(100.0)}, {S1: pos("A", "C"), S2: pos("A", "C"), S3: pos("C")}),
    ]
    for prices, port in cases:
        env = mini_env(prices)
        for cost in (0, 25, 300):
            assert engine.check_engine(env.run(port, cost), env.replay(port, cost)) == [], (cost, port)


# ===========================================================================
# the database
# ===========================================================================

def test_run_table_knows_the_p_e_version_and_the_universe():
    con = db.connect()
    _OPEN.append(con)
    db.create_schema(con)
    assert [c[0] for c in db.columns(con, "RUN")] == ["run_id", "top_n", "value_weight", "cost_bps", "buffer", "pe_mode", "universe"]
    ids = {pf.get_or_create_run(con, 20, 0.5, 10, 1.0, m, u) for m in config.EPS_MODES for u in config.UNIVERSES}
    assert len(ids) == 6 and pf.get_or_create_run(con, 20, 0.5, 10, 1.0, "no_lag", "all") in ids
    assert pf.get_or_create_run(con, 20, 0.5, 10, 1.0) == pf.get_or_create_run(con, 20, 0.5, 10, 1.0, "lagged", "survivors")
    for bad in (("nonsense", "survivors"), ("lagged", "everyone"), (None, "all")):
        try:
            con.execute("INSERT INTO RUN (top_n, value_weight, cost_bps, pe_mode, universe) VALUES (5, 0.5, 10, ?, ?)", bad)
        except sqlite3.IntegrityError:
            continue
        raise AssertionError(bad)


def old_run_database(with_buffer=True):
    """A database as an earlier stage would have left it."""
    old = sqlite3.connect(":memory:")
    old.execute("PRAGMA foreign_keys = ON")
    buffer_sql = db.BUFFER_COLUMN + "," if with_buffer else ""
    old.executescript(db.SCHEMA.replace(db.RUN_COLUMNS, f"""
        run_id INTEGER PRIMARY KEY AUTOINCREMENT,
        top_n INTEGER NOT NULL CHECK (top_n >= 1 AND top_n <= 500),
        value_weight REAL NOT NULL CHECK (value_weight >= 0 AND value_weight <= 1),
        cost_bps REAL NOT NULL CHECK (cost_bps >= 0 AND cost_bps <= 500),
        {buffer_sql}
        pe_mode TEXT NOT NULL DEFAULT 'lagged' CHECK (pe_mode IN ('lagged', 'today'))"""))
    old.execute("INSERT INTO STOCK VALUES ('A', 'A', 1, 0)")
    cols = "top_n, value_weight, cost_bps, buffer, pe_mode" if with_buffer else "top_n, value_weight, cost_bps, pe_mode"
    marks = "20, 0.5, 10, 1.5, 'today'" if with_buffer else "20, 0.5, 10, 'today'"
    old.execute(f"INSERT INTO RUN ({cols}) VALUES ({marks})")
    old.execute("INSERT INTO RUN (top_n, value_weight, cost_bps) VALUES (10, 0.3, 5)")
    old.execute("INSERT INTO HOLDING VALUES (2, '2020-01-31', 'A', 1.0)")
    old.execute("INSERT INTO RESULT VALUES (1, '2020-02-03', 0.01, 0.001, 0.009, 0.005)")
    old.commit()
    return old


def test_a_stage_5_database_is_rebuilt_without_losing_runs_or_their_rows():
    old = old_run_database()
    _OPEN.append(old)
    assert "universe" not in [c[0] for c in db.columns(old, "RUN")]
    db.create_schema(old)
    db.create_schema(old)                                           # a second time: nothing more to do
    assert db.foreign_keys_on(old)
    assert old.execute("SELECT run_id, top_n, buffer, pe_mode, universe FROM RUN ORDER BY run_id").fetchall() == [
        (1, 20, 1.5, "today", "survivors"), (2, 10, 1.0, "lagged", "survivors")]
    assert old.execute("SELECT run_id, ticker FROM HOLDING").fetchall() == [(2, "A")]
    assert old.execute("SELECT run_id, net_return FROM RESULT").fetchall() == [(1, 0.009)]
    assert old.execute("PRAGMA foreign_key_check").fetchall() == []
    assert "RUN_new" not in db.table_names(old)
    old.execute("INSERT INTO RUN (top_n, value_weight, cost_bps, pe_mode, universe) VALUES (5, 0.5, 10, 'no_lag', 'all')")
    assert old.execute("SELECT MAX(run_id) FROM RUN").fetchone()[0] == 3                 # numbering carries on
    try:
        old.execute("INSERT INTO HOLDING VALUES (99, '2020-01-31', 'A', 1.0)")
    except sqlite3.IntegrityError:
        pass
    else:
        raise AssertionError("foreign keys must still be enforced after the rebuild")


def test_a_stage_3_database_without_the_buffer_column_gets_every_migration():
    old = old_run_database(with_buffer=False)
    _OPEN.append(old)
    db.create_schema(old)
    assert [c[0] for c in db.columns(old, "RUN")] == ["run_id", "top_n", "value_weight", "cost_bps", "buffer", "pe_mode", "universe"]
    assert old.execute("SELECT run_id, buffer, pe_mode FROM RUN ORDER BY run_id").fetchall() == [(1, 1.0, "today"), (2, 1.0, "lagged")]


def test_the_rebuild_leaves_a_current_schema_alone():
    con = db.connect()
    _OPEN.append(con)
    db.create_schema(con)
    con.execute("INSERT INTO RUN (top_n, value_weight, cost_bps) VALUES (5, 0.5, 10)")
    assert db._rebuild_run_if_old(con) is False
    assert con.execute("SELECT COUNT(*) FROM RUN").fetchone()[0] == 1


# ===========================================================================
# bias.py
# ===========================================================================

def test_summarize_by_hand():
    s = bias.summarize([1.0, 2.0, 3.0])
    assert s["n"] == 3 and close(s["mean"], 2.0) and close(s["se"], 1 / math.sqrt(3)) and close(s["t"], 2.0 * math.sqrt(3))
    one = bias.summarize([0.5])
    assert one["mean"] == 0.5 and one["se"] != one["se"] and one["t"] != one["t"]
    flat_ = bias.summarize([0.1, 0.1, 0.1])
    assert close(flat_["se"], 0.0) and flat_["t"] != flat_["t"]
    try:
        bias.summarize([])
    except ValueError:
        pass
    else:
        raise AssertionError("an empty list has no summary")


def test_bias_verdict_needs_both_statistical_and_practical_size():
    assert bias.bias_verdict(0.05, 0.01) == "INFLATED"
    assert bias.bias_verdict(0.05, 0.03) == "within noise"            # not more than two standard errors
    assert bias.bias_verdict(0.004, 0.0001) == "within noise"         # significant but under half a percent a year
    assert bias.bias_verdict(0.006, 0.0001) == "INFLATED"
    assert bias.bias_verdict(-0.05, 0.01) == "DEFLATED" and bias.bias_verdict(-0.004, 0.0001) == "within noise"
    assert bias.bias_verdict(-0.05, 0.03) == "within noise" and bias.bias_verdict(-0.02, 0.01) == "within noise"
    assert bias.bias_verdict(0.05, float("nan")) == "NOT ENOUGH RUNS"
    assert bias.bias_verdict(0.02, 0.01) == "within noise" and bias.bias_verdict(0.0201, 0.01) == "INFLATED"     # exactly two se is not enough


def test_run_variant_reports_what_it_held():
    market, con, env = world()
    surv = bias.run_variant(con, env, "lagged", "survivors", 5, 0.5, 1.0, 10)
    assert surv["dead_held"] == 0 and surv["dead_weight"] == 0 and surv["delisted_sales"] == 0
    full = bias.run_variant(con, env, "lagged", "all", 5, 0.5, 1.0, 10, with_ic=True)
    assert 0 < full["dead_held"] <= 6 and 0 < full["dead_weight"] < 1 and "ic" in full and "ic_t" in full and "ic" not in surv
    direct = engine.metrics(env.run(pf.build_portfolio(stage3_run.all_signals(con), 5, 0.5, 1.0), 10))
    assert close(surv["net_cagr"], direct["net"]["cagr"]) and close(surv["gross_cagr"], direct["gross"]["cagr"])
    assert surv["net_cagr"] < surv["gross_cagr"]
    assert set(surv) >= {"portfolio", "periods", "metrics", "sharpe", "pe_mode", "universe", "rows"}
    assert surv["pe_mode"] == "lagged" and full["universe"] == "all"


def test_evaluate_world_identities_with_a_graveyard():
    market, con, env = world()
    r = bias.evaluate_world(con, env, top_n=5)
    assert close(r["inflation_no_lag"], r["no_lag"]["net_cagr"] - r["lagged"]["net_cagr"])
    assert close(r["inflation_today"], r["today"]["net_cagr"] - r["lagged"]["net_cagr"])
    assert close(r["survivorship"], r["lagged"]["net_cagr"] - r["all"]["net_cagr"])
    assert close(r["total"], r["today"]["net_cagr"] - r["all"]["net_cagr"])
    assert close(r["total"], r["inflation_today"] + r["survivorship"])
    assert r["all"]["universe"] == "all" and r["today"]["universe"] == "survivors" and r["all"] is not r["lagged"]


def test_without_a_graveyard_there_is_no_survivorship_bias_by_construction():
    market, con, env = world(n_dead=0)
    r = bias.evaluate_world(con, env, top_n=5)
    assert r["survivorship"] == 0 and r["all"] is r["lagged"] and close(r["total"], r["inflation_today"])


def test_shortcuts_inflate_only_when_earnings_move_prices():
    """Deterministic worlds: no link, no bias; a strong link, both shortcuts help and hindsight helps most."""
    null = {}
    for reaction in (0.0, 3.0):
        _, con, env = world(n_stocks=100, seed=1, reaction=reaction, n_dead=0)
        null[reaction] = bias.evaluate_world(con, env)
    assert abs(null[0.0]["inflation_no_lag"]) < 0.015 and abs(null[0.0]["inflation_today"]) < 0.015
    assert 0.01 < null[3.0]["inflation_no_lag"] < 0.05
    assert null[3.0]["inflation_today"] > 0.04 and null[3.0]["inflation_today"] > null[3.0]["inflation_no_lag"]


def test_worlds_over_seeds_and_survivorship_over_seeds_return_one_result_per_seed():
    res = bias.worlds_over_seeds([1, 2], n_stocks=20, n_dead=4, top_n=5)
    assert len(res) == 2 and all(set(r) >= {"inflation_no_lag", "inflation_today", "survivorship", "total"} for r in res)
    assert res[0]["lagged"]["net_cagr"] != res[1]["lagged"]["net_cagr"]                  # different seeds, different worlds
    surv = bias.survivorship_over_seeds([1, 2], n_dead=4, n_stocks=20, top_n=5)
    assert len(surv) == 2 and all(set(r) == {"survivorship", "dead_held", "delisted_sales", "dead_weight"} for r in surv)
    assert all(close(a["survivorship"], b["survivorship"]) for a, b in zip(surv, res))


# ===========================================================================
# the run script
# ===========================================================================

def test_ladder_lines_show_every_version_and_the_total():
    market, con, env = world()
    lines, res = stage6_run.evaluate_lines(con, env, 0.5, True, top_n=5)
    text = "\n".join(lines)
    for words in ("honest: EPS from filing date", "no filing lag", "hindsight", "ALL stocks", "naive backtest", "failed stocks at some point", "IC"):
        assert words in text, words
    assert f"{res['inflation_no_lag'] * 100:+.2f}%" in text and f"{res['inflation_today'] * 100:+.2f}%" in text
    assert f"{-res['survivorship'] * 100:+.2f}%" in text and f"{res['total'] * 100:+.2f}%" in text
    market0, con0, env0 = world(n_dead=0)
    text0 = "\n".join(stage6_run.evaluate_lines(con0, env0, 0.5, False, top_n=5, with_ic=False)[0])
    assert "ALL stocks" not in text0 and "naive backtest" not in text0 and "(t " not in text0


def test_seeds_line_reports_mean_spread_and_verdict():
    line, verdict, s = stage6_run.seeds_line("x", [0.05, 0.06, 0.07, 0.05])
    assert verdict == "INFLATED" and "+5.75%" in line and "INFLATED" in line and close(s["mean"], 0.0575)
    assert stage6_run.seeds_line("x", [-0.02, 0.02, 0.01, -0.01])[1] == "within noise"


def test_try_cases_give_the_documented_answers():
    assert len(stage6_run.try_cases()) == 6
    out = {n: stage6_run.run_try_case(n) for n in range(1, 7)}
    assert all(lesson for _, _, lesson in out.values())
    t1 = out[1][1]
    assert "10.0" in t1 and "8.0" in t1 and "5.7" in t1 and t1.count("LOOK-AHEAD") == 2
    t2 = out[2][1]
    assert "a tie" in t2 and "Y, the one that is about to rise" in t2 and "5.7" in t2
    t3 = out[3][1]
    assert "violations: 0" in t3 and t3.count("violations: 2") == 2
    t4 = out[4][1]
    assert "-33.33%" in t4 and "A delisted" in t4 and "-67%" in t4 and "B:100%" in t4
    t5 = out[5][1]
    assert "FROZEN" in t5 and "unpriced 1, delisted 0" in t5
    t6 = out[6][1]
    assert "10 stocks in today's list and 3 that failed" in t6 and "13 stocks" in t6 and "D001" in t6


def test_try_mode_menu_and_direct_pick():
    assert quiet(stage6_run.try_mode, 3)[0] == 0
    assert quiet(stage6_run.try_mode, 99)[0] == 1
    with mock.patch("builtins.input", side_effect=["zzz", "0", "7", "2", ""]):
        code, out = quiet(stage6_run.try_mode)
    assert code == 0 and out.count("Not on the list") == 3 and "[2] Hindsight picks" in out
    with mock.patch("builtins.input", side_effect=EOFError):
        assert quiet(stage6_run.try_mode)[0] == 0


def test_graveyard_info_and_table():
    market, con, env = world()
    info, rows, runs = stage6_run.graveyard_info(con, top_n=5)
    assert [e["ticker"] for e in info] == sorted(market["truth"]["dead_tickers"])
    for e in info:
        assert e["last_date"] == market["truth"]["delist_date"][e["ticker"]] and -1 < e["fall"] <= 0 and e["days_from_peak"] >= 0
        assert e["months_held_1.0"] >= 0 and e["sold_at_failure_1.0"] <= 1 and e["sold_at_failure_0.5"] <= 1
    assert sum(e["sold_at_failure_1.0"] for e in info) == sum(len(p.delisted) for p in runs[1.0][1])
    text = "\n".join(stage6_run.graveyard_lines(info))
    assert "6 failed stocks" in text and "D001" in text and "blend / value" in text


def test_graveyard_story_follows_a_failed_stock_to_its_end():
    market, con, env = world()
    info, rows, runs = stage6_run.graveyard_info(con, top_n=5)
    t = next(e["ticker"] for e in info if e["months_held_1.0"] > 0)
    lines = stage6_run.story_lines(con, rows, runs, t.lower())
    text = "\n".join(lines)
    assert lines[0].startswith(f"{t}: the last") and "CHEAPER" in text and "held by blend / value" in text
    body = [l for l in lines[2:-1]]
    assert 1 <= len(body) <= 9 and all(l.strip().startswith("20") for l in body)
    assert any(l.rstrip().endswith("yes") for l in body) or any("yes /" in l for l in body)
    assert "No price history" in stage6_run.story_lines(con, rows, runs, "ZZZZ")[0]


def test_graveyard_mode_direct_and_interactive():
    market, con, env = world()
    code, out = quiet(stage6_run.graveyard_mode, con, "d001")
    assert code == 0 and "D001: the last" in out
    code, out = quiet(stage6_run.graveyard_mode, con, "S001")
    assert code == 1 and "not a failed stock" in out and "D001" in out
    with mock.patch("builtins.input", side_effect=["S001", "d002", ""]):
        code, out = quiet(stage6_run.graveyard_mode, con)
    assert code == 0 and "6 failed stocks" in out and "Not a failed stock" in out and out.count("the last") == 1
    with mock.patch("builtins.input", side_effect=EOFError):
        assert quiet(stage6_run.graveyard_mode, con)[0] == 0
    market0, con0, env0 = world(n_dead=0)
    code, out = quiet(stage6_run.graveyard_mode, con0)
    assert code == 1 and "no failed stocks" in out


def small_factory(reaction):
    return stage3_run.build_world(n_stocks=20, seed=5, defects=False, reaction=reaction, n_dead=4)[1]


def test_ladder_mode_one_shot_and_interactive():
    code, out = quiet(stage6_run.ladder_mode, small_factory, 2.0, 0.5, 4)
    assert code == 0 and "honest: EPS from filing date" in out and "ALL stocks" in out
    inputs = ["abc", "", "9", "", "2", "0.5", "q", "", ""]
    with mock.patch("builtins.input", side_effect=inputs):
        code, out = quiet(stage6_run.ladder_mode, small_factory)
    assert code == 0 and "numbers only" in out and "must be between" in out and out.count("honest: EPS from filing date") == 1
    with mock.patch("builtins.input", side_effect=EOFError):
        assert quiet(stage6_run.ladder_mode, small_factory)[0] == 0


def test_main_argument_handling():
    bad = (["--reaction", "9"], ["--reaction", "-1"], ["--dead", "99"], ["--dead", "-1"], ["--seeds", "1"],
           ["--ladder", "2.0"], ["--ladder", "x", "0.5"], ["--ladder", "5", "0.5"], ["--ladder", "2", "1.5"],
           ["--graveyard", "D001", "D002"], ["--graveyard", "D999", "--clean"], ["--graveyard", "S001", "--clean"], ["--try", "99"])
    for argv in bad:
        assert quiet(stage6_run.main, argv)[0] == 1, argv
    code, out = quiet(stage6_run.main, ["--graveyard", "D001", "--clean"])
    assert code == 0 and "D001: the last" in out
    code, out = quiet(stage6_run.main, ["--try", "3"])
    assert code == 0 and "violations: 0" in out


def test_a_database_without_failed_stocks_says_so():
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "old.db")
        market, con = stage3_run.build_world(n_stocks=15, seed=2, defects=False)
        out = db.connect(path)
        con.backup(out)
        out.close()
        con.close()
        code, text = quiet(stage6_run.main, ["--graveyard", "--db", path])
        assert code == 1 and "no failed stocks" in text


def test_demo_succeeds_saves_four_runs_and_the_database_can_be_reopened():
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "bias.db")
        code, out = quiet(stage6_run.demo, clean=True, db_path=path, n_seeds=2, n_dead=6)
        assert code == 0, out[-1200:]
        assert "Everything behaved as designed" in out and "verdicts are shown but not checked" in out
        assert "the honest version is clean and both shortcuts are caught" in out and "agrees on every date" in out
        assert "4 distinct runs in the database (as expected)" in out and "Check question" in out
        con = sqlite3.connect(path)
        assert con.execute("SELECT COUNT(*) FROM RUN").fetchone()[0] == 4
        assert con.execute("SELECT COUNT(*) FROM RESULT").fetchone()[0] == 4 * 143
        assert sorted(con.execute("SELECT pe_mode, universe FROM RUN").fetchall()) == [("lagged", "all"), ("lagged", "survivors"), ("no_lag", "survivors"), ("today", "survivors")]
        assert con.execute("SELECT COUNT(*) FROM HOLDING").fetchone()[0] > 1000
        con.close()
        code, out = quiet(stage6_run.main, ["--graveyard", "D001", "--db", path])
        assert code == 0 and "D001: the last" in out
        code, out = quiet(stage6_run.main, ["--ladder", "2", "0.5", "--db", path])
        assert code == 0 and "ignored" in out and "honest: EPS from filing date" in out


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
