"""app_logic.py: everything the Streamlit app computes, with no Streamlit in it, so it can be tested like the rest.

The app (app.py) is only a face: it collects choices, calls these functions and draws what they return.
Every function takes and returns plain Python values (numbers, strings, lists of dicts), never a database cursor.

    open_synthetic(...)   a fake market, built in memory          -> World
    open_database(path)   a database a --live run saved earlier   -> World (opened READ-ONLY)
    run_one(...)          one backtest                            -> dict
    curve_rows / metric_rows / holding_rows / period_rows         tables for the screen
    bias_summary(...)     the four runs that separate the biases -> dict
    seeds_summary(...)    the same over several fake worlds       -> list of dict
    graveyard_rows / graveyard_story / price_history              failed stocks
"""
import math
import os
import sqlite3
from dataclasses import dataclass, field

import align
import bias
import config
import db
import engine
import portfolio as pf
import stage3_run
import stage6_run


class AppError(Exception):
    """A problem the person using the app can fix (a missing file, a value out of range). Its text is shown as is."""


@dataclass
class World:
    con: sqlite3.Connection
    env: engine.Env
    label: str
    synthetic: bool
    n_stocks: int                 # stocks in today's list
    n_dead: int                   # failed stocks (not in today's list)
    first_date: str
    last_date: str
    params: dict = field(default_factory=dict)       # how a synthetic world was made (empty for a database)


# ---------------------------------------------------------------------------
# opening a world
# ---------------------------------------------------------------------------

def _describe(con, label, synthetic, params):
    stocks = con.execute("SELECT COUNT(*) FROM STOCK WHERE is_benchmark = 0 AND in_sp500 = 1").fetchone()[0]
    dead = con.execute("SELECT COUNT(*) FROM STOCK WHERE is_benchmark = 0 AND in_sp500 = 0").fetchone()[0]
    first, last = con.execute("SELECT MIN(date), MAX(date) FROM PRICE").fetchone()
    if not stocks or first is None:
        raise AppError("That database has no prices yet. Collect some first (stage2_run.py --live).")
    try:
        env = engine.Env(con)
    except Exception as e:  # noqa: BLE001
        raise AppError(f"Could not set up the trading calendar: {e}") from e
    return World(con, env, label, synthetic, stocks, dead, first, last, dict(params))


def _copy_to_shared_memory(con):
    """Streamlit may use a connection from a different thread than the one that made it, which sqlite refuses by
    default. A copy opened with check_same_thread=False is safe because the app only ever READS from it."""
    shared = sqlite3.connect(":memory:", check_same_thread=False)
    con.backup(shared)
    con.close()
    shared.execute("PRAGMA foreign_keys = ON")
    return shared


def open_synthetic(edge=config.SYNTH_DEFAULT_EDGE, seed=config.SYNTH_SEED, reaction=config.SYNTH_DEFAULT_REACTION,
                   n_dead=config.SYNTH_DEFAULT_DEAD, n_stocks=config.SYNTH_N_STOCKS, defects=False):
    """A fake market with a planted edge (see synthetic.py). Raises AppError for values the generator refuses."""
    try:
        _, con = stage3_run.build_world(edge=edge, seed=int(seed), defects=defects, n_stocks=int(n_stocks),
                                        reaction=reaction, n_dead=int(n_dead))
    except ValueError as e:
        raise AppError(str(e)) from e
    params = {"edge": edge, "seed": int(seed), "reaction": reaction, "n_dead": int(n_dead),
              "n_stocks": int(n_stocks), "defects": defects}
    label = (f"Fake market: edge {edge * 100:.0f}%, earnings reaction {reaction:g}, {int(n_dead)} failed stocks, "
             f"seed {int(seed)}")
    return _describe(_copy_to_shared_memory(con), label, True, params)


def open_database(path):
    """Open a database file from a --live run. Read-only: the app never changes your data."""
    path = (path or "").strip()
    if not path:
        raise AppError("Type the path of a database file.")
    if not os.path.isfile(path):
        raise AppError(f"No file at {path}. Run  python stage2_run.py --live --db {path}  on your computer first.")
    try:
        con = sqlite3.connect(f"file:{path}?mode=ro", uri=True, check_same_thread=False)
        names = {r[0] for r in con.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    except sqlite3.DatabaseError as e:
        raise AppError(f"{path} is not a SQLite database ({e}).") from e
    missing = {"STOCK", "PRICE", "FUNDAMENTAL"} - names
    if missing:
        con.close()
        raise AppError(f"{path} is not a backtester database (missing {', '.join(sorted(missing))}).")
    con.execute("PRAGMA foreign_keys = ON")
    return _describe(con, f"Database file: {os.path.basename(path)}", False, {"path": path})


# ---------------------------------------------------------------------------
# one backtest
# ---------------------------------------------------------------------------

def check_choices(top_n, value_weight, buffer, cost_bps, delay, pe_mode, universe):
    """Refuse a value the engine would refuse, in words the person can act on."""
    try:
        pf._check_params(top_n, value_weight, buffer)
    except ValueError as e:
        raise AppError(str(e)) from e
    if not 0 <= cost_bps <= config.MAX_COST_BPS:
        raise AppError(f"Trading cost must be between 0 and {config.MAX_COST_BPS} basis points.")
    if not 0 <= delay <= config.MAX_EXEC_DELAY_DAYS:
        raise AppError(f"The trade delay must be between 0 and {config.MAX_EXEC_DELAY_DAYS} trading days.")
    if pe_mode not in config.EPS_MODES:
        raise AppError(f"P/E version must be one of {config.EPS_MODES}.")
    if universe not in config.UNIVERSES:
        raise AppError(f"Universe must be one of {config.UNIVERSES}.")


def run_one(world, top_n=config.DEFAULT_TOP_N, value_weight=config.DEFAULT_VALUE_WEIGHT, buffer=config.DEFAULT_BUFFER,
            cost_bps=config.DEFAULT_COST_BPS, delay=config.EXEC_DELAY_DAYS, pe_mode="lagged", universe="survivors"):
    """One backtest. Returns {'periods', 'portfolio', 'metrics', 'ledger_problems', 'choices', 'dead_held'}."""
    check_choices(top_n, value_weight, buffer, cost_bps, delay, pe_mode, universe)
    rows = stage3_run.all_signals(world.con, tickers=align.tickers_for(world.con, universe), pe_mode=pe_mode)
    port = pf.build_portfolio(rows, top_n, value_weight, buffer)
    periods = world.env.run(port, cost_bps, delay)
    if len(periods) < 2:
        raise AppError("Not enough months with signals to run a backtest. Try a smaller portfolio or other data.")
    dead = bias.dead_tickers(world.con)
    return {
        "periods": periods, "portfolio": port, "metrics": engine.metrics(periods),
        "ledger_problems": engine.check_engine(periods, world.env.replay(port, cost_bps, delay)),
        "choices": {"top_n": top_n, "value_weight": value_weight, "buffer": buffer, "cost_bps": cost_bps,
                    "delay": delay, "pe_mode": pe_mode, "universe": universe},
        "dead_held": len({t for p in periods for t in p.after if t in dead}),
    }


def curve_rows(periods):
    """Growth of 1 for the strategy (before and after costs) and for SPY, one row per result date."""
    gross = net = bench = 1.0
    out = []
    for p in periods:
        gross *= 1.0 + p.gross
        net *= 1.0 + p.net
        bench *= 1.0 + p.benchmark
        out.append({"date": p.date, "after costs": net, "before costs": gross, "SPY": bench})
    return out


def _pct(x):
    return None if x != x else x


def metric_rows(m):
    """The headline table: one row per measure, a column for before costs, after costs and the benchmark."""
    def three(key):
        return [_pct(m["gross"][key]), _pct(m["net"][key]), _pct(m["benchmark"][key])]
    rows = [
        ("Total return", "pct", three("total_return")),
        ("Return per year (CAGR)", "pct", three("cagr")),
        ("Volatility per year", "pct", three("volatility")),
        (f"Sharpe (risk-free {config.RISK_FREE_RATE * 100:.0f}%)", "num", three("sharpe")),
        ("Worst fall from a peak", "pct", three("max_drawdown")),
    ]
    return [{"measure": name, "kind": kind, "before costs": v[0], "after costs": v[1], "SPY": v[2]} for name, kind, v in rows]


def summary_facts(m):
    """The numbers that are about the strategy against SPY rather than one column, as (label, value, kind) triples."""
    return [
        ("Beats SPY by, per year (after costs)", m["excess_cagr_net"], "pct"),
        ("Cost drag, per year", m["cost_drag_cagr"], "pct"),
        ("Beta to SPY", m["beta"], "num"),
        ("Information ratio", m["information_ratio"], "num"),
        ("Months better than SPY", m["hit_rate"], "pct"),
        ("Typical monthly turnover", m["avg_traded_fraction"], "pct"),
        ("Stocks sold because their data ended", m["delisted_sales"], "int"),
        ("Purchases skipped (no price)", m["skipped_buys"], "int"),
    ]


def signal_dates(result):
    """Month-ends with something held, newest last."""
    return [d for d, positions in sorted(result["portfolio"].items()) if positions]


def holding_rows(result, signal_date):
    """What the portfolio decided at one month-end, and what trading it cost. Returns (rows, facts)."""
    positions = result["portfolio"].get(signal_date)
    if positions is None:
        raise AppError(f"{signal_date} is not a month-end in this backtest.")
    period = next((p for p in result["periods"] if p.signal_date == signal_date), None)
    rows = [{"ticker": p.ticker, "target weight": p.weight, "rank": p.place, "score": p.score,
             "kept from last month": p.kept} for p in positions]
    facts = {"signal_date": signal_date, "trade_date": period.date if period else None}
    if period:
        facts.update(traded_fraction=period.traded_fraction, cost=period.cost, delisted=list(period.delisted),
                     skipped=list(period.skipped), unpriced=period.unpriced, cash=period.cash)
    return rows, facts


# ---------------------------------------------------------------------------
# bias lab
# ---------------------------------------------------------------------------

def bias_summary(world, top_n=config.DEFAULT_TOP_N, value_weight=config.DEFAULT_VALUE_WEIGHT,
                 buffer=config.DEFAULT_BUFFER, cost_bps=config.DEFAULT_COST_BPS):
    """The ladder: honest, no filing lag, today's EPS, and (if there are failed stocks) honest on all stocks."""
    check_choices(top_n, value_weight, buffer, cost_bps, config.EXEC_DELAY_DAYS, "lagged", "survivors")
    r = bias.evaluate_world(world.con, world.env, top_n, value_weight, buffer, cost_bps)
    steps = [("Honest: EPS from filing date, today's S&P 500 list", r["lagged"]),
             ("No filing lag (quarter-end date)", r["no_lag"]),
             ("Hindsight: today's EPS on old prices", r["today"])]
    if world.n_dead:
        steps.append(("Honest P/E on ALL stocks, failed ones included", r["all"]))
    rows = [{"version": name, "per year after costs": v["net_cagr"], "Sharpe": v["sharpe"],
             "failed stocks held": v["dead_held"], "sold when data ended": v["delisted_sales"]} for name, v in steps]
    return {"rows": rows, "inflation_no_lag": r["inflation_no_lag"], "inflation_today": r["inflation_today"],
            "survivorship": r["survivorship"] if world.n_dead else None,
            "total": r["total"] if world.n_dead else None, "has_failed_stocks": world.n_dead > 0}


def seeds_summary(world, seeds, top_n=config.DEFAULT_TOP_N, value_weight=config.DEFAULT_VALUE_WEIGHT,
                  buffer=config.DEFAULT_BUFFER, cost_bps=config.DEFAULT_COST_BPS, progress=None):
    """The ladder over several fake worlds with the same settings as `world` (only the seed changes).
    Returns one row per kind of bias: mean, standard error, runs, verdict."""
    if not world.synthetic:
        raise AppError("Repeating over seeds needs a fake market (a real database has only one world).")
    if len(seeds) < 2:
        raise AppError("Use at least 2 seeds: one run has no spread to measure luck against.")
    check_choices(top_n, value_weight, buffer, cost_bps, config.EXEC_DELAY_DAYS, "lagged", "survivors")
    p = world.params
    results = []
    for i, s in enumerate(seeds):
        results += bias.worlds_over_seeds([s], edge=p["edge"], reaction=p["reaction"], n_dead=p["n_dead"],
                                          top_n=top_n, value_weight=value_weight, buffer=buffer, cost_bps=cost_bps,
                                          n_stocks=p["n_stocks"], defects=p["defects"])
        if progress:
            progress(i + 1, len(seeds))
    kinds = [("No filing lag", "inflation_no_lag"), ("Today's EPS (hindsight)", "inflation_today")]
    if p["n_dead"]:
        kinds += [("Survivorship (failed stocks left out)", "survivorship"), ("All shortcuts together", "total")]
    out = []
    for name, key in kinds:
        s = bias.summarize([r[key] for r in results])
        out.append({"bias": name, "mean per year": s["mean"], "standard error": s["se"], "worlds": s["n"],
                    "verdict": bias.bias_verdict(s["mean"], s["se"])})
    return out


# ---------------------------------------------------------------------------
# failed stocks
# ---------------------------------------------------------------------------

def graveyard_rows(world, top_n=config.DEFAULT_TOP_N, buffer=config.DEFAULT_BUFFER, cost_bps=config.DEFAULT_COST_BPS):
    """Per failed stock: when it stopped, how far it fell, and how the blend and pure value treated it.
    Returns (rows, handle); pass the handle to graveyard_story."""
    if not world.n_dead:
        raise AppError("This world has no failed stocks. Raise 'failed stocks' in the sidebar, or use a database that has some.")
    check_choices(top_n, 0.5, buffer, cost_bps, config.EXEC_DELAY_DAYS, "lagged", "all")
    info, rows, runs = stage6_run.graveyard_info(world.con, top_n, buffer, cost_bps)
    w1, w2 = config.DEFAULT_VALUE_WEIGHT, 1.0
    table = [{"ticker": e["ticker"], "stopped trading": e["last_date"], "fell from peak": e["fall"],
              "days from peak": e["days_from_peak"], "months held (blend)": e[f"months_held_{w1}"],
              "months held (pure value)": e[f"months_held_{w2}"],
              "sold at failure (blend)": e[f"sold_at_failure_{w1}"],
              "sold at failure (pure value)": e[f"sold_at_failure_{w2}"]} for e in info]
    return sorted(table, key=lambda r: r["stopped trading"]), (rows, runs)


def graveyard_story(world, handle, ticker):
    rows, runs = handle
    if ticker.upper() not in bias.dead_tickers(world.con):
        raise AppError(f"{ticker} is not a failed stock in this world.")
    return stage6_run.story_lines(world.con, rows, runs, ticker)


def failed_tickers(world):
    return sorted(bias.dead_tickers(world.con))


def price_history(world, ticker):
    """[(date, price)] for one stock, flagged prices left out."""
    return [(d, p) for d, p in world.con.execute(
        "SELECT date, adj_close FROM PRICE WHERE ticker = ? AND flagged = 0 ORDER BY date", (ticker,))]


# ---------------------------------------------------------------------------
# formatting shared by the screen and the tests
# ---------------------------------------------------------------------------

def fmt(value, kind):
    """Text for one number: 'pct' as +12.3%-style without the sign, 'num' with two decimals, 'int' as a whole number."""
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return "n/a"
    if kind == "pct":
        return f"{value * 100:.1f}%"
    if kind == "int":
        return f"{int(value):d}"
    return f"{value:.2f}"
