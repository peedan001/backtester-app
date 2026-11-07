"""stage5_run.py: walk through Stage 5 (the engine and the metrics).

    python stage5_run.py --demo                      # the full walkthrough, no internet needed
    python stage5_run.py --demo --db results.db      # also save prices, EPS, signals, holdings and results to a file
    python stage5_run.py --run                       # interactive: choose N, value weight, buffer and cost, see the results
    python stage5_run.py --run 20 0.5 1.5 10         # the same in one go: N, weight on value, buffer, cost in bps
    python stage5_run.py --trade                     # interactive: pick a month and watch one rebalance, stock by stock
    python stage5_run.py --trade 2019-06-28          # the same in one go (a month-end signal date)
    python stage5_run.py --try                       # seven hand-made situations that show each engine rule
Add --db FILE to --run / --trade to use a database you already have instead of a fresh synthetic world.
"""
import argparse
import datetime as dt
import sys

import align
import collect
import config
import db
import engine
import portfolio as pf
import stage3_run
import stage4_run
from portfolio import Position


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

Env = engine.Env      # prices and the calendar, loaded once (lives in engine.py so other modules can share it)


def pc(x, width=8):
    return f"{'n/a':>{width}}" if x != x else f"{x * 100:{width - 1}.2f}%"


def num(x, width=6):
    return f"{'n/a':>{width}}" if x != x else f"{x:{width}.2f}"


def metrics_lines(m, rf=config.RISK_FREE_RATE):
    L = [f"   {'':10s} {'total':>8} {'per year':>8} {'vol':>8} {'Sharpe':>6} {'worst drop':>10}"]
    for key, label in (("gross", "no costs"), ("net", "after costs"), ("benchmark", "SPY")):
        x = m[key]
        L.append(f"   {label:10s} {pc(x['total_return'])} {pc(x['cagr'])} {pc(x['volatility'])} {num(x['sharpe'])} {pc(x['max_drawdown'], 10)}")
    L.append(f"   costs take {m['cost_drag_cagr'] * 100:.2f}% a year; after costs the strategy beats SPY by {m['excess_cagr_net'] * 100:+.2f}% a year "
             f"({m['excess_cagr_gross'] * 100:+.2f}% before costs)")
    L.append(f"   beta to SPY {num(m['beta'], 4).strip()}, information ratio {num(m['information_ratio'], 4).strip()}, "
             f"beat SPY in {m['hit_rate'] * 100:.0f}% of {m['periods']} months")
    L.append(f"   average trade: {num(m['avg_traded_fraction'], 4).strip()} of the portfolio a month (buys + sells), "
             f"costing {pc(m['avg_cost_per_month'], 6).strip()} a month")
    L.append(f"   Sharpe uses a fixed risk-free rate of {rf * 100:.1f}% a year (a config setting, not a real T-bill series)")
    if m["skipped_buys"] or m["unpriced_marks"]:
        L.append(f"   data gaps: {m['skipped_buys']} purchases skipped for want of a price, {m['unpriced_marks']} holdings kept at last value")
    return L


def run_lines(rows, env, top_n, value_weight, buffer, cost_bps, delay=config.EXEC_DELAY_DAYS):
    """The full result of one set of choices, as printable lines. Returns (lines, problems)."""
    port = pf.build_portfolio(rows, top_n, value_weight, buffer)
    periods = env.run(port, cost_bps, delay)
    if len(periods) < 2:
        return ["Not enough months with signals to run a backtest."], ["no result rows"]
    bad = engine.check_engine(periods, env.replay(port, cost_bps, delay))
    m = engine.metrics(periods)
    stats = pf.portfolio_stats(port, top_n)
    trade_note = (f"trade at the close {delay} trading day(s) after each month-end signal" if delay
                  else "trade at the SAME close as the signal (optimistic: you could not know the signal yet)")
    L = [f"Top {top_n}, {value_weight:.2f} weight on value, buffer {buffer:.2f}, cost {cost_bps:g} bps per side; {trade_note}.",
         f"   {periods[0].date} to {periods[-1].date}: {m['periods']} monthly return periods"]
    L += metrics_lines(m)
    if stats["steady_months"]:
        L.append(f"   Stage 4 preview of costs: {pf.estimated_cost_drag(stats, cost_bps) * 100:.2f}% a year; "
                 f"the engine measured {m['cost_drag_cagr'] * 100:.2f}% (it trades from DRIFTED weights, so it trades more)")
    L.append(f"   arithmetic check against an independent share-counting ledger: "
             f"{'agrees on every date' if not bad else str(len(bad)) + ' PROBLEMS: ' + bad[0]}")
    return L, bad


# ---------------------------------------------------------------------------
# --trade: one rebalance, stock by stock
# ---------------------------------------------------------------------------

def trade_lines(periods, signal_date, cost_bps, limit=14):
    by_signal = {p.signal_date: i for i, p in enumerate(periods)}
    if signal_date not in by_signal:
        return [f"{signal_date} is not a signal date that was traded."]
    i = by_signal[signal_date]
    p = periods[i]
    L = [f"Signal at the close of {p.signal_date}; the trade happens at the close of {p.date}."]
    if i == 0:
        L.append("   This is the first purchase: we start in cash, so everything is bought. No return yet.")
    else:
        L.append(f"   Over the period since {periods[i - 1].date} the portfolio earned {p.gross * 100:+.2f}% (SPY {p.benchmark * 100:+.2f}%).")
    names = sorted(set(p.before) | set(p.after), key=lambda t: (-abs(p.after.get(t, 0.0) - p.before.get(t, 0.0)), t))
    L.append(f"   {'stock':7s} {'drifted':>8} {'target':>8} {'trade':>8}")
    for t in names[:limit]:
        b, a = p.before.get(t, 0.0), p.after.get(t, 0.0)
        L.append(f"   {t:7s} {b * 100:7.2f}% {a * 100:7.2f}% {(a - b) * 100:+7.2f}%   {'BUY' if a > b else 'SELL'}")
    if len(names) > limit:
        L.append(f"   ... and {len(names) - limit} more small trades")
    if p.skipped:
        L.append(f"   could not buy (no usable price): {', '.join(p.skipped)}; the rest were re-split")
    L.append(f"   traded {p.traded_fraction:.3f} of the portfolio (bought + sold). At {cost_bps:g} bps per side the cost is "
             f"{p.cost * 100:.4f}% of the portfolio: {cost_bps:g} bps x {p.traded_fraction:.3f}.")
    L.append(f"   net return of the period: (1 {p.gross * 100:+.2f}%) x (1 -{p.cost * 100:.4f}%) = {p.net * 100:+.2f}%")
    return L


def trade_mode(rows, env, date=None, top_n=config.DEFAULT_TOP_N, value_weight=config.DEFAULT_VALUE_WEIGHT,
               buffer=config.DEFAULT_BUFFER, cost_bps=config.DEFAULT_COST_BPS):
    port = pf.build_portfolio(rows, top_n, value_weight, buffer)
    periods = env.run(port, cost_bps)
    signal_dates = [p.signal_date for p in periods]
    if not signal_dates:
        print("No trades to show.")
        return 1
    if date is not None:
        try:
            d = stage4_run.month_on_or_before(signal_dates, date)
        except ValueError as e:
            print(f"Cannot use that date: {e}")
            return 1
        print("\n".join(trade_lines(periods, d, cost_bps)))
        return 0
    print(f"Pick a signal month-end (for example {signal_dates[-13]}; trades run {signal_dates[0]} to {signal_dates[-1]}). "
          f"Top {top_n}, buffer {buffer}, {cost_bps} bps. Press Enter to quit.")
    while True:
        try:
            raw = input("signal month-end> ").strip()
        except EOFError:
            break
        if not raw:
            break
        try:
            d = stage4_run.month_on_or_before(signal_dates, raw)
        except ValueError as e:
            print(f"  {e if 'before' in str(e) else 'Please type a date like 2019-06-28.'}")
            continue
        print()
        print("\n".join(trade_lines(periods, d, cost_bps)))
        print()
    return 0


# ---------------------------------------------------------------------------
# --run: choose everything, see the results
# ---------------------------------------------------------------------------

def run_mode(rows, env, top_n=None, value_weight=None, buffer=None, cost_bps=None):
    if top_n is not None:
        lines, bad = run_lines(rows, env, top_n, value_weight, buffer, cost_bps)
        print("\n".join(lines))
        return 1 if bad else 0
    print("Choose your strategy. Press Enter to keep the default in brackets; type q to quit.")
    n, w, b, c = config.DEFAULT_TOP_N, config.DEFAULT_VALUE_WEIGHT, config.DEFAULT_BUFFER, config.DEFAULT_COST_BPS
    while True:
        try:
            raw = [input(f"stocks to hold N [{n}]> ").strip(),
                   input(f"weight on value, 0 to 1 [{w}]> ").strip(),
                   input(f"buffer, {config.MIN_BUFFER} to {config.MAX_BUFFER} [{b}]> ").strip(),
                   input(f"cost in bps per side, 0 to {config.MAX_COST_BPS} [{c}]> ").strip()]
        except EOFError:
            break
        if any(x.lower() == "q" for x in raw):
            break
        try:
            n2, w2 = int(raw[0]) if raw[0] else n, float(raw[1]) if raw[1] else w
            b2, c2 = float(raw[2]) if raw[2] else b, float(raw[3]) if raw[3] else c
        except ValueError:
            print("  Please type numbers only (N must be a whole number).")
            continue
        try:
            pf._check_params(n2, w2, b2)
            engine._check_params(c2, config.EXEC_DELAY_DAYS)
        except ValueError as e:
            print(f"  {e}")
            continue
        n, w, b, c = n2, w2, b2, c2
        print()
        print("\n".join(run_lines(rows, env, n, w, b, c)[0]))
        print()
    return 0


# ---------------------------------------------------------------------------
# --try: tiny hand-made worlds, one engine rule each
# ---------------------------------------------------------------------------

S1, T1, S2, T2, S3, T3 = "2020-01-31", "2020-02-03", "2020-02-28", "2020-03-02", "2020-03-31", "2020-04-01"
CAL = [S1, T1, S2, T2, S3, T3]


def mini_world(prices, benchmark="SPY"):
    """prices: {ticker: {date: price}}. A real (in-memory) database, so the engine runs exactly as on real data."""
    con = db.connect()
    db.create_schema(con)
    collect.register_stocks(con, [(t, t, 1, 1 if t == benchmark else 0) for t in prices])
    for t, series in prices.items():
        for d, p in series.items():
            con.execute("INSERT INTO PRICE (ticker, date, adj_close) VALUES (?, ?, ?)", (t, d, p))
    con.commit()
    env = Env(con)
    return con, env


def pos(*tickers):
    return [Position(t, 1.0 / len(tickers), i + 1, 0.5, False) for i, t in enumerate(tickers)]


def flat(value, dates=CAL):
    return {d: value for d in dates}


def show_periods(periods, cost_bps):
    L = [f"  {'trade day':10s} {'signal':10s} {'gross':>7} {'cost':>7} {'net':>7} {'SPY':>7} {'traded':>6}  holds after"]
    for p in periods:
        held = " ".join(f"{t}:{w * 100:.0f}%" for t, w in sorted(p.after.items())) or "cash"
        L.append(f"  {p.date:10s} {p.signal_date:10s} {p.gross * 100:+6.2f}% {p.cost * 100:6.3f}% {p.net * 100:+6.2f}% "
                 f"{p.benchmark * 100:+6.2f}% {p.traded_fraction:6.2f}  {held}")
    return L


def _case_timing():
    con, env = mini_world({"SPY": flat(100.0), "A": {S1: 100.0, T1: 110.0, S2: 110.0, T2: 110.0, S3: 110.0, T3: 110.0}})
    port = {S1: pos("A"), S2: pos("A")}
    out = []
    for delay in (1, 0):
        per = env.run(port, 0, delay)
        out.append(f"  trade {delay} day(s) after the signal: A is bought at {per[0].date}'s close")
        out += show_periods(per, 0)
    con.close()
    return "\n".join(out)


def _case_drift():
    con, env = mini_world({"SPY": flat(100.0), "A": {**flat(100.0), T2: 200.0, S3: 200.0, T3: 200.0, S2: 150.0},
                           "B": flat(100.0)})
    port = {S1: pos("A", "B"), S2: pos("A", "B")}
    per = env.run(port, 100)
    p = per[1]
    out = ["  same two stocks both months, cost 100 bps per side:"] + show_periods(per, 100)
    out.append(f"  A doubled, B did not, so before the second trade the weights had drifted to "
               f"A {p.before['A'] * 100:.1f}% / B {p.before['B'] * 100:.1f}%. Back to 50/50 means selling A and buying B:")
    out.append(f"  traded {p.traded_fraction:.3f} of the portfolio, cost 100 bps x {p.traded_fraction:.3f} = {p.cost * 100:.3f}%. "
               "Stage 4's count of changed names would have said 0 trades.")
    con.close()
    return "\n".join(out)


def _case_swap():
    con, env = mini_world({"SPY": flat(100.0), **{t: flat(100.0) for t in "ABCD"}})
    port = {S1: pos("A", "B"), S2: pos("C", "D")}
    per = env.run(port, 50)
    out = ["  cost 50 bps per side; prices never move, so only costs show:"] + show_periods(per, 50)
    out.append("  The first row buys 100% from cash (traded 1.00). The second sells 100% and buys 100% (traded 2.00),")
    out.append(f"  so it costs 2 x 50 bps = {per[1].cost * 100:.2f}%. Wealth after both: {engine.net_wealth_path(per)[-1]:.5f}.")
    con.close()
    return "\n".join(out)


def _case_cash():
    con, env = mini_world({"SPY": {S1: 100.0, T1: 100.0, S2: 100.0, T2: 100.0, S3: 105.0, T3: 110.0},
                           "A": {S1: 100.0, T1: 100.0, S2: 105.0, T2: 105.0, S3: 106.0, T3: 106.0}})
    port = {S1: pos("A"), S2: [], S3: pos("A")}
    per = env.run(port, 10)
    out = ["  month 2 has no eligible stock, so the target is empty:"] + show_periods(per, 10)
    out.append("  The engine sells everything and sits in cash. Cash earns nothing here, while SPY kept moving.")
    con.close()
    return "\n".join(out)


def _case_no_price():
    con, env = mini_world({"SPY": flat(100.0), "A": flat(100.0), "B": flat(100.0), "C": {T2: 100.0, S3: 100.0, T3: 100.0}})
    port = {S1: pos("A", "B", "C")}
    per = env.run(port, 0)
    out = ["  the target is A, B, C, but C has no price before March:"] + show_periods(per, 0)
    out.append(f"  C could not be bought (skipped: {', '.join(per[0].skipped)}). A and B share the whole portfolio, 50% each.")
    con.close()
    return "\n".join(out)


def _case_unpriced():
    con, env = mini_world({"SPY": flat(100.0),
                           "A": {S1: 100.0, T1: 100.0, S2: 110.0, T2: 120.0, S3: 120.0, T3: 120.0},
                           "B": {S1: 100.0, T1: 100.0}})
    port = {S1: pos("A", "B"), S2: pos("A", "B")}
    per = env.run(port, 0)
    out = ["  B's prices stop after February (no price within 5 days of the next trade day):"] + show_periods(per, 0)
    out.append(f"  A rose 20%, B has no price to mark ({per[1].unpriced} unpriced holding), so B is FROZEN at its last value: "
               f"the period return is {per[1].gross * 100:+.1f}% = half of 20% + half of 0%.")
    out.append("  It is neither sold nor topped up (we cannot trade at a price we do not have), so it keeps drifting out of balance until it is priced again.")
    con.close()
    return "\n".join(out)


def _case_ledger():
    con, env = mini_world({"SPY": flat(100.0), "A": {**flat(100.0), T2: 200.0, S3: 200.0, T3: 200.0, S2: 150.0}, "B": flat(100.0)})
    port = {S1: pos("A", "B"), S2: pos("A", "B"), S3: pos("B")}
    per = env.run(port, 100)
    ledger = env.replay(port, 100)
    out = ["  the engine tracks the VALUE of each holding; the ledger counts SHARES and cash. Net wealth after each trade:"]
    for p, w, (d, w2) in zip(per, engine.net_wealth_path(per), ledger):
        out.append(f"  {d}   engine {w:.8f}   share ledger {w2:.8f}")
    out.append(f"  problems found: {engine.check_engine(per, ledger) or 'none'}")
    con.close()
    return "\n".join(out)


def try_cases():
    return [
        ("Trade the day AFTER the signal", _case_timing,
         "The 10% jump between the signal close and the next close belongs to whoever was already in. Trading at the signal close "
         "would claim it for free, which a real investor could not."),
        ("Weights drift, so even the same names mean trades", _case_drift,
         "Winners grow and losers shrink inside the month. Putting the weights back to equal is a trade, and trades cost money."),
        ("A full swap costs twice the bps", _case_swap,
         "Cost is charged on buys and sells both. Replacing every stock trades 200% of the portfolio."),
        ("No signal means cash", _case_cash,
         "An empty target means sell everything. Cash earns nothing in this engine, so a long gap drags on results."),
        ("A stock with no price cannot be bought", _case_no_price,
         "The engine never invents a price. It skips the stock and splits the portfolio over what it can actually buy."),
        ("A held stock with no price is frozen", _case_unpriced,
         "Without a price we cannot say the stock fell, and we cannot trade it. The engine freezes it at its last value, counts it and reports it, never hides it."),
        ("Two different calculations must agree", _case_ledger,
         "A second, independently written share-counting ledger gives the same wealth on every date. If it ever disagreed, one of the two would have a bug."),
    ]


def run_try_case(n):
    name, builder, lesson = try_cases()[n - 1]
    return name, builder(), lesson


def try_mode(arg=None):
    cases = try_cases()
    pick = None if arg is None else [arg]
    while True:
        if pick is None:
            print("\nSituations that show each engine rule:")
            for i, c in enumerate(cases, 1):
                print(f"  {i}. {c[0]}")
            try:
                raw = input("Pick a number (or press Enter to quit): ").strip()
            except EOFError:
                break
            if not raw:
                break
            if not raw.isdigit() or not 1 <= int(raw) <= len(cases):
                print("  Not on the list. Try again.")
                continue
            n = int(raw)
        else:
            if not pick:
                break
            n = pick.pop(0)
            if not 1 <= n <= len(cases):
                print(f"Valid choices are 1 to {len(cases)}.")
                return 1
        name, text, lesson = run_try_case(n)
        print(f"\n[{n}] {name}")
        print(text)
        print(f"  why: {lesson}")
        if pick is not None and not pick:
            break
    return 0


# ---------------------------------------------------------------------------
# --demo
# ---------------------------------------------------------------------------

def returns_lookahead(con, cutoff, top_n, value_weight, buffer, cost_bps, scramble):
    """Scramble everything after `cutoff`, re-run the whole pipeline, and compare the result rows."""
    def rows_of(c):
        rows = stage3_run.all_signals(c)
        return {p.date: (p.gross, p.net, p.benchmark) for p in Env(c).run(pf.build_portfolio(rows, top_n, value_weight, buffer), cost_bps)}
    before = rows_of(con)
    copy = stage3_run.scrambled_copy(con, cutoff, scramble)
    after = rows_of(copy)
    copy.close()
    out = {"changed_up_to_cutoff": 0, "changed_after_cutoff": 0, "rows_up_to_cutoff": 0}
    for d, vals in before.items():
        same = d in after and all(abs(a - b) < 1e-12 for a, b in zip(vals, after[d]))
        if d <= cutoff:
            out["rows_up_to_cutoff"] += 1
            out["changed_up_to_cutoff"] += not same
        else:
            out["changed_after_cutoff"] += not same
    return out


def null_world_verdict(excesses):
    """Is the average excess return in worlds with NO planted edge within the noise? Returns (mean, standard_error, ok).
    ok is False when the average is more than two standard errors (plus half a percent of slack) away from zero."""
    n = len(excesses)
    if n < 3:
        raise ValueError("need at least three worlds to judge noise")
    mean = sum(excesses) / n
    se = (sum((x - mean) ** 2 for x in excesses) / (n - 1)) ** 0.5 / n ** 0.5
    return mean, se, abs(mean) < 2 * se + 0.005


def world_excess(edge, seed, clean, top_n, value_weight):
    """Gross excess return over SPY (yearly, no costs) in a fresh synthetic world."""
    _, con = stage3_run.build_world(edge=edge, seed=seed, defects=not clean)
    try:
        rows = stage3_run.all_signals(con)
        periods = Env(con).run(pf.build_portfolio(rows, top_n, value_weight, 1.0), 0)
        return engine.metrics(periods)["excess_cagr_gross"]
    finally:
        con.close()


def demo(edge=config.SYNTH_DEFAULT_EDGE, seed=config.SYNTH_SEED, clean=False, db_path=None):
    print("=" * 72)
    print("STAGE 5: the engine and the metrics")
    print("=" * 72)
    ok_all = True
    N, W, C = config.DEFAULT_TOP_N, config.DEFAULT_VALUE_WEIGHT, config.DEFAULT_COST_BPS

    print("\n--- Step 1: portfolios from Stage 4, prices from Stage 2 ---")
    market, con = stage3_run.build_world(edge=edge, seed=seed, defects=not clean)
    rows = stage3_run.all_signals(con)
    env = Env(con)
    port = pf.build_portfolio(rows, N, W, 1.0)
    n_stocks = con.execute("SELECT COUNT(*) FROM STOCK WHERE is_benchmark = 0").fetchone()[0]
    print(f"  synthetic market: {n_stocks} stocks, planted edge {edge * 100:.1f}% a year, "
          f"{'clean data' if clean else 'with planted data defects'}")
    print(f"  top {N}, {W:.2f} weight on value, no buffer, {C} bps per side")

    print("\n--- Step 2: the trade happens AFTER the signal ---")
    periods = env.run(port, C)
    for p in periods[:3]:
        print(f"  signal at close of {p.signal_date} -> trade at close of {p.date}")
    gaps = {(dt.date.fromisoformat(p.date) - dt.date.fromisoformat(p.signal_date)).days for p in periods}
    print(f"  calendar days between signal and trade: {sorted(gaps)} (weekends and holidays make it more than 1)")
    ok = all(p.date > p.signal_date for p in periods)
    ok_all &= ok
    print(f"  every trade is after its signal: {'yes' if ok else 'NO!'}")

    print("\n--- Step 3: results, with and without costs, next to SPY ---")
    lines, bad = run_lines(rows, env, N, W, 1.0, C)
    print("\n".join("  " + l if i == 0 else l for i, l in enumerate(lines)))
    ok_all &= not bad

    print("\n--- Step 4: drift makes real costs higher than the Stage 4 preview ---")
    for b in (1.0, 1.5, 2.0):
        pt = pf.build_portfolio(rows, N, W, b)
        m = engine.metrics(env.run(pt, C))
        st = pf.portfolio_stats(pt, N)
        print(f"  buffer {b:.1f}: preview traded {st['avg_traded_fraction']:.3f} -> engine traded {m['avg_traded_fraction']:.3f} a month; "
              f"cost a year: preview {pf.estimated_cost_drag(st, C) * 100:.2f}%, measured {m['cost_drag_cagr'] * 100:.2f}%")

    print("\n--- Step 5: what costs do (net return per year; SPY for comparison) ---")
    bench = engine.metrics(env.run(port, 0))["benchmark"]["cagr"]
    costs = (0, 10, 25, 50, 100)
    print(f"  {'':12s}" + "".join(f"{c:>9d} bps" for c in costs) + f"   (SPY {bench * 100:.2f}%)")
    for b in (1.0, 1.5, 2.0):
        pt = pf.build_portfolio(rows, N, W, b)
        print(f"  buffer {b:<5.1f}  " + "".join(f"{engine.metrics(env.run(pt, c))['net']['cagr'] * 100:>12.2f}%" for c in costs))

    print("\n--- Step 6: how much does the trade-day assumption matter? ---")
    for delay in (0, 1, 2, 5):
        m = engine.metrics(env.run(port, C, delay))
        print(f"  trade {delay} day(s) after the signal: net {m['net']['cagr'] * 100:.2f}% a year, Sharpe {m['net']['sharpe']:.2f}")
    print("  (day 0 is the optimistic assumption: it trades at the very close the signal was computed from.")
    print("   Here the differences are small and not in one direction, because this synthetic market has no jump on that")
    print("   particular day. That is a property of the fake data, not a reason to relax the rule on real data.)")

    print("\n--- Step 7: does the engine invent profit where there is none? ---")
    seeds = (1, 2, 3, 4)
    nulls = [world_excess(0.0, s, clean, N, W) for s in seeds]
    mean, se, ok = null_world_verdict(nulls)
    print("  no planted edge, seeds " + ", ".join(map(str, seeds)) + ": excess return per year before costs: "
          + ", ".join(f"{x * 100:+.2f}%" for x in nulls))
    print(f"  average {mean * 100:+.2f}%, standard error {se * 100:.2f}%")
    ok_all &= ok
    print(f"  {'no false profit: the average is within the noise' if ok else 'FALSE PROFIT in a world with no edge: something is wrong'}")
    strong = world_excess(0.15, config.SYNTH_SEED, clean, N, W)
    ok = strong > 0.02
    ok_all &= ok
    print(f"  planted edge of 15% a year: excess {strong * 100:+.2f}% a year before costs  {'(the strategy finds it)' if ok else 'NOT FOUND!'}")

    print("\n--- Step 8: the look-ahead test, once more, for the results themselves ---")
    cutoff = stage3_run.resolve_month_end(con, "2019-06-30")
    for what in (("prices", "eps"), ("prices",), ("eps",)):
        ex = returns_lookahead(con, cutoff, N, W, 1.0, C, what)
        verdict = stage4_run.lookahead_verdict(ex)
        ok_all &= verdict == "OK"
        print(f"  scramble {' + '.join(what):12s} after {cutoff}: {ex['changed_up_to_cutoff']} of {ex['rows_up_to_cutoff']} earlier result rows changed, "
              f"{ex['changed_after_cutoff']} later ones changed  {verdict}")

    print("\n--- Step 9: save the results (safe to repeat) ---")
    for b in (1.0, 1.5):
        pt = pf.build_portfolio(rows, N, W, b)
        per = env.run(pt, C)
        run = pf.get_or_create_run(con, N, W, C, b)
        pf.store_holdings(con, run, pt)
        first = engine.store_results(con, run, per)
        again = engine.store_results(con, run, per)
        loaded = engine.load_results(con, run)
        same = len(loaded) == len(per) and all(abs(r["net"] - p.net) < 1e-12 for r, p in zip(loaded, per))
        idem = again["inserted"] == 0 and again["updated"] == 0 and again["deleted"] == 0
        ok_all &= same and idem
        print(f"  run {run} (buffer {b}): first {first}; again {again}  {'idempotent, and reads back the same' if idem and same else 'PROBLEM!'}")
    if db_path:
        out = db.connect(db_path)
        con.backup(out)
        out.close()
        print(f"  saved everything to {db_path}")

    print("\n" + "=" * 72)
    print("Everything behaved as designed." if ok_all else "SOMETHING BEHAVED UNEXPECTEDLY: see the lines above.")
    print("Check question: Step 4 shows that ignoring drift makes costs look 20 to 70 percent lower, and Step 6 shows the")
    print("one-day delay barely matters in this fake market. Explain why the delay still matters on real data, and why")
    print("choosing whichever delay gives the best number would be cheating.")
    con.close()
    return 0 if ok_all else 1


# ---------------------------------------------------------------------------

def main(argv):
    ap = argparse.ArgumentParser(description="Stage 5: the engine and the metrics")
    ap.add_argument("--demo", action="store_true")
    ap.add_argument("--clean", action="store_true", help="no planted data defects")
    ap.add_argument("--edge", type=float, default=config.SYNTH_DEFAULT_EDGE)
    ap.add_argument("--seed", type=int, default=config.SYNTH_SEED)
    ap.add_argument("--db", default=None)
    ap.add_argument("--run", nargs="*", default=None, metavar="N WEIGHT BUFFER COST")
    ap.add_argument("--trade", nargs="*", default=None, metavar="DATE")
    ap.add_argument("--try", dest="try_", nargs="?", const="ask", default=None)
    a = ap.parse_args(argv)

    if a.try_ is not None:
        return try_mode(None if a.try_ == "ask" else int(a.try_))
    if a.run is not None:
        if len(a.run) not in (0, 4):
            print("Use --run with four numbers (for example --run 20 0.5 1.5 10), or with nothing.")
            return 1
        try:
            args = (int(a.run[0]), float(a.run[1]), float(a.run[2]), float(a.run[3])) if a.run else (None,) * 4
            if a.run:
                pf._check_params(*args[:3])
                engine._check_params(args[3], config.EXEC_DELAY_DAYS)
        except ValueError as e:
            print(f"Cannot use those numbers: {e}")
            return 1
        con = stage3_run.open_world(a)
        try:
            return run_mode(stage3_run.all_signals(con), Env(con), *args)
        finally:
            con.close()
    if a.trade is not None:
        if len(a.trade) not in (0, 1):
            print("Use --trade with one date (for example --trade 2019-06-28), or with nothing.")
            return 1
        try:
            if a.trade:
                dt.date.fromisoformat(a.trade[0])
        except ValueError:
            print(f"Cannot use that date: {a.trade[0]!r} is not a date like 2019-06-28.")
            return 1
        con = stage3_run.open_world(a)
        try:
            return trade_mode(stage3_run.all_signals(con), Env(con), a.trade[0] if a.trade else None)
        finally:
            con.close()
    return demo(edge=a.edge, seed=a.seed, clean=a.clean, db_path=a.db)


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
