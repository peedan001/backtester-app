"""stage6_run.py: walk through Stage 6 (bias: how much do shortcuts inflate a backtest?).

    python stage6_run.py --demo                      # the full walkthrough (about two minutes), no internet needed
    python stage6_run.py --demo --seeds 3            # quicker: fewer repeated worlds (verdicts are then not checked)
    python stage6_run.py --demo --db bias.db         # also save the four runs (and everything under them) to a file
    python stage6_run.py --ladder                    # interactive: choose how strongly earnings move prices, see the bias ladder
    python stage6_run.py --ladder 2.0 0.5            # the same in one go: earnings reaction, weight on value
    python stage6_run.py --graveyard                 # interactive: browse the failed stocks and how the strategy treated them
    python stage6_run.py --graveyard D001            # the story of one failed stock
    python stage6_run.py --try                       # six hand-made situations that show each bias mechanism
Add --db FILE to --ladder / --graveyard to use a database you already have (it must contain failed stocks to show survivorship).
"""
import argparse
import calendar
import datetime as dt
import sys

import align
import bias
import collect
import config
import db
import engine
import portfolio as pf
import signals
import stage3_run
import stage5_run
from portfolio import Position

pc = stage5_run.pc


def pp(x):
    """A signed percentage-point difference."""
    return f"{'n/a':>7}" if x != x else f"{x * 100:+6.2f}%"


# ---------------------------------------------------------------------------
# the ladder: each shortcut, and what it added
# ---------------------------------------------------------------------------

def ladder_lines(res, value_weight, n_dead):
    L = [f"   {'version':45s} {'net / year':>10} {'vs honest':>10}   signal quality (IC, t)"]

    def ic(v):
        return f"{v['ic']:+.3f} (t {v['ic_t']:+.1f})" if "ic" in v and v["ic"] == v["ic"] else ""

    L.append(f"   {'honest: EPS from filing date, today' + chr(39) + 's list':45s} {pc(res['lagged']['net_cagr'], 10)} {'':>10}   {ic(res['lagged'])}")
    L.append(f"   {'EPS from the quarter end (no filing lag)':45s} {pc(res['no_lag']['net_cagr'], 10)} {pp(res['inflation_no_lag']):>10}   {ic(res['no_lag'])}")
    L.append(f"   {'today' + chr(39) + 's EPS on every date (hindsight)':45s} {pc(res['today']['net_cagr'], 10)} {pp(res['inflation_today']):>10}   {ic(res['today'])}")
    if n_dead:
        L.append(f"   {'honest, but ALL stocks (the failed ones too)':45s} {pc(res['all']['net_cagr'], 10)} {pp(-res['survivorship']):>10}   {ic(res['all'])}")
        L.append(f"   naive backtest (today's EPS, today's list) against the fully honest one: it overstates the return by "
                 f"{res['total'] * 100:+.2f}% a year")
        L.append(f"   the honest run held {res['all']['dead_held']} failed stocks at some point (about {res['all']['dead_weight'] * 100:.0f}% of the portfolio "
                 f"on average) and sold {res['all']['delisted_sales']} of them at their last price")
    return L


def evaluate_lines(con, env, value_weight, n_dead, top_n=config.DEFAULT_TOP_N, buffer=config.DEFAULT_BUFFER,
                   cost_bps=config.DEFAULT_COST_BPS, with_ic=True):
    res = bias.evaluate_world(con, env, top_n, value_weight, buffer, cost_bps, with_ic=with_ic)
    head = (f"Top {top_n}, {value_weight:.2f} weight on value, buffer {buffer:.2f}, {cost_bps:g} bps per side. "
            f"Same strategy, same costs; only the shortcut changes:")
    return [head] + ladder_lines(res, value_weight, n_dead), res


def seeds_line(label, values):
    s = bias.summarize(values)
    verdict = bias.bias_verdict(s["mean"], s["se"])
    return (f"   {label:34s} average {s['mean'] * 100:+6.2f}% a year, standard error {s['se'] * 100:4.2f}%, "
            f"range {min(values) * 100:+5.1f}% to {max(values) * 100:+5.1f}%  -> {verdict}"), verdict, s


# ---------------------------------------------------------------------------
# --ladder (interactive)
# ---------------------------------------------------------------------------

def ladder_mode(con_factory, reaction=None, value_weight=None, n_dead=config.SYNTH_DEFAULT_DEAD):
    """con_factory(reaction) -> a connection. One-shot when reaction is given, otherwise a prompt loop."""
    def show(r, w):
        con = con_factory(r)
        try:
            lines, _ = evaluate_lines(con, engine.Env(con), w, bool(bias.dead_tickers(con)))
        finally:
            con.close()
        print("\n".join(lines))

    if reaction is not None:
        show(reaction, value_weight)
        return 0
    print("Choose a world. Press Enter to keep the default in brackets; type q to quit. Each run takes several seconds.")
    r, w = config.SYNTH_DEFAULT_REACTION, config.DEFAULT_VALUE_WEIGHT
    while True:
        try:
            raw = [input(f"how strongly earnings move prices, 0 to {config.SYNTH_MAX_REACTION:g} [{r}]> ").strip(),
                   input(f"weight on value, 0 to 1 [{w}]> ").strip()]
        except EOFError:
            break
        if any(x.lower() == "q" for x in raw):
            break
        try:
            r2, w2 = float(raw[0]) if raw[0] else r, float(raw[1]) if raw[1] else w
        except ValueError:
            print("  Please type numbers only.")
            continue
        if not 0 <= r2 <= config.SYNTH_MAX_REACTION or not 0 <= w2 <= 1:
            print(f"  The reaction must be between 0 and {config.SYNTH_MAX_REACTION:g} and the weight between 0 and 1.")
            continue
        r, w = r2, w2
        print()
        show(r, w)
        print()
    return 0


# ---------------------------------------------------------------------------
# --graveyard: the failed stocks
# ---------------------------------------------------------------------------

def graveyard_info(con, top_n=config.DEFAULT_TOP_N, buffer=config.DEFAULT_BUFFER, cost_bps=config.DEFAULT_COST_BPS):
    """For every failed stock: when it stopped trading, how far it fell, and how two strategies treated it
    (the blend and pure value, both on the full universe with honest P/E)."""
    dead = sorted(bias.dead_tickers(con))
    env = engine.Env(con)
    rows = stage3_run.all_signals(con, tickers=align.all_tickers(con))
    runs = {}
    for w in (config.DEFAULT_VALUE_WEIGHT, 1.0):
        port = pf.build_portfolio(rows, top_n, w, buffer)
        runs[w] = (port, env.run(port, cost_bps))
    info = []
    for t in dead:
        last = env.prices.last_date(t)
        dates = env.prices._dates[t]
        prices = env.prices._prices[t]
        window = [p for d, p in zip(dates, prices) if (dt.date.fromisoformat(last) - dt.date.fromisoformat(d)).days <= 365]
        peak_i = max(range(len(window)), key=lambda i: window[i])
        entry = {"ticker": t, "last_date": last, "last_price": prices[-1], "peak": window[peak_i],
                 "fall": prices[-1] / window[peak_i] - 1.0, "days_from_peak": len(window) - 1 - peak_i}
        for w, (port, periods) in runs.items():
            entry[f"months_held_{w}"] = sum(1 for p in periods if t in p.after)
            entry[f"sold_at_failure_{w}"] = sum(1 for p in periods if t in p.delisted)
        info.append(entry)
    return info, rows, runs


def graveyard_lines(info):
    w1, w2 = config.DEFAULT_VALUE_WEIGHT, 1.0
    L = [f"   {'stock':6s} {'stopped trading':>15} {'fell from peak':>14} {'in last':>9}   {'months held':>13} {'sold at failure':>16}",
         f"   {'':6s} {'':>15} {'':>14} {'':>9}   {'blend / value':>13} {'blend / value':>16}"]
    for e in sorted(info, key=lambda e: e["last_date"]):
        L.append(f"   {e['ticker']:6s} {e['last_date']:>15} {e['fall'] * 100:13.0f}% {e['days_from_peak']:>7d}d   "
                 f"{e[f'months_held_{w1}']:>6d} / {e[f'months_held_{w2}']:<5d} {e[f'sold_at_failure_{w1}']:>8d} / {e[f'sold_at_failure_{w2}']:<5d}")
    held = sum(1 for e in info if e[f"months_held_{w2}"])
    L.append(f"   {len(info)} failed stocks; the pure-value strategy bought {held} of them at some point "
             f"and was still holding {sum(e[f'sold_at_failure_{w2}'] for e in info)} when they stopped trading.")
    return L


def story_lines(con, rows, runs, ticker, months=9):
    """The last months of a failed stock's life: its price, its signals, and whether each strategy held it."""
    ticker = ticker.upper()
    mine = [r for r in rows if r.ticker == ticker and r.price is not None]
    if not mine:
        return [f"No price history for {ticker} at any month-end."]
    mine = mine[-months:]
    by_date = {}
    for r in rows:
        by_date.setdefault(r.date, []).append(r)
    w1, w2 = config.DEFAULT_VALUE_WEIGHT, 1.0
    L = [f"{ticker}: the last {len(mine)} month-ends before it stopped trading",
         f"   {'month-end':10s} {'price':>8} {'P/E':>6} {'cheap rank':>10} {'momentum rank':>13} {'place (blend)':>14}   held by blend / value"]
    for r in mine:
        place = ""
        order = signals.scores_for_date(by_date[r.date], r.date, w1)
        for i, (_, x) in enumerate(order, 1):
            if x.ticker == ticker:
                place = f"{i} of {len(order)}"
        held = [("yes" if any(p.ticker == ticker for p in runs[w][0].get(r.date, [])) else "no") for w in (w1, w2)]
        pe = f"{r.pe:6.1f}" if r.pe is not None else f"{'-':>6}"
        cheap = f"{r.pe_rank:10.2f}" if r.pe_rank is not None else f"{'-':>10}"
        mom = f"{r.momentum_rank:13.2f}" if r.momentum_rank is not None else f"{'-':>13}"
        L.append(f"   {r.date:10s} {r.price:8.2f} {pe} {cheap} {mom} {place:>14}   {held[0]:>5} / {held[1]}")
    L.append("   As the price falls the stock looks CHEAPER (low P/E). A value signal reads that as a bargain; a momentum signal reads it as weakness.")
    return L


def graveyard_mode(con, ticker=None):
    dead = bias.dead_tickers(con)
    if not dead:
        print("This database has no failed stocks (every stock is in today's list), so there is no graveyard to browse.")
        return 1
    info, rows, runs = graveyard_info(con)
    if ticker is not None:
        if ticker.upper() not in dead:
            print(f"{ticker} is not a failed stock here. The failed stocks are: {', '.join(sorted(dead))}")
            return 1
        print("\n".join(story_lines(con, rows, runs, ticker)))
        return 0
    print("\n".join(graveyard_lines(info)))
    print("Type a failed stock (for example " + sorted(dead)[0] + ") to see its last months. Press Enter to quit.")
    while True:
        try:
            raw = input("failed stock> ").strip()
        except EOFError:
            break
        if not raw:
            break
        if raw.upper() not in dead:
            print(f"  Not a failed stock. Try one of: {', '.join(sorted(dead)[:6])} ...")
            continue
        print()
        print("\n".join(story_lines(con, rows, runs, raw)))
        print()
    return 0


# ---------------------------------------------------------------------------
# --try: tiny hand-made worlds, one mechanism each
# ---------------------------------------------------------------------------

def _month_ends(first_year, last_year):
    out = []
    for y in range(first_year, last_year + 1):
        for m in range(1, 13):
            d = dt.date(y, m, calendar.monthrange(y, m)[1])
            while d.weekday() >= 5:
                d -= dt.timedelta(days=1)
            out.append(d.isoformat())
    return out


def _quarters(first, last, eps_for, lag=40):
    """[(period_end, filed, eps)] for quarter ends from `first` to `last` (inclusive, 'YYYY-Qn'); eps_for(index) gives the EPS."""
    out, i = [], 0
    y, q = first
    while (y, q) <= last:
        end = dt.date(y, q * 3, calendar.monthrange(y, q * 3)[1])
        out.append((end.isoformat(), (end + dt.timedelta(days=lag)).isoformat(), eps_for(y, q, i)))
        i += 1
        y, q = (y, q + 1) if q < 4 else (y + 1, 1)
    return out


def _eps_con(eps_by_ticker, prices_by_ticker):
    """A tiny database: a flat benchmark on every month-end of 2018 to 2020, plus the given EPS and prices."""
    con = db.connect()
    db.create_schema(con)
    tickers = sorted(set(eps_by_ticker) | set(prices_by_ticker))
    collect.register_stocks(con, [(t, t, 1, 0) for t in tickers] + [(config.BENCHMARK_TICKER, "benchmark", 0, 1)])
    for d in _month_ends(2018, 2020):
        con.execute("INSERT INTO PRICE (ticker, date, adj_close) VALUES (?, ?, 100)", (config.BENCHMARK_TICKER, d))
    for t, series in prices_by_ticker.items():
        for d, p in series.items():
            con.execute("INSERT INTO PRICE (ticker, date, adj_close) VALUES (?, ?, ?)", (t, d, p))
    for t, rows in eps_by_ticker.items():
        for period_end, filed, eps in rows:
            con.execute("INSERT INTO FUNDAMENTAL (ticker, period_end, available_date, eps) VALUES (?, ?, ?, ?)", (t, period_end, filed, eps))
    con.commit()
    return con


def _hindsight_con():
    """X never changes; Y's earnings double from the first quarter of 2020 and its price follows."""
    x = _quarters((2018, 4), (2020, 3), lambda y, q, i: 1.0)
    y = _quarters((2018, 4), (2020, 3), lambda y, q, i: 1.0 if (y, q) <= (2019, 4) else 2.0)
    return _eps_con({"X": x, "Y": y}, {"X": {"2019-12-31": 40.0, "2020-09-30": 40.0}, "Y": {"2019-12-31": 40.0, "2020-09-30": 70.0}})


def _case_three_views():
    eps = _quarters((2019, 1), (2020, 2), lambda y, q, i: {0: 1.0, 1: 1.0, 2: 1.0, 3: 1.0, 4: 2.0, 5: 3.0}[i])
    con = _eps_con({"X": eps}, {})
    book = align.EpsBook(con)
    date, price = "2020-04-30", 40.0
    out = [f"  stock X earned 1.0 a quarter through 2019, then 2.0 (filed 2020-05-10) and 3.0 (filed 2020-08-09). Price on {date}: {price:.0f}.",
           f"  {'version':28s} {'trailing EPS':>12} {'P/E':>6}   uses a filing dated"]
    for mode, label in (("lagged", "lagged (honest)"), ("no_lag", "no filing lag"), ("today", "today's EPS")):
        ttm, why = book.ttm("X", date, mode)
        flag = "   <- after " + date + ": LOOK-AHEAD" if ttm.latest_filed > date else ""
        out.append(f"  {label:28s} {ttm.value:12.1f} {price / ttm.value:6.1f}   {ttm.latest_filed}{flag}")
    out.append("  The same stock and the same day give three different P/E ratios, depending on what you allow yourself to know.")
    con.close()
    return "\n".join(out)


def _case_hindsight_winner():
    con = _hindsight_con()
    out = ["  X and Y are identical until the end of 2019: same price (40), same earnings. Then Y's earnings double and its price rises to 70 by September 2020.",
           f"  {'version':18s} {'P/E of X':>9} {'P/E of Y':>9}   {'who looks cheaper at 2019-12-31?'}"]
    for mode, label in (("lagged", "lagged (honest)"), ("today", "today's EPS")):
        rows = {r.ticker: r for r in signals.add_ranks(align.build_panel(con, tickers=["X", "Y"], dates=["2019-12-31"], pe_mode=mode))}
        x, y = rows["X"], rows["Y"]
        verdict = "a tie: nobody could tell them apart" if x.pe_rank == y.pe_rank else ("Y, the one that is about to rise" if y.pe_rank > x.pe_rank else "X")
        out.append(f"  {label:18s} {x.pe:9.1f} {y.pe:9.1f}   {verdict}")
    out.append("  Today's EPS 'knows' that Y's earnings are about to double, so it calls Y a bargain and Y then rises. That is not skill.")
    con.close()
    return "\n".join(out)


def _case_audit():
    con = _hindsight_con()
    out = ["  the look-ahead audit checks that every date a signal relied on is on or before its own month-end:"]
    for mode, label in (("lagged", "lagged (honest)"), ("no_lag", "no filing lag"), ("today", "today's EPS")):
        rows = align.build_panel(con, tickers=["X", "Y"], dates=["2019-12-31"], pe_mode=mode)
        bad = align.audit_lookahead(rows)
        detail = f"  e.g. {bad[0][0]} used {bad[0][2]} dated {bad[0][3]}" if bad else ""
        out.append(f"  {label:18s} violations: {len(bad)}{detail}")
    con.close()
    return "\n".join(out)


def _delist_world(prices, port, cost=0):
    con, env = stage5_run.mini_world(prices)
    try:
        return env.run(port, cost)
    finally:
        con.close()


def _case_delisting():
    S1, T1, S2, T2, S3, T3 = stage5_run.S1, stage5_run.T1, stage5_run.S2, stage5_run.T2, stage5_run.S3, stage5_run.T3
    prices = {"SPY": stage5_run.flat(100.0), "B": stage5_run.flat(100.0),
              "A": {S1: 100.0, T1: 100.0, S2: 90.0, "2020-03-04": 30.0}}
    port = {S1: stage5_run.pos("A", "B"), S2: stage5_run.pos("A", "B"), S3: stage5_run.pos("B")}
    per = _delist_world(prices, port, 10)
    out = ["  A trades until 2020-03-04, last price 30 (it was 100). Nothing is quoted after that:"] + stage5_run.show_periods(per, 10)
    p = per[-1]
    out.append(f"  On {p.date} A's data has ended ({', '.join(p.delisted)} delisted): it is marked at its last price, 30, which is a "
               f"{(30 / 90 - 1) * 100:.0f}% fall from the last price we marked before, and sold. The cash waits for the next rebalance.")
    return "\n".join(out)


def _case_pause():
    S1, T1, S2, T2, S3, T3 = stage5_run.S1, stage5_run.T1, stage5_run.S2, stage5_run.T2, stage5_run.S3, stage5_run.T3
    prices = {"SPY": stage5_run.flat(100.0), "A": stage5_run.flat(100.0),
              "B": {S1: 100.0, T1: 100.0, S2: 100.0, T2: 100.0, "2020-04-20": 100.0}}
    port = {S1: stage5_run.pos("A", "B"), S2: stage5_run.pos("A", "B"), S3: stage5_run.pos("A", "B")}
    per = _delist_world(prices, port, 10)
    out = ["  B has no prices on 2020-03-31 or 2020-04-01, but trades again on 2020-04-20:"] + stage5_run.show_periods(per, 10)
    p = per[-1]
    out.append(f"  On {p.date} B is FROZEN (unpriced {p.unpriced}, delisted {len(p.delisted)}), not sold: because a later price exists, this is a gap, not a failure.")
    return "\n".join(out)


def _case_survivors_only():
    market, con = stage3_run.build_world(n_stocks=10, seed=3, defects=False, n_dead=3)
    try:
        surv, everything = align.tickers_for(con, "survivors"), align.tickers_for(con, "all")
        prices = align.PriceBook(con)
        out = [f"  a small world: {len(surv)} stocks in today's list and {len(everything) - len(surv)} that failed along the way.",
               f"  today's list      : {len(surv)} stocks, e.g. {', '.join(surv[:4])}, ...",
               f"  everything        : {len(everything)} stocks, adding {', '.join(sorted(set(everything) - set(surv)))}"]
        for t in sorted(set(everything) - set(surv)):
            last = prices.last_date(t)
            out.append(f"  {t} stopped trading on {last} at {prices.last_price(t):.2f}; a data source that only knows today's list never shows it.")
        return "\n".join(out)
    finally:
        con.close()


def try_cases():
    return [
        ("Three P/E ratios for the same stock on the same day", _case_three_views,
         "'Lagged' uses what had been filed. 'No lag' uses a quarter that had ended but not yet been reported. 'Today' uses the future."),
        ("Hindsight picks tomorrow's winner", _case_hindsight_winner,
         "Dividing an old price by today's earnings quietly tells you which companies grew. Those are the ones whose prices rose."),
        ("The audit catches the look-ahead", _case_audit,
         "The honest version passes. Both shortcuts rely on filings dated after the month-end, and the audit counts every one."),
        ("A stock whose data ends is sold at its last price", _case_delisting,
         "The loss is realised, not hidden. The engine never invents a worse price, and never lets a dead stock sit in the portfolio forever."),
        ("A pause is not a failure", _case_pause,
         "Only a series that really ends is a delisting. A gap that resumes later is frozen, because a price will come back."),
        ("Today's list never shows the failures", _case_survivors_only,
         "A data source that starts from today's index cannot show you stocks that left it. That is how survivorship bias hides."),
    ]


def run_try_case(n):
    name, builder, lesson = try_cases()[n - 1]
    return name, builder(), lesson


def try_mode(arg=None):
    cases = try_cases()
    pick = None if arg is None else [arg]
    while True:
        if pick is None:
            print("\nSituations that show each bias mechanism:")
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

def demo(edge=config.SYNTH_DEFAULT_EDGE, seed=config.SYNTH_SEED, clean=False, db_path=None,
         reaction=config.SYNTH_DEFAULT_REACTION, n_dead=config.SYNTH_DEFAULT_DEAD, n_seeds=5):
    print("=" * 72)
    print("STAGE 6: bias, or how much do shortcuts inflate a backtest?")
    print("=" * 72)
    ok_all = True
    N, W, C, B = config.DEFAULT_TOP_N, config.DEFAULT_VALUE_WEIGHT, config.DEFAULT_COST_BPS, config.DEFAULT_BUFFER
    judge = n_seeds >= 5
    seeds = list(range(1, n_seeds + 1))

    print("\n--- Step 1: a world with earnings that move prices, and a graveyard ---")
    market, con = stage3_run.build_world(edge=edge, seed=seed, defects=not clean, reaction=reaction, n_dead=n_dead)
    tr = market["truth"]
    print(f"  {len(tr['live_tickers'])} stocks in today's list + {len(tr['dead_tickers'])} that failed along the way; "
          f"a filing moves the price by {reaction:g} x the change in trailing EPS; planted edge {edge * 100:.1f}% a year.")
    for t in tr["dead_tickers"][:3]:
        kind = "abruptly" if tr["crash_days"][t] < 100 else "gradually"
        print(f"  {t} stopped trading on {tr['delist_date'][t]} after falling {1 - 2.718281828 ** -tr['crash'][t]:.0%} over {tr['crash_days'][t]} trading days ({kind})")
    print(f"  a data source that starts from today's list would hand you the {len(tr['live_tickers'])} survivors and none of the failures.")

    print("\n--- Step 2: the look-ahead audit on each P/E version ---")
    counts = {}
    for mode in config.EPS_MODES:
        rows = stage3_run.all_signals(con, tickers=align.default_tickers(con), pe_mode=mode)
        counts[mode] = len(align.audit_lookahead(rows))
        print(f"  {mode:8s} {counts[mode]:6d} look-ahead violations")
    ok = counts["lagged"] == 0 and counts["no_lag"] > 0 and counts["today"] > 0
    ok_all &= ok
    print(f"  {'the honest version is clean and both shortcuts are caught' if ok else 'THE AUDIT DID NOT BEHAVE AS EXPECTED'}")

    print("\n--- Step 3: the bias ladder in this one world ---")
    env = engine.Env(con)
    lines, res = evaluate_lines(con, env, W, n_dead > 0, N, B, C)
    print("\n".join("  " + l if i == 0 else l for i, l in enumerate(lines)))
    print("  One world is one draw. The next steps repeat it, because a single number can mislead in either direction.")

    print(f"\n--- Step 4: the same ladder in {n_seeds} fresh worlds, with and without a link between earnings and prices ---")
    verdicts = {}
    for r_ in (0.0, reaction):
        results = bias.worlds_over_seeds(seeds, edge=edge, reaction=r_, n_dead=0, top_n=N, value_weight=W, buffer=B, cost_bps=C)
        print(f"  earnings {'do NOT move' if r_ == 0 else 'move'} prices (reaction {r_:g}):")
        for key, label in (("inflation_no_lag", "EPS from the quarter end"), ("inflation_today", "today's EPS")):
            line, verdict, _ = seeds_line(label, [x[key] for x in results])
            print(line)
            verdicts[(r_, key)] = verdict
    if judge and reaction > 0:
        ok = (verdicts[(0.0, "inflation_no_lag")] == "within noise" and verdicts[(0.0, "inflation_today")] == "within noise"
              and verdicts[(reaction, "inflation_no_lag")] == "INFLATED" and verdicts[(reaction, "inflation_today")] == "INFLATED")
        ok_all &= ok
        print(f"  {'no link, no bias; with a link, both shortcuts inflate the result' if ok else 'THE BIAS DID NOT BEHAVE AS EXPECTED'}")
    elif not judge:
        print(f"  (with fewer than 5 worlds the verdicts are shown but not checked)")

    print(f"\n--- Step 5: survivorship, by how many stocks fail and by what the strategy buys ---")
    surv = {}
    for label, nd, w in (("blend, 20 failures", 20, W), ("blend, 40 failures", 40, W), ("pure value, 20 failures", 20, 1.0), ("pure momentum, 20 failures", 20, 0.0)):
        results = bias.survivorship_over_seeds(seeds, n_dead=nd, value_weight=w, top_n=N, buffer=B, cost_bps=C, edge=edge, reaction=reaction)
        line, verdict, s = seeds_line(label, [x["survivorship"] for x in results])
        surv[label] = verdict
        print(line + f"   (failed stocks sold at their last price: {sum(x['delisted_sales'] for x in results) / len(results):.0f} per world)")
    if judge:
        ok = surv["pure value, 20 failures"] == "INFLATED"
        ok_all &= ok
        print(f"  {'a value strategy buys the stocks that look cheapest as they fail, so it needs the graveyard most' if ok else 'THE SURVIVORSHIP RESULT WAS NOT AS EXPECTED'}")
    print("  Limit of this fake graveyard: failures here fall only in their last weeks or months. Real failing stocks are often weak")
    print("  for much longer, so real survivorship bias is likely larger than what is measured here. Do not carry these numbers over.")

    print("\n--- Step 6: the headline a careless backtest would report ---")
    naive, honest = res["today"]["net_cagr"], res["all"]["net_cagr"]
    print(f"  naive  (today's EPS, today's list): {naive * 100:6.2f}% a year")
    print(f"  honest (filed EPS, every stock):    {honest * 100:6.2f}% a year")
    print(f"  the shortcuts add {res['total'] * 100:+.2f}% a year, in this one world. Nothing about the strategy changed.")

    print("\n--- Step 7: a failed stock held to the end, at the engine level ---")
    pure = bias.run_variant(con, env, "lagged", "all", N, 1.0, B, C)
    sales = [(p.date, t) for p in pure["periods"] for t in p.delisted]
    prices = env.prices
    print(f"  pure value on every stock: {len(sales)} failed stocks were still held when their data ended")
    for date, t in sales[:3]:
        last = prices.last_date(t)
        dates, ps = prices._dates[t], prices._prices[t]
        peak = max(p for d, p in zip(dates, ps) if (dt.date.fromisoformat(last) - dt.date.fromisoformat(d)).days <= 365)
        print(f"  {t}: last price {prices.last_price(t):.2f} on {last} ({prices.last_price(t) / peak - 1:+.0%} from its peak that year); sold at that price on {date}")
    ok = bool(sales) or n_dead == 0
    ok_all &= ok
    print(f"  arithmetic check on that run against the share ledger: "
          f"{'agrees on every date' if not engine.check_engine(pure['periods'], env.replay(pure['portfolio'], C)) else 'PROBLEM'}")

    print("\n--- Step 8: save the four runs (safe to repeat) ---")
    for key, label in (("lagged", "honest"), ("no_lag", "no filing lag"), ("today", "today's EPS"), ("all", "honest, all stocks")):
        v = res[key]
        run = pf.get_or_create_run(con, N, W, C, B, v["pe_mode"], v["universe"])
        pf.store_holdings(con, run, v["portfolio"])
        first = engine.store_results(con, run, v["periods"])
        again = engine.store_results(con, run, v["periods"])
        same = len(engine.load_results(con, run)) == len(v["periods"])
        idem = again["inserted"] == 0 and again["updated"] == 0 and again["deleted"] == 0
        ok_all &= idem and same
        print(f"  run {run} ({label}: pe_mode {v['pe_mode']}, universe {v['universe']}): {first['inserted']} rows; again: {'unchanged' if idem else 'CHANGED!'}")
    n_runs = con.execute("SELECT COUNT(*) FROM RUN").fetchone()[0]
    ok = n_runs == (4 if n_dead else 3)
    ok_all &= ok
    print(f"  {n_runs} distinct runs in the database {'(as expected)' if ok else '(UNEXPECTED)'}")
    if db_path:
        out = db.connect(db_path)
        con.backup(out)
        out.close()
        print(f"  saved everything to {db_path}")

    print("\n" + "=" * 72)
    print("Everything behaved as designed." if ok_all else "SOMETHING BEHAVED UNEXPECTEDLY: see the lines above.")
    print("Check question: a vendor's backtest uses today's S&P 500 list and P/E ratios that are recomputed with current earnings.")
    print("Name the two biases, say which way each pushes the result, and explain why the MOMENTUM half of our strategy")
    print("would have protected us from one kind of failed stock but not another.")
    con.close()
    return 0 if ok_all else 1


# ---------------------------------------------------------------------------

def open_world(a, reaction=None):
    if a.db:
        import os
        if os.path.exists(a.db):
            con = db.connect(a.db)
            if "STOCK" not in db.table_names(con):
                raise SystemExit(f"{a.db} is not a backtester database")
            db.create_schema(con)
            return con
    r = a.reaction if reaction is None else reaction
    return stage3_run.build_world(edge=a.edge, seed=a.seed, defects=not a.clean, reaction=r, n_dead=a.dead)[1]


def main(argv):
    ap = argparse.ArgumentParser(description="Stage 6: bias")
    ap.add_argument("--demo", action="store_true")
    ap.add_argument("--clean", action="store_true", help="no planted data defects")
    ap.add_argument("--edge", type=float, default=config.SYNTH_DEFAULT_EDGE)
    ap.add_argument("--seed", type=int, default=config.SYNTH_SEED)
    ap.add_argument("--reaction", type=float, default=config.SYNTH_DEFAULT_REACTION)
    ap.add_argument("--dead", type=int, default=config.SYNTH_DEFAULT_DEAD)
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--db", default=None)
    ap.add_argument("--ladder", nargs="*", default=None, metavar="REACTION WEIGHT")
    ap.add_argument("--graveyard", nargs="*", default=None, metavar="TICKER")
    ap.add_argument("--try", dest="try_", nargs="?", const="ask", default=None)
    a = ap.parse_args(argv)

    if not 0 <= a.reaction <= config.SYNTH_MAX_REACTION:
        print(f"--reaction must be between 0 and {config.SYNTH_MAX_REACTION:g}")
        return 1
    if not 0 <= a.dead <= config.SYNTH_MAX_DEAD:
        print(f"--dead must be between 0 and {config.SYNTH_MAX_DEAD}")
        return 1
    if a.seeds < 2:
        print("--seeds must be at least 2 (a spread needs two or more worlds)")
        return 1

    if a.try_ is not None:
        return try_mode(None if a.try_ == "ask" else int(a.try_))
    if a.ladder is not None:
        if len(a.ladder) not in (0, 2):
            print("Use --ladder with two numbers (for example --ladder 2.0 0.5), or with nothing.")
            return 1
        r = w = None
        if a.ladder:
            try:
                r, w = float(a.ladder[0]), float(a.ladder[1])
            except ValueError:
                print("Cannot use those numbers: they must be numbers.")
                return 1
            if not 0 <= r <= config.SYNTH_MAX_REACTION or not 0 <= w <= 1:
                print(f"The reaction must be between 0 and {config.SYNTH_MAX_REACTION:g} and the weight between 0 and 1.")
                return 1
        if a.db and r is not None:
            print("(the earnings reaction is built into a saved database, so the reaction you typed is ignored)")
        return ladder_mode(lambda rr: open_world(a, rr), r, w, a.dead)
    if a.graveyard is not None:
        if len(a.graveyard) not in (0, 1):
            print("Use --graveyard with one ticker (for example --graveyard D001), or with nothing.")
            return 1
        con = open_world(a)
        try:
            return graveyard_mode(con, a.graveyard[0] if a.graveyard else None)
        finally:
            con.close()
    return demo(edge=a.edge, seed=a.seed, clean=a.clean, db_path=a.db, reaction=a.reaction, n_dead=a.dead, n_seeds=a.seeds)


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
