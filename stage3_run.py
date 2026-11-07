"""stage3_run.py: walk through Stage 3 (align and signals).

    python stage3_run.py --demo                      # the full walkthrough, no internet needed
    python stage3_run.py --demo --edge 0.15          # a world with a strong planted edge
    python stage3_run.py --demo --edge 0 --clean     # the null world, no planted defects
    python stage3_run.py --demo --db signals.db      # also save prices, EPS and signals to a file
    python stage3_run.py --explain                   # interactive: pick a stock and a date, see every step
    python stage3_run.py --explain S010 2019-06-28   # the same, in one go
    python stage3_run.py --top                       # interactive: move the value/momentum weight, watch the top 10 change
    python stage3_run.py --top 2019-06-28 0.7        # one list: month-end, weight
    python stage3_run.py --try                       # interactive: ten situations where a signal must NOT be computed
Add --db FILE to --explain / --top to use a database you already have instead of a fresh synthetic world.
"""
import argparse
import datetime as dt
import os
import sys

import numpy as np

import align
import collect
import config
import db
import signals
import synthetic


# ---------------------------------------------------------------------------
# building a world
# ---------------------------------------------------------------------------

def build_world(edge=config.SYNTH_DEFAULT_EDGE, seed=config.SYNTH_SEED, defects=True,
                n_stocks=config.SYNTH_N_STOCKS, value_share=0.5, reaction=0.0, n_dead=0):
    """A synthetic market, collected into an in-memory database. Returns (market, con).
    Failed stocks (n_dead > 0) are collected too, but marked as NOT in today's list (in_sp500 = 0): a real data
    source would not hand them to you, which is exactly what makes survivorship bias hard to see."""
    market = synthetic.make_market(n_stocks=n_stocks, seed=seed, edge=edge, defects=defects, value_share=value_share,
                                   reaction=reaction, n_dead=n_dead)
    con = db.connect(":memory:")
    db.create_schema(con)
    fetchers = synthetic.fake_fetchers(market)
    collect.collect_all(con, fetchers)
    dead = market["truth"]["dead_tickers"]
    if dead:
        names = {t: name for t, name, _, _ in market["stocks"]}
        collect.register_stocks(con, [(t, names[t], 0, 0) for t in dead])
        collect.collect_prices(con, fetchers["prices"], dead, config.DATA_START, config.END_DATE)
        collect.collect_eps(con, fetchers["eps"], dead)
        collect.validate_prices(con)
    return market, con


def all_signals(con, tickers=None, dates=None, pe_mode="lagged"):
    """Panel plus ranks in one call."""
    rows = align.build_panel(con, tickers=tickers, dates=dates, pe_mode=pe_mode)
    return signals.add_ranks(rows)


# ---------------------------------------------------------------------------
# checking against what the synthetic generator stored
# ---------------------------------------------------------------------------

def truth_check(rows, market, rel_tol=1e-3):
    """Compare P/E and momentum from the public feeds with the numbers the generator used.
    Returns a dict. 'differs' lists tickers whose availability or values are not within tolerance,
    with a short description, so the cause can be found."""
    tr = market["truth"]
    t_idx = {t: i for i, t in enumerate(tr["tickers"])}
    m_idx = {d: k for k, d in enumerate(tr["month_ends"])}
    compared = 0
    worst_pe = worst_mom = 0.0
    differs = {}
    for r in rows:
        k, i = m_idx[r.date], t_idx[r.ticker]
        if k < config.MIN_HISTORY_MONTHS:
            continue
        true_pe, true_mom = tr["pe"][k][i], tr["momentum"][k][i]
        have_pe, have_mom = r.pe is not None, r.momentum is not None
        if have_pe != (not np.isnan(true_pe)) or have_mom != (not np.isnan(true_mom)):
            differs.setdefault(r.ticker, set()).add("availability")
            continue
        compared += 1
        if have_pe:
            rel = abs(r.pe / true_pe - 1.0)
            worst_pe = max(worst_pe, rel)
            if rel > rel_tol:
                differs.setdefault(r.ticker, set()).add("P/E value")
        if have_mom:
            diff = abs(r.momentum - true_mom)
            worst_mom = max(worst_mom, diff)
            if diff > rel_tol:
                differs.setdefault(r.ticker, set()).add("momentum value")
    return {"compared": compared, "worst_pe": worst_pe, "worst_mom": worst_mom,
            "differs": {t: sorted(v) for t, v in differs.items()}}


def scrambled_copy(con, cutoff, scramble=("prices", "eps")):
    """A copy of the database in which everything that happens AFTER the cutoff is scrambled:
    each price multiplied by an irregular factor between 1 and 4, and/or earnings flipped in sign and tripled.
    The prices must change IRREGULARLY: multiplying every price by the same number would leave every rank
    unchanged, and a portfolio depends on ranks only, so such a test would prove nothing. The caller closes it."""
    copy = db.connect(":memory:")
    con.backup(copy)
    copy.execute("PRAGMA foreign_keys = ON")
    if "prices" in scramble:
        copy.execute("UPDATE PRICE SET adj_close = adj_close * (1.0 + ((rowid * 7919) % 13) / 4.0) WHERE date > ?", (cutoff,))
    if "eps" in scramble:
        copy.execute("UPDATE FUNDAMENTAL SET eps = -eps * 3 WHERE available_date > ?", (cutoff,))
    copy.commit()
    return copy


def lookahead_experiment(con, cutoff, tickers=None, scramble=("prices", "eps")):
    """THE look-ahead test. Scramble everything that happens AFTER the cutoff in a copy of the database.
    Every signal dated on or before the cutoff must be exactly unchanged. Signals after the cutoff must change
    (otherwise the test proves nothing). `scramble` chooses what to scramble, so each source of possible
    leakage can be tested on its own."""
    before = all_signals(con, tickers)
    scrambled = scrambled_copy(con, cutoff, scramble)
    after = all_signals(scrambled, tickers)
    scrambled.close()

    def key(r):
        return (r.ticker, r.date, r.price, r.pe, r.momentum, r.pe_rank, r.momentum_rank, tuple(r.reasons))
    unchanged_before = changed_before = changed_after = n_before = 0
    for a, b in zip(before, after):
        assert (a.ticker, a.date) == (b.ticker, b.date)
        same = key(a) == key(b)
        if a.date <= cutoff:
            n_before += 1
            unchanged_before += same
            changed_before += not same
        else:
            changed_after += not same
    return {"rows_up_to_cutoff": n_before, "changed_up_to_cutoff": changed_before,
            "changed_after_cutoff": changed_after, "unchanged_up_to_cutoff": unchanged_before}


def information_coefficients(con, rows, value_weight):
    """For each month: the rank correlation between the combined score and the NEXT month's return.
    Positive on average means the score points at tomorrow's winners. (A measurement, not a backtest.)"""
    prices = align.PriceBook(con)
    all_me = align.month_end_dates(con)
    nxt = {d: all_me[i + 1] for i, d in enumerate(all_me[:-1])}
    by_date = {}
    for r in rows:
        if r.date in nxt and r.pe_rank is not None and r.momentum_rank is not None:
            by_date.setdefault(r.date, []).append(r)
    ics = []
    for date, rs in sorted(by_date.items()):
        scores, rets = [], []
        for r in rs:
            q0, _ = prices.quote(r.ticker, date)
            q1, _ = prices.quote(r.ticker, nxt[date])
            if q0 is None or q1 is None:
                continue
            scores.append(signals.combined_score(r, value_weight))
            rets.append(q1.price / q0.price - 1.0)
        if len(scores) >= 20:
            a, b = signals.pct_rank(scores), signals.pct_rank(rets)
            ics.append(float(np.corrcoef(a, b)[0, 1]))
    return ics


def ic_summary(ics):
    """mean IC and its t-statistic (mean / standard error)."""
    x = np.array(ics)
    if len(x) < 3:
        return float("nan"), float("nan")
    mean, sd = float(x.mean()), float(x.std(ddof=1))
    if sd < 1e-12:                                 # every month identical: no spread to measure luck against
        return mean, float("inf") * np.sign(mean) if mean != 0 else 0.0
    return mean, mean / (sd / np.sqrt(len(x)))


# ---------------------------------------------------------------------------
# --explain: one stock, one month, every step
# ---------------------------------------------------------------------------

def resolve_month_end(con, date):
    """The latest month-end on or before `date`, or None."""
    me = [d for d in align.month_end_dates(con) if d <= date]
    return me[-1] if me else None


def explain_lines(con, ticker, date, value_weight=config.DEFAULT_VALUE_WEIGHT):
    """The full working for one stock at one month-end, as printable lines."""
    ticker = ticker.upper()
    if con.execute("SELECT 1 FROM STOCK WHERE ticker = ?", (ticker,)).fetchone() is None:
        return [f"No stock called {ticker} in the database."]
    me = resolve_month_end(con, date)
    if me is None:
        return [f"{date} is before the first month-end in the data."]
    all_me = align.month_end_dates(con)
    k = all_me.index(me)
    eps, prices = align.EpsBook(con), align.PriceBook(con)
    L = []
    L.append(f"{ticker} as of the month-end {me} (month {k + 1} of {len(all_me)})"
             + ("" if me == date else f"   [you asked for {date}; the month-end on or before it is used]"))

    L.append("")
    L.append("1) EARNINGS: what could the world see on that day?")
    rows = eps.rows(ticker)
    visible = [r for r in rows if r[1] <= me]
    hidden = [r for r in rows if r[1] > me]
    used = set(r[0] for r in visible[-config.MIN_EPS_QUARTERS:])
    shown = visible[-6:] + hidden[:2]
    L.append("   quarter end    filed on       EPS      status")
    for period_end, filed, e in shown:
        if filed > me:
            gap = (dt.date.fromisoformat(filed) - dt.date.fromisoformat(me)).days
            status = f"HIDDEN: filed {gap} days after this date, so it cannot be used"
        elif period_end in used:
            status = "used in the trailing twelve months"
        else:
            status = "visible, but older than the four used"
        L.append(f"   {period_end}   {filed}   {e:7.4f}   {status}")
    ttm, why = eps.ttm(ticker, me)
    if ttm is None:
        L.append(f"   -> no trailing twelve months EPS: {why}")
    else:
        parts = " + ".join(f"{r[2]:.4f}" for r in visible[-config.MIN_EPS_QUARTERS:])
        L.append(f"   -> TTM EPS = {parts} = {ttm.value:.4f}   (latest filing used: {ttm.latest_filed})")
        if why:
            L.append(f"   -> but: {why}")

    L.append("")
    L.append("2) PRICE on the month-end")
    quote, qwhy = prices.quote(ticker, me)
    if quote is None:
        L.append(f"   -> no usable price: {qwhy}")
    else:
        carried = "" if quote.carried_days == 0 else f"  (carried forward {quote.carried_days} day(s) from {quote.price_date})"
        L.append(f"   -> {quote.price:.4f}{carried}")

    L.append("")
    L.append("3) P/E = price / TTM EPS")
    pe = quote.price / ttm.value if (quote is not None and ttm is not None and not why) else None
    L.append(f"   -> {quote.price:.4f} / {ttm.value:.4f} = {pe:.2f}" if pe is not None else "   -> not computed (see above)")

    L.append("")
    L.append("4) MOMENTUM = price a month ago / price 12 months ago - 1   (the latest month is skipped on purpose)")
    if k < config.MIN_HISTORY_MONTHS:
        L.append(f"   -> {align.REASON_HISTORY}")
        mom = None
    else:
        d1, d12 = all_me[k - config.MOMENTUM_SKIP_MONTHS], all_me[k - config.MOMENTUM_LOOKBACK_MONTHS]
        q1, w1 = prices.quote(ticker, d1)
        q12, w12 = prices.quote(ticker, d12)
        L.append(f"   price a month ago   ({d1}): " + (f"{q1.price:.4f}" if q1 else f"unusable, {w1}"))
        L.append(f"   price 12 months ago ({d12}): " + (f"{q12.price:.4f}" if q12 else f"unusable, {w12}"))
        mom = q1.price / q12.price - 1.0 if (q1 and q12) else None
        L.append(f"   -> {mom * 100:+.1f}%" if mom is not None else "   -> not computed (see above)")

    L.append("")
    L.append("5) RANKS against the other stocks on the same day")
    panel = all_signals(con, dates=[me])
    mine = next((r for r in panel if r.ticker == ticker), None)
    n_pe = sum(1 for r in panel if r.pe is not None)
    n_mom = sum(1 for r in panel if r.momentum is not None)
    if mine is None or mine.pe_rank is None:
        L.append("   P/E rank:      none")
    else:
        L.append(f"   P/E rank:      {mine.pe_rank:.2f}   (cheaper than {mine.pe_rank * 100:.0f}% of the {n_pe} stocks with a P/E)")
    if mine is None or mine.momentum_rank is None:
        L.append("   momentum rank: none")
    else:
        L.append(f"   momentum rank: {mine.momentum_rank:.2f}   (stronger than {mine.momentum_rank * 100:.0f}% of the {n_mom} stocks with momentum)")
    if mine is not None and mine.eligible:
        score = signals.combined_score(mine, value_weight)
        order = signals.scores_for_date(panel, me, value_weight)
        place = [r.ticker for _, r in order].index(ticker) + 1
        L.append(f"   combined score at weight {value_weight:.2f} on value: {score:.3f}   (place {place} of {len(order)} eligible)")
    else:
        L.append("   -> NOT ELIGIBLE this month" + (": " + "; ".join(mine.reasons) if mine and mine.reasons else ""))
    return L


def explain_mode(con, ticker=None, date=None):
    if ticker and date:
        print("\n".join(explain_lines(con, ticker, date)))
        return 0
    tickers = align.default_tickers(con)
    me = align.month_end_dates(con)
    print(f"Pick a stock (for example {tickers[0]}..{tickers[-1]}) and a date between {me[0]} and {me[-1]}. Enter to quit.")
    while True:
        try:
            t = input("stock> ").strip()
            if not t:
                break
            d = input(f"date (YYYY-MM-DD, Enter for {me[-13]})> ").strip() or me[-13]
        except EOFError:
            break
        try:
            dt.date.fromisoformat(d)
        except ValueError:
            print("  That is not a date like 2019-06-28.")
            continue
        print()
        print("\n".join(explain_lines(con, t, d)))
        print()
    return 0


# ---------------------------------------------------------------------------
# --top: move the weight, watch the list change
# ---------------------------------------------------------------------------

def top_lines(rows, date, weight, n=10):
    order = signals.scores_for_date(rows, date, weight)
    L = [f"Top {n} of {len(order)} eligible stocks on {date}, weight {weight:.2f} on value "
         f"({1 - weight:.2f} on momentum):",
         "   place  stock   score   P/E rank  mom rank     P/E   momentum"]
    for i, (score, r) in enumerate(order[:n], 1):
        L.append(f"   {i:4d}   {r.ticker:6s}  {score:.3f}   {r.pe_rank:7.2f}  {r.momentum_rank:8.2f}  {r.pe:6.1f}   {r.momentum * 100:+7.1f}%")
    return L, [r.ticker for _, r in order[:n]]


def top_mode(con, date=None, weight=None, n=10):
    me = align.month_end_dates(con)
    date = resolve_month_end(con, date) if date else me[-13]
    if date is None:
        print("That date is before the first month-end in the data.")
        return 1
    rows = all_signals(con, dates=[date])
    if weight is not None:
        lines, _ = top_lines(rows, date, weight, n)
        print("\n".join(lines))
        return 0
    print(f"Month-end {date}. Type a weight from 0 (all momentum) to 1 (all value). Enter to quit.")
    previous = None
    while True:
        try:
            raw = input("weight> ").strip()
        except EOFError:
            break
        if not raw:
            break
        try:
            w = float(raw)
            if not 0 <= w <= 1:
                raise ValueError
        except ValueError:
            print("  Please type a number between 0 and 1, for example 0.7.")
            continue
        lines, names = top_lines(rows, date, w, n)
        print("\n".join(lines))
        if previous is not None:
            print(f"   {len(set(names) - set(previous))} of the {n} names are new compared with the last list.")
        previous = names
    return 0


# ---------------------------------------------------------------------------
# --try: situations where a signal must NOT be computed
# ---------------------------------------------------------------------------

QUARTERS = ["2017-03-31", "2017-06-30", "2017-09-30", "2017-12-31", "2018-03-31", "2018-06-30",
            "2018-09-30", "2018-12-31", "2019-03-31", "2019-06-30", "2019-09-30"]


def _business_days(start, end):
    d, last = dt.date.fromisoformat(start), dt.date.fromisoformat(end)
    while d <= last:
        if d.weekday() < 5:
            yield d.isoformat()
        d += dt.timedelta(days=1)


def default_eps(negative_from=None):
    rows = []
    for i, q in enumerate(QUARTERS):
        filed = (dt.date.fromisoformat(q) + dt.timedelta(days=40)).isoformat()
        e = round(1.0 + 0.1 * i, 4)
        if negative_from and q >= negative_from:
            e = -0.5
        rows.append((q, filed, e))
    return rows


def mini_world(eps_rows=None, drop=(), override=None, validate=False):
    """A tiny, fully hand-controlled world: one stock AAA and a benchmark, daily prices 2018-2019."""
    con = db.connect(":memory:")
    db.create_schema(con)
    collect.register_stocks(con, [("AAA", "Test Corp", 1, 0), (config.BENCHMARK_TICKER, "Benchmark", 0, 1)])
    for i, d in enumerate(_business_days("2018-01-01", "2019-12-31")):
        con.execute("INSERT INTO PRICE (ticker, date, adj_close) VALUES (?, ?, ?)",
                    (config.BENCHMARK_TICKER, d, round(100 * 1.0003 ** i, 4)))
        if d in drop:
            continue
        price = (override or {}).get(d, round(50 * 1.0004 ** i, 4))
        con.execute("INSERT INTO PRICE (ticker, date, adj_close) VALUES (?, ?, ?)", ("AAA", d, price))
    for q, filed, e in (default_eps() if eps_rows is None else eps_rows):
        con.execute("INSERT INTO FUNDAMENTAL (ticker, period_end, available_date, eps) VALUES ('AAA', ?, ?, ?)", (q, filed, e))
    con.commit()
    if validate:
        collect.validate_prices(con)
    return con


def _range(a, b):
    return [d for d in _business_days(a, b)]


def try_cases():
    base = default_eps()
    late = [(q, "2019-07-10" if q == "2019-03-31" else f, e) for q, f, e in base]
    return [
        ("A healthy stock",
         dict(), ["2019-06-28"],
         "Everything is available, so both signals are computed."),
        ("A loss-making company (trailing EPS below zero)",
         dict(eps_rows=default_eps(negative_from="2018-06-30")), ["2019-06-28"],
         "A negative P/E means nothing, so the stock is excluded that month (your choice in the design)."),
        ("The latest quarter is filed AFTER the month-end",
         dict(eps_rows=late), ["2019-06-28", "2019-07-31"],
         "On 28 June the quarter is not public yet, so the older four quarters are used. By 31 July it is, and the number changes. This is look-ahead protection."),
        ("One quarter is missing from the filings",
         dict(eps_rows=[r for r in base if r[0] != "2018-12-31"]), ["2019-06-28"],
         "Four quarters that are not consecutive would cover 15 months, not 12, so the TTM is refused."),
        ("The company stopped filing months ago",
         dict(eps_rows=[r for r in base if r[0] <= "2018-09-30"]), ["2019-06-28"],
         "The latest quarter is 271 days old. Old news is not a basis for a P/E."),
        ("Fewer than four quarters on file",
         dict(eps_rows=[r for r in base if "2018-09-30" <= r[0] <= "2019-03-31"]), ["2019-06-28"],
         "TTM needs four quarters."),
        ("The month-end price is missing, but one from 2 days earlier exists",
         dict(drop=("2019-06-27", "2019-06-28")), ["2019-06-28"],
         "Carried forward (limit: 5 days). The row says how many days, so you can count how often it happens."),
        ("The month-end price is missing and the last one is 9 days old",
         dict(drop=tuple(_range("2019-06-20", "2019-06-28"))), ["2019-06-28"],
         "Too old to carry, so there is no P/E that month. A momentum date that falls here would also be refused."),
        ("A flagged price (a 10x spike) sits exactly on the month-end",
         dict(override={"2019-06-28": 3100.0}, validate=True), ["2019-06-28"],
         "The validator flagged it in Stage 2, and the signals refuse to use it. The flag has consequences."),
        ("A date with less than 12 months of price history",
         dict(), ["2018-10-31"],
         "Momentum needs 12 earlier month-ends. P/E can still be computed, but the stock is not eligible."),
    ]


def describe_row(r):
    pe = f"P/E {r.pe:.1f} (price {r.price:.2f} / TTM EPS {r.ttm_eps:.2f}, latest filing used {r.eps_filed})" if r.pe is not None else "P/E none"
    if r.price is not None and r.carried_days:
        pe += f" [price carried {r.carried_days} day(s) from {r.price_date}]"
    mom = f"momentum {r.momentum * 100:+.1f}%" if r.momentum is not None else "momentum none"
    verdict = "ELIGIBLE" if r.eligible else "EXCLUDED: " + "; ".join(r.reasons)
    return f"  {r.date}: {pe}\n              {mom}  ->  {verdict}"


def run_try_case(n):
    name, kwargs, dates, lesson = try_cases()[n - 1]
    con = mini_world(**kwargs)
    rows = align.build_panel(con, dates=dates)
    text = "\n".join(describe_row(r) for r in rows)
    return name, text, lesson


def try_mode(arg=None):
    cases = try_cases()
    pick = None if arg is None else [arg]
    while True:
        if pick is None:
            print("\nSituations where a signal needs care:")
            for i, c in enumerate(cases, 1):
                print(f"  {i:2d}. {c[0]}")
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

def demo(edge=config.SYNTH_DEFAULT_EDGE, seed=config.SYNTH_SEED, clean=False, value_share=0.5, db_path=None):
    print("=" * 72)
    print("STAGE 3: align and signals")
    print("=" * 72)
    ok_all = True

    print("\n--- Step 1: a synthetic world, collected through the Stage 2 pipeline ---")
    market, con = build_world(edge=edge, seed=seed, defects=not clean, value_share=value_share)
    tr = market["truth"]
    print(f"  {len(tr['tickers'])} stocks, planted edge {edge * 100:.1f}% a year, "
          f"{'with' if market['defects'] else 'without'} planted data defects")

    print("\n--- Step 2: one stock, one month, every step (the same output as --explain) ---")
    me = resolve_month_end(con, "2019-06-30")
    print("\n".join("  " + line for line in explain_lines(con, "S010", me)))

    print("\n--- Step 3: the whole panel: every stock at every month-end ---")
    rows = all_signals(con)
    s = align.summarize(rows)
    print(f"  {s['rows']} stock-months, {s['eligible']} eligible ({s['eligible'] / s['rows'] * 100:.1f}%), "
          f"{s['carried']} used a carried-forward price")
    print("  why the others were not usable:")
    for why, n in s["reasons"].most_common():
        print(f"     {n:5d}  {why}")

    print("\n--- Step 4: what did the planted data defects do to the signals? ---")
    touched = {}
    for r in rows:
        extra = [w for w in r.reasons if w != align.REASON_HISTORY and w != align.REASON_LOSS]
        if extra or r.carried_days:
            touched.setdefault(r.ticker, []).append(r)
    if market["defects"]:
        for d in market["defects"]:
            hit = touched.get(d["ticker"], [])
            if hit:
                what = sorted({(w if w else "price carried forward") for r in hit for w in (r.reasons or [None])})
                print(f"  {d['kind']:18s} {d['ticker']}: {len(hit)} stock-month(s) affected ({'; '.join(what)})")
            else:
                print(f"  {d['kind']:18s} {d['ticker']}: no effect on any signal (the bad row did not fall on a month-end date)")
    else:
        print("  no defects planted")

    print("\n--- Step 5: do the signals match what the generator knew? ---")
    check = truth_check(rows, market)
    explained = set(touched)
    unexplained = {t: v for t, v in check["differs"].items() if t not in explained}
    print(f"  compared {check['compared']} stock-months: worst P/E difference {check['worst_pe']:.1e}, "
          f"worst momentum difference {check['worst_mom']:.1e}")
    print("  (the feed rounds prices to 4 decimals, so differences around 1e-5 are expected)")
    for t, v in sorted(check["differs"].items()):
        print(f"  {t} differs ({', '.join(v)})" + (": explained by a planted defect above" if t in explained else ": UNEXPLAINED"))
    print(f"  {'Every difference is explained.' if not unexplained else 'There are UNEXPLAINED differences!'}")
    ok_all = ok_all and not unexplained

    print("\n--- Step 6: the look-ahead test ---")
    cutoff = resolve_month_end(con, "2019-06-30")
    ex = lookahead_experiment(con, cutoff)
    print(f"  We scramble everything after {cutoff}: prices multiplied by an irregular 1 to 4, earnings flipped and tripled.")
    print(f"  signals dated up to {cutoff}: {ex['changed_up_to_cutoff']} of {ex['rows_up_to_cutoff']} changed")
    print(f"  signals dated after:           {ex['changed_after_cutoff']} changed (proof that the scramble really bit)")
    clean_ok = ex["changed_up_to_cutoff"] == 0 and ex["changed_after_cutoff"] > 0
    audit = align.audit_lookahead(rows)
    print(f"  audit of every date each row relied on: {len(audit)} violation(s)")
    ok_all = ok_all and clean_ok and not audit
    print(f"  {'No signal looks into the future.' if clean_ok and not audit else 'LOOK-AHEAD DETECTED!'}")

    print("\n--- Step 7: do the signals point at next month's winners? (the planted edge) ---")
    print("  Each month: rank correlation between the score and the NEXT month's return (the 'IC').")
    for label, w in (("value only", 1.0), ("momentum only", 0.0), ("blend 50/50", 0.5)):
        mean, t = ic_summary(information_coefficients(con, rows, w))
        print(f"     {label:14s} mean IC {mean:+.3f}   t-statistic {t:+.1f}")
    print("  A t-statistic beyond about +-2 is hard to explain by luck. With a 5% edge and 11 years of data")
    print("  it is borderline. Try --edge 0.15 (clear) or --edge 0 (nothing).")

    print("\n--- Step 8: save the signals (safe to repeat) ---")
    first = signals.store_signals(con, rows)
    again = signals.store_signals(con, rows)
    idem = again["inserted"] == 0 and again["updated"] == 0
    ok_all = ok_all and idem
    print(f"  first run:  {first}")
    print(f"  second run: {again}   {'idempotent' if idem else 'NOT idempotent!'}")
    if db_path:
        out = db.connect(db_path)
        con.backup(out)
        out.close()
        print(f"  saved prices, EPS and signals to {db_path}")

    print("\n" + "=" * 72)
    print("Everything behaved as designed." if ok_all else "SOMETHING BEHAVED UNEXPECTEDLY: see the lines above.")
    print("Check question: on 28 June, a company's results for the quarter ended 31 March are not public yet")
    print("(they come out on 10 July). Which EPS figures should the backtester use for 28 June, and why would")
    print("using the 10 July figure for 28 June make the strategy look better than it really was?")
    con.close()
    return 0 if ok_all else 1


# ---------------------------------------------------------------------------

def open_world(a):
    if a.db and os.path.exists(a.db):
        con = db.connect(a.db)
        if "STOCK" not in db.table_names(con):
            raise SystemExit(f"{a.db} is not a backtester database")
        return con
    return build_world(edge=a.edge, seed=a.seed, defects=not a.clean)[1]


def main(argv):
    ap = argparse.ArgumentParser(description="Stage 3: align and signals")
    ap.add_argument("--demo", action="store_true")
    ap.add_argument("--clean", action="store_true", help="no planted data defects")
    ap.add_argument("--edge", type=float, default=config.SYNTH_DEFAULT_EDGE)
    ap.add_argument("--seed", type=int, default=config.SYNTH_SEED)
    ap.add_argument("--value-share", type=float, default=0.5)
    ap.add_argument("--db", default=None)
    ap.add_argument("--explain", nargs="*", default=None, metavar="TICKER DATE")
    ap.add_argument("--top", nargs="*", default=None, metavar="DATE WEIGHT")
    ap.add_argument("--try", dest="try_", nargs="?", const="ask", default=None)
    a = ap.parse_args(argv)

    if a.try_ is not None:
        return try_mode(None if a.try_ == "ask" else int(a.try_))
    if a.explain is not None:
        if len(a.explain) not in (0, 2):
            print("Use --explain with a stock and a date (for example --explain S010 2019-06-28), or with nothing.")
            return 1
        con = open_world(a)
        try:
            return explain_mode(con, *a.explain) if a.explain else explain_mode(con)
        finally:
            con.close()
    if a.top is not None:
        if len(a.top) not in (0, 2):
            print("Use --top with a date and a weight (for example --top 2019-06-28 0.7), or with nothing.")
            return 1
        con = open_world(a)
        try:
            return top_mode(con, a.top[0], float(a.top[1])) if a.top else top_mode(con)
        finally:
            con.close()
    return demo(edge=a.edge, seed=a.seed, clean=a.clean, value_share=a.value_share, db_path=a.db)


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
