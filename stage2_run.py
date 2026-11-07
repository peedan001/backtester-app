"""stage2_run.py: walk through Stage 2 (collect, with a synthetic market that has a planted edge).

    python stage2_run.py --demo                 # the full walkthrough, no internet needed
    python stage2_run.py --demo --clean         # same, with no planted data defects
    python stage2_run.py --demo --edge 0        # the null world: no edge planted at all
    python stage2_run.py --demo --db backtest.db   # also save the synthetic world to a file
    python stage2_run.py --explore              # interactive: pick an edge size, see how visible it is
    python stage2_run.py --try                  # interactive: break the data feed, watch the collector cope
    python stage2_run.py --check                # LIVE (your computer): test each real data source alone
    python stage2_run.py --live --limit 25      # LIVE (your computer): collect real data for 25 companies

Options for --check and --live:  --universe-csv F  --prices-csv F  --eps-csv F  (CSV backups)
"""
import argparse
import datetime as dt
import sys

import numpy as np

import collect
import config
import db
import synthetic


# ---------------------------------------------------------------------------
# checking the planted defects against what the collector did
# ---------------------------------------------------------------------------

def verify_defects(con, market, reports):
    """For each planted defect, did the pipeline do what it should? Returns [(defect, ok, what_we_saw)]."""
    out = []
    for d in market["defects"]:
        t, kind = d["ticker"], d["kind"]
        row_date = d.get("row_date", d["date"])
        has_price = con.execute("SELECT 1 FROM PRICE WHERE ticker=? AND date=?", (t, row_date)).fetchone() is not None
        flagged = collect.flagged_rows(con, t)
        if kind == "spike":
            ok = len(flagged) >= 2 and any("jump" in r[3] for r in flagged) and has_price
            seen = f"{len(flagged)} rows flagged, row kept" if ok else "not flagged as expected"
        elif kind == "gap":
            ok = any("gap" in r[3] for r in flagged)
            seen = f"flagged: {[r[3] for r in flagged][0]}" if ok else "gap not flagged"
        elif kind in ("zero_price", "text_price", "infinite_price", "bad_date"):
            ok = not has_price and reports["prices"].rejected_total >= 1
            seen = "row refused, not stored" if ok else "row was stored (bad!)"
        elif kind == "duplicate_rows":
            n = con.execute("SELECT COUNT(*) FROM PRICE WHERE ticker=? AND date=?", (t, row_date)).fetchone()[0]
            ok = n == 1 and reports["prices"].duplicates_in_feed >= 5
            seen = "stored once, repeats noticed" if ok else "duplicates mishandled"
        elif kind == "eps_before_period":
            row = con.execute("SELECT 1 FROM FUNDAMENTAL WHERE ticker=? AND period_end=?", (t, d["date"])).fetchone()
            ok = row is None and "EPS filed before its quarter ended" in reports["eps"].rejected
            seen = "row refused (would be look-ahead)" if ok else "bad EPS row was stored (bad!)"
        else:
            ok, seen = False, "unknown defect kind"
        out.append((d, ok, seen))
    return out


def quintile_line(market):
    sp = synthetic.planted_quintile_spread(market)
    return f"{sp * 100:+.1f}% a year"


# ---------------------------------------------------------------------------
# --demo
# ---------------------------------------------------------------------------

def demo(edge=config.SYNTH_DEFAULT_EDGE, seed=config.SYNTH_SEED, value_share=0.5, clean=False, db_path=":memory:"):
    print("=" * 72)
    print("STAGE 2: collect (with a synthetic market that has a planted edge)")
    print("=" * 72)

    print("\n--- Step 1: build a fake market where we KNOW the answer ---")
    market = synthetic.make_market(edge=edge, seed=seed, value_share=value_share, defects=not clean)
    tr = market["truth"]
    print(f"  {len(tr['tickers'])} fake stocks + a benchmark, {len(tr['days'])} trading days "
          f"({tr['days'][0]} to {tr['days'][-1]}), seed {seed}")
    print(f"  planted edge: {edge * 100:.1f}% a year for a stock with a perfect score "
          f"(value part {tr['value_edge'] * 100:.1f}%, momentum part {tr['momentum_edge'] * 100:.1f}%)")
    print("  rule: each month, stocks with a low P/E and strong past momentum get extra drift next month.")
    print("        The ranks use only information known at that month-end. No peeking.")
    print(f"  proof the edge is in the data: holding the best-scoring fifth minus the worst fifth earned "
          f"{quintile_line(market)}")
    null = synthetic.make_market(edge=0.0, seed=seed, value_share=value_share, defects=False)
    print(f"  same stocks with NO edge planted:                                          {quintile_line(null)}")
    print("  (single runs are noisy: about +-3% a year. Try --explore to see how noise hides small edges.)")

    if market["defects"]:
        print("\n--- Step 2: we also broke the raw data on purpose ---")
        for i, d in enumerate(market["defects"], 1):
            print(f"  {i}. {d['kind']:18s} {d['ticker']}: {d['detail']}")
            print(f"     should be: {d['expected']}")
    else:
        print("\n--- Step 2: clean data (no defects planted) ---")

    print("\n--- Step 3: collect it into the database ---")
    con = db.connect(db_path)
    db.create_schema(con)
    res = collect.collect_all(con, synthetic.fake_fetchers(market))
    print(res["prices"].summary())
    print(res["eps"].summary())
    print(f"  validator flagged: {dict(res['flags']) or 'nothing'}  (flagged rows are KEPT, never deleted)")

    ok_all = True
    if market["defects"]:
        print("\n--- Step 4: scoreboard, did the pipeline treat each defect correctly? ---")
        for d, ok, seen in verify_defects(con, market, res):
            ok_all = ok_all and ok
            print(f"  [{'OK' if ok else 'XX'}] {d['kind']:18s} {d['ticker']}: {seen}")
    else:
        bad = res["prices"].rejected_total + res["eps"].rejected_total + sum(res["flags"].values())
        ok_all = bad == 0
        print(f"\n--- Step 4: clean data should produce no rejections and no flags: "
              f"{'confirmed' if ok_all else f'{bad} unexpected problems'} ---")

    print("\n--- Step 5: run the collector AGAIN (it must be safe to repeat) ---")
    res2 = collect.collect_all(con, synthetic.fake_fetchers(market))
    idem = res2["prices"].inserted == 0 and res2["prices"].updated == 0 and \
        res2["eps"].inserted == 0 and res2["eps"].updated == 0
    ok_all = ok_all and idem
    print(f"  prices: {res2['prices'].inserted} inserted, {res2['prices'].updated} updated, "
          f"{res2['prices'].unchanged} unchanged;  EPS: {res2['eps'].inserted} inserted, "
          f"{res2['eps'].updated} updated, {res2['eps'].unchanged} unchanged")
    print(f"  {'Idempotent: nothing new, nothing doubled.' if idem else 'NOT idempotent!'}")

    if db_path != ":memory:":
        print(f"\n  Saved the synthetic world to {db_path} (Stage 3 and later can read it).")

    print("\n" + "=" * 72)
    print("Everything behaved as designed." if ok_all else "SOMETHING BEHAVED UNEXPECTEDLY: see the XX lines.")
    print("Check question: in the null world (edge 0) the best fifth still sometimes beats the worst fifth.")
    print("Why is one run of a backtest never enough to prove a strategy works?")
    return 0 if ok_all else 1


# ---------------------------------------------------------------------------
# --explore
# ---------------------------------------------------------------------------

def explore_one(edge, seeds=range(1, 9), value_share=0.5):
    vals = [synthetic.planted_quintile_spread(
        synthetic.make_market(seed=s, edge=edge, value_share=value_share, defects=False)) for s in seeds]
    vals = np.array(vals)
    print(f"\n  planted edge {edge * 100:.1f}% a year, {len(vals)} different fake worlds (10 years of data each):")
    print("  best-fifth minus worst-fifth, per world: " + "  ".join(f"{v * 100:+5.1f}%" for v in vals))
    print(f"  average {vals.mean() * 100:+.1f}%   lowest {vals.min() * 100:+.1f}%   highest {vals.max() * 100:+.1f}%   "
          f"worlds where it looked NEGATIVE: {int((vals < 0).sum())} of {len(vals)}")
    if edge == 0:
        print("  No edge planted, yet some worlds look positive and some negative. That is pure luck.")
    elif vals.min() < 0:
        print("  Even with a real edge, some worlds show a loss. A single backtest can lie in both directions.")
    else:
        print("  The edge is big enough that every world shows it. Real edges are rarely this loud.")
    return vals


def explore(arg=None):
    if arg is not None:
        explore_one(arg)
        return 0
    print("Pick a planted edge (a yearly rate, e.g. 0, 0.02, 0.05, 0.10). I will build 8 fake worlds with that edge")
    print("and show how clearly (or badly) it shows up. Press Enter to quit.")
    while True:
        try:
            raw = input("edge> ").strip()
        except EOFError:
            break
        if not raw:
            break
        try:
            edge = float(raw)
            if not 0 <= edge <= 0.5:
                raise ValueError
        except ValueError:
            print("  Please type a number between 0 and 0.5 (for example 0.05 means 5% a year).")
            continue
        explore_one(edge)
    return 0


# ---------------------------------------------------------------------------
# --try : break the feed, watch the collector cope
# ---------------------------------------------------------------------------

def _clean_prices(n=30, start=dt.date(2020, 1, 6)):
    rows, d, p = [], start, 100.0
    while len(rows) < n:
        if d.weekday() < 5:
            rows.append((d.isoformat(), round(p, 4)))
            p *= 1.001
        d += dt.timedelta(days=1)
    return rows


def _clean_eps():
    return [("2019-03-31", "2019-05-10", 1.0), ("2019-06-30", "2019-08-09", 1.1),
            ("2019-09-30", "2019-11-08", 1.2), ("2019-12-31", "2020-02-14", 1.3)]


def _case(name, lesson, prices=None, eps=None, boom=False, empty=False):
    return {"name": name, "lesson": lesson, "prices": prices, "eps": eps, "boom": boom, "empty": empty}


def _try_cases():
    c = _clean_prices()

    def change(i, value, date=None):
        r = list(c)
        r[i] = (date or r[i][0], value)
        return r
    spike = change(12, c[12][1] * 10)
    gap = c[:10] + c[20:]
    dup = c + [c[5], c[6]]
    eps_bad = _clean_eps()
    eps_bad[1] = ("2019-06-30", "2019-06-01", 1.1)
    return [
        _case("A price is 10 times too big for one day",
              "Kept but FLAGGED on the jump in and the jump out. A real crash can look the same, so a human decides.",
              prices=spike),
        _case("A price of 0.0",
              "Refused by the database (a zero price would break every return calculation).",
              prices=change(7, 0.0)),
        _case("The text 'N/A' where a price should be",
              "Refused. A loosely typed database would have stored the text and broken the maths later.",
              prices=change(7, "N/A")),
        _case("An infinite price",
              "Refused. Infinity passes 'bigger than 0' but ruins every average.",
              prices=change(7, float("inf"))),
        _case("An impossible date (2020-13-45)",
              "Refused. A bad date would break every date comparison.",
              prices=change(7, 100.0, date="2020-13-45")),
        _case("Two rows repeated inside the feed",
              "Stored once. The repeats are counted so you know the source is messy.",
              prices=dup),
        _case("Ten days of prices missing",
              "Kept and FLAGGED: the first price after the gap says how long it was.",
              prices=gap),
        _case("EPS 'filed' before its quarter ended",
              "Refused. This is exactly the look-ahead bias the whole project is built to avoid.",
              prices=c, eps=eps_bad),
        _case("One ticker's download crashes (the other is fine)",
              "The crash is recorded for that ticker only. The good ticker still loads: one failure never stops the rest.",
              prices=c, boom=True),
        _case("A ticker returns nothing at all",
              "Recorded as a failure ('empty feed'), never silently treated as success.",
              prices=c, empty=True),
    ]


def run_try_case(case):
    """Run one broken feed through the collector. Returns the text to print."""
    con = db.connect(":memory:")
    db.create_schema(con)
    collect.register_stocks(con, [("AAA", "Test A", 1, 0), ("BBB", "Test B", 1, 0)])
    clean = _clean_prices()
    lines = []

    def fetch_prices(ticker, start, end):
        if ticker == "AAA":
            if case["boom"]:
                raise ConnectionError("the server hung up")
            if case["empty"]:
                return []
            return case["prices"]
        return clean

    def fetch_eps(ticker):
        return case["eps"] if (ticker == "AAA" and case["eps"]) else _clean_eps()

    pr = collect.collect_prices(con, fetch_prices, ["AAA", "BBB"], "2019-01-01", "2021-12-31")
    ep = collect.collect_eps(con, fetch_eps, ["AAA", "BBB"])
    flags = collect.validate_prices(con)
    lines.append(pr.summary())
    if case["eps"]:
        lines.append(ep.summary())
    fl = collect.flagged_rows(con, "AAA")
    lines.append(f"  flagged on AAA: {[(r[1], r[3]) for r in fl] or 'nothing'}")
    n_a = con.execute("SELECT COUNT(*) FROM PRICE WHERE ticker='AAA'").fetchone()[0]
    n_b = con.execute("SELECT COUNT(*) FROM PRICE WHERE ticker='BBB'").fetchone()[0]
    lines.append(f"  rows stored: AAA {n_a}, BBB {n_b}  (BBB had a perfectly clean feed)")
    return "\n".join(lines)


def try_mode(arg=None):
    cases = _try_cases()
    pick = None if arg is None else [arg]
    while True:
        if pick is None:
            print("\nBreak the data feed (the collector should cope with each one):")
            for i, c in enumerate(cases, 1):
                print(f"  {i:2d}. {c['name']}")
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
        case = cases[n - 1]
        print(f"\n[{n}] {case['name']}")
        print(run_try_case(case))
        print(f"  why: {case['lesson']}")
        if pick is not None and not pick:
            break
    return 0


# ---------------------------------------------------------------------------
# --check and --live (need the internet: run these on YOUR computer)
# ---------------------------------------------------------------------------

def build_live_fetchers(universe_csv=None, prices_csv=None, eps_csv=None):
    """Real fetchers, each with its CSV file as the backup when one is given."""
    universe = collect.fetch_sp500_wikipedia
    prices = collect.make_yahoo_prices_fetcher()
    eps = collect.make_edgar_eps_fetcher()
    if universe_csv or prices_csv or eps_csv:
        csv_f = collect.make_csv_fetchers(universe_csv, prices_csv, eps_csv)
        if universe_csv:
            universe = collect.with_backup(universe, csv_f["universe"])
        if prices_csv:
            prices = collect.with_backup(prices, csv_f["prices"])
        if eps_csv:
            eps = collect.with_backup(eps, csv_f["eps"])
    return {"universe": universe, "prices": prices, "eps": eps}


def check_mode(fetchers):
    print("Testing each live data source on its own (one ticker, a few seconds each)...\n")
    results = collect.run_checks(fetchers)
    for r in results:
        print(f"  [{'OK' if r['ok'] else 'FAIL'}] {r['name']:9s} {r['message']}")
        if r["fix"]:
            print(f"         fix: {r['fix']}")
    all_ok = all(r["ok"] for r in results)
    print("\nAll sources work. Next:  python stage2_run.py --live --limit 25" if all_ok
          else "\nFix the FAIL lines above, then run --check again.")
    return 0 if all_ok else 1


def live_mode(fetchers, db_path, limit):
    con = db.connect(db_path)
    db.create_schema(con)

    def progress(i, total, ticker):
        print(f"\r  {i}/{total} {ticker:8s}", end="", flush=True)

    print(f"Collecting real data into {db_path}" + (f" (first {limit} companies)" if limit else "") + "...")
    res = collect.collect_all(con, fetchers, limit=limit, progress=progress)
    print("\n")
    print(res["prices"].summary())
    print(res["eps"].summary())
    print(f"validator flagged: {dict(res['flags']) or 'nothing'}")
    rows = collect.flagged_rows(con)
    for r in rows[:15]:
        print(f"   flagged {r[0]} {r[1]} {r[2]:.2f}  {r[3]}")
    if len(rows) > 15:
        print(f"   ... and {len(rows) - 15} more flagged rows (kept, never deleted)")
    print("\nReview the failures and flags above. Real data is messy: flagged rows are for YOU to look at.")
    return 0


# ---------------------------------------------------------------------------

def main(argv):
    ap = argparse.ArgumentParser(description="Stage 2: collect")
    ap.add_argument("--demo", action="store_true")
    ap.add_argument("--clean", action="store_true", help="no planted data defects")
    ap.add_argument("--edge", type=float, default=config.SYNTH_DEFAULT_EDGE)
    ap.add_argument("--seed", type=int, default=config.SYNTH_SEED)
    ap.add_argument("--value-share", type=float, default=0.5)
    ap.add_argument("--db", default=None)
    ap.add_argument("--explore", nargs="?", const="ask", default=None)
    ap.add_argument("--try", dest="try_", nargs="?", const="ask", default=None)
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--live", action="store_true")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--universe-csv", default=None)
    ap.add_argument("--prices-csv", default=None)
    ap.add_argument("--eps-csv", default=None)
    a = ap.parse_args(argv)

    if a.explore is not None:
        return explore(None if a.explore == "ask" else float(a.explore))
    if a.try_ is not None:
        return try_mode(None if a.try_ == "ask" else int(a.try_))
    if a.check or a.live:
        fetchers = build_live_fetchers(a.universe_csv, a.prices_csv, a.eps_csv)
        if a.check:
            return check_mode(fetchers)
        return live_mode(fetchers, a.db or config.DB_PATH, a.limit)
    return demo(edge=a.edge, seed=a.seed, value_share=a.value_share, clean=a.clean,
                db_path=a.db or ":memory:")


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
