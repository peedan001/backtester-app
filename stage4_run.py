"""stage4_run.py: walk through Stage 4 (portfolio construction).

    python stage4_run.py --demo                    # the full walkthrough, no internet needed
    python stage4_run.py --demo --db port.db       # also save prices, EPS, signals, runs and holdings to a file
    python stage4_run.py --build                   # interactive: choose N, value weight and buffer, see the portfolio's behaviour
    python stage4_run.py --build 20 0.5 1.5        # the same in one go: N, weight on value, buffer
    python stage4_run.py --changes                 # interactive: pick a month and a buffer, see who was kept, sold and bought
    python stage4_run.py --changes 2019-06-28 1.5  # the same in one go: month-end, buffer
    python stage4_run.py --try                     # nine hand-made situations that show each selection rule
Add --db FILE to --build / --changes to use a database you already have instead of a fresh synthetic world.
"""
import argparse
import datetime as dt
import sys

import align
import config
import db
import portfolio as pf
import signals
import stage3_run
from align import Row


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def eligible_counts(rows):
    counts = {}
    for r in rows:
        if r.pe_rank is not None and r.momentum_rank is not None:
            counts[r.date] = counts.get(r.date, 0) + 1
    return counts


def positions_table(positions, limit=None):
    lines = ["   place  stock   weight   score   status"]
    for p in positions[:limit]:
        lines.append(f"   {p.place:5d}  {p.ticker:6s}  {p.weight * 100:5.2f}%  {p.score:.3f}   {'kept' if p.kept else 'NEW'}")
    if limit is not None and len(positions) > limit:
        lines.append(f"   ... and {len(positions) - limit} more")
    return lines


def summary_line(label, stats, cost_bps=config.DEFAULT_COST_BPS):
    return (f"   {label:22s} {stats['avg_replaced']:5.1f} names ({stats['avg_replaced_fraction'] * 100:3.0f}%) replaced a month, "
            f"traded {stats['avg_traded_fraction']:.2f}, "
            f"cost at {cost_bps} bps about {pf.estimated_cost_drag(stats, cost_bps) * 100:.2f}% a year")


def lookahead_verdict(result):
    """'OK' only when the past is untouched AND the scramble demonstrably changed the future.
    If nothing changed anywhere, the test proved nothing, and that is reported as such, not as a pass."""
    if result["changed_up_to_cutoff"] > 0:
        return "LOOK-AHEAD!"
    if result["changed_after_cutoff"] == 0:
        return "INCONCLUSIVE (the scramble changed nothing)"
    return "OK"


def portfolio_lookahead(con, cutoff, top_n, value_weight, buffer, scramble=("prices", "eps")):
    """Scramble the future in a copy of the database, rebuild the portfolio, and compare.
    Holdings dated on or before the cutoff must be identical; later holdings must change."""
    def key(positions):
        return tuple((p.ticker, p.weight, p.place, p.score, p.kept) for p in positions)
    before = pf.build_portfolio(stage3_run.all_signals(con), top_n, value_weight, buffer)
    copy = stage3_run.scrambled_copy(con, cutoff, scramble)
    after = pf.build_portfolio(stage3_run.all_signals(copy), top_n, value_weight, buffer)
    copy.close()
    out = {"months_up_to_cutoff": 0, "changed_up_to_cutoff": 0, "changed_after_cutoff": 0}
    for d in before:
        same = key(before[d]) == key(after[d])
        if d <= cutoff:
            out["months_up_to_cutoff"] += 1
            out["changed_up_to_cutoff"] += not same
        else:
            out["changed_after_cutoff"] += not same
    return out


# ---------------------------------------------------------------------------
# --changes: who was kept, sold and bought in one month
# ---------------------------------------------------------------------------

def changes_lines(rows, port, date, top_n, value_weight, buffer):
    dates = sorted(port)
    if date not in port:
        return [f"{date} is not a month-end in the data."]
    i = dates.index(date)
    now = port[date]
    before = port[dates[i - 1]] if i > 0 else []
    by_ticker = {r.ticker: r for r in rows if r.date == date}
    scored = signals.scores_for_date(list(by_ticker.values()), date, value_weight)
    places = {r.ticker: (k, s) for k, (s, r) in enumerate(scored, 1)}
    limit = pf.buffer_limit(top_n, buffer)
    L = [f"Month-end {date}: top {top_n}, weight {value_weight:.2f} on value, buffer {buffer:.2f} "
         f"(a holding is kept while it stays within the top {limit}); {len(scored)} stocks eligible"]
    if not now:
        L.append("   Nothing is held this month: no stock has both signals.")
        return L
    prev_place = {p.ticker: p.place for p in before}
    kept = [p for p in now if p.kept]
    bought = [p for p in now if not p.kept]
    sold = [p for p in before if p.ticker not in {q.ticker for q in now}]
    L.append(f"   KEPT   ({len(kept)})  " + ", ".join(f"{p.ticker} {prev_place.get(p.ticker, '?')}->{p.place}" for p in kept)
             if kept else "   KEPT   (0)")
    L.append(f"   SOLD   ({len(sold)})")
    for p in sold:
        if p.ticker in places:
            k, _ = places[p.ticker]
            why = (f"now place {k}, beyond the buffer limit {limit}" if buffer > 1.0 and len(scored) > top_n
                   else f"now place {k}, outside the top {top_n}")
        else:
            r = by_ticker.get(p.ticker)
            why = "no longer eligible: " + ("; ".join(r.reasons) if r and r.reasons else "no signal")
        L.append(f"            {p.ticker}: was place {p.place}, {why}")
    L.append(f"   BOUGHT ({len(bought)})")
    for p in bought:
        L.append(f"            {p.ticker}: place {p.place}, score {p.score:.3f}")
    old_w, new_w = pf.weights(before), pf.weights(now)
    L.append(f"   traded {pf.traded_fraction(old_w, new_w):.2f} of the portfolio (buys + sells), "
             f"one-way turnover {pf.one_way_turnover(old_w, new_w) * 100:.0f}%")
    if buffer > 1.0:
        worse = [p for p in kept if p.place > top_n]
        if worse:
            L.append(f"   The buffer kept {len(worse)} stock(s) outside the top {top_n}: "
                     + ", ".join(f"{p.ticker} (place {p.place})" for p in worse))
    return L


def month_on_or_before(months, text):
    """The latest month-end on or before the date `text`. Raises ValueError for a text that is not a date,
    or a date before the first month-end (a plain text comparison would quietly accept 'abc')."""
    dt.date.fromisoformat(text)
    earlier = [m for m in months if m <= text]
    if not earlier:
        raise ValueError(f"{text} is before the first month-end in the data")
    return earlier[-1]


def changes_mode(rows, top_n=config.DEFAULT_TOP_N, value_weight=config.DEFAULT_VALUE_WEIGHT, date=None, buffer=None):
    me = sorted({r.date for r in rows})
    if date is not None:
        try:
            date = month_on_or_before(me, date)
        except ValueError as e:
            print(f"Cannot use that date: {e}")
            return 1
        port = pf.build_portfolio(rows, top_n, value_weight, buffer)
        print("\n".join(changes_lines(rows, port, date, top_n, value_weight, buffer)))
        return 0
    print(f"Pick a month-end (for example {me[-13]}; the data runs {me[0]} to {me[-1]}) and a buffer from "
          f"{config.MIN_BUFFER} to {config.MAX_BUFFER}. Press Enter on the month-end prompt to quit.")
    while True:
        try:
            raw_d = input("month-end> ").strip()
            if not raw_d:
                break
            raw_b = input("buffer (Enter for 1.5)> ").strip()
        except EOFError:
            break
        try:
            b = float(raw_b) if raw_b else 1.5
            if not config.MIN_BUFFER <= b <= config.MAX_BUFFER:
                raise ValueError(f"the buffer must be between {config.MIN_BUFFER} and {config.MAX_BUFFER}")
            d = month_on_or_before(me, raw_d)
        except ValueError as e:
            print(f"  {e if ('buffer' in str(e) or 'before' in str(e)) else 'Please type a number for the buffer and a date like 2019-06-28.'}")
            continue
        port = pf.build_portfolio(rows, top_n, value_weight, b)
        print()
        print("\n".join(changes_lines(rows, port, d, top_n, value_weight, b)))
        print()
    return 0


# ---------------------------------------------------------------------------
# --build: choose the parameters, see how the portfolio behaves
# ---------------------------------------------------------------------------

def build_lines(rows, top_n, value_weight, buffer, cost_bps=config.DEFAULT_COST_BPS):
    port = pf.build_portfolio(rows, top_n, value_weight, buffer)
    stats = pf.portfolio_stats(port, top_n)
    bad = pf.check_invariants(port, rows, top_n, value_weight, buffer)
    L = [f"Top {top_n}, {value_weight:.2f} weight on value ({1 - value_weight:.2f} on momentum), buffer {buffer:.2f}:",
         f"   invested from {stats['first_invested']} ({stats['empty_months']} early months have no signals, so no holdings)",
         f"   {stats['avg_names']:.1f} stocks held on average; {stats['partial_months']} months hold fewer than {top_n}"]
    if stats["steady_months"]:
        L.append(f"   each month about {stats['avg_replaced']:.1f} stocks are replaced ({stats['avg_replaced_fraction'] * 100:.0f}% of the portfolio)")
        L.append(f"   traded fraction (buys + sells) {stats['avg_traded_fraction']:.2f}; one-way turnover {stats['avg_one_way_turnover'] * 100:.0f}%")
        L.append(f"   a preview of costs: at {cost_bps} bps that is about {pf.estimated_cost_drag(stats, cost_bps) * 100:.2f}% a year "
                 f"(real costs in Stage 5 will be a little higher because holdings drift from equal weights)")
    L.append(f"   rule check: {'all rules hold' if not bad else str(len(bad)) + ' VIOLATIONS: ' + bad[0]}")
    last = max(d for d in port if port[d])
    L.append(f"   the most recent portfolio ({last}), first 8 of {len(port[last])}:")
    L.extend("   " + line for line in positions_table(port[last], 8))
    return L, bad


def build_mode(rows, top_n=None, value_weight=None, buffer=None):
    if top_n is not None:
        lines, bad = build_lines(rows, top_n, value_weight, buffer)
        print("\n".join(lines))
        return 1 if bad else 0
    print("Choose your portfolio. Press Enter to keep the default in brackets; type q to quit.")
    n, w, b = config.DEFAULT_TOP_N, config.DEFAULT_VALUE_WEIGHT, config.DEFAULT_BUFFER
    while True:
        try:
            raw = [input(f"stocks to hold N [{n}]> ").strip(),
                   input(f"weight on value, 0 to 1 [{w}]> ").strip(),
                   input(f"buffer, {config.MIN_BUFFER} to {config.MAX_BUFFER} [{b}]> ").strip()]
        except EOFError:
            break
        if any(x.lower() == "q" for x in raw):
            break
        try:
            n2, w2, b2 = int(raw[0]) if raw[0] else n, float(raw[1]) if raw[1] else w, float(raw[2]) if raw[2] else b
            pf._check_params(n2, w2, b2)
        except ValueError as e:
            print(f"  {e if 'must' in str(e) else 'Please type numbers only.'}")
            continue
        n, w, b = n2, w2, b2
        print()
        print("\n".join(build_lines(rows, n, w, b)[0]))
        print()
    return 0


# ---------------------------------------------------------------------------
# --try: nine hand-made situations
# ---------------------------------------------------------------------------

def _month(order, ties=()):
    """Eligible stocks in the given order (best first) as Rows whose scores fall from high to low.
    Names in `ties` share the score of the name before them."""
    rows, score = [], 0.95
    for i, name in enumerate(order):
        if i > 0 and name not in ties:
            score -= 0.05
        rows.append(Row(name, "d", pe=1.0, momentum=0.1, pe_rank=score, momentum_rank=score))
    return rows


def try_cases():
    return [
        ("Plenty of stocks, no buffer", 3, 1.0, set(), "ABCDEFGH",
         "A pure top-3: the three best scores, one third of the money each."),
        ("Fewer eligible stocks than N", 3, 1.0, set(), "AB",
         "Only two stocks qualify, so we hold those two at 50% each: fully invested, a smaller portfolio (your choice in the design)."),
        ("Exactly N eligible", 3, 1.0, set(), "ABC",
         "The boundary: N stocks qualify, all of them are held."),
        ("Nothing is eligible", 3, 1.0, {"A"}, "",
         "No signals, no holdings: the month is all cash, and the next month starts from nothing."),
        ("The buffer keeps a stock that has slipped", 3, 1.5, set("ABC"), "DBCAEF",
         "A is now only place 4, but 4 is inside the buffer (1.5 x 3, rounded down = 4), so it stays. B, C and A fill all three seats, "
         "and D, the best score, is NOT bought. That is the price of fewer trades. Try the same month with buffer 1.0."),
        ("A holding falls outside the buffer", 3, 1.5, set("ABC"), "DBCEAF",
         "A is at place 5, beyond the limit 4, so it is sold. B and C are kept, and D takes the empty seat."),
        ("A holding stops being eligible", 3, 2.0, set("ABC"), "BCDE",
         "A lost its signal (for example, it made a loss). Even a generous buffer cannot keep a stock that cannot be scored."),
        ("The buffer never grows the portfolio", 3, 2.0, set("ABC"), "ABCD",
         "Everything held is still in the top places, and there are only three seats. D is not added."),
        ("Equal scores are split by ticker", 1, 1.0, set(), "BA",
         "Both stocks have the same score. The tie goes to the alphabetically first ticker, so the answer is always the same."),
    ]


def run_try_case(n):
    name, top_n, buffer, previous, order, lesson = try_cases()[n - 1]
    ties = {"A"} if n == 9 else set()                       # case 9: A gets exactly the same score as B, and is listed AFTER it
    rows = _month(order, ties=ties)
    scored = signals.scores_for_date(rows, "d", 1.0)
    positions = pf.select_holdings(scored, previous, top_n, buffer)
    L = [f"  top {top_n}, buffer {buffer}; held last month: {''.join(sorted(previous)) or 'nothing'}; "
         f"eligible, best first: {' '.join(f'{r.ticker}({s:.2f})' for s, r in scored) or 'none'}"]
    if positions:
        L.append("  holds: " + ", ".join(f"{p.ticker} ({p.weight * 100:.1f}%, place {p.place}, {'kept' if p.kept else 'bought'})" for p in positions))
    else:
        L.append("  holds: nothing (all cash)")
    return name, "\n".join(L), lesson


def try_mode(arg=None):
    cases = try_cases()
    pick = None if arg is None else [arg]
    while True:
        if pick is None:
            print("\nSituations that show each selection rule:")
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

def demo(edge=config.SYNTH_DEFAULT_EDGE, seed=config.SYNTH_SEED, clean=False, db_path=None):
    print("=" * 72)
    print("STAGE 4: portfolio construction")
    print("=" * 72)
    ok_all = True
    N, W = config.DEFAULT_TOP_N, config.DEFAULT_VALUE_WEIGHT

    print("\n--- Step 1: the signals from Stage 3 ---")
    market, con = stage3_run.build_world(edge=edge, seed=seed, defects=not clean)
    rows = stage3_run.all_signals(con)
    elig = eligible_counts(rows)
    print(f"  {len(market['truth']['tickers'])} stocks, {len(elig)} month-ends with signals, "
          f"{min(elig.values())} to {max(elig.values())} eligible stocks per month")

    print(f"\n--- Step 2: one month's portfolio (top {N}, {W:.0%} weight on value, no buffer) ---")
    port = pf.build_portfolio(rows, N, W, 1.0)
    month = "2019-06-28"
    print(f"  {month}: {elig[month]} eligible stocks, the best {N} are held at {100 / N:.0f}% each")
    print("\n".join("  " + line for line in positions_table(port[month], 8)))

    print("\n--- Step 3: fewer eligible stocks than N: hold what exists ---")
    big = pf.build_portfolio(rows, 100, W, 1.0)
    stats = pf.portfolio_stats(big, 100)
    sample = big[month]
    print(f"  With N=100, {stats['partial_months']} of {stats['invested_months']} invested months hold fewer than 100 stocks.")
    print(f"  On {month}: {elig[month]} are eligible, so {len(sample)} are held at {sample[0].weight * 100:.2f}% each, fully invested.")

    print("\n--- Step 4: the buffer, and what it does to turnover ---")
    print("  Rebuilding from scratch every month vs keeping holdings that slip, same signals, same N:")
    by_buffer = {}
    for b in (1.0, 1.25, 1.5, 2.0):
        by_buffer[b] = pf.portfolio_stats(pf.build_portfolio(rows, N, W, b), N)
        print(summary_line(f"buffer {b:.2f}", by_buffer[b]))
    drop = 1 - by_buffer[1.5]["avg_traded_fraction"] / by_buffer[1.0]["avg_traded_fraction"]
    print(f"  A buffer of 1.5 cuts trading by {drop * 100:.0f}%. Costs fall, but you now hold some stocks whose signal has faded.")
    print("  Stage 5 will show whether that pays off after returns.")

    print("\n--- Step 5: turnover depends on the signal too ---")
    for w, label in ((0.0, "all momentum"), (0.5, "50 / 50"), (1.0, "all value")):
        s = pf.portfolio_stats(pf.build_portfolio(rows, N, w, 1.0), N)
        print(summary_line(label, s))
    print("  (In this synthetic world P/E ranks barely move from month to month, so value churns least. Real P/E ranks")
    print("   usually move more. Do not carry these numbers over to real data.)")

    print("\n--- Step 6: every construction rule, checked independently ---")
    combos = [(20, 0.5, 1.0), (20, 0.5, 1.5), (20, 0.0, 2.0), (20, 1.0, 1.25), (10, 0.7, 1.0), (50, 0.5, 1.5), (100, 0.5, 1.0)]
    total_bad = 0
    for n, w, b in combos:
        bad = pf.check_invariants(pf.build_portfolio(rows, n, w, b), rows, n, w, b)
        total_bad += len(bad)
        print(f"  N={n:3d} weight on value {w:.2f} buffer {b:.2f}: {'all rules hold' if not bad else str(len(bad)) + ' VIOLATIONS: ' + bad[0]}")
    ok_all = ok_all and total_bad == 0

    print("\n--- Step 7: the look-ahead test, again, for holdings ---")
    cutoff = stage3_run.resolve_month_end(con, "2019-06-30")
    for what in (("prices", "eps"), ("prices",), ("eps",)):
        ex = portfolio_lookahead(con, cutoff, N, W, 1.5, scramble=what)
        verdict = lookahead_verdict(ex)
        ok_all = ok_all and verdict == "OK"
        print(f"  scramble {' + '.join(what):13s} after {cutoff}: {ex['changed_up_to_cutoff']} of {ex['months_up_to_cutoff']} earlier portfolios "
              f"changed, {ex['changed_after_cutoff']} later ones changed  {verdict}")

    print("\n--- Step 8: save the runs (safe to repeat) ---")
    ids = {}
    for b in (1.0, 1.5):
        run = pf.get_or_create_run(con, N, W, buffer=b)
        ids[b] = run
        p = pf.build_portfolio(rows, N, W, b)
        first = pf.store_holdings(con, run, p)
        again = pf.store_holdings(con, run, p)
        idem = again["inserted"] == 0 and again["updated"] == 0 and again["deleted"] == 0
        same_id = pf.get_or_create_run(con, N, W, buffer=b) == run
        ok_all = ok_all and idem and same_id
        print(f"  run {run} (buffer {b}): first {first}; again {again}  {'idempotent' if idem and same_id else 'NOT idempotent!'}")
    if db_path:
        out = db.connect(db_path)
        con.backup(out)
        out.close()
        print(f"  saved everything to {db_path}")

    print("\n" + "=" * 72)
    print("Everything behaved as designed." if ok_all else "SOMETHING BEHAVED UNEXPECTEDLY: see the lines above.")
    print("Check question: a buffer of 1.5 cuts trading by more than half, yet it can lower OR raise the return you")
    print("keep after costs. Give one reason it could lower it, and one reason it could raise it.")
    con.close()
    return 0 if ok_all else 1


# ---------------------------------------------------------------------------

def main(argv):
    ap = argparse.ArgumentParser(description="Stage 4: portfolio construction")
    ap.add_argument("--demo", action="store_true")
    ap.add_argument("--clean", action="store_true", help="no planted data defects")
    ap.add_argument("--edge", type=float, default=config.SYNTH_DEFAULT_EDGE)
    ap.add_argument("--seed", type=int, default=config.SYNTH_SEED)
    ap.add_argument("--db", default=None)
    ap.add_argument("--build", nargs="*", default=None, metavar="N WEIGHT BUFFER")
    ap.add_argument("--changes", nargs="*", default=None, metavar="DATE BUFFER")
    ap.add_argument("--try", dest="try_", nargs="?", const="ask", default=None)
    a = ap.parse_args(argv)

    if a.try_ is not None:
        return try_mode(None if a.try_ == "ask" else int(a.try_))
    if a.build is not None:
        if len(a.build) not in (0, 3):
            print("Use --build with three numbers (for example --build 20 0.5 1.5), or with nothing.")
            return 1
        try:
            args = (int(a.build[0]), float(a.build[1]), float(a.build[2])) if a.build else (None, None, None)
            if a.build:
                pf._check_params(*args)
        except ValueError as e:
            print(f"Cannot use those numbers: {e}")
            return 1
        con = stage3_run.open_world(a)
        try:
            return build_mode(stage3_run.all_signals(con), *args)
        finally:
            con.close()
    if a.changes is not None:
        if len(a.changes) not in (0, 2):
            print("Use --changes with a date and a buffer (for example --changes 2019-06-28 1.5), or with nothing.")
            return 1
        try:
            if a.changes:
                dt.date.fromisoformat(a.changes[0])
                pf._check_params(config.DEFAULT_TOP_N, buffer=float(a.changes[1]))
        except ValueError as e:
            print(f"Cannot use those values: {e}")
            return 1
        con = stage3_run.open_world(a)
        try:
            rows = stage3_run.all_signals(con)
            if a.changes:
                return changes_mode(rows, date=a.changes[0], buffer=float(a.changes[1]))
            return changes_mode(rows)
        finally:
            con.close()
    return demo(edge=a.edge, seed=a.seed, clean=a.clean, db_path=a.db)


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
