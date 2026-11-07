"""portfolio.py: from monthly scores to the stocks we hold.

EVERY MONTH-END
  1. take the eligible stocks, best combined score first (place 1 = best)
  2. choose the holdings:
       * fewer eligible stocks than top_n  -> hold ALL of them (the portfolio is simply smaller)
       * otherwise                         -> top_n stocks, with the BUFFER rule below
  3. weight them equally: each of the k chosen stocks gets 1/k  (fully invested, no cash)

THE BUFFER (hysteresis). buffer = 1.0 means a pure top-N: rebuilt from scratch every month.
With buffer = 1.5 and top_n = 20, a stock we already hold is KEPT while it is still in the top 30 places,
even if it has slipped out of the top 20. Empty seats are then filled by the best-placed stocks we do not
hold. Fewer trades mean lower costs, at the price of holding some stocks whose signal has faded.

WHAT THIS MODULE DOES NOT DO: returns, trade prices and costs (Stage 5). The turnover numbers here compare
TARGET weights, so they ignore that held stocks drift away from 1/k during the month. Stage 5 uses the drifted
weights, because that is what must actually be traded.
"""
import math
from dataclasses import dataclass

import config
import signals


@dataclass(frozen=True)
class Position:
    ticker: str
    weight: float
    place: int           # rank among this month's eligible stocks (1 = best score)
    score: float
    kept: bool           # True if we already held it last month (False = newly bought)


def _check_params(top_n, value_weight=None, buffer=None):
    if int(top_n) != top_n or not config.MIN_TOP_N <= top_n <= config.MAX_TOP_N:
        raise ValueError(f"top_n must be a whole number between {config.MIN_TOP_N} and {config.MAX_TOP_N}")
    if value_weight is not None and not 0.0 <= value_weight <= 1.0:
        raise ValueError("value_weight must be between 0 and 1")
    if buffer is not None and not config.MIN_BUFFER <= buffer <= config.MAX_BUFFER:
        raise ValueError(f"buffer must be between {config.MIN_BUFFER} and {config.MAX_BUFFER}")


def buffer_limit(top_n, buffer):
    """The worst place at which a current holding is still kept. A small epsilon guards against 1.1 * 20 = 22.000000000000004."""
    return int(math.floor(buffer * top_n + 1e-9))


# ---------------------------------------------------------------------------
# one month
# ---------------------------------------------------------------------------

def select_holdings(scored, previous, top_n, buffer=config.DEFAULT_BUFFER):
    """Choose this month's holdings.
    scored    [(score, row), ...] eligible stocks, best first (signals.scores_for_date)
    previous  the tickers held last month (empty at the start, or after a month with nothing to hold)
    Returns a list of Position, best place first."""
    _check_params(top_n, buffer=buffer)
    previous = set(previous)
    n = len(scored)
    if n == 0:
        return []
    places = {row.ticker: i for i, (_, row) in enumerate(scored, 1)}
    if n <= top_n:
        chosen = set(range(1, n + 1))                                    # hold everything that exists
    else:
        limit = buffer_limit(top_n, buffer)
        kept = sorted(places[t] for t in previous if t in places and places[t] <= limit)
        chosen = set(kept[:top_n])
        for p in range(1, n + 1):                                        # fill the empty seats, best place first
            if len(chosen) >= top_n:
                break
            chosen.add(p)
    weight = 1.0 / len(chosen)
    return [Position(scored[p - 1][1].ticker, weight, p, scored[p - 1][0], scored[p - 1][1].ticker in previous)
            for p in sorted(chosen)]


def weights(positions):
    return {p.ticker: p.weight for p in positions}


# ---------------------------------------------------------------------------
# all months
# ---------------------------------------------------------------------------

def build_portfolio(rows, top_n=config.DEFAULT_TOP_N, value_weight=config.DEFAULT_VALUE_WEIGHT,
                    buffer=config.DEFAULT_BUFFER):
    """{month-end: [Position, ...]} for every month-end in rows (signals.add_ranks must already have run).
    Months run in time order, because the buffer rule depends on what was held the month before.
    A month with nothing eligible is an empty list (all cash), and the next month starts from nothing."""
    _check_params(top_n, value_weight, buffer)
    by_date = {}
    for r in rows:
        by_date.setdefault(r.date, []).append(r)
    portfolio, previous = {}, set()
    for date in sorted(by_date):
        scored = signals.scores_for_date(by_date[date], date, value_weight)
        positions = select_holdings(scored, previous, top_n, buffer)
        portfolio[date] = positions
        previous = {p.ticker for p in positions}
    return portfolio


# ---------------------------------------------------------------------------
# turnover
# ---------------------------------------------------------------------------

def traded_fraction(old, new):
    """Buys plus sells, as a fraction of the portfolio, when moving from weights `old` to `new` (dicts ticker -> weight).
    Trading costs are charged on this: a full swap of every holding is 2.0 (sell 100%, buy 100%);
    building the portfolio from cash is 1.0."""
    return sum(abs(new.get(t, 0.0) - old.get(t, 0.0)) for t in set(old) | set(new))


def one_way_turnover(old, new):
    """Half of the traded fraction: the share of the portfolio that was replaced."""
    return traded_fraction(old, new) / 2.0


def names_changed(old_positions, new_positions):
    """(bought, sold): the tickers that entered and the tickers that left."""
    old = {p.ticker for p in old_positions}
    new = {p.ticker for p in new_positions}
    return new - old, old - new


def portfolio_stats(portfolio, top_n):
    """Summary of a built portfolio. The averages use 'steady' months only: months that are invested and follow an
    invested month. The first purchase from cash is not turnover in the ordinary sense, so it is left out."""
    dates = sorted(portfolio)
    invested = [d for d in dates if portfolio[d]]
    steady = [d for i, d in enumerate(dates) if i > 0 and portfolio[d] and portfolio[dates[i - 1]]]
    replaced, replaced_frac, traded, one_way = [], [], [], []
    for d in steady:
        prev = portfolio[dates[dates.index(d) - 1]]
        bought, _ = names_changed(prev, portfolio[d])
        replaced.append(len(bought))
        replaced_frac.append(len(bought) / len(portfolio[d]))
        tf = traded_fraction(weights(prev), weights(portfolio[d]))
        traded.append(tf)
        one_way.append(tf / 2.0)

    def avg(xs):
        return sum(xs) / len(xs) if xs else float("nan")
    return {
        "months": len(dates),
        "invested_months": len(invested),
        "empty_months": len(dates) - len(invested),
        "partial_months": sum(1 for d in invested if len(portfolio[d]) < top_n),
        "first_invested": invested[0] if invested else None,
        "steady_months": len(steady),
        "avg_names": avg([len(portfolio[d]) for d in invested]),
        "avg_replaced": avg(replaced),
        "avg_replaced_fraction": avg(replaced_frac),
        "avg_traded_fraction": avg(traded),
        "avg_one_way_turnover": avg(one_way),
    }


def estimated_cost_drag(stats, cost_bps):
    """A preview of what costs will take per year: cost rate x traded fraction x 12 months.
    It uses target weights (no drift), so Stage 5's real number will be a little higher."""
    return cost_bps / 10000.0 * stats["avg_traded_fraction"] * config.MONTHS_PER_YEAR


# ---------------------------------------------------------------------------
# checking a portfolio against the rules
# ---------------------------------------------------------------------------

def check_invariants(portfolio, rows, top_n, value_weight, buffer):
    """Every rule the construction must satisfy, checked independently of how the portfolio was built.
    Returns a list of violation messages (empty = all good)."""
    bad = []
    by_date = {}
    for r in rows:
        by_date.setdefault(r.date, []).append(r)
    limit = buffer_limit(top_n, buffer)
    previous = {}
    for date in sorted(portfolio):
        positions = portfolio[date]
        scored = signals.scores_for_date(by_date.get(date, []), date, value_weight)
        eligible = {row.ticker: (i, s) for i, (s, row) in enumerate(scored, 1)}
        tickers = [p.ticker for p in positions]
        if len(set(tickers)) != len(tickers):
            bad.append(f"{date}: a stock is held twice")
        if len(positions) > top_n:
            bad.append(f"{date}: {len(positions)} holdings is more than top_n={top_n}")
        if not positions:
            if eligible:
                bad.append(f"{date}: nothing held although {len(eligible)} stocks are eligible")
            previous = {}
            continue
        if abs(sum(p.weight for p in positions) - 1.0) > 1e-9:
            bad.append(f"{date}: weights add up to {sum(p.weight for p in positions)}")
        if any(abs(p.weight - 1.0 / len(positions)) > 1e-12 for p in positions):
            bad.append(f"{date}: weights are not equal")
        for p in positions:
            if p.ticker not in eligible:
                bad.append(f"{date}: {p.ticker} is held but not eligible")
                continue
            place, score = eligible[p.ticker]
            if p.place != place or abs(p.score - score) > 1e-12:
                bad.append(f"{date}: {p.ticker} has the wrong place or score")
            if p.kept != (p.ticker in previous):
                bad.append(f"{date}: {p.ticker} has the wrong 'kept' flag")
            if p.kept and place > limit and len(eligible) > top_n:
                bad.append(f"{date}: {p.ticker} was kept at place {place}, beyond the buffer limit {limit}")
        if len(eligible) <= top_n:
            if set(tickers) != set(eligible):
                bad.append(f"{date}: fewer eligible stocks than top_n, but not all of them are held")
        else:
            if len(positions) != top_n:
                bad.append(f"{date}: {len(positions)} holdings, expected exactly top_n={top_n}")
            held_places = {p.place for p in positions}
            for p in positions:
                if not p.kept and any(q not in held_places for q in range(1, p.place)):
                    bad.append(f"{date}: {p.ticker} was bought at place {p.place} while a better-placed stock was left out")
            if buffer == 1.0 and {p.place for p in positions} != set(range(1, top_n + 1)):
                bad.append(f"{date}: with no buffer the holdings must be exactly the top {top_n}")
        previous = {p.ticker: p.weight for p in positions}
    return bad


# ---------------------------------------------------------------------------
# storage
# ---------------------------------------------------------------------------

def get_or_create_run(con, top_n, value_weight, cost_bps=config.DEFAULT_COST_BPS, buffer=config.DEFAULT_BUFFER,
                      pe_mode="lagged", universe="survivors"):
    """The run_id for this exact set of choices. Asking twice gives the same id (no duplicate runs)."""
    args = (top_n, value_weight, cost_bps, buffer, pe_mode, universe)
    row = con.execute("SELECT run_id FROM RUN WHERE top_n = ? AND value_weight = ? AND cost_bps = ? "
                      "AND buffer = ? AND pe_mode = ? AND universe = ?", args).fetchone()
    if row:
        return row[0]
    cur = con.execute("INSERT INTO RUN (top_n, value_weight, cost_bps, buffer, pe_mode, universe) "
                      "VALUES (?, ?, ?, ?, ?, ?)", args)
    con.commit()
    return cur.lastrowid


def store_holdings(con, run_id, portfolio):
    """Write the portfolio to HOLDING for this run. Safe to repeat. Rows that are no longer part of the portfolio
    are removed, so the table always equals what was just built. Returns the counts."""
    new = {(d, p.ticker): p.weight for d, positions in portfolio.items() for p in positions}
    old = {(d, t): w for d, t, w in con.execute(
        "SELECT rebalance_date, ticker, weight FROM HOLDING WHERE run_id = ?", (run_id,))}
    counts = {"inserted": 0, "updated": 0, "unchanged": 0, "deleted": 0}
    for key in set(old) - set(new):
        con.execute("DELETE FROM HOLDING WHERE run_id = ? AND rebalance_date = ? AND ticker = ?", (run_id,) + key)
        counts["deleted"] += 1
    for key, w in new.items():
        if key not in old:
            con.execute("INSERT INTO HOLDING (run_id, rebalance_date, ticker, weight) VALUES (?, ?, ?, ?)", (run_id,) + key + (w,))
            counts["inserted"] += 1
        elif old[key] != w:
            con.execute("UPDATE HOLDING SET weight = ? WHERE run_id = ? AND rebalance_date = ? AND ticker = ?", (w, run_id) + key)
            counts["updated"] += 1
        else:
            counts["unchanged"] += 1
    con.commit()
    return counts


def load_holdings(con, run_id):
    """{month-end: {ticker: weight}} as stored for this run (months with no holdings are absent)."""
    out = {}
    for d, t, w in con.execute("SELECT rebalance_date, ticker, weight FROM HOLDING WHERE run_id = ? "
                               "ORDER BY rebalance_date, ticker", (run_id,)):
        out.setdefault(d, {})[t] = w
    return out
