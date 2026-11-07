"""signals.py: from raw P/E and momentum to ranks, and from ranks to one combined score.

DIRECTION: both ranks run from 0 to 1 and HIGHER IS BETTER.
    pe_rank        1 = the cheapest stock that month (lowest P/E), 0 = the most expensive
    momentum_rank  1 = the strongest past return,                  0 = the weakest
Ranking is relative to the other stocks on the SAME month-end. Today's rank says nothing about last month's.

Each signal is ranked among the stocks that HAVE that signal that month. A stock is only eligible to be bought
when it has both (align.Row.eligible), so a loss-making company is out for that month, not ranked last.

COMBINED SCORE (computed on demand, never stored, because it depends on the slider):
    score = value_weight * pe_rank + (1 - value_weight) * momentum_rank
    value_weight 1.0 = pure value, 0.0 = pure momentum, 0.5 = an equal blend.
"""
import numpy as np

import config


def pct_rank(values):
    """Rank each value from 0 (lowest) to 1 (highest). Ties share the average rank. None/NaN stay NaN.
    One value on its own gets 0.5 (it is neither high nor low)."""
    x = np.array([np.nan if v is None else v for v in values], dtype=float)
    out = np.full(len(x), np.nan)
    ok = np.flatnonzero(~np.isnan(x))
    m = len(ok)
    if m == 0:
        return out
    if m == 1:
        out[ok] = 0.5
        return out
    order = ok[np.argsort(x[ok], kind="stable")]
    sorted_vals = x[order]
    ranks = np.empty(m)
    i = 0
    while i < m:
        j = i
        while j + 1 < m and sorted_vals[j + 1] == sorted_vals[i]:
            j += 1
        ranks[i:j + 1] = (i + j) / 2.0                      # tied values share the middle rank
        i = j + 1
    out[order] = ranks / (m - 1)
    return out


def add_ranks(rows):
    """Fill pe_rank and momentum_rank on every Row, month by month. Returns the same rows."""
    by_date = {}
    for r in rows:
        by_date.setdefault(r.date, []).append(r)
    for date_rows in by_date.values():
        cheap = 1.0 - pct_rank([r.pe for r in date_rows])      # low P/E -> high rank
        strong = pct_rank([r.momentum for r in date_rows])
        for r, c, s in zip(date_rows, cheap, strong):
            r.pe_rank = None if np.isnan(c) else float(c)
            r.momentum_rank = None if np.isnan(s) else float(s)
    return rows


def combined_score(row, value_weight):
    """The blend of the two ranks. The row must be eligible (have both ranks)."""
    if not 0.0 <= value_weight <= 1.0:
        raise ValueError("value_weight must be between 0 and 1")
    if row.pe_rank is None or row.momentum_rank is None:
        raise ValueError(f"{row.ticker} {row.date} has no score: it is missing a signal")
    return value_weight * row.pe_rank + (1.0 - value_weight) * row.momentum_rank


def scores_for_date(rows, date, value_weight=config.DEFAULT_VALUE_WEIGHT):
    """[(score, row), ...] for the eligible stocks on one month-end, best first.
    Ties are broken by ticker so the same inputs always give the same list."""
    scored = [(combined_score(r, value_weight), r) for r in rows
              if r.date == date and r.pe_rank is not None and r.momentum_rank is not None]
    scored.sort(key=lambda t: (-t[0], t[1].ticker))
    return scored


# ---------------------------------------------------------------------------
# storage
# ---------------------------------------------------------------------------

def store_signals(con, rows):
    """Write the raw values and ranks to the SIGNAL table. Safe to repeat (upsert).
    Returns {'inserted', 'updated', 'unchanged'}."""
    counts = {"inserted": 0, "updated": 0, "unchanged": 0}
    for r in rows:
        values = (r.pe, r.momentum, r.pe_rank, r.momentum_rank)
        existing = con.execute("SELECT pe, momentum, pe_rank, momentum_rank FROM SIGNAL "
                               "WHERE ticker = ? AND rebalance_date = ?", (r.ticker, r.date)).fetchone()
        if existing is not None and tuple(existing) == values:
            counts["unchanged"] += 1
            continue
        con.execute(
            "INSERT INTO SIGNAL (ticker, rebalance_date, pe, momentum, pe_rank, momentum_rank) "
            "VALUES (?, ?, ?, ?, ?, ?) ON CONFLICT(ticker, rebalance_date) DO UPDATE SET "
            "pe = excluded.pe, momentum = excluded.momentum, "
            "pe_rank = excluded.pe_rank, momentum_rank = excluded.momentum_rank",
            (r.ticker, r.date) + values)
        counts["updated" if existing is not None else "inserted"] += 1
    con.commit()
    return counts


def load_signals(con, date):
    """[(ticker, pe, momentum, pe_rank, momentum_rank), ...] stored for one month-end, ordered by ticker."""
    return con.execute(
        "SELECT ticker, pe, momentum, pe_rank, momentum_rank FROM SIGNAL WHERE rebalance_date = ? ORDER BY ticker",
        (date,)).fetchall()
