"""align.py: the point-in-time join.

THE RULE: on a given month-end, you may use only what the world could see on that day.
  * a price is usable if it is dated on or before the month-end (and not too old, and not flagged)
  * an EPS figure is usable only if its available_date (the filing date) is on or before the month-end

For each stock and month-end this module builds one Row with:
    P/E       = month-end price / trailing-twelve-month (TTM) EPS        (low = cheap)
    momentum  = price at the previous month-end / price 12 month-ends ago - 1    ("12-1")
and, whenever one of them cannot be computed honestly, the REASON (nothing is silently filled in).

Rules, all from config.py:
  price carry     a missing month-end price may be replaced by one at most MAX_CARRY_DAYS old
  flagged prices  a price flagged as a jump (bad tick, 10x spike) is never used: treated as missing
                  (a 'gap' flag only says the PREVIOUS price was long ago; the price itself is fine)
  TTM EPS         the 4 latest quarters (by quarter end) that are already filed; they must be consecutive
                  (a missing quarter would make 'trailing twelve months' cover 15 months), the latest
                  must not be stale, and the total must be above 0 (a loss has no meaningful P/E)

Stage 6 adds two ways to BREAK these rules on purpose, so their cost can be measured (never use them for real results):
  pe_mode 'no_lag'   an EPS figure is used from the day its quarter ENDS, ignoring the 30 to 55 days it took to be filed
  pe_mode 'today'    every date uses the latest EPS ever filed (what you get if you divide old prices by today's EPS)
audit_lookahead() flags both, because they rely on filing dates after the month-end.

One honest caveat: adjusted closes are rebuilt by the data source after every later dividend and split.
Ratios between two dates are right, but the LEVEL of an old adjusted price is not what anyone saw on that day.
This cannot leak a future return, because the adjustment is the same factor for all dates before the event.
"""
import bisect
import datetime as dt
from collections import Counter, defaultdict
from dataclasses import dataclass, field

import config

REASON_NO_PRICE = "price missing: none on file"
REASON_OLD_PRICE = "price missing: latest one is too old"
REASON_FLAGGED = "price flagged as a jump"
REASON_FEW_Q = "EPS: fewer than 4 quarters filed"
REASON_GAP_Q = "EPS: a quarter is missing"
REASON_STALE = "EPS: stale (no recent filing)"
REASON_LOSS = "EPS: loss (TTM EPS <= 0)"
REASON_HISTORY = "momentum: less than 12 months of history"


def _d(s):
    return dt.date.fromisoformat(s)


# ---------------------------------------------------------------------------
# the month-end calendar
# ---------------------------------------------------------------------------

def month_end_dates(con):
    """The last trading day of each month, taken from the benchmark's own dates.

    A month whose last date is well before the calendar month-end AND has no later month is treated
    as unfinished (for example, live data up to the 14th) and is dropped, because a half month is not a month-end."""
    bench = con.execute("SELECT ticker FROM STOCK WHERE is_benchmark = 1").fetchall()
    if len(bench) != 1:
        raise ValueError(f"expected exactly one benchmark in STOCK, found {len(bench)}")
    rows = con.execute(
        "SELECT MAX(date) FROM PRICE WHERE ticker = ? GROUP BY substr(date, 1, 7) ORDER BY 1", (bench[0][0],)
    ).fetchall()
    dates = [r[0] for r in rows]
    if dates:
        last = _d(dates[-1])
        next_month = (last.replace(day=28) + dt.timedelta(days=4)).replace(day=1)
        if (next_month - last).days - 1 > 4:            # more than 4 calendar days left in the month
            dates.pop()
    return dates


# ---------------------------------------------------------------------------
# prices as seen on a date
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Quote:
    price: float
    price_date: str
    carried_days: int          # 0 = the price is dated exactly on the requested day


class PriceBook:
    """All prices in memory, looked up by (ticker, date) with the carry and flag rules."""

    def __init__(self, con):
        self._dates = defaultdict(list)
        self._prices = defaultdict(list)
        self._jump = defaultdict(list)
        for ticker, date, price, reason in con.execute(
                "SELECT ticker, date, adj_close, flag_reason FROM PRICE ORDER BY ticker, date"):
            self._dates[ticker].append(date)
            self._prices[ticker].append(price)
            self._jump[ticker].append(reason is not None and "jump" in reason)

    def quote(self, ticker, date, max_carry=None):
        """Return (Quote, None) or (None, reason)."""
        max_carry = config.MAX_CARRY_DAYS if max_carry is None else max_carry
        dates = self._dates.get(ticker)
        if not dates:
            return None, REASON_NO_PRICE
        i = bisect.bisect_right(dates, date) - 1            # latest price dated on or before `date`
        if i < 0:
            return None, REASON_NO_PRICE
        carried = (_d(date) - _d(dates[i])).days
        if carried > max_carry:
            return None, REASON_OLD_PRICE
        if self._jump[ticker][i]:
            return None, REASON_FLAGGED
        return Quote(self._prices[ticker][i], dates[i], carried), None

    def last_date(self, ticker):
        """The date of the last price on file for the stock, or None."""
        dates = self._dates.get(ticker)
        return dates[-1] if dates else None

    def last_price(self, ticker):
        """The last price on file that is not flagged as a jump, or None."""
        for price, jump in zip(reversed(self._prices.get(ticker, [])), reversed(self._jump.get(ticker, []))):
            if not jump:
                return price
        return None

    def delisted(self, ticker, date, max_carry=None):
        """True when the stock's data has ENDED: its last price is older than the carry limit as of `date`.
        (A gap in the middle of a series is not a delisting: the data resumes later, so the last price is recent.)"""
        max_carry = config.MAX_CARRY_DAYS if max_carry is None else max_carry
        last = self.last_date(ticker)
        return last is not None and (_d(date) - _d(last)).days > max_carry


# ---------------------------------------------------------------------------
# earnings as seen on a date
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Ttm:
    value: float
    quarters: tuple            # the four quarter ends used, oldest first
    latest_filed: str          # the latest filing date among them: this is what must be on or before the as-of date


class EpsBook:
    """All EPS figures in memory, looked up as they were visible on a date."""

    def __init__(self, con):
        self._rows = defaultdict(list)
        for ticker, period_end, available, eps in con.execute(
                "SELECT ticker, period_end, available_date, eps FROM FUNDAMENTAL ORDER BY ticker, period_end"):
            self._rows[ticker].append((period_end, available, eps))

    def rows(self, ticker):
        return list(self._rows.get(ticker, []))

    def ttm(self, ticker, date, mode="lagged"):
        """Return (Ttm, None), (None, reason), or for a loss (Ttm, REASON_LOSS) so callers can still show the number.
        mode 'lagged' is the honest rule. 'no_lag' and 'today' deliberately look ahead (see the module docstring)."""
        if mode not in config.EPS_MODES:
            raise ValueError(f"pe_mode must be one of {config.EPS_MODES}")
        rows = self._rows.get(ticker, [])
        if mode == "lagged":
            visible = [r for r in rows if r[1] <= date]                          # filed on or before `date`
        elif mode == "no_lag":
            visible = [r for r in rows if r[0] <= date]                          # quarter has ended (not yet filed!)
        else:
            visible = rows                                                       # everything, including the future
        if len(visible) < config.MIN_EPS_QUARTERS:
            return None, REASON_FEW_Q
        last4 = visible[-config.MIN_EPS_QUARTERS:]                               # latest by quarter end
        if (_d(last4[-1][0]) - _d(last4[0][0])).days > config.MAX_TTM_SPAN_DAYS:
            return None, REASON_GAP_Q
        if mode != "today" and (_d(date) - _d(last4[-1][0])).days > config.MAX_EPS_STALENESS_DAYS:
            return None, REASON_STALE
        ttm = Ttm(sum(r[2] for r in last4), tuple(r[0] for r in last4), max(r[1] for r in last4))
        if ttm.value <= 0:
            return ttm, REASON_LOSS
        return ttm, None


# ---------------------------------------------------------------------------
# the panel: one Row per stock per month-end
# ---------------------------------------------------------------------------

@dataclass
class Row:
    ticker: str
    date: str
    price: float = None
    price_date: str = None
    carried_days: int = 0
    ttm_eps: float = None
    eps_filed: str = None                       # latest filing date used for the TTM
    pe: float = None
    momentum: float = None
    mom_price_dates: tuple = ()                 # dates of the two prices behind the momentum
    pe_rank: float = None                       # filled by signals.add_ranks: 1 = cheapest
    momentum_rank: float = None                 # filled by signals.add_ranks: 1 = strongest
    reasons: list = field(default_factory=list)

    @property
    def eligible(self):
        """Both signals exist, so the stock can be scored and bought this month."""
        return self.pe is not None and self.momentum is not None


def default_tickers(con):
    """Today's list: the stocks that are still in the index now (survivors)."""
    return [r[0] for r in con.execute(
        "SELECT ticker FROM STOCK WHERE is_benchmark = 0 AND in_sp500 = 1 ORDER BY ticker")]


def all_tickers(con):
    """Everything that ever traded in the data, including stocks that have since left the index."""
    return [r[0] for r in con.execute("SELECT ticker FROM STOCK WHERE is_benchmark = 0 ORDER BY ticker")]


def tickers_for(con, universe):
    if universe not in config.UNIVERSES:
        raise ValueError(f"universe must be one of {config.UNIVERSES}")
    return default_tickers(con) if universe == "survivors" else all_tickers(con)


def build_panel(con, tickers=None, dates=None, prices=None, eps=None, pe_mode="lagged"):
    """Raw signals for each stock at each month-end in `dates` (default: all month-ends).
    Ranks are not filled here: that is signals.add_ranks. Rows are ordered by date, then ticker.
    pe_mode: 'lagged' (honest, the default); 'no_lag' and 'today' look ahead on purpose, to measure the bias."""
    if pe_mode not in config.EPS_MODES:
        raise ValueError(f"pe_mode must be one of {config.EPS_MODES}")
    all_me = month_end_dates(con)
    index = {d: i for i, d in enumerate(all_me)}
    dates = list(all_me if dates is None else dates)
    for d in dates:
        if d not in index:
            raise ValueError(f"{d} is not a month-end in the data")
    tickers = list(default_tickers(con) if tickers is None else tickers)
    prices = prices or PriceBook(con)
    eps = eps or EpsBook(con)

    rows = []
    for date in sorted(dates):
        k = index[date]
        for ticker in tickers:
            row = Row(ticker, date)
            # ---- P/E: month-end price over TTM EPS ----
            quote, why = prices.quote(ticker, date)
            if quote is None:
                row.reasons.append(why)
            else:
                row.price, row.price_date, row.carried_days = quote.price, quote.price_date, quote.carried_days
            ttm, why = eps.ttm(ticker, date, pe_mode)
            if ttm is not None:
                row.ttm_eps, row.eps_filed = ttm.value, ttm.latest_filed
            if why:
                row.reasons.append(why)
            if row.price is not None and ttm is not None and why is None:
                row.pe = row.price / ttm.value
            # ---- momentum: price a month ago over price 12 months ago ----
            if k < config.MIN_HISTORY_MONTHS:
                row.reasons.append(REASON_HISTORY)
            else:
                recent_date = all_me[k - config.MOMENTUM_SKIP_MONTHS]
                old_date = all_me[k - config.MOMENTUM_LOOKBACK_MONTHS]
                recent, why1 = prices.quote(ticker, recent_date)
                old, why2 = prices.quote(ticker, old_date)
                for label, why in (("t-1", why1), ("t-12", why2)):
                    if why:
                        row.reasons.append(f"momentum ({label}): {why}")
                if recent is not None and old is not None:
                    row.momentum = recent.price / old.price - 1.0
                    row.mom_price_dates = (recent.price_date, old.price_date)
            rows.append(row)
    return rows


def summarize(rows):
    """Counts for a report: how many rows, how many usable, how many used a carried price, and why the rest are not."""
    reasons = Counter()
    for r in rows:
        for why in r.reasons:
            reasons[why] += 1
    return {
        "rows": len(rows),
        "eligible": sum(1 for r in rows if r.eligible),
        "carried": sum(1 for r in rows if r.carried_days > 0),
        "reasons": reasons,
    }


def audit_lookahead(rows):
    """Every date a row relied on must be on or before its own month-end. Returns the violations (empty = clean)."""
    bad = []
    for r in rows:
        used = [("price", r.price_date), ("EPS filing", r.eps_filed)] + [("momentum price", d) for d in r.mom_price_dates]
        for what, d in used:
            if d is not None and d > r.date:
                bad.append((r.ticker, r.date, what, d))
    return bad
