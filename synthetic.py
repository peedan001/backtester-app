"""synthetic.py: a fake stock market where we KNOW the true answer.

Why: if the backtester cannot find an edge that we planted, it cannot be trusted
to find a real one. And when we plant nothing (edge = 0), it must find nothing.

HOW THE PLANTED EDGE WORKS (no look-ahead, by construction)
  1. EPS is generated first, on its own. Each quarterly EPS has a filing date
     (period end + 30 to 55 days). Prices play no part in it.
  2. Prices are then built month by month. At each month-end d we compute, from
     data known at d only, the same two signals the backtester will use:
        momentum(d) = price(previous month-end) / price(12 month-ends ago) - 1     ("12-1")
        pe(d)       = price(d) / (sum of the 4 latest EPS filed on or before d)
     Stocks are ranked 0..1 on each: value rank vr = 1 for the cheapest (lowest
     positive P/E), momentum rank mr = 1 for the strongest.
  3. Next month, each stock gets an extra yearly drift of
        value_edge * (2*vr - 1)  +  momentum_edge * (2*mr - 1)
     with value_edge = edge * value_share and momentum_edge = edge * (1 - value_share).
     So a stock that is both cheapest and strongest earns +edge a year more than
     normal, the worst earns -edge, and the middle earns 0. The edge averages to
     zero across the universe, so it does not inflate the market return.
  4. Everything else is noise: a market factor, a beta per stock, and each
     stock's own random moves.

TWO OPTIONAL FEATURES FOR STAGE 6 (both off by default, so earlier stages produce exactly the same market):
  reaction > 0   a price jump on each filing date, in proportion to the change in trailing EPS that the
                 filing reveals (so earnings news moves prices, as it does in real markets). This is what
                 makes look-ahead in EPS profitable, and therefore measurable.
  n_dead > 0     a graveyard: extra stocks that are NOT in today's list. Each one decays for about nine months
                 and then stops trading. A backtest on today's list never sees them; one on everything does.

The same month-end signal definitions are used again in Stage 4 (signals.py),
which will be checked against the numbers stored here in market["truth"].

PLANTED DEFECTS (defects=True): a few deliberate data problems in the raw feeds
(a price spike, a zero price, duplicate rows, a gap, text instead of a number,
an infinite price, an impossible date, EPS "filed" before its quarter ended).
Each is recorded in market["defects"] with what the pipeline SHOULD do about it.
The clean data underneath is identical with or without defects.
"""
import calendar
import datetime as dt
import math

import numpy as np

import config

BENCHMARK_NAME = "Synthetic Market Index"


# ---------------------------------------------------------------------------
# calendar helpers
# ---------------------------------------------------------------------------

def business_days(start, end):
    """All Monday-to-Friday dates from start to end inclusive (numpy datetime64[D]).
    Synthetic calendar: no holidays."""
    days = np.arange(np.datetime64(start), np.datetime64(end) + np.timedelta64(1, "D"), dtype="datetime64[D]")
    weekday = (days.astype("int64") + 3) % 7          # Monday = 0; 1970-01-01 was a Thursday (= 3)
    return days[weekday < 5]


def month_end_indices(days):
    """Index of the last business day of each calendar month."""
    months = days.astype("datetime64[M]")
    last = np.nonzero(months[1:] != months[:-1])[0]    # index just before the month changes
    return np.append(last, len(days) - 1)


def quarter_ends(first_year, end_date):
    """Quarter-end dates (Mar 31, Jun 30, Sep 30, Dec 31) up to end_date."""
    end = dt.date.fromisoformat(end_date)
    out = []
    for year in range(first_year, end.year + 1):
        for month in (3, 6, 9, 12):
            d = dt.date(year, month, calendar.monthrange(year, month)[1])
            if d <= end:
                out.append(d)
    return np.array(out, dtype="datetime64[D]")


# ---------------------------------------------------------------------------
# the signals, exactly as defined in the docstring
# ---------------------------------------------------------------------------

def pct_rank(x):
    """Rank each value 0..1 among the non-NaN values (lowest = 0, highest = 1). NaN stays NaN."""
    x = np.asarray(x, dtype=float)
    out = np.full(len(x), np.nan)
    ok = ~np.isnan(x)
    m = int(ok.sum())
    if m == 0:
        return out
    if m == 1:
        out[ok] = 0.5
        return out
    order = np.argsort(x[ok], kind="stable")
    ranks = np.empty(m)
    ranks[order] = np.arange(m)
    out[ok] = ranks / (m - 1)
    return out


def ttm_eps_at(eps_q, avail_q, date):
    """Trailing-twelve-month EPS known on `date`: the sum of the 4 latest quarters filed
    on or before that date. NaN if fewer than 4 quarters have been filed.
    eps_q and avail_q have shape (quarters, stocks); quarters are in time order."""
    n = eps_q.shape[1]
    counts = (avail_q <= date).sum(axis=0)             # quarters filed so far, per stock
    cs = np.vstack([np.zeros((1, n)), np.cumsum(eps_q, axis=0)])
    idx = np.arange(n)
    ttm = cs[counts, idx] - cs[np.maximum(counts - 4, 0), idx]
    ttm[counts < 4] = np.nan
    return ttm


# ---------------------------------------------------------------------------
# the generator
# ---------------------------------------------------------------------------

def _make_eps(rng, n, qends):
    nq = len(qends)
    level0 = np.exp(rng.normal(1.2, 0.5, n))                      # yearly EPS, about $3 to $4
    growth = rng.normal(0.02, 0.04, n)                            # yearly EPS growth (about the median price growth)
    shocks = rng.normal(0.0, 0.06, (nq, n))
    level = level0 * np.exp(np.cumsum(growth / 4 + shocks, axis=0))
    eps = level / 4 * (1 + 0.05 * rng.normal(size=(nq, n)))

    # a few loss-making stretches, so the P/E rule "negative earnings are excluded" is exercised
    n_loss = max(1, int(round(0.05 * n)))
    for i in rng.choice(n, n_loss, replace=False):
        start_q = int(rng.integers(12, nq - 8))
        eps[start_q:start_q + 6, i] = -level[start_q:start_q + 6, i] / 4 * 0.5

    eps = np.round(eps, 4)     # the feed publishes 4 decimals; the truth must use exactly what the feed shows
    lag_days = rng.integers(30, 56, size=(nq, n))                 # days from quarter end to filing
    avail = qends[:, None] + lag_days.astype("timedelta64[D]")
    return eps, avail


FAR_FUTURE = np.datetime64("2100-01-01")


def _earnings_jumps(eps_q, avail_q, days, reaction):
    """Log price jumps on filing days. A filing reveals the new trailing-twelve-month (TTM) EPS; the jump is
    reaction x (TTM change as a share of the old TTM), clipped, then centred so that across stocks the AVERAGE
    simple return of the jumps is exactly zero in every quarter (no stock group drifts away from the market).
    Returns an array (days, stocks), almost all zeros."""
    nq, n = eps_q.shape
    T = len(days)
    jump = np.zeros((T, n))
    cs = np.vstack([np.zeros((1, n)), np.cumsum(eps_q, axis=0)])
    clip = config.SYNTH_REACTION_CLIP
    for q in range(4, nq):
        ttm_now = cs[q + 1] - cs[q - 3]                      # EPS of quarters q-3 .. q
        ttm_before = cs[q] - cs[q - 4]                       # EPS of quarters q-4 .. q-1
        change = (ttm_now - ttm_before) / np.maximum(np.abs(ttm_before), 0.5)
        j = np.clip(reaction * change, -clip, clip)
        day = np.searchsorted(days, avail_q[q])              # first trading day on or after the filing date
        ok = (day >= 1) & (day < T)
        if not ok.any():
            continue
        j[ok] -= np.log(np.mean(np.exp(j[ok])))              # centre: mean simple jump = 0
        jump[day[ok], np.nonzero(ok)[0]] += j[ok]
    return jump


def _graveyard(seed, n_dead, days, me_idx):
    """When each failed stock stops trading (a day index), how far it falls first (total log decline) and over how many
    days. Some failures are gradual (months of decline that a momentum signal can see coming) and some abrupt
    (a few weeks, which it cannot). Uses its own random stream, so adding a graveyard never disturbs the live stocks'
    random numbers."""
    rng = np.random.default_rng(seed + 5000)
    K = len(me_idx)
    last_month = K - 4
    first_month = config.SYNTH_FIRST_DELIST_MONTH
    months = rng.integers(first_month, last_month + 1, n_dead)
    delist = np.empty(n_dead, dtype=int)
    for i, k in enumerate(months):
        lo, hi = me_idx[k - 1] + 1, me_idx[k]                # somewhere inside month k
        delist[i] = int(rng.integers(lo, hi + 1))
    crash = rng.uniform(*config.SYNTH_CRASH_RANGE, n_dead)
    abrupt = rng.random(n_dead) < config.SYNTH_ABRUPT_SHARE
    window = np.where(abrupt, rng.integers(config.SYNTH_ABRUPT_DAYS[0], config.SYNTH_ABRUPT_DAYS[1] + 1, n_dead),
                      config.SYNTH_CRASH_DAYS)
    return delist, crash, window


def make_market(n_stocks=config.SYNTH_N_STOCKS, seed=config.SYNTH_SEED,
                edge=config.SYNTH_DEFAULT_EDGE, value_share=0.5, defects=True, reaction=0.0, n_dead=0):
    """Build a synthetic market. See the module docstring for the design.

    edge         extra yearly return for a stock with a perfect score (0 = null world)
    value_share  1.0 = value edge only, 0.0 = momentum edge only, 0.5 = both equally
    defects      plant the deliberate data problems in the raw feeds
    reaction     0 = earnings do not move prices; r > 0 = a filing moves the price by r x the change in trailing EPS
    n_dead       failed stocks added on top of n_stocks (they are not in today's list and stop trading before the end)
    """
    if not 10 <= n_stocks <= 1000:
        raise ValueError("n_stocks must be between 10 and 1000")
    if not 0 <= edge <= 0.5:
        raise ValueError("edge must be between 0 and 0.5 (a yearly rate, so 0.05 = 5%)")
    if not 0 <= value_share <= 1:
        raise ValueError("value_share must be between 0 and 1")
    if not 0 <= reaction <= config.SYNTH_MAX_REACTION:
        raise ValueError(f"reaction must be between 0 and {config.SYNTH_MAX_REACTION}")
    if int(n_dead) != n_dead or not 0 <= n_dead <= config.SYNTH_MAX_DEAD:
        raise ValueError(f"n_dead must be a whole number between 0 and {config.SYNTH_MAX_DEAD}")
    n_dead = int(n_dead)

    rng = np.random.default_rng(seed)
    n_live = n_stocks
    n = n_stocks + n_dead
    value_edge = edge * value_share
    momentum_edge = edge * (1 - value_share)

    days = business_days(config.DATA_START, config.END_DATE)
    T = len(days)
    me_idx = month_end_indices(days)
    K = len(me_idx)

    qends = quarter_ends(2011, config.END_DATE)
    eps_q, avail_q = _make_eps(rng, n, qends)

    # ---- the graveyard: failed stocks decay, then stop trading ---------------
    alive = np.ones((T, n), dtype=bool)                      # False once a stock no longer trades
    extra = np.zeros((T, n))                                 # extra daily log drift (the decay before a failure)
    last_day = np.full(n, T - 1)
    crash = np.zeros(n)
    crash_days = np.zeros(n, dtype=int)
    if n_dead:
        delist, fall, window = _graveyard(seed, n_dead, days, me_idx)
        for j, i in enumerate(range(n_live, n)):
            dd = int(delist[j])
            last_day[i], crash[i], crash_days[i] = dd, fall[j], int(window[j])
            alive[dd + 1:, i] = False
            start = max(1, dd - int(window[j]) + 1)
            extra[start:dd + 1, i] = -fall[j] / (dd - start + 1)
            avail_q[:, i] = np.where(avail_q[:, i] > days[dd], FAR_FUTURE, avail_q[:, i])   # no filings after failure
    jump = _earnings_jumps(eps_q, avail_q, days, reaction) if reaction > 0 else np.zeros((T, n))

    # ---- noise: a market factor, a beta per stock, own moves per stock -----
    # Volatility drag: a stock with yearly volatility v loses v^2/2 of LOG growth, so its
    # expected SIMPLE return still equals beta x market return. Without this correction,
    # random stocks would beat the index on average, and the null world (edge = 0) would
    # show false alpha.
    mkt_vol_d = config.SYNTH_MARKET_VOL / math.sqrt(252)
    mkt_noise = rng.normal(0.0, mkt_vol_d, T)
    beta = np.clip(rng.normal(1.0, 0.25, n), 0.3, 1.8)
    idio_vol = rng.uniform(*config.SYNTH_IDIO_VOL, n)
    idio_vol_d = idio_vol / math.sqrt(252)
    log_drift = beta * config.SYNTH_MARKET_RETURN / 252 - 0.5 * ((beta * mkt_vol_d) ** 2 + idio_vol_d ** 2)
    base = mkt_noise[:, None] * beta[None, :] + rng.normal(size=(T, n)) * idio_vol_d + log_drift
    base[0] = 0.0
    mkt_log = config.SYNTH_MARKET_RETURN / 252 - 0.5 * mkt_vol_d ** 2 + mkt_noise

    # ---- starting prices: P/E differs by stock, around a typical value -----
    ttm0 = ttm_eps_at(eps_q, avail_q, days[0])
    pe0 = np.exp(rng.normal(math.log(config.SYNTH_START_PRICE_PE), 0.45, n))
    P = np.zeros((T, n))
    P[0] = pe0 * ttm0

    # ---- build prices month by month; signals use data up to the month-end --
    pe_hist = np.full((K, n), np.nan)
    mom_hist = np.full((K, n), np.nan)
    vr_hist = np.full((K, n), np.nan)
    mr_hist = np.full((K, n), np.nan)
    drift_hist = np.zeros((K, n))            # yearly planted drift that applies to the month AFTER month-end k

    def fill(a, b, drift_daily):
        """Fill days a..b inclusive using log returns plus a daily drift."""
        lr = base[a:b + 1] + drift_daily + extra[a:b + 1] + jump[a:b + 1]
        P[a:b + 1] = P[a - 1] * np.exp(np.cumsum(lr, axis=0))

    fill(1, me_idx[0], 0.0)                  # first (partial) month: no signals yet
    for k in range(K):
        d = me_idx[k]
        ttm = ttm_eps_at(eps_q, avail_q, days[d])
        pe = np.where(ttm > 0, P[d] / np.where(ttm > 0, ttm, 1.0), np.nan)
        pe = np.where(alive[d], pe, np.nan)                   # a stock that no longer trades has no signal
        pe_hist[k] = pe
        if k >= 12:
            mom_hist[k] = np.where(alive[d], P[me_idx[k - 1]] / P[me_idx[k - 12]] - 1.0, np.nan)
        vr_hist[k] = 1.0 - pct_rank(pe)                       # 1 = cheapest
        mr_hist[k] = pct_rank(mom_hist[k])                    # 1 = strongest
        drift = (value_edge * np.nan_to_num(2 * vr_hist[k] - 1, nan=0.0)
                 + momentum_edge * np.nan_to_num(2 * mr_hist[k] - 1, nan=0.0))
        drift_hist[k] = drift
        if k + 1 < K:
            fill(me_idx[k] + 1, me_idx[k + 1], drift / 252)

    # signal-only score (does NOT depend on edge): what a perfect-knowledge strategy ranks by
    score = (value_share * np.nan_to_num(vr_hist, nan=0.5)
             + (1 - value_share) * np.nan_to_num(mr_hist, nan=0.5))

    spy = 100.0 * np.exp(np.cumsum(np.where(np.arange(T) == 0, 0.0, mkt_log)))

    live_tickers = [f"S{i + 1:03d}" for i in range(n_live)]
    dead_tickers = [f"D{i + 1:03d}" for i in range(n_dead)]
    tickers = live_tickers + dead_tickers
    date_str = [str(x) for x in days]
    stocks = [(t, f"Synthetic Corp {i + 1:03d}", 1, 0) for i, t in enumerate(live_tickers)]
    stocks += [(t, f"Failed Corp {i + 1:03d}", 0, 0) for i, t in enumerate(dead_tickers)]
    stocks.append((config.BENCHMARK_TICKER, BENCHMARK_NAME, 0, 1))

    prices = {t: [(date_str[j], round(float(P[j, i]), 4)) for j in range(last_day[i] + 1)] for i, t in enumerate(tickers)}
    prices[config.BENCHMARK_TICKER] = [(date_str[j], round(float(spy[j]), 4)) for j in range(T)]
    eps = {t: [(str(qends[q]), str(avail_q[q, i]), round(float(eps_q[q, i]), 4)) for q in range(len(qends))
               if avail_q[q, i] < FAR_FUTURE]
           for i, t in enumerate(tickers)}

    market = {
        "stocks": stocks,
        "prices": prices,
        "eps": eps,
        "defects": [],
        "truth": {
            "seed": seed, "edge": edge, "value_share": value_share,
            "value_edge": value_edge, "momentum_edge": momentum_edge,
            "tickers": tickers, "live_tickers": live_tickers, "dead_tickers": dead_tickers,
            "reaction": reaction, "n_dead": n_dead,
            "alive": alive,                                 # (days, stocks): does the stock still trade that day?
            "delist_date": {t: date_str[last_day[n_live + j]] for j, t in enumerate(dead_tickers)},
            "crash": {t: float(crash[n_live + j]) for j, t in enumerate(dead_tickers)},   # total log decline before failure
            "crash_days": {t: int(crash_days[n_live + j]) for j, t in enumerate(dead_tickers)},   # how many days that fall took
            "days": date_str,
            "price_matrix": P,                              # clean prices, shape (days, stocks)
            "month_end_idx": me_idx,
            "month_ends": [date_str[j] for j in me_idx],
            "pe": pe_hist, "momentum": mom_hist,
            "value_rank": vr_hist, "momentum_rank": mr_hist,
            "score": score, "planted_drift": drift_hist,
        },
    }
    if defects:
        plant_defects(market, seed + 1000)
    return market


# ---------------------------------------------------------------------------
# planted defects
# ---------------------------------------------------------------------------

def plant_defects(market, seed):
    """Damage the raw feeds in market (in place) and record what should happen.

    expected outcomes:
      'flagged'        rows kept, flagged by the validator
      'rejected'       the database refuses the row
      'deduplicated'   repeated rows are stored once
    """
    rng = np.random.default_rng(seed)
    tickers = market["truth"].get("live_tickers", market["truth"]["tickers"])     # defects go into live stocks only
    chosen = [str(t) for t in rng.choice(tickers, 8, replace=False)]
    out = market["defects"]

    def pick_row(ticker, lo=600, hi=2800):
        return int(rng.integers(lo, min(hi, len(market["prices"][ticker]) - 40)))

    # 1. a price spike: one price is 10 times too big for one day
    t = chosen[0]
    j = pick_row(t)
    d, p = market["prices"][t][j]
    market["prices"][t][j] = (d, round(p * 10, 4))
    out.append({"kind": "spike", "ticker": t, "date": d, "expected": "flagged",
                "detail": f"price on {d} is 10x too big for one day (two jumps: in and out)"})

    # 2. a zero price
    t = chosen[1]
    j = pick_row(t)
    d, _ = market["prices"][t][j]
    market["prices"][t][j] = (d, 0.0)
    out.append({"kind": "zero_price", "ticker": t, "date": d, "expected": "rejected",
                "detail": f"price on {d} is 0.0"})

    # 3. duplicate rows: five days appear twice in the feed
    t = chosen[2]
    j = pick_row(t)
    dups = list(market["prices"][t][j:j + 5])
    market["prices"][t].extend(dups)
    out.append({"kind": "duplicate_rows", "ticker": t, "date": dups[0][0], "expected": "deduplicated",
                "detail": f"5 rows starting {dups[0][0]} are repeated at the end of the feed"})

    # 4. a gap: 15 business days of prices are missing
    t = chosen[3]
    j = pick_row(t)
    first, last = market["prices"][t][j][0], market["prices"][t][j + 14][0]
    del market["prices"][t][j:j + 15]
    out.append({"kind": "gap", "ticker": t, "date": last, "expected": "flagged",
                "detail": f"15 business days missing, {first} to {last}"})

    # 5. EPS "filed" before its quarter ended (look-ahead!)
    t = chosen[4]
    q = 20
    period_end, _, e = market["eps"][t][q]
    bad_avail = str(np.datetime64(period_end) - np.timedelta64(10, "D"))
    market["eps"][t][q] = (period_end, bad_avail, e)
    out.append({"kind": "eps_before_period", "ticker": t, "date": period_end, "expected": "rejected",
                "detail": f"EPS for quarter ending {period_end} claims it was filed on {bad_avail}"})

    # 6. text instead of a number
    t = chosen[5]
    j = pick_row(t)
    d, _ = market["prices"][t][j]
    market["prices"][t][j] = (d, "N/A")
    out.append({"kind": "text_price", "ticker": t, "date": d, "expected": "rejected",
                "detail": f"price on {d} is the text 'N/A'"})

    # 7. infinity
    t = chosen[6]
    j = pick_row(t)
    d, _ = market["prices"][t][j]
    market["prices"][t][j] = (d, float("inf"))
    out.append({"kind": "infinite_price", "ticker": t, "date": d, "expected": "rejected",
                "detail": f"price on {d} is infinite"})

    # 8. an impossible date
    t = chosen[7]
    j = pick_row(t)
    d, p = market["prices"][t][j]
    market["prices"][t][j] = ("2020-13-45", p)
    out.append({"kind": "bad_date", "ticker": t, "date": "2020-13-45", "row_date": d, "expected": "rejected",
                "detail": f"the row for {d} carries the impossible date 2020-13-45"})
    return out


# ---------------------------------------------------------------------------
# a sanity check on the planted edge, independent of the backtester
# ---------------------------------------------------------------------------

def planted_quintile_spread(market, quantile=0.2):
    """Using the CLEAN prices and the signal-only score, hold the top quintile (best score)
    and the bottom quintile each month, equal weight, and return the average yearly
    return of the top minus the bottom. About 0 in a null world; positive when an edge is planted.

    This is not a backtest (no costs, perfect knowledge of the stored ranks); it only
    proves that the edge really is in the data."""
    tr = market["truth"]
    P, me_idx, score = tr["price_matrix"], tr["month_end_idx"], tr["score"]
    K = len(me_idx)
    diffs = []
    alive = tr.get("alive")
    for k in range(13, K - 1):                       # signals exist from month-end 12
        r = P[me_idx[k + 1]] / P[me_idx[k]] - 1.0
        idx = np.arange(P.shape[1]) if alive is None else np.nonzero(alive[me_idx[k + 1]])[0]    # only stocks still trading
        order = idx[np.argsort(score[k][idx], kind="stable")]
        m = max(1, int(round(quantile * len(order))))
        diffs.append(r[order[-m:]].mean() - r[order[:m]].mean())
    return float(np.mean(diffs) * 12)


# ---------------------------------------------------------------------------
# fake fetchers: they look like the real ones, but read from a synthetic market
# ---------------------------------------------------------------------------

def fake_fetchers(market):
    """Return {'universe', 'prices', 'eps'} callables with the same signatures as the live fetchers."""
    def universe():
        return [(t, name) for t, name, in_sp500, is_bench in market["stocks"] if in_sp500 and not is_bench]

    def prices(ticker, start, end):
        if ticker not in market["prices"]:
            raise KeyError(f"no such ticker: {ticker}")
        return [(d, p) for d, p in market["prices"][ticker] if start <= d <= end]

    def eps(ticker):
        if ticker not in market["eps"]:
            raise KeyError(f"no such ticker: {ticker}")
        return list(market["eps"][ticker])

    return {"universe": universe, "prices": prices, "eps": eps}
