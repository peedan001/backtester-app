"""engine.py: turn month-end target portfolios into a return series, with and without costs, next to the benchmark.

The rules (every one is a decision made in the design talk, and a test below):
  * The signal is computed at the month-end close. The trade happens at the close `delay` trading days later
    (default 1: the next trading day), so we never trade at a price we only knew after computing the signal.
  * Between trades nothing is bought or sold. Each holding's value drifts with its own price, so the weights drift
    away from equal. At the next trade we measure the trade from the DRIFTED weights to the new targets.
  * Cost = cost_bps / 10000 x (value bought + value sold) / portfolio value. Buys and sells both pay.
  * A held stock whose data has ENDED (its last price is older than the carry limit and nothing follows) is
    DELISTED: it is marked at its last price, sold at that price, and the money waits in cash for the next
    rebalance. A loss before the end is therefore realised, never hidden.
  * A stock with no usable price on the trade day cannot be traded. If we do not hold it, it is skipped and the rest
    are re-split. If we hold it, it is FROZEN: it keeps its last known value (return 0 that period), is neither
    sold nor topped up, and the rebalance applies to everything else.
  * Cash earns nothing. The benchmark is bought on the first trade day and held, with no costs.
  * One result row per trade day. Its returns cover the period since the previous trade day, earned by the
    portfolio held during that period. The first row is the inception row: return 0, and it pays the entry cost.

`replay_with_shares` re-does the same story by counting shares and cash instead of tracking values, as an
independent cross-check of the arithmetic.
"""
import bisect
import math
from dataclasses import dataclass, field

import align
import config

NAN = float("nan")


# ---------------------------------------------------------------------------
# validation and calendar
# ---------------------------------------------------------------------------

def _check_params(cost_bps, delay):
    if not isinstance(cost_bps, (int, float)) or isinstance(cost_bps, bool) or not 0 <= cost_bps <= config.MAX_COST_BPS:
        raise ValueError(f"cost_bps must be a number between 0 and {config.MAX_COST_BPS}")
    if int(delay) != delay or not 0 <= delay <= config.MAX_EXEC_DELAY_DAYS:
        raise ValueError(f"delay must be a whole number of trading days between 0 and {config.MAX_EXEC_DELAY_DAYS}")


def trading_days(con):
    """Every date the benchmark traded, in order. This is the calendar for execution dates."""
    bench = con.execute("SELECT ticker FROM STOCK WHERE is_benchmark = 1").fetchall()
    if len(bench) != 1:
        raise ValueError(f"expected exactly one benchmark in STOCK, found {len(bench)}")
    return [r[0] for r in con.execute("SELECT date FROM PRICE WHERE ticker = ? ORDER BY date", (bench[0][0],))], bench[0][0]


def execution_date(calendar, signal_date, delay=config.EXEC_DELAY_DAYS):
    """The trading day `delay` days after the signal date, or None if the data ends first."""
    i = bisect.bisect_left(calendar, signal_date)
    if i >= len(calendar) or calendar[i] != signal_date:
        raise ValueError(f"{signal_date} is not a trading day of the benchmark")
    j = i + int(delay)
    return calendar[j] if j < len(calendar) else None


def schedule(portfolio, calendar, delay=config.EXEC_DELAY_DAYS):
    """[(signal_date, trade_date, {ticker: weight})] in time order, up to where the data ends."""
    out = []
    for signal_date in sorted(portfolio):
        trade_date = execution_date(calendar, signal_date, delay)
        if trade_date is None:
            break
        out.append((signal_date, trade_date, {p.ticker: p.weight for p in portfolio[signal_date]}))
    return out


# ---------------------------------------------------------------------------
# the simulation
# ---------------------------------------------------------------------------

@dataclass
class Period:
    date: str                  # the trade day that closes this period
    signal_date: str           # the month-end whose signal was traded on this date
    gross: float               # return of the portfolio held over the period, before costs
    cost: float                # trading cost paid on this date, as a fraction of portfolio value
    net: float                 # return after that cost
    benchmark: float
    traded_fraction: float     # (bought + sold) / portfolio value
    n_held: int                # stocks held after this trade
    skipped: tuple = ()        # target stocks that could not be bought (no usable price)
    unpriced: int = 0          # held stocks with no usable price at this date (frozen at last value)
    before: dict = field(default_factory=dict)   # drifted weights just before the trade (share of total wealth)
    after: dict = field(default_factory=dict)    # weights right after the trade (share of total wealth, frozen names included)
    cash: float = 0.0          # share of wealth in cash after the trade
    delisted: tuple = ()       # held stocks whose data had ended: sold at their last price on this date


def _rebalance_plan(target, quote_at):
    """Drop stocks without a price today and re-split what is left. Returns (weights, prices, skipped)."""
    prices, skipped = {}, []
    for t in sorted(target):
        q, _ = quote_at(t)
        if q is None:
            skipped.append(t)
        else:
            prices[t] = q.price
    total = sum(target[t] for t in prices)
    weights = {t: target[t] / total for t in prices} if prices and total > 0 else {}
    return weights, prices, tuple(skipped)


def run_backtest(portfolio, prices, calendar, benchmark, cost_bps=config.DEFAULT_COST_BPS, delay=config.EXEC_DELAY_DAYS):
    """Simulate. `portfolio` is {signal_date: [Position]} from portfolio.build_portfolio; `prices` is an align.PriceBook."""
    _check_params(cost_bps, delay)
    periods = []
    value = {}                     # ticker -> value of the holding, in units of the starting wealth
    last_price = {}                # ticker -> the price that value was last marked at
    cash = 1.0                     # we start in cash, with a wealth of 1
    wealth = 1.0                   # wealth after the last trade (costs already deducted: this is the NET wealth)
    started = False
    prev_bench = None

    for signal_date, date, target in schedule(portfolio, calendar, delay):
        if not started and not target:
            continue                                         # nothing held yet and nothing to buy
        bq, _ = prices.quote(benchmark, date, 0)
        if bq is None:
            raise ValueError(f"no benchmark price on {date}")

        frozen, delisted = set(), []
        for t in list(value):                                # mark what we hold to today's prices
            q, _ = prices.quote(t, date)
            if q is None and prices.delisted(t, date):
                final = prices.last_price(t)                 # the data has ended: mark at the last price, then sell it
                if final is not None:
                    value[t] *= final / last_price[t]
                    last_price[t] = final
                    delisted.append(t)
                else:
                    frozen.add(t)
            elif q is None:
                frozen.add(t)
            else:
                value[t] *= q.price / last_price[t]
                last_price[t] = q.price
        total = sum(value.values()) + cash
        gross = total / wealth - 1.0                         # 0 on the first row: nothing was held yet
        bench_ret = bq.price / prev_bench - 1.0 if started else 0.0
        before = {t: v / total for t, v in value.items()}

        weights, new_prices, skipped = _rebalance_plan(target, lambda t: prices.quote(t, date))
        skipped = tuple(t for t in skipped if t not in frozen)       # a held stock is frozen, not "skipped"
        frozen_value = sum(value[t] for t in frozen)
        investable = total - frozen_value                    # cash plus every holding we can actually trade
        tradable = {t: v for t, v in value.items() if t not in frozen}
        # Value bought plus value sold. Buying from cash counts as buying everything; moving cash to cash costs nothing.
        traded = sum(abs(weights.get(t, 0.0) * investable - tradable.get(t, 0.0)) for t in set(weights) | set(tradable))
        cost_value = cost_bps / 10000.0 * traded
        investable -= cost_value
        value = {t: value[t] for t in frozen}
        last_price = {t: last_price[t] for t in frozen}
        if weights:
            value.update({t: w * investable for t, w in weights.items()})
            last_price.update(new_prices)
            cash = 0.0
        else:
            cash = investable
        wealth = total - cost_value
        cost_frac = cost_value / total if total > 0 else 0.0
        net = (1.0 + gross) * (1.0 - cost_frac) - 1.0
        started = True
        prev_bench = bq.price
        periods.append(Period(date, signal_date, gross, cost_frac, net, bench_ret,
                              traded / total if total > 0 else 0.0, len(value), skipped, len(frozen), before,
                              {t: v / wealth for t, v in value.items()}, cash / wealth, tuple(sorted(delisted))))
    return periods


def replay_with_shares(portfolio, prices, calendar, benchmark, cost_bps=config.DEFAULT_COST_BPS, delay=config.EXEC_DELAY_DAYS):
    """The same story told by counting shares and cash. Returns [(date, wealth after the trade)].
    Written separately from run_backtest on purpose: if the two disagree, one of them has a bug."""
    _check_params(cost_bps, delay)
    shares, mark, cash = {}, {}, 1.0                          # start in cash, wealth 1
    started = False
    out = []
    for signal_date, date, target in schedule(portfolio, calendar, delay):
        if not started and not target:
            continue
        frozen = set()
        for t in shares:
            q, _ = prices.quote(t, date)
            if q is None and prices.delisted(t, date) and prices.last_price(t) is not None:
                mark[t] = prices.last_price(t)               # data ended: it trades at its last price (and is sold below)
            elif q is None:
                frozen.add(t)                                # no price: cannot trade it, keep its last mark
            else:
                mark[t] = q.price
        wealth = cash + sum(n * mark[t] for t, n in shares.items())
        investable = wealth - sum(shares[t] * mark[t] for t in frozen)
        want, px, _ = _rebalance_plan(target, lambda t: prices.quote(t, date))
        now_value = {t: n * mark[t] for t, n in shares.items() if t not in frozen}
        names = set(want) | set(now_value)
        bought = sum(max(want.get(t, 0.0) * investable - now_value.get(t, 0.0), 0.0) for t in names)
        sold = sum(max(now_value.get(t, 0.0) - want.get(t, 0.0) * investable, 0.0) for t in names)
        fee = cost_bps / 10000.0 * (bought + sold)
        investable -= fee
        wealth -= fee
        kept = {t: shares[t] for t in frozen}
        kept_mark = {t: mark[t] for t in frozen}
        shares = {**kept, **{t: want[t] * investable / px[t] for t in want}}
        mark = {**kept_mark, **px}
        cash = 0.0 if want else investable
        started = True
        out.append((date, wealth))
    return out


def net_wealth_path(periods):
    """Wealth after each row, starting from 1, compounding the NET returns."""
    path, w = [], 1.0
    for p in periods:
        w *= 1.0 + p.net
        path.append(w)
    return path


def check_engine(periods, replay, tol=1e-9):
    """Cross-check: the share ledger must give the same net wealth on every date. Returns a list of problems."""
    bad = []
    path = net_wealth_path(periods)
    if len(replay) != len(periods):
        return [f"the engine has {len(periods)} rows but the ledger has {len(replay)}"]
    for p, w, (d, w2) in zip(periods, path, replay):
        if p.date != d:
            bad.append(f"dates differ: {p.date} vs {d}")
        elif abs(w - w2) > tol:
            bad.append(f"{d}: engine wealth {w:.10f} vs share ledger {w2:.10f}")
        if not -1e-12 <= p.cost <= 1.0:
            bad.append(f"{p.date}: cost {p.cost} out of range")
        if p.net > p.gross + 1e-12 or abs((1 + p.gross) * (1 - p.cost) - 1 - p.net) > 1e-12:
            bad.append(f"{p.date}: net return is not (1+gross)(1-cost)-1")
        if abs(sum(p.after.values()) + p.cash - 1.0) > 1e-9:
            bad.append(f"{p.date}: holdings plus cash after the trade add to {sum(p.after.values()) + p.cash}")
        if sum(p.before.values()) > 1.0 + 1e-9:
            bad.append(f"{p.date}: drifted weights add to {sum(p.before.values())}, more than the whole portfolio")
    return bad


# ---------------------------------------------------------------------------
# metrics
# ---------------------------------------------------------------------------

def _mean(xs):
    return sum(xs) / len(xs)


def _sd(xs):
    if len(xs) < 2:
        return NAN
    m = _mean(xs)
    return math.sqrt(sum((x - m) ** 2 for x in xs) / (len(xs) - 1))


def growth(returns):
    w = 1.0
    for r in returns:
        w *= 1.0 + r
    return w


def cagr(returns, n_years):
    """Compound yearly growth rate of a list of period returns over n_years."""
    if n_years <= 0:
        return NAN
    return growth(returns) ** (1.0 / n_years) - 1.0


def max_drawdown(returns):
    """The worst fall from a previous peak, as a negative fraction (0 if wealth never fell). Wealth starts at 1."""
    w = peak = 1.0
    worst = 0.0
    for r in returns:
        w *= 1.0 + r
        peak = max(peak, w)
        worst = min(worst, w / peak - 1.0)
    return worst


def sharpe(returns, rf_annual=config.RISK_FREE_RATE, periods_per_year=config.MONTHS_PER_YEAR):
    """(average excess return per period) / (volatility per period), scaled to a year by sqrt(periods)."""
    sd = _sd(returns)
    if not sd == sd or sd < 1e-12:
        return NAN
    rf = (1.0 + rf_annual) ** (1.0 / periods_per_year) - 1.0
    return (_mean(returns) - rf) / sd * math.sqrt(periods_per_year)


def beta(returns, bench):
    if len(returns) != len(bench) or len(returns) < 2:
        return NAN
    mb, mr = _mean(bench), _mean(returns)
    var = sum((b - mb) ** 2 for b in bench)
    if var < 1e-18:
        return NAN
    return sum((r - mr) * (b - mb) for r, b in zip(returns, bench)) / var


def information_ratio(returns, bench):
    diff = [r - b for r, b in zip(returns, bench)]
    sd = _sd(diff)
    if not sd == sd or sd < 1e-12:
        return NAN
    return _mean(diff) / sd * math.sqrt(config.MONTHS_PER_YEAR)


def metrics(periods, rf_annual=config.RISK_FREE_RATE):
    """Headline numbers for gross, net and the benchmark. The inception row (return 0, entry cost) counts toward
    total return and CAGR but is not a return period, so it is left out of volatility, Sharpe and the like."""
    if not 0 <= rf_annual <= config.MAX_RISK_FREE_RATE:
        raise ValueError(f"the risk-free rate must be between 0 and {config.MAX_RISK_FREE_RATE}")
    if len(periods) < 2:
        raise ValueError("need at least two result rows (an inception row and one return period)")
    years = (len(periods) - 1) / config.MONTHS_PER_YEAR
    series = {"gross": [p.gross for p in periods], "net": [p.net for p in periods], "benchmark": [p.benchmark for p in periods]}
    out = {"years": years, "periods": len(periods) - 1}
    for name, rets in series.items():
        body = rets[1:]
        out[name] = {
            "total_return": growth(rets) - 1.0,
            "cagr": cagr(rets, years),
            "volatility": _sd(body) * math.sqrt(config.MONTHS_PER_YEAR),
            "sharpe": sharpe(body, rf_annual),
            "max_drawdown": max_drawdown(rets),
        }
    net, bench = series["net"][1:], series["benchmark"][1:]
    out["excess_cagr_net"] = out["net"]["cagr"] - out["benchmark"]["cagr"]
    out["excess_cagr_gross"] = out["gross"]["cagr"] - out["benchmark"]["cagr"]
    out["cost_drag_cagr"] = out["gross"]["cagr"] - out["net"]["cagr"]
    out["beta"] = beta(net, bench)
    out["information_ratio"] = information_ratio(net, bench)
    out["hit_rate"] = sum(1 for r, b in zip(net, bench) if r > b) / len(net)
    # Same idea as Stage 4's "steady months": rows that hold something AND follow a row that held something.
    # The inception purchase (everything bought from cash) and re-entries after a stay in cash are not typical trading.
    steady = [p for prev, p in zip(periods, periods[1:]) if prev.n_held > 0 and p.n_held > 0]
    out["avg_traded_fraction"] = _mean([p.traded_fraction for p in steady]) if steady else NAN
    out["avg_cost_per_month"] = _mean([p.cost for p in steady]) if steady else NAN
    out["unpriced_marks"] = sum(p.unpriced for p in periods)
    out["skipped_buys"] = sum(len(p.skipped) for p in periods)
    out["delisted_sales"] = sum(len(p.delisted) for p in periods)
    return out


# ---------------------------------------------------------------------------
# storage (RESULT table)
# ---------------------------------------------------------------------------

def store_results(con, run_id, periods):
    """Write the rows to RESULT for this run. Safe to repeat; rows no longer in the result are removed."""
    new = {p.date: (p.gross, p.cost, p.net, p.benchmark) for p in periods}
    old = {d: tuple(v) for d, *v in con.execute(
        "SELECT date, gross_return, cost, net_return, benchmark_return FROM RESULT WHERE run_id = ?", (run_id,))}
    counts = {"inserted": 0, "updated": 0, "unchanged": 0, "deleted": 0}
    for d in set(old) - set(new):
        con.execute("DELETE FROM RESULT WHERE run_id = ? AND date = ?", (run_id, d))
        counts["deleted"] += 1
    for d, vals in new.items():
        if d not in old:
            con.execute("INSERT INTO RESULT (run_id, date, gross_return, cost, net_return, benchmark_return) "
                        "VALUES (?, ?, ?, ?, ?, ?)", (run_id, d) + vals)
            counts["inserted"] += 1
        elif old[d] != vals:
            con.execute("UPDATE RESULT SET gross_return = ?, cost = ?, net_return = ?, benchmark_return = ? "
                        "WHERE run_id = ? AND date = ?", vals + (run_id, d))
            counts["updated"] += 1
        else:
            counts["unchanged"] += 1
    con.commit()
    return counts


def load_results(con, run_id):
    return [dict(zip(("date", "gross", "cost", "net", "benchmark"), r)) for r in con.execute(
        "SELECT date, gross_return, cost, net_return, benchmark_return FROM RESULT WHERE run_id = ? ORDER BY date", (run_id,))]


class Env:
    """Prices and the trading calendar, loaded once, so many runs can share them."""

    def __init__(self, con):
        self.calendar, self.bench = trading_days(con)
        self.prices = align.PriceBook(con)

    def run(self, port, cost_bps, delay=config.EXEC_DELAY_DAYS):
        return run_backtest(port, self.prices, self.calendar, self.bench, cost_bps, delay)

    def replay(self, port, cost_bps, delay=config.EXEC_DELAY_DAYS):
        return replay_with_shares(port, self.prices, self.calendar, self.bench, cost_bps, delay)


def backtest(con, portfolio, cost_bps=config.DEFAULT_COST_BPS, delay=config.EXEC_DELAY_DAYS):
    """Convenience: load prices and the calendar from the database, then run."""
    calendar, bench = trading_days(con)
    return run_backtest(portfolio, align.PriceBook(con), calendar, bench, cost_bps, delay)
