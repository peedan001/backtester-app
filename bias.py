"""bias.py: measure how much each shortcut inflates a backtest.

A backtest can be flattered in many ways. This module runs the SAME strategy with the SAME settings and costs,
changing only the shortcut, so the difference in yearly return is the cost of that shortcut:

    honest        EPS used from its filing date, universe = everything that existed          ('lagged', 'all')
    no filing lag EPS used from the day its quarter ends (30 to 55 days before anyone could see it)
    today's EPS   every date uses the latest EPS ever filed                                  (hindsight)
    survivors     only stocks still in the index today (the failed ones never appear)

The result is only meaningful in a world where we KNOW how the data was made, which is why it runs on the
synthetic market. A real data source cannot tell you what the shortcut cost, because it cannot show you the
honest version.
"""
import math

import align
import config
import engine
import portfolio as pf
import stage3_run

NAN = float("nan")


def dead_tickers(con):
    """Stocks that traded but are not in today's list (the failed ones)."""
    return {r[0] for r in con.execute("SELECT ticker FROM STOCK WHERE is_benchmark = 0 AND in_sp500 = 0")}


def run_variant(con, env, pe_mode, universe, top_n, value_weight, buffer, cost_bps, with_ic=False):
    """One backtest. Returns the portfolio, the result rows, the metrics, and how much it leaned on failed stocks."""
    rows = stage3_run.all_signals(con, tickers=align.tickers_for(con, universe), pe_mode=pe_mode)
    port = pf.build_portfolio(rows, top_n, value_weight, buffer)
    periods = env.run(port, cost_bps)
    if len(periods) < 2:
        raise ValueError("not enough months with signals to run a backtest")
    dead = dead_tickers(con)
    m = engine.metrics(periods)
    held_dead = {t for p in periods for t in p.after if t in dead}
    out = {
        "pe_mode": pe_mode, "universe": universe, "rows": rows, "portfolio": port, "periods": periods, "metrics": m,
        "net_cagr": m["net"]["cagr"], "gross_cagr": m["gross"]["cagr"], "sharpe": m["net"]["sharpe"],
        "dead_held": len(held_dead),
        "dead_weight": sum(sum(w for t, w in p.after.items() if t in dead) for p in periods) / len(periods),
        "delisted_sales": m["delisted_sales"],
    }
    if with_ic:
        out["ic"], out["ic_t"] = stage3_run.ic_summary(stage3_run.information_coefficients(con, rows, value_weight))
    return out


def evaluate_world(con, env, top_n=config.DEFAULT_TOP_N, value_weight=config.DEFAULT_VALUE_WEIGHT,
                   buffer=config.DEFAULT_BUFFER, cost_bps=config.DEFAULT_COST_BPS, with_ic=False):
    """The four runs that separate the biases, and the yearly return each shortcut added.
    Every 'inflation' is a difference in NET return per year (costs included), in the same world."""
    args = (top_n, value_weight, buffer, cost_bps)
    honest = run_variant(con, env, "lagged", "survivors", *args, with_ic=with_ic)
    no_lag = run_variant(con, env, "no_lag", "survivors", *args, with_ic=with_ic)
    today = run_variant(con, env, "today", "survivors", *args, with_ic=with_ic)
    everything = run_variant(con, env, "lagged", "all", *args, with_ic=with_ic) if dead_tickers(con) else honest
    return {
        "lagged": honest, "no_lag": no_lag, "today": today, "all": everything,
        "inflation_no_lag": no_lag["net_cagr"] - honest["net_cagr"],
        "inflation_today": today["net_cagr"] - honest["net_cagr"],
        "survivorship": honest["net_cagr"] - everything["net_cagr"],
        "total": today["net_cagr"] - everything["net_cagr"],          # the naive backtest against the fully honest one
    }


def worlds_over_seeds(seeds, edge=config.SYNTH_DEFAULT_EDGE, reaction=config.SYNTH_DEFAULT_REACTION,
                      n_dead=config.SYNTH_DEFAULT_DEAD, top_n=config.DEFAULT_TOP_N, value_weight=config.DEFAULT_VALUE_WEIGHT,
                      buffer=config.DEFAULT_BUFFER, cost_bps=config.DEFAULT_COST_BPS, n_stocks=config.SYNTH_N_STOCKS,
                      defects=False):
    """evaluate_world on a fresh synthetic world for each seed. One run proves nothing, so we repeat."""
    out = []
    for seed in seeds:
        _, con = stage3_run.build_world(edge=edge, seed=seed, defects=defects, n_stocks=n_stocks,
                                        reaction=reaction, n_dead=n_dead)
        try:
            out.append(evaluate_world(con, engine.Env(con), top_n, value_weight, buffer, cost_bps))
        finally:
            con.close()
    return out


def survivorship_over_seeds(seeds, n_dead=config.SYNTH_DEFAULT_DEAD, value_weight=config.DEFAULT_VALUE_WEIGHT,
                            top_n=config.DEFAULT_TOP_N, buffer=config.DEFAULT_BUFFER, cost_bps=config.DEFAULT_COST_BPS,
                            edge=config.SYNTH_DEFAULT_EDGE, reaction=config.SYNTH_DEFAULT_REACTION,
                            n_stocks=config.SYNTH_N_STOCKS, defects=False):
    """Only the two runs that isolate survivorship (honest P/E on today's list, and on everything), per seed.
    Returns a list of {'survivorship', 'dead_held', 'delisted_sales'}."""
    out = []
    for seed in seeds:
        _, con = stage3_run.build_world(edge=edge, seed=seed, defects=defects, n_stocks=n_stocks,
                                        reaction=reaction, n_dead=n_dead)
        try:
            env = engine.Env(con)
            args = ("lagged", None, top_n, value_weight, buffer, cost_bps)
            survivors = run_variant(con, env, "lagged", "survivors", top_n, value_weight, buffer, cost_bps)
            everything = run_variant(con, env, "lagged", "all", top_n, value_weight, buffer, cost_bps)
            out.append({"survivorship": survivors["net_cagr"] - everything["net_cagr"],
                        "dead_held": everything["dead_held"], "delisted_sales": everything["delisted_sales"],
                        "dead_weight": everything["dead_weight"]})
        finally:
            con.close()
    return out


def summarize(values):
    """mean, standard error of the mean, and t (= mean / standard error) of a list of numbers."""
    n = len(values)
    if n == 0:
        raise ValueError("no values to summarize")
    mean = sum(values) / n
    if n < 2:
        return {"n": n, "mean": mean, "se": NAN, "t": NAN}
    sd = math.sqrt(sum((v - mean) ** 2 for v in values) / (n - 1))
    se = sd / math.sqrt(n)
    return {"n": n, "mean": mean, "se": se, "t": mean / se if se > 1e-12 else NAN}


def bias_verdict(mean, se, floor=0.005):
    """Is the average difference really different from zero? 'INFLATED' needs the average to be more than two
    standard errors above zero AND at least `floor` (half a percent a year); otherwise it is within the noise."""
    if se != se:
        return "NOT ENOUGH RUNS"
    if mean > 2 * se and mean > floor:
        return "INFLATED"
    if mean < -2 * se and mean < -floor:
        return "DEFLATED"
    return "within noise"
