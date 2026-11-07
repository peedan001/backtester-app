# Backtester study guide: low P/E + momentum

This guide is for explaining the project aloud: what it does, why it is built the way it is, what it found, and where it cannot be trusted. Each section opens on its own. The check questions and their answers are near the end; say the answer aloud first, then compare.

## What the project is

A backtester answers one question: *if I had followed this rule in the past, what would have happened?* The rule here is to buy the stocks that are cheap (low price-to-earnings, P/E) and have been rising (momentum), hold about 20 of them, and rebalance every month.

The hard part is not the rule. It is making the answer honest. A backtest is easy to fool in ways that make a rule look better than it was, and this project is built to **measure** how much each kind of fooling is worth. It does that inside a fake market where the true answer is planted, so a method can be checked against a known truth.

Three biases are the heart of it:

- **Look-ahead bias:** using information that was not yet public on the trading date.
- **Survivorship bias:** testing only on stocks that still exist today.
- **Luck:** one run of a noisy process proves nothing, so results are repeated across many fake worlds.

## The three designs

**Conceptual design (what the parts are):** data comes in, is cleaned and stored, is turned into signals, signals become a portfolio, the portfolio is traded through a simulator, and the results are measured and drawn.

**Logical design (what is stored and how it connects):** seven tables.
- `STOCK`: one row per company (is it in today's S&P 500 list? is it the SPY benchmark?)
- `PRICE`: daily adjusted close, with a flag for suspicious prices
- `FUNDAMENTAL`: quarterly EPS with **two dates**, `period_end` and `available_date`
- `SIGNAL`: P/E and momentum per stock per month-end
- `RUN`: one row per set of choices (N, value weight, cost, buffer, P/E version, universe)
- `HOLDING`: what each run held at each month-end
- `RESULT`: gross, cost, net and benchmark return per period

**Physical design (how it is built):** SQLite for storage, plain Python modules, one `*_run.py` walkthrough per stage with a `--demo`, and a Streamlit app on top. Every module has tests, and the tests were themselves tested by breaking the code on purpose (mutation checks).

## Stage by stage

| Stage | Files | What it adds |
|---|---|---|
| 1 | `config.py`, `db.py` | The schema, with rules the database enforces itself (foreign keys, value checks) |
| 2 | `collect.py`, `synthetic.py` | Collectors with a quarantine for bad rows, and a fake market with a planted edge and stored truth |
| 3 | `align.py`, `signals.py` | Point-in-time joins: for each date, only what was public. P/E, momentum, ranks |
| 4 | `portfolio.py` | Top N by blended rank, equal weight, optional buffer |
| 5 | `engine.py` | Trades at the next close, drifting weights, costs on traded value, metrics, an independent share ledger as a cross-check |
| 6 | `bias.py`, `synthetic.py` | Earnings that move prices, failed stocks, three P/E versions, delisting rule, repeat-over-seeds verdicts |
| 7 | `app.py`, `app_logic.py` | The Streamlit app |

## How the engine works, in one paragraph

On each month-end it ranks stocks, picks the top N, and *plans* to buy them. The trade happens at the **next trading day's close** (you cannot trade at the price you computed the signal from). Between trades, each holding drifts with its own price, so the real weights are no longer 1/N. Cost is basis points times the value bought plus the value sold, charged on each side. A stock whose data ends is sold at its last price and the money waits as cash until the next rebalance. A second, independent engine counts shares instead of weights; the two must agree on wealth on every date.

## What the bias measurements showed

All numbers are per year, after costs, in the fake market (edge 5%, earnings reaction 2).

| Bias | Result | Verdict |
|---|---|---|
| No filing lag (quarter-end date) | +2.02% (±0.32) over 5 worlds | inflated |
| Today's EPS on old prices | +4.43% (±0.69) over 5 worlds | inflated |
| Survivorship, blend, 20 failed stocks | +0.20% | within noise |
| Survivorship, blend, 40 failed stocks | +2.71% | inflated |
| Survivorship, pure value, 20 failed | +3.58% | inflated |
| Survivorship, pure momentum, 20 failed | +0.30% | within noise |
| Both shortcuts when earnings do **not** move prices | about +0.2% | within noise |

The last row matters most: when earnings carry no information about prices, the shortcuts cost nothing. The bias exists only to the extent that the information leaked is *useful*.

One caveat: the fake graveyard models only the final decline of a failing company. Real failures also include years of slow underperformance and sudden acquisitions, so real survivorship bias is probably larger than shown.

## Using the app

Start with `streamlit run app.py`. The sidebar picks the data: a fake market (with sliders for the planted edge, how strongly earnings move prices, and how many failed stocks) or a database file saved by a live run. Press **Run backtest** to see growth of 1 against SPY, the headline numbers with and without costs, and what the portfolio held in any month. The **Bias lab** runs the ladder and repeats it over several fake worlds. The **Graveyard** follows a single failed stock to its end.

Try this: set the earnings reaction to 0 and run the ladder. The shortcuts should shrink to noise. Then set it to 3 and watch them grow. That is the whole lesson in one slider.

## Getting real data

This app never contacts Yahoo or EDGAR. A separate step on your own computer does:

1. `python stage2_run.py --check` tests each source with one ticker.
2. Set `EDGAR_USER_AGENT="Your Name your.email@example.com"` (the SEC requires it; use your own details).
3. `python stage2_run.py --live --limit 25 --db real.db` collects 25 companies (then more).
4. In the app, choose **Database file** and type `real.db`.

Why split it? Collection is slow, can be blocked, and changes from day to day. A saved database gives the same answer every time, so results can be compared and explained. The live fetchers were written and tested against recorded samples, but could not be run against the real services during development, so expect to fix small things (a changed Yahoo layout, a ticker EDGAR spells differently) on the first run.

## Check questions and answers

### 1. Why does FUNDAMENTAL have two dates (period_end and available_date)?

`period_end` is when the quarter ended (31 March). `available_date` is when the world could first see the number (a filing around 10 May, about 40 days later). A backtest standing on a given date may only use rows whose `available_date` is on or before that date. If it keyed on `period_end` instead, it would act on results that nobody had yet. The two dates are the whole defence against look-ahead bias. Measured in Stage 6: using `period_end` added about +2% a year in a world where earnings move prices.

### 2. In the null world the best fifth still sometimes beats the worst fifth. Why is one run never enough?

Returns are noisy, so even when the signal has no power, one run is one random draw and some draws favour the signal by luck. A single good result cannot be told apart from a lucky one. The fix is to repeat the experiment: many seeds (many worlds), then compare the average result to its standard error. In the null world the average over many worlds settles near zero and the spread tells you how big a lucky run can be; a real edge is one that stays clear of that spread.

### 3. On 28 June the March quarter is not public until 10 July. Which EPS should be used, and why does the 10 July figure flatter the strategy?

Use the last four quarters that had been **filed by 28 June**, which ends at the December quarter (filed in February or March). The 10 July number did not exist on 28 June. Using it lets the backtest see good news before the price reacts: companies whose earnings are about to jump look cheap right then, and their prices rise after the filing, so the cheap-looking stocks "win". The strategy appears to predict price moves it was actually reading ahead of time. The measured inflation was +2.02% a year.

### 4. A buffer of 1.5 cuts trading by more than half, yet it can lower or raise the return after costs. Give one reason for each.

*Lower it:* a buffer keeps stocks whose scores have faded, so the portfolio holds names that are no longer the best candidates and the signal is diluted. If the planted edge is strong, the return given up can exceed the cost saved.

*Raise it:* every trade pays on both sides, and ranks near the cutoff are mostly noise, so swapping the 20th stock for the 21st is a coin flip that costs money. The buffer avoids those pointless swaps, and if the cost saved is larger than the signal lost, the return after costs goes up.

### 5. Ignoring drift makes costs look 20 to 70 percent lower, and a one-day delay barely matters in the fake market. Why does the delay still matter on real data, and why is picking the best delay cheating?

In the fake market each day's noise is independent of the signal, so waiting a day changes little. In real data there is information in the gap: earnings and news announced after the month-end, overnight gaps, and short-term reversal in momentum, all land between the signal and the trade. And the month-end close that produced the signal cannot be traded at, so assuming it can gives a result nobody could have earned.

Choosing the delay that gives the best number is cheating because the same data is used to both choose and judge. Among several delays, the best looks good partly by luck, and that luck will not repeat. The delay should be decided from how trading actually works (next available close) before seeing results, and any alternatives reported with it.

### 6. A vendor's backtest uses today's S&P 500 list and P/E recomputed with current earnings. Name the two biases, which way each pushes, and why momentum protects against one kind of failed stock but not another.

*Survivorship bias:* today's list contains only companies that survived, so the failures (the losers) are missing. This pushes returns **up**.

*Look-ahead bias:* recomputing P/E with current earnings gives the old price a denominator from the future. Companies that later grew earnings look cheap in the past, and those are the ones that rose. This pushes returns **up**, and in our fake market was the larger effect (+4.43% against +2.02% for the milder version).

Momentum protects against **gradual** failures: a company declining for nine months has a falling price, so a 12-month momentum signal ranks it low and the strategy is out before the end. It does not protect against **abrupt** failures (a collapse over a few weeks), because the signal skips the latest month and looks backward over a year in which the stock may have done fine. The value half is worse: a collapsing stock looks cheap, so value *buys* it. Measured: pure value held 10 to 15 failed stocks to the end and showed +3.58% of survivorship bias; pure momentum showed +0.30%, within noise; the blend, in between, showed +0.20% at 20 failed stocks and +2.71% at 40.

### 7. Why does the app read a saved database instead of fetching live data?

Fetching is slow, rate-limited and can be blocked, and a live source changes over time, so two runs a week apart could disagree for reasons unrelated to the strategy. A saved database gives the same data every time, so a result can be reproduced, compared and explained, and the same app code works on a fake market and on real data. It also keeps the point-in-time dates fixed: what was filed when is recorded once, not re-fetched with later revisions mixed in.

## Limits to say out loud

- Everything above was measured in a **fake** market, by design. It shows the method works and how large each bias *can* be. It says nothing about whether low P/E plus momentum earns money in real markets.
- The live Yahoo and EDGAR fetchers are untested against the real services.
- Even with real data, a list of today's S&P 500 members is itself survivorship-biased. Historical index membership would be needed to remove it, and that data is usually paid.
- Costs are a flat number of basis points. Real costs depend on size, liquidity and the spread, and are higher for small and failing stocks.
- Sharpe uses a fixed 2% risk-free rate, not a real T-bill series.

## Glossary

- **P/E:** price divided by trailing twelve-month earnings per share. Low means cheap.
- **Momentum:** return over the last 12 months, skipping the latest month.
- **Rank blend:** each stock's cheapness rank and momentum rank, mixed by the value-weight slider.
- **CAGR:** compound growth rate per year.
- **Sharpe:** return above the risk-free rate, divided by volatility.
- **Drawdown:** the fall from a previous peak.
- **Basis point (bp):** 0.01%.
- **Point-in-time:** using only what was public on the date in question.
- **Standard error:** how much an average would wobble if the experiment were repeated.
- **Mutation check:** deliberately breaking the code to prove the tests notice.
