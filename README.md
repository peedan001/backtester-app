# Backtester: low P/E + momentum

A teaching backtester that measures how much common shortcuts inflate a strategy's results. It runs on a synthetic S&P-500-style market where the true answer is planted, and on real data you collect yourself.

## Quick start

```
pip install numpy pandas streamlit        # streamlit 1.50 or newer
streamlit run app.py
```

The sidebar offers a fake market (no internet needed) or a database file. Press **Run backtest**, then try the **Bias lab**.

For real data, on your own computer (the live sources need internet):

```
pip install yfinance
Set EDGAR_USER_AGENT="Your Name your.email@example.com"
python stage2_run.py --check
python stage2_run.py --live --limit 25 --db real.db
streamlit run app.py          # then Source: Database file, path real.db
```

The live fetchers have not been run against the real services yet. Expect small fixes on the first run.

## What is where

| File | Job |
|---|---|
| `config.py` | Every setting in one place |
| `db.py` | SQLite schema and migrations |
| `collect.py` | Fetchers (Yahoo prices, EDGAR EPS, CSV backup), validation, quarantine |
| `synthetic.py` | Fake market: planted edge, earnings reaction, failed stocks, stored truth |
| `align.py`, `signals.py` | Point-in-time P/E and momentum, ranks |
| `portfolio.py` | Top N, equal weight, buffer |
| `engine.py` | Trading simulation, costs, metrics, share-ledger cross-check |
| `bias.py` | The ladder of look-ahead and survivorship runs, repeat-over-seeds verdicts |
| `app_logic.py` | Everything the app computes (no Streamlit, tested) |
| `app.py` | The Streamlit face |
| `stage1_run.py` to `stage6_run.py` | A guided walkthrough per stage (`--demo`, interactive modes, `--try`) |
| `STUDY_GUIDE.md` | Explanations, results, check questions with answers |
| `tests/` | One test file per stage plus the app |

## Walkthroughs

```
python stage6_run.py --demo        # the bias story, about 3 minutes
python stage6_run.py --ladder      # choose how strongly earnings move prices
python stage6_run.py --graveyard   # browse failed stocks
python stage5_run.py --try         # hand-checkable trading cases
```

Each stage's `--demo` ends with a check question; the answers are in `STUDY_GUIDE.md`.

## Tests

```
python -W error -m pytest -q tests
```

The tests were checked by mutation: the code was deliberately broken in many ways, and every meaningful break had to make a test fail.

## Honest limits

Results in the fake market show that the method works and how large each bias can be. They are not evidence about real markets. A list of today's S&P 500 members is survivorship-biased even with real prices. Costs are a flat basis-point charge, and the Sharpe ratio uses a fixed risk-free rate.
