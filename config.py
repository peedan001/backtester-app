"""config.py: the catalogue. Every constant of the project lives here, once.

If a number appears in two places in the code, it belongs here instead.
"""

# ---- Time window ---------------------------------------------------------
START_DATE = "2015-01-01"   # first day of the backtest
END_DATE = "2025-12-31"     # last day of the backtest
# Signals need history (12 months of prices, 4 quarters of EPS), so we collect
# data from earlier than START_DATE.
DATA_START = "2013-01-01"

# ---- Benchmark -----------------------------------------------------------
# SPY (an ETF that tracks the S&P 500). Its ADJUSTED close includes dividends,
# so it is a fair "total return" benchmark.
BENCHMARK_TICKER = "SPY"

# ---- Signals -------------------------------------------------------------
MOMENTUM_LOOKBACK_MONTHS = 12   # look back 12 months ...
MOMENTUM_SKIP_MONTHS = 1        # ... but skip the latest month (short-term reversal)
EPS_LAG_DAYS = 45               # quarterly EPS usable only 45 days after quarter end
# Point-in-time hygiene rules used when aligning data (Stage 3):
MAX_CARRY_DAYS = 5              # a missing month-end price may be replaced by one at most 5 calendar days old
MIN_EPS_QUARTERS = 4            # trailing-twelve-month EPS needs 4 quarters
MAX_TTM_SPAN_DAYS = 290         # the 4 quarters must be consecutive: first to last period end is about 273 days
MAX_EPS_STALENESS_DAYS = 200    # the latest quarter may be at most 200 days old (normally 91 + a filing lag of 55 at most)
MIN_HISTORY_MONTHS = MOMENTUM_LOOKBACK_MONTHS   # momentum needs 12 earlier month-ends

# ---- Portfolio defaults (the app lets the user change these) -------------
DEFAULT_TOP_N = 20              # stocks held each month
DEFAULT_VALUE_WEIGHT = 0.5      # 0 = all momentum, 1 = all P/E ("value")
DEFAULT_COST_BPS = 10           # trading cost, in basis points (1 bp = 0.01%)
DEFAULT_BUFFER = 1.0            # 1.0 = a pure top-N every month; 1.5 = keep a holding while it stays in the top 1.5 x N

# ---- Limits the database enforces ----------------------------------------
MIN_TOP_N = 1
MAX_TOP_N = 500
MAX_COST_BPS = 500              # 5%: anything higher is almost surely a typo
MIN_BUFFER = 1.0                # a buffer below 1 would drop names that are still in the top N
MAX_BUFFER = 5.0

# ---- Annualisation --------------------------------------------------------
TRADING_DAYS_PER_YEAR = 252
MONTHS_PER_YEAR = 12

# ---- Engine (Stage 5) ------------------------------------------------------
EXEC_DELAY_DAYS = 1             # trade at the close this many trading days AFTER the month-end signal (1 = next trading day)
MAX_EXEC_DELAY_DAYS = 5
RISK_FREE_RATE = 0.02           # a fixed yearly rate for Sharpe, not a real T-bill series; shown wherever Sharpe is shown
MAX_RISK_FREE_RATE = 0.20

# ---- Data checks (validate step: flag, never delete) ---------------------
JUMP_THRESHOLD = 0.40       # flag a day whose log return is bigger than +-0.40 (about -33% / +49%)
MAX_GAP_DAYS = 7            # flag a price row that follows a gap of more than 7 calendar days

# ---- Synthetic market (planted truth) ------------------------------------
SYNTH_N_STOCKS = 100
SYNTH_SEED = 7
SYNTH_DEFAULT_EDGE = 0.05   # extra return per year for a stock with a perfect score
SYNTH_MARKET_RETURN = 0.08  # yearly
SYNTH_MARKET_VOL = 0.16     # yearly
SYNTH_IDIO_VOL = (0.20, 0.40)   # each stock's own yearly volatility is drawn from this range
SYNTH_START_PRICE_PE = 18.0     # typical starting P/E; each stock differs around it

# ---- Live data -----------------------------------------------------------
# SEC EDGAR asks every caller to identify itself. Set an environment variable:
#   EDGAR_USER_AGENT="Your Name your.email@example.com"
EDGAR_USER_AGENT_ENV = "EDGAR_USER_AGENT"
EDGAR_PAUSE_SECONDS = 0.15  # SEC allows at most 10 requests a second
YAHOO_PAUSE_SECONDS = 0.2   # be polite: Yahoo blocks an IP that asks too fast
SP500_URL = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
BROWSER_UA = "Mozilla/5.0 (compatible; backtester-learning-project)"
HTTP_TIMEOUT_SECONDS = 20

# ---- Files ---------------------------------------------------------------
DB_PATH = "backtest.db"

# ---- Tables (names only; the columns are in db.py) -----------------------
TABLES = ["STOCK", "PRICE", "FUNDAMENTAL", "SIGNAL", "RUN", "HOLDING", "RESULT"]


# ---- Stage 6: bias ---------------------------------------------------------
EPS_MODES = ("lagged", "no_lag", "today")   # honest | EPS used the day its quarter ends | EPS from the END of the data (hindsight)
UNIVERSES = ("survivors", "all")             # today's list only | everything that existed (needs a graveyard to differ)
SYNTH_DEFAULT_REACTION = 2.0                 # price jump on a filing date = reaction x change in trailing EPS (0 = no link)
SYNTH_MAX_REACTION = 3.0
SYNTH_REACTION_CLIP = 0.25                   # a single filing never moves the log price by more than this (stays under JUMP_THRESHOLD)
SYNTH_DEFAULT_DEAD = 20                      # failed stocks added to the synthetic world by the bias demos
SYNTH_MAX_DEAD = 50
SYNTH_CRASH_DAYS = 190                       # a GRADUAL failure: business days of decline before the stock stops trading (about 9 months)
SYNTH_ABRUPT_SHARE = 0.5                     # share of failures that are ABRUPT: the whole fall happens in a few weeks
SYNTH_ABRUPT_DAYS = (10, 30)                 # length of an abrupt fall, in business days
SYNTH_CRASH_RANGE = (0.9, 2.3)               # total log decline over those days: prices fall by 60% to 90%
SYNTH_FIRST_DELIST_MONTH = 30                # failures happen between this month-end and a few months before the end
