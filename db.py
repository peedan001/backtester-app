"""db.py: the schema. The database enforces the rules, so a bug later cannot
quietly store bad data.

Rules enforced here (each one is a line in the SQL below):
  * one price per (ticker, date)            -> composite primary key
  * no price/EPS/holding for an unknown ticker -> foreign keys
  * prices must be above 0                   -> CHECK
  * EPS becomes available only AFTER its quarter ends -> CHECK (anti look-ahead)
  * weights between 0 and 1, costs not negative       -> CHECK
  * numbers must really be numbers, and finite        -> CHECK typeof(...) and < 1e12
    (SQLite is loosely typed: without this, the text 'N/A' or infinity is stored
    happily in a REAL column, because text compares as "bigger than" any number)
  * dates must be real ISO dates (YYYY-MM-DD)         -> CHECK (uses IS, not =,
    because a CHECK that evaluates to NULL passes, and date('2020-13-45') is NULL)
  * bad data is FLAGGED, never deleted                -> flagged / flag_reason columns
"""
import sqlite3

import config

# The RUN.buffer column is defined once, so a new database and an older one that gets it added are identical.
BUFFER_COLUMN = ("buffer REAL NOT NULL DEFAULT %s CHECK (buffer >= %s AND buffer <= %s "
                 "AND typeof(buffer) IN ('real', 'integer'))"
                 % (config.DEFAULT_BUFFER, config.MIN_BUFFER, config.MAX_BUFFER))

# The RUN table's columns, written once so a new database and a rebuilt older one are identical.
# top_n, value_weight, cost_bps, buffer: the strategy's choices. pe_mode: which P/E version the signals used.
# universe: 'survivors' = today's list only, 'all' = everything that existed (differs only when the data has failed stocks).
RUN_COLUMNS = """
    run_id        INTEGER PRIMARY KEY AUTOINCREMENT,
    top_n         INTEGER NOT NULL CHECK (top_n >= %(min_n)d AND top_n <= %(max_n)d),
    value_weight  REAL NOT NULL CHECK (value_weight >= 0 AND value_weight <= 1),
    cost_bps      REAL NOT NULL CHECK (cost_bps >= 0 AND cost_bps <= %(max_cost)d),
    %(buffer_col)s,
    pe_mode       TEXT NOT NULL DEFAULT 'lagged' CHECK (pe_mode IN %(pe_modes)s),
    universe      TEXT NOT NULL DEFAULT 'survivors' CHECK (universe IN %(universes)s)
""" % {
    "min_n": config.MIN_TOP_N,
    "max_n": config.MAX_TOP_N,
    "max_cost": config.MAX_COST_BPS,
    "buffer_col": BUFFER_COLUMN,
    "pe_modes": "(" + ", ".join(repr(m) for m in config.EPS_MODES) + ")",
    "universes": "(" + ", ".join(repr(u) for u in config.UNIVERSES) + ")",
}

SCHEMA = """
CREATE TABLE IF NOT EXISTS STOCK (
    ticker        TEXT PRIMARY KEY,
    name          TEXT NOT NULL,
    in_sp500      INTEGER NOT NULL DEFAULT 1 CHECK (in_sp500 IN (0, 1)),
    is_benchmark  INTEGER NOT NULL DEFAULT 0 CHECK (is_benchmark IN (0, 1))
);

CREATE TABLE IF NOT EXISTS PRICE (
    ticker       TEXT NOT NULL REFERENCES STOCK(ticker),
    date         TEXT NOT NULL CHECK (date(date) IS date),
    adj_close    REAL NOT NULL CHECK (adj_close > 0 AND adj_close < 1e12
                                      AND typeof(adj_close) IN ('real', 'integer')),
    flagged      INTEGER NOT NULL DEFAULT 0 CHECK (flagged IN (0, 1)),
    flag_reason  TEXT,
    PRIMARY KEY (ticker, date)
);

-- EPS is stored with TWO dates: when the quarter ended, and when the world
-- could first see the number. Signals may only use it from available_date.
CREATE TABLE IF NOT EXISTS FUNDAMENTAL (
    ticker          TEXT NOT NULL REFERENCES STOCK(ticker),
    period_end      TEXT NOT NULL CHECK (date(period_end) IS period_end),
    available_date  TEXT NOT NULL CHECK (date(available_date) IS available_date),
    eps             REAL NOT NULL CHECK (abs(eps) < 1e12 AND typeof(eps) IN ('real', 'integer')),
    PRIMARY KEY (ticker, period_end),
    CHECK (available_date >= period_end)
);

-- Ranks only. The combined score depends on the app slider, so it is computed
-- on demand, never stored.
CREATE TABLE IF NOT EXISTS SIGNAL (
    ticker          TEXT NOT NULL REFERENCES STOCK(ticker),
    rebalance_date  TEXT NOT NULL CHECK (date(rebalance_date) IS rebalance_date),
    pe              REAL CHECK (pe IS NULL OR (abs(pe) < 1e12 AND typeof(pe) IN ('real', 'integer'))),
    momentum        REAL CHECK (momentum IS NULL OR (abs(momentum) < 1e12 AND typeof(momentum) IN ('real', 'integer'))),
    pe_rank         REAL CHECK (pe_rank IS NULL OR (pe_rank >= 0 AND pe_rank <= 1)),
    momentum_rank   REAL CHECK (momentum_rank IS NULL OR (momentum_rank >= 0 AND momentum_rank <= 1)),
    PRIMARY KEY (ticker, rebalance_date)
);

-- One RUN = one set of choices (portfolio size, signal weight, cost, buffer, P/E version, universe).
-- buffer: a holding is kept while it stays within the top (buffer x top_n) places; 1.0 = pure top-N.
CREATE TABLE IF NOT EXISTS RUN (%(run_columns)s);

CREATE TABLE IF NOT EXISTS HOLDING (
    run_id          INTEGER NOT NULL REFERENCES RUN(run_id),
    rebalance_date  TEXT NOT NULL CHECK (date(rebalance_date) IS rebalance_date),
    ticker          TEXT NOT NULL REFERENCES STOCK(ticker),
    weight          REAL NOT NULL CHECK (weight > 0 AND weight <= 1),
    PRIMARY KEY (run_id, rebalance_date, ticker)
);

CREATE TABLE IF NOT EXISTS RESULT (
    run_id            INTEGER NOT NULL REFERENCES RUN(run_id),
    date              TEXT NOT NULL CHECK (date(date) IS date),
    -- A simple return can never be minus 100 percent or worse; all four must be real numbers.
    gross_return      REAL NOT NULL CHECK (gross_return > -1 AND gross_return < 1e12 AND typeof(gross_return) IN ('real', 'integer')),
    cost              REAL NOT NULL CHECK (cost >= 0 AND cost < 1e12 AND typeof(cost) IN ('real', 'integer')),
    net_return        REAL NOT NULL CHECK (net_return > -1 AND net_return < 1e12 AND typeof(net_return) IN ('real', 'integer')),
    benchmark_return  REAL NOT NULL CHECK (benchmark_return > -1 AND benchmark_return < 1e12 AND typeof(benchmark_return) IN ('real', 'integer')),
    PRIMARY KEY (run_id, date)
);
""" % {
    "min_n": config.MIN_TOP_N,
    "max_n": config.MAX_TOP_N,
    "max_cost": config.MAX_COST_BPS,
    "run_columns": RUN_COLUMNS,
}


def connect(path=":memory:"):
    """Open a database with foreign keys switched ON (SQLite has them off by default)."""
    con = sqlite3.connect(path)
    con.execute("PRAGMA foreign_keys = ON")
    return con


def create_schema(con):
    """Create all tables. Safe to run twice. A database saved by an earlier stage gets the new columns added,
    so nobody has to regenerate their data when the schema grows."""
    con.executescript(SCHEMA)
    con.execute("PRAGMA foreign_keys = ON")  # executescript can reset it; make sure
    if "buffer" not in [c[0] for c in columns(con, "RUN")]:           # added in Stage 4
        con.execute("ALTER TABLE RUN ADD COLUMN " + BUFFER_COLUMN)
    con.commit()
    _rebuild_run_if_old(con)                                          # new pe_mode value and universe column: Stage 6


def _rebuild_run_if_old(con):
    """SQLite cannot change a CHECK constraint in place, so a RUN table from before Stage 6 (which allowed only
    'lagged' and 'today' and had no universe column) is rebuilt: new table, copy every row with its run_id, swap.
    Foreign keys are switched off for the swap so HOLDING and RESULT keep pointing at the same run_ids."""
    sql = con.execute("SELECT sql FROM sqlite_master WHERE type = 'table' AND name = 'RUN'").fetchone()[0]
    if "universe" in sql and "no_lag" in sql:
        return False
    con.executescript(
        "PRAGMA foreign_keys = OFF; BEGIN;"
        "CREATE TABLE RUN_new (" + RUN_COLUMNS + ");"
        "INSERT INTO RUN_new (run_id, top_n, value_weight, cost_bps, buffer, pe_mode) "
        "SELECT run_id, top_n, value_weight, cost_bps, buffer, pe_mode FROM RUN;"
        "DROP TABLE RUN; ALTER TABLE RUN_new RENAME TO RUN; COMMIT;")
    con.execute("PRAGMA foreign_keys = ON")
    problems = con.execute("PRAGMA foreign_key_check").fetchall()
    if problems:
        raise sqlite3.IntegrityError(f"rebuilding RUN broke {len(problems)} foreign key(s)")
    return True


def table_names(con):
    rows = con.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%' ORDER BY name"
    ).fetchall()
    return [r[0] for r in rows]


def columns(con, table):
    """Return (name, type, not_null, primary_key_position) for each column."""
    return [(r[1], r[2], bool(r[3]), r[5]) for r in con.execute(f"PRAGMA table_info({table})")]


def foreign_keys_on(con):
    return con.execute("PRAGMA foreign_keys").fetchone()[0] == 1
