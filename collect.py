"""collect.py: get data into the database, safely.

Design rules (each one is tested):
  * IDEMPOTENT      running the collector twice gives the same database (upsert, never duplicate)
  * ISOLATED        one failing ticker never stops the others; its error is recorded
  * COUNTED         every row is inserted, updated, unchanged or REJECTED, and rejections are
                    counted by reason, never hidden
  * FLAG, NEVER DELETE   suspicious prices stay in the table with flagged = 1 and a reason
  * INJECTED FETCHERS    the collector never touches the internet itself. It is handed fetch
                    functions, so tests use fake data and the live versions plug in on your computer.

Fetch function shapes (the contract between the collector and any data source):
    universe()                      -> [(ticker, name), ...]
    prices(ticker, start, end)      -> [(date 'YYYY-MM-DD', adjusted close), ...]
    eps(ticker)                     -> [(period_end, available_date, eps), ...]
"""
import csv
import datetime as dt
import io
import json
import math
import os
import sqlite3
import time
import urllib.request
from collections import Counter
from dataclasses import dataclass, field
from html.parser import HTMLParser

import config


# ---------------------------------------------------------------------------
# registering stocks
# ---------------------------------------------------------------------------

def register_stocks(con, stocks):
    """Add or update rows in STOCK. stocks: (ticker, name, in_sp500, is_benchmark). Returns the count."""
    n = 0
    for ticker, name, in_sp500, is_bench in stocks:
        con.execute(
            "INSERT INTO STOCK (ticker, name, in_sp500, is_benchmark) VALUES (?, ?, ?, ?) "
            "ON CONFLICT(ticker) DO UPDATE SET name = excluded.name, "
            "in_sp500 = excluded.in_sp500, is_benchmark = excluded.is_benchmark",
            (ticker, name, int(in_sp500), int(is_bench)))
        n += 1
    con.commit()
    return n


# ---------------------------------------------------------------------------
# a report of what happened
# ---------------------------------------------------------------------------

@dataclass
class Report:
    table: str
    tickers_ok: int = 0
    inserted: int = 0
    updated: int = 0
    unchanged: int = 0
    duplicates_in_feed: int = 0
    rejected: Counter = field(default_factory=Counter)
    failed: dict = field(default_factory=dict)       # ticker -> error text

    @property
    def rejected_total(self):
        return sum(self.rejected.values())

    def summary(self):
        lines = [f"{self.table}: {self.tickers_ok} ticker(s) loaded, {len(self.failed)} failed",
                 f"   rows: {self.inserted} inserted, {self.updated} updated, {self.unchanged} unchanged, "
                 f"{self.rejected_total} rejected, {self.duplicates_in_feed} repeated inside the feed"]
        for reason, n in self.rejected.most_common():
            lines.append(f"   rejected x{n}: {reason}")
        for t, msg in list(self.failed.items())[:10]:
            lines.append(f"   FAILED {t}: {msg}")
        if len(self.failed) > 10:
            lines.append(f"   ... and {len(self.failed) - 10} more failures")
        return "\n".join(lines)


def _clean_number(value):
    """Turn a feed value into a finite float, or raise ValueError('not a number' / 'not a finite number')."""
    if isinstance(value, bool):
        raise ValueError("not a number")
    if isinstance(value, (int, float)):
        x = float(value)
    elif isinstance(value, str):
        try:
            x = float(value.strip())
        except ValueError:
            raise ValueError("not a number") from None
    else:
        raise ValueError("not a number")
    if not math.isfinite(x):
        raise ValueError("not a finite number")
    return x


def _explain(err):
    """Turn a database error into a short, human reason."""
    msg = str(err)
    if "FOREIGN KEY" in msg:
        return "unknown ticker"
    if "available_date >= period_end" in msg:
        return "EPS filed before its quarter ended"
    if "CHECK constraint failed: adj_close" in msg:
        return "price not a positive finite number"
    if "CHECK constraint failed: abs(eps)" in msg:
        return "EPS not a finite number"
    if "date(" in msg and "IS" in msg:
        return "invalid date"
    return msg[:80]


def _is_registered(con, ticker):
    return con.execute("SELECT 1 FROM STOCK WHERE ticker = ?", (ticker,)).fetchone() is not None


# ---------------------------------------------------------------------------
# collecting prices and EPS
# ---------------------------------------------------------------------------

def collect_prices(con, fetch, tickers, start=config.DATA_START, end=config.END_DATE, progress=None):
    """Fetch and store prices. fetch(ticker, start, end) -> [(date, adj_close), ...]."""
    report = Report("PRICE")
    tickers = list(tickers)
    for i, ticker in enumerate(tickers):
        if progress:
            progress(i + 1, len(tickers), ticker)
        if not _is_registered(con, ticker):
            report.failed[ticker] = "not registered in STOCK"
            continue
        try:
            rows = list(fetch(ticker, start, end))
        except Exception as e:                                   # one failure never stops the rest
            report.failed[ticker] = f"{type(e).__name__}: {e}"
            continue
        if not rows:
            report.failed[ticker] = "empty feed (no rows returned)"
            continue
        seen = set()
        for row in rows:
            try:
                date, raw = row
            except (ValueError, TypeError):                      # wrong number of fields, or not a row at all
                report.rejected["malformed row"] += 1
                continue
            try:
                value = _clean_number(raw)
            except ValueError as e:
                report.rejected[str(e)] += 1
                continue
            if date in seen:
                report.duplicates_in_feed += 1
            seen.add(date)
            try:
                existing = con.execute("SELECT adj_close FROM PRICE WHERE ticker = ? AND date = ?",
                                       (ticker, date)).fetchone()
                if existing is not None and existing[0] == value:
                    report.unchanged += 1
                    continue
                con.execute(
                    "INSERT INTO PRICE (ticker, date, adj_close) VALUES (?, ?, ?) "
                    "ON CONFLICT(ticker, date) DO UPDATE SET adj_close = excluded.adj_close",
                    (ticker, date, value))
                if existing is None:
                    report.inserted += 1
                else:
                    report.updated += 1
            except sqlite3.IntegrityError as e:
                report.rejected[_explain(e)] += 1
        con.commit()
        report.tickers_ok += 1
    return report


def collect_eps(con, fetch, tickers, progress=None):
    """Fetch and store quarterly EPS. fetch(ticker) -> [(period_end, available_date, eps), ...]."""
    report = Report("FUNDAMENTAL")
    tickers = list(tickers)
    for i, ticker in enumerate(tickers):
        if progress:
            progress(i + 1, len(tickers), ticker)
        if not _is_registered(con, ticker):
            report.failed[ticker] = "not registered in STOCK"
            continue
        try:
            rows = list(fetch(ticker))
        except Exception as e:
            report.failed[ticker] = f"{type(e).__name__}: {e}"
            continue
        if not rows:
            report.failed[ticker] = "empty feed (no rows returned)"
            continue
        seen = set()
        for row in rows:
            try:
                period_end, available, raw = row
            except (ValueError, TypeError):
                report.rejected["malformed row"] += 1
                continue
            try:
                value = _clean_number(raw)
            except ValueError as e:
                report.rejected[str(e)] += 1
                continue
            if period_end in seen:
                report.duplicates_in_feed += 1
            seen.add(period_end)
            try:
                existing = con.execute(
                    "SELECT available_date, eps FROM FUNDAMENTAL WHERE ticker = ? AND period_end = ?",
                    (ticker, period_end)).fetchone()
                if existing is not None and existing == (available, value):
                    report.unchanged += 1
                    continue
                con.execute(
                    "INSERT INTO FUNDAMENTAL (ticker, period_end, available_date, eps) VALUES (?, ?, ?, ?) "
                    "ON CONFLICT(ticker, period_end) DO UPDATE SET "
                    "available_date = excluded.available_date, eps = excluded.eps",
                    (ticker, period_end, available, value))
                if existing is None:
                    report.inserted += 1
                else:
                    report.updated += 1
            except sqlite3.IntegrityError as e:
                report.rejected[_explain(e)] += 1
        con.commit()
        report.tickers_ok += 1
    return report


# ---------------------------------------------------------------------------
# validation: flag, never delete
# ---------------------------------------------------------------------------

def validate_prices(con, jump=config.JUMP_THRESHOLD, max_gap=config.MAX_GAP_DAYS):
    """Flag suspicious prices. Nothing is deleted. Flags are recomputed from scratch each run,
    so running it twice gives the same answer.

    jump: a day whose log return is bigger than +-jump (a 10x bad tick, an unadjusted split)
    gap:  a row that follows more than max_gap calendar days with no price
    A 12-sigma day can be real, which is why we flag for a human to look and never remove.
    Returns a Counter: {'jump': n, 'gap': n}."""
    con.execute("UPDATE PRICE SET flagged = 0, flag_reason = NULL")
    summary = Counter()
    tickers = [r[0] for r in con.execute("SELECT DISTINCT ticker FROM PRICE ORDER BY ticker")]
    for ticker in tickers:
        prev_d = prev_p = None
        for d, p in con.execute("SELECT date, adj_close FROM PRICE WHERE ticker = ? ORDER BY date", (ticker,)).fetchall():
            reasons = []
            if prev_d is not None:
                gap = (dt.date.fromisoformat(d) - dt.date.fromisoformat(prev_d)).days
                if gap > max_gap:
                    reasons.append(f"gap {gap} days")
                    summary["gap"] += 1
                r = math.log(p / prev_p)
                if abs(r) > jump:
                    reasons.append(f"jump {r:+.2f} log")
                    summary["jump"] += 1
            if reasons:
                con.execute("UPDATE PRICE SET flagged = 1, flag_reason = ? WHERE ticker = ? AND date = ?",
                            ("; ".join(reasons), ticker, d))
            prev_d, prev_p = d, p
    con.commit()
    return summary


def flagged_rows(con, ticker=None):
    """[(ticker, date, adj_close, reason), ...] for flagged prices."""
    sql = "SELECT ticker, date, adj_close, flag_reason FROM PRICE WHERE flagged = 1"
    args = ()
    if ticker:
        sql += " AND ticker = ?"
        args = (ticker,)
    return con.execute(sql + " ORDER BY ticker, date", args).fetchall()


# ---------------------------------------------------------------------------
# orchestration: the same path for the demo and for live data
# ---------------------------------------------------------------------------

def collect_all(con, fetchers, tickers=None, limit=None, start=config.DATA_START, end=config.END_DATE,
                progress=None):
    """Universe -> register -> prices (+ benchmark) -> EPS -> validate.
    fetchers: {'universe', 'prices', 'eps'}. Returns a dict of reports."""
    universe = list(fetchers["universe"]())
    if tickers is None:
        tickers = [t for t, _ in universe]
    if limit:
        tickers = tickers[:limit]
    names = dict(universe)
    register_stocks(con, [(t, names.get(t, t), 1, 0) for t in tickers])
    register_stocks(con, [(config.BENCHMARK_TICKER, "Benchmark (S&P 500 proxy)", 0, 1)])
    prices = collect_prices(con, fetchers["prices"], tickers + [config.BENCHMARK_TICKER], start, end, progress)
    eps = collect_eps(con, fetchers["eps"], tickers, progress)
    flags = validate_prices(con)
    return {"prices": prices, "eps": eps, "flags": flags, "tickers": tickers}


# ---------------------------------------------------------------------------
# live fetchers (run these on YOUR computer; the cloud workspace cannot reach these sites)
# ---------------------------------------------------------------------------

def _http_get(url, headers=None, timeout=config.HTTP_TIMEOUT_SECONDS):
    req = urllib.request.Request(url, headers=headers or {})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return resp.read()


class _ConstituentsParser(HTMLParser):
    """Reads the first HTML table with id='constituents' into a list of rows of cell text."""

    def __init__(self):
        super().__init__()
        self.in_table = False
        self.in_cell = False
        self.rows = []
        self._row = None
        self._text = []

    def handle_starttag(self, tag, attrs):
        if tag == "table" and dict(attrs).get("id") == "constituents":
            self.in_table = True
        elif self.in_table and tag == "tr":
            self._row = []
        elif self.in_table and tag in ("td", "th"):
            self.in_cell = True
            self._text = []

    def handle_endtag(self, tag):
        if tag == "table" and self.in_table:
            self.in_table = False
        elif self.in_table and tag in ("td", "th") and self.in_cell:
            self.in_cell = False
            if self._row is not None:
                self._row.append("".join(self._text).strip())
        elif self.in_table and tag == "tr" and self._row is not None:
            self.rows.append(self._row)
            self._row = None

    def handle_data(self, data):
        if self.in_cell:
            self._text.append(data)


def parse_sp500_html(html):
    """Parse Wikipedia's 'List of S&P 500 companies' page into [(ticker, name)].
    Tickers use '-' instead of '.' (BRK.B -> BRK-B), the style Yahoo and the SEC use."""
    p = _ConstituentsParser()
    p.feed(html)
    if not p.rows:
        raise ValueError("could not find the constituents table (the page layout may have changed)")
    header = [h.lower() for h in p.rows[0]]
    try:
        i_sym, i_name = header.index("symbol"), header.index("security")
    except ValueError:
        raise ValueError(f"unexpected table columns: {p.rows[0]}") from None
    out = [(r[i_sym].replace(".", "-"), r[i_name]) for r in p.rows[1:] if len(r) > max(i_sym, i_name) and r[i_sym]]
    if not out:
        raise ValueError("the constituents table was empty")
    return out


def fetch_sp500_wikipedia(http_get=_http_get):
    html = http_get(config.SP500_URL, {"User-Agent": config.BROWSER_UA}).decode("utf-8", "replace")
    return parse_sp500_html(html)


def make_yahoo_prices_fetcher(pause=config.YAHOO_PAUSE_SECONDS):
    """Daily ADJUSTED closes (splits and dividends included) from Yahoo via the yfinance package."""
    def fetch(ticker, start, end):
        try:
            import yfinance as yf
        except ImportError:
            raise RuntimeError("yfinance is not installed: run  pip install yfinance") from None
        end_excl = (dt.date.fromisoformat(end) + dt.timedelta(days=1)).isoformat()   # Yahoo's end is exclusive
        df = yf.download(ticker, start=start, end=end_excl, auto_adjust=True, progress=False, threads=False)
        time.sleep(pause)
        return yahoo_frame_to_rows(df, ticker)
    return fetch


def yahoo_frame_to_rows(df, ticker):
    """Turn a yfinance DataFrame into [(date, close)]. Handles both the flat and the two-level column layout."""
    if df is None or len(df) == 0:
        return []
    close = df["Close"]
    if hasattr(close, "columns"):                       # newer yfinance: (field, ticker) columns
        close = close[ticker] if ticker in close.columns else close.iloc[:, 0]
    return [(idx.strftime("%Y-%m-%d"), float(v)) for idx, v in close.dropna().items()]


# ---- SEC EDGAR: real filing dates, so the EPS is truly point-in-time -------

def parse_edgar_eps(facts):
    """Turn an EDGAR 'companyfacts' JSON into quarterly EPS rows (period_end, available_date, eps).

    * quarters come from 10-Q filings (3-month figures only)
    * the fourth quarter is never reported on its own, so it is derived:
          Q4 = full-year EPS (10-K) - the three quarters inside that year
      This is an approximation (the share count can change during the year).
    * for each period we keep the EARLIEST filing. Later filings may restate the number,
      and using a restated figure on an earlier date would be look-ahead bias.
    * available_date is the filing date ('filed')."""
    usgaap = facts.get("facts", {}).get("us-gaap", {})
    entries = None
    for concept in ("EarningsPerShareDiluted", "EarningsPerShareBasic"):
        entries = usgaap.get(concept, {}).get("units", {}).get("USD/shares")
        if entries:
            break
    if not entries:
        return []

    quarters, annuals = {}, {}
    for e in entries:
        try:
            start = dt.date.fromisoformat(e["start"])
            end = dt.date.fromisoformat(e["end"])
            filed = dt.date.fromisoformat(e["filed"])
            val = float(e["val"])
        except (KeyError, ValueError, TypeError):
            continue
        days = (end - start).days
        form = str(e.get("form", ""))
        if 80 <= days <= 100 and form.startswith("10-Q"):
            target = quarters
        elif 350 <= days <= 380 and form.startswith("10-K"):
            target = annuals
        else:
            continue
        if end not in target or filed < target[end][0]:
            target[end] = (filed, val, start)

    rows = {end: (filed, val) for end, (filed, val, _) in quarters.items()}
    for end, (filed, val, start) in annuals.items():
        inside = [(qe, qv) for qe, (qf, qv, qs) in quarters.items()
                  if qs >= start - dt.timedelta(days=5) and qe <= end - dt.timedelta(days=60)]
        if len(inside) == 3 and end not in rows:
            rows[end] = (filed, val - sum(v for _, v in inside))
    return [(end.isoformat(), filed.isoformat(), round(val, 4)) for end, (filed, val) in sorted(rows.items())]


def make_edgar_eps_fetcher(user_agent=None, http_get=_http_get, pause=config.EDGAR_PAUSE_SECONDS):
    """Quarterly EPS with real filing dates from SEC EDGAR. SEC requires you to identify yourself:
    pass user_agent='Your Name your.email@example.com' or set the EDGAR_USER_AGENT environment variable."""
    state = {"cik": None}

    def ua():
        agent = user_agent or os.environ.get(config.EDGAR_USER_AGENT_ENV)
        if not agent:
            raise RuntimeError(
                f"SEC EDGAR needs to know who you are. Set the environment variable "
                f"{config.EDGAR_USER_AGENT_ENV}=\"Your Name your.email@example.com\" and try again.")
        return agent

    def cik_for(ticker):
        if state["cik"] is None:
            raw = http_get("https://www.sec.gov/files/company_tickers.json", {"User-Agent": ua()})
            state["cik"] = {v["ticker"].upper(): int(v["cik_str"]) for v in json.loads(raw).values()}
            time.sleep(pause)
        try:
            return state["cik"][ticker.upper()]
        except KeyError:
            raise KeyError(f"{ticker} not found in the SEC ticker list") from None

    def fetch(ticker):
        cik = cik_for(ticker)
        raw = http_get(f"https://data.sec.gov/api/xbrl/companyfacts/CIK{cik:010d}.json", {"User-Agent": ua()})
        time.sleep(pause)
        return parse_edgar_eps(json.loads(raw))
    return fetch


# ---- CSV backups -----------------------------------------------------------

def _read_csv(path):
    with open(path, newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        reader.fieldnames = [h.strip().lower() for h in (reader.fieldnames or [])]
        return list(reader)


def _require(rows, columns, path):
    if rows and not set(columns) <= set(rows[0]):
        raise ValueError(f"{path} must have the columns {columns}; it has {sorted(rows[0])}")


def load_universe_csv(path):
    """CSV with columns: ticker,name  ->  [(ticker, name)]."""
    rows = _read_csv(path)
    _require(rows, ["ticker"], path)
    return [(r["ticker"].strip().upper(), (r.get("name") or r["ticker"]).strip()) for r in rows if r["ticker"].strip()]


def load_prices_csv(path):
    """CSV with columns: date,ticker,adj_close  ->  {ticker: [(date, value), ...]} (values left as text;
    the collector checks them)."""
    rows = _read_csv(path)
    _require(rows, ["date", "ticker", "adj_close"], path)
    out = {}
    for r in rows:
        out.setdefault(r["ticker"].strip().upper(), []).append((r["date"].strip(), r["adj_close"]))
    return out


def load_eps_csv(path):
    """CSV with columns: ticker,period_end,eps and optionally available_date  ->  ({ticker: rows}, n_estimated).
    If available_date is missing or empty, we ESTIMATE it as period_end + EPS_LAG_DAYS and count how many
    rows were estimated, because an estimated date is weaker than a real filing date."""
    rows = _read_csv(path)
    _require(rows, ["ticker", "period_end", "eps"], path)
    out, estimated = {}, 0
    for r in rows:
        period_end = r["period_end"].strip()
        avail = (r.get("available_date") or "").strip()
        if not avail:
            try:
                avail = (dt.date.fromisoformat(period_end) + dt.timedelta(days=config.EPS_LAG_DAYS)).isoformat()
                estimated += 1
            except ValueError:
                avail = ""                                 # the collector will reject it, with a reason
        out.setdefault(r["ticker"].strip().upper(), []).append((period_end, avail, r["eps"]))
    return out, estimated


def make_csv_fetchers(universe_path=None, prices_path=None, eps_path=None):
    """Fetchers that read from CSV files instead of the internet (same signatures as the live ones)."""
    prices = load_prices_csv(prices_path) if prices_path else {}
    eps = load_eps_csv(eps_path)[0] if eps_path else {}
    universe_rows = load_universe_csv(universe_path) if universe_path else None

    def universe():
        if universe_rows is None:
            raise RuntimeError("no universe CSV given")
        return universe_rows

    def in_range(d, start, end):
        try:
            dt.date.fromisoformat(d)
        except ValueError:
            return True                  # not a real date: let the collector reject it and COUNT it
        return start <= d <= end

    def fetch_prices(ticker, start, end):
        return [(d, v) for d, v in prices.get(ticker.upper(), []) if in_range(d, start, end)]

    def fetch_eps(ticker):
        return eps.get(ticker.upper(), [])
    return {"universe": universe, "prices": fetch_prices, "eps": fetch_eps}


def with_backup(primary, backup):
    """Use primary; if it raises or returns nothing, use backup. fn.used counts which one answered."""
    used = Counter()

    def fetch(*args):
        try:
            rows = list(primary(*args))
            if rows:
                used["primary"] += 1
                return rows
        except Exception:                                   # noqa: BLE001  (we fall back on purpose)
            pass
        used["backup"] += 1
        return backup(*args)
    fetch.used = used
    return fetch


# ---------------------------------------------------------------------------
# diagnosing live sources
# ---------------------------------------------------------------------------

FIXES = {
    "universe": "Check your internet, or pass --universe-csv path/to/universe.csv (columns: ticker,name).",
    "prices": "Check your internet and that yfinance is installed (pip install yfinance). Yahoo blocks an IP that "
              "asks too fast: wait a few minutes. Or pass --prices-csv path/to/prices.csv (date,ticker,adj_close).",
    "eps": f"Set {config.EDGAR_USER_AGENT_ENV}=\"Your Name your.email@example.com\" (SEC requires it) and check your "
           "internet. Or pass --eps-csv path/to/eps.csv (ticker,period_end,available_date,eps).",
}


def run_checks(fetchers, sample="AAPL", days=30):
    """Try each source alone, on one ticker. Returns [{'name', 'ok', 'message', 'fix'}]."""
    end = dt.date.today()
    start = (end - dt.timedelta(days=days)).isoformat()
    attempts = [
        ("universe", lambda: fetchers["universe"](), lambda out: f"{len(out)} companies, first: {out[0][0] if out else '-'}"),
        ("prices", lambda: fetchers["prices"](sample, start, end.isoformat()), lambda out: f"{len(out)} daily prices for {sample}"),
        ("eps", lambda: fetchers["eps"](sample), lambda out: f"{len(out)} quarters of EPS for {sample}"),
    ]
    results = []
    for name, call, describe in attempts:
        try:
            out = list(call())
            ok = len(out) > 0
            msg = describe(out) if ok else "the source answered but returned nothing"
        except Exception as e:                              # noqa: BLE001
            ok, msg = False, f"{type(e).__name__}: {e}"
        results.append({"name": name, "ok": ok, "message": msg, "fix": "" if ok else FIXES[name]})
    return results
