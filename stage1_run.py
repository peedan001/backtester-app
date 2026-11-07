"""stage1_run.py: walk through Stage 1 (catalogue and schema).

    python stage1_run.py --demo        # the full walkthrough, no internet needed
    python stage1_run.py --try         # interactive: pick a bad row, watch it get rejected
    python stage1_run.py --try 3       # run bad row number 3 directly
"""
import sqlite3
import sys

import config
import db

# Each case: (title, sql, params, should_succeed, lesson)
# Cases run in order on ONE database, so earlier good rows set the scene.
CASES = [
    ("GOOD: add stock AAPL",
     "INSERT INTO STOCK (ticker, name) VALUES (?, ?)", ("AAPL", "Apple Inc."), True,
     "A normal row goes in."),
    ("GOOD: AAPL price on 2020-01-02",
     "INSERT INTO PRICE (ticker, date, adj_close) VALUES (?, ?, ?)", ("AAPL", "2020-01-02", 74.0), True,
     "A normal row goes in."),
    ("BAD: the same price again (same ticker and date)",
     "INSERT INTO PRICE (ticker, date, adj_close) VALUES (?, ?, ?)", ("AAPL", "2020-01-02", 75.0), False,
     "Composite primary key (ticker, date): one value per stock per day, so a double load cannot double count."),
    ("BAD: price for a stock we never added (ZZZZ)",
     "INSERT INTO PRICE (ticker, date, adj_close) VALUES (?, ?, ?)", ("ZZZZ", "2020-01-02", 10.0), False,
     "Foreign key: no price can exist without its stock."),
    ("BAD: price of 0",
     "INSERT INTO PRICE (ticker, date, adj_close) VALUES (?, ?, ?)", ("AAPL", "2020-01-03", 0.0), False,
     "CHECK adj_close > 0: a zero price would make a return divide by zero."),
    ("BAD: impossible date 2020-13-45",
     "INSERT INTO PRICE (ticker, date, adj_close) VALUES (?, ?, ?)", ("AAPL", "2020-13-45", 74.0), False,
     "CHECK date is a real ISO date: typos like this would break every date comparison later."),
    ("BAD: date written 2020/01/03 instead of 2020-01-03",
     "INSERT INTO PRICE (ticker, date, adj_close) VALUES (?, ?, ?)", ("AAPL", "2020/01/03", 74.0), False,
     "ISO text dates sort correctly as text; other formats would silently sort wrong."),
    ("BAD: EPS 'available' BEFORE its quarter ended (look-ahead!)",
     "INSERT INTO FUNDAMENTAL (ticker, period_end, available_date, eps) VALUES (?, ?, ?, ?)",
     ("AAPL", "2020-03-31", "2020-03-01", 0.65), False,
     "CHECK available_date >= period_end: the database itself refuses numbers that 'know the future'."),
    ("GOOD: EPS available 45 days AFTER the quarter ended",
     "INSERT INTO FUNDAMENTAL (ticker, period_end, available_date, eps) VALUES (?, ?, ?, ?)",
     ("AAPL", "2020-03-31", "2020-05-15", 0.65), True,
     "The honest version: the number appears only once the world could see it."),
    ("BAD: signal weight 1.5 (must be between 0 and 1)",
     "INSERT INTO RUN (top_n, value_weight, cost_bps) VALUES (?, ?, ?)", (20, 1.5, 10), False,
     "CHECK value_weight between 0 and 1: the slider can never save nonsense."),
    ("BAD: negative trading cost",
     "INSERT INTO RUN (top_n, value_weight, cost_bps) VALUES (?, ?, ?)", (20, 0.5, -5), False,
     "CHECK cost_bps >= 0: a negative cost would make trading look like free money."),
    ("BAD: a portfolio of 0 stocks",
     "INSERT INTO RUN (top_n, value_weight, cost_bps) VALUES (?, ?, ?)", (0, 0.5, 10), False,
     "CHECK top_n between 1 and 500."),
    ("BAD: the text 'N/A' as a price",
     "INSERT INTO PRICE (ticker, date, adj_close) VALUES (?, ?, ?)", ("AAPL", "2020-01-06", "N/A"), False,
     "Added in Stage 2: SQLite is loosely typed and would store text in a number column. CHECK typeof() stops it."),
    ("BAD: an infinite price",
     "INSERT INTO PRICE (ticker, date, adj_close) VALUES (?, ?, ?)", ("AAPL", "2020-01-07", float("inf")), False,
     "Added in Stage 2: infinity passes 'price > 0' but would ruin every average. CHECK price < 1e12 stops it."),
]


def attempt(con, case):
    title, sql, params, should_succeed, lesson = case
    try:
        con.execute(sql, params)
        con.commit()
        outcome = "ACCEPTED"
        error = ""
    except sqlite3.Error as e:
        outcome = "REJECTED"
        error = str(e)
    ok = (outcome == "ACCEPTED") == should_succeed
    return outcome, error, ok


def show_case(i, case, outcome, error, ok):
    title, _, params, _, lesson = case
    print(f"\n[{i}] {title}")
    print(f"    values : {params}")
    print(f"    result : {outcome}" + (f"  ({error})" if error else ""))
    print(f"    why    : {lesson}")
    if not ok:
        print("    !! UNEXPECTED: the database behaved differently from what we designed")


def fresh_db():
    con = db.connect(":memory:")
    db.create_schema(con)
    return con


def demo():
    print("=" * 70)
    print("STAGE 1: catalogue and schema")
    print("=" * 70)

    print("\n--- The catalogue (config.py): every constant in one place ---")
    for name in dir(config):
        if name.isupper() and name != "TABLES":
            print(f"  {name:28s} = {getattr(config, name)!r}")

    con = fresh_db()
    print("\n--- The tables (db.py) ---")
    print(f"  foreign keys switched on: {db.foreign_keys_on(con)}")
    for t in db.table_names(con):
        cols = db.columns(con, t)
        pk = [c[0] for c in sorted((c for c in cols if c[3] > 0), key=lambda c: c[3])]
        print(f"  {t:12s} columns: {', '.join(c[0] for c in cols)}")
        print(f"  {'':12s} primary key: ({', '.join(pk)})")

    print("\n--- Trying good and bad rows: the database enforces the rules ---")
    all_ok = True
    for i, case in enumerate(CASES, 1):
        outcome, error, ok = attempt(con, case)
        all_ok = all_ok and ok
        show_case(i, case, outcome, error, ok)

    print("\n" + "=" * 70)
    print("Every row behaved as designed." if all_ok else "SOMETHING BEHAVED UNEXPECTEDLY: see the !! lines.")
    print("Check question: why does FUNDAMENTAL have TWO dates (period_end and available_date)?")
    print("(Answer in the README/study guide at the end of the project. Try saying it aloud first.)")
    return 0 if all_ok else 1


def try_mode(arg):
    con = fresh_db()
    # Run the GOOD setup rows silently so the BAD cases have something to collide with.
    for case in CASES:
        if case[3]:
            attempt(con, case)
    bad = [(i, c) for i, c in enumerate(CASES, 1) if not c[3]]

    if arg is not None:
        picks = [int(arg)]
    else:
        picks = None

    while True:
        if picks is None:
            print("\nBad rows you can try (the database should reject each one):")
            for i, c in bad:
                print(f"  {i:2d}. {c[0]}")
            try:
                choice = input("Pick a number (or press Enter to quit): ").strip()
            except EOFError:
                break
            if not choice:
                break
            if not choice.isdigit() or int(choice) not in dict(bad):
                print("  Not one of the bad rows listed. Try again.")
                continue
            n = int(choice)
        else:
            if not picks:
                break
            n = picks.pop(0)
            if n not in dict(bad):
                print(f"Row {n} is not a bad row. Valid: {[i for i, _ in bad]}")
                return 1
        case = CASES[n - 1]
        outcome, error, ok = attempt(con, case)
        show_case(n, case, outcome, error, ok)
        if picks is not None and not picks:
            break
    return 0


def main(argv):
    if "--try" in argv:
        idx = argv.index("--try")
        arg = argv[idx + 1] if idx + 1 < len(argv) and argv[idx + 1].isdigit() else None
        return try_mode(arg)
    return demo()


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
