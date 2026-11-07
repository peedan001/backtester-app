"""pytest helper: close every SQLite connection a test module opened once that module finishes.

Several tests open throwaway in-memory databases. Closing them explicitly in every test is noisy, so this records each
sqlite3.connect() call and closes the lot at the end of the module (cached worlds stay usable for the whole module).
Without it, Python warns 'unclosed database' when they are garbage-collected, which strict mode (-W error) turns into failures.
"""
import sqlite3

import pytest


@pytest.fixture(scope="module", autouse=True)
def _close_sqlite_connections_at_module_end():
    real_connect = sqlite3.connect
    opened = []

    def recording_connect(*args, **kwargs):
        con = real_connect(*args, **kwargs)
        opened.append(con)
        return con

    sqlite3.connect = recording_connect
    try:
        yield
    finally:
        sqlite3.connect = real_connect
        for con in opened:
            try:
                con.close()
            except sqlite3.Error:
                pass
