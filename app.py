"""app.py: the Backtester as a Streamlit app.

    pip install streamlit
    streamlit run app.py

All the thinking lives in app_logic.py (tested). This file only collects choices and draws the answers.
Data comes from the sidebar: a fake market built on the spot, or a database file saved by a --live run
(python stage2_run.py --live --db backtest.db). The app never fetches from Yahoo or EDGAR itself.
"""
import os

import pandas as pd
import streamlit as st

import app_logic as L
import config

HERE = os.path.dirname(os.path.abspath(__file__))

st.set_page_config(page_title="Backtester: low P/E + momentum", layout="wide")


# ---------------------------------------------------------------------------
# cached work: the same choices always give the same answer, so they are computed once
# ---------------------------------------------------------------------------

@st.cache_resource(show_spinner="Building the market...")
def get_world(kind, path, mtime, edge, seed, reaction, n_dead):
    if kind == "database":
        return L.open_database(path)
    return L.open_synthetic(edge=edge, seed=seed, reaction=reaction, n_dead=n_dead)


@st.cache_data(show_spinner="Running the backtest...")
def backtest(world_key, choices):
    return L.run_one(get_world(*world_key), **dict(choices))


@st.cache_data(show_spinner="Running four backtests...")
def bias_ladder(world_key, top_n, value_weight, cost_bps):
    return L.bias_summary(get_world(*world_key), top_n=top_n, value_weight=value_weight, cost_bps=cost_bps)


@st.cache_data(show_spinner="Building fake worlds and running the ladder in each (about 6 seconds per world)...")
def bias_seeds(world_key, n_worlds, top_n, value_weight, cost_bps):
    seeds = list(range(1, n_worlds + 1))
    return L.seeds_summary(get_world(*world_key), seeds, top_n=top_n, value_weight=value_weight, cost_bps=cost_bps)


@st.cache_data(show_spinner="Running the failed-stock backtests...")
def graveyard(world_key, top_n):
    table, handle = L.graveyard_rows(get_world(*world_key), top_n=top_n)
    return table, handle


def percent_columns(df, columns):
    return df.style.format({c: "{:.1%}" for c in columns}, na_rep="n/a")


# ---------------------------------------------------------------------------
# sidebar: where the data comes from
# ---------------------------------------------------------------------------

st.sidebar.title("Data")
source = st.sidebar.radio("Source", ["Fake market", "Database file"],
                          help="The fake market has a planted edge, so we know the right answer. A database file comes "
                               "from running stage2_run.py --live on your own computer.")
if source == "Fake market":
    edge = st.sidebar.slider("Planted edge (extra return per year for a perfect score)", 0.0, 0.15,
                             config.SYNTH_DEFAULT_EDGE, 0.01, format="%.2f")
    reaction = st.sidebar.slider("Earnings reaction (0 = earnings do not move prices)", 0.0, config.SYNTH_MAX_REACTION,
                                 config.SYNTH_DEFAULT_REACTION, 0.5)
    n_dead = st.sidebar.slider("Failed stocks (not in today's list)", 0, config.SYNTH_MAX_DEAD, config.SYNTH_DEFAULT_DEAD, 5)
    seed = st.sidebar.number_input("Seed (a different seed is a different random world)", 1, 9999, config.SYNTH_SEED, 1)
    world_key = ("synthetic", "", 0, edge, int(seed), reaction, int(n_dead))
else:
    path = st.sidebar.text_input("Path to the database file", value=config.DB_PATH)
    mtime = os.path.getmtime(path) if os.path.isfile(path) else 0
    world_key = ("database", path, mtime, 0.0, 0, 0.0, 0)

try:
    world = get_world(*world_key)
except L.AppError as e:
    st.sidebar.error(str(e))
    st.title("Backtester")
    st.info("Pick a data source in the sidebar.")
    st.stop()

st.sidebar.success(world.label)
st.sidebar.caption(f"{world.n_stocks} stocks in today's list, {world.n_dead} failed stocks, prices {world.first_date} to {world.last_date}.")

st.title("Backtester: low P/E + momentum")
tab_bt, tab_bias, tab_grave, tab_how = st.tabs(["Backtest", "Bias lab", "Graveyard", "How it works"])


# ---------------------------------------------------------------------------
# tab 1: one backtest
# ---------------------------------------------------------------------------

with tab_bt:
    with st.form("backtest_form"):
        c1, c2, c3 = st.columns(3)
        top_n = c1.slider("Stocks held (N)", 1, max(world.n_stocks, 2), min(config.DEFAULT_TOP_N, world.n_stocks))
        value_weight = c1.slider("Weight on cheap P/E (0 = all momentum, 1 = all value)", 0.0, 1.0, config.DEFAULT_VALUE_WEIGHT, 0.05)
        cost_bps = c2.slider("Trading cost (basis points per side)", 0, 100, config.DEFAULT_COST_BPS)
        buffer = c2.slider("Buffer (1.0 = fresh top N each month)", config.MIN_BUFFER, 3.0, config.DEFAULT_BUFFER, 0.1)
        delay = c3.slider("Trade delay (trading days after the signal)", 0, config.MAX_EXEC_DELAY_DAYS, config.EXEC_DELAY_DAYS)
        pe_mode = c3.selectbox("P/E version", list(config.EPS_MODES), help="lagged = honest. no_lag and today look ahead on purpose.")
        universe = c3.selectbox("Stocks considered", list(config.UNIVERSES),
                                help="survivors = today's list only. all = also the stocks that later failed.")
        submitted = st.form_submit_button("Run backtest", type="primary")
    if submitted:
        st.session_state["choices"] = dict(top_n=top_n, value_weight=value_weight, buffer=buffer, cost_bps=cost_bps,
                                           delay=delay, pe_mode=pe_mode, universe=universe)
    choices = st.session_state.get("choices")
    if not choices:
        st.info("Set your choices and press **Run backtest**. Identical choices are served from a cache, so going back is instant.")
    else:
        try:
            result = backtest(world_key, tuple(sorted(choices.items())))
        except L.AppError as e:
            st.error(str(e))
            st.stop()
        if choices["pe_mode"] != "lagged":
            st.warning("This run uses information that did not exist on the trading date (see the P/E version). "
                       "Its result is inflated on purpose. Compare it with the honest run in the Bias lab.")
        if result["ledger_problems"]:
            st.error("The independent share-counting check disagrees with the engine: " + "; ".join(result["ledger_problems"][:3]))
        else:
            st.caption("Engine check passed: an independent share-counting ledger gives the same wealth on every date.")
        m = result["metrics"]
        a, b, c, d = st.columns(4)
        a.metric("Return per year (after costs)", L.fmt(m["net"]["cagr"], "pct"), f"{(m['excess_cagr_net']) * 100:+.1f} pts vs SPY")
        b.metric("SPY per year", L.fmt(m["benchmark"]["cagr"], "pct"))
        c.metric("Sharpe (after costs)", L.fmt(m["net"]["sharpe"], "num"))
        d.metric("Worst fall", L.fmt(m["net"]["max_drawdown"], "pct"))
        curve = pd.DataFrame(L.curve_rows(result["periods"]))
        curve["date"] = pd.to_datetime(curve["date"])
        st.subheader("Growth of 1")
        st.line_chart(curve.set_index("date")[["after costs", "before costs", "SPY"]])
        left, right = st.columns(2)
        with left:
            st.subheader("Headline numbers")
            table = pd.DataFrame(L.metric_rows(m))
            shown = table[["measure"]].copy()
            for col in ("before costs", "after costs", "SPY"):
                shown[col] = [L.fmt(v, k) for v, k in zip(table[col], table["kind"])]
            st.dataframe(shown, hide_index=True, width="stretch")
        with right:
            st.subheader("Against SPY and trading")
            facts = pd.DataFrame([{"measure": n, "value": L.fmt(v, k)} for n, v, k in L.summary_facts(m)])
            st.dataframe(facts, hide_index=True, width="stretch")
        st.subheader("What did it hold?")
        dates = L.signal_dates(result)
        date = st.select_slider("Month-end signal", options=dates, value=dates[len(dates) // 2])
        rows, facts = L.holding_rows(result, date)
        st.write(f"Signal on **{facts['signal_date']}**, traded at the close of **{facts['trade_date']}**. "
                 f"Traded {facts.get('traded_fraction', 0) * 100:.0f}% of the portfolio, cost {facts.get('cost', 0) * 1e4:.1f} basis points of its value."
                 + (f" Sold because data ended: {', '.join(facts['delisted'])}." if facts.get("delisted") else ""))
        st.dataframe(pd.DataFrame(rows).style.format({"target weight": "{:.1%}", "score": "{:.2f}"}),
                     hide_index=True, width="stretch")


# ---------------------------------------------------------------------------
# tab 2: bias lab
# ---------------------------------------------------------------------------

with tab_bias:
    st.write("The same strategy, four ways. Only the honest line is something you could have traded. "
             "Each step below it shows what a shortcut would have added.")
    c1, c2, c3 = st.columns(3)
    b_n = c1.number_input("Stocks held (N)", 1, max(world.n_stocks, 2), min(config.DEFAULT_TOP_N, world.n_stocks), key="b_n")
    b_w = c2.slider("Weight on cheap P/E", 0.0, 1.0, config.DEFAULT_VALUE_WEIGHT, 0.05, key="b_w")
    b_c = c3.slider("Trading cost (bps)", 0, 100, config.DEFAULT_COST_BPS, key="b_c")
    if st.button("Run the bias ladder", type="primary"):
        st.session_state["ladder"] = (int(b_n), b_w, b_c)
    if "ladder" in st.session_state:
        try:
            ladder = bias_ladder(world_key, *st.session_state["ladder"])
        except L.AppError as e:
            st.error(str(e))
        else:
            ladder_table = pd.DataFrame(ladder["rows"])
            st.dataframe(percent_columns(ladder_table, ["per year after costs"]).format({"Sharpe": "{:.2f}"}),
                         hide_index=True, width="stretch")
            a, b, c, d = st.columns(4)
            a.metric("No filing lag adds", f"{ladder['inflation_no_lag'] * 100:+.2f}% / yr")
            b.metric("Today's EPS adds", f"{ladder['inflation_today'] * 100:+.2f}% / yr")
            if ladder["has_failed_stocks"]:
                c.metric("Survivorship adds", f"{ladder['survivorship'] * 100:+.2f}% / yr")
                d.metric("Naive backtest vs honest", f"{ladder['total'] * 100:+.2f}% / yr")
            else:
                st.info("No failed stocks in this world, so survivorship cannot be measured. Add some in the sidebar.")
            st.caption("This is ONE world. A single difference can be luck: see the repeat below before you trust it.")
    st.subheader("Is it luck? Repeat over many worlds")
    if not world.synthetic:
        st.info("Repeating needs a fake market: a real database holds only one world.")
    else:
        n_worlds = st.slider("Number of fake worlds", 2, 8, 3)
        if st.button("Repeat the ladder"):
            st.session_state["seeds"] = (n_worlds, int(b_n), b_w, b_c)
        if "seeds" in st.session_state:
            repeated = pd.DataFrame(bias_seeds(world_key, *st.session_state["seeds"]))
            st.dataframe(percent_columns(repeated, ["mean per year", "standard error"]), hide_index=True, width="stretch")
            st.caption("INFLATED means the average is more than two standard errors above zero AND above half a percent a year. "
                       "With only a few worlds the standard error is rough: use 5 or more before you believe a verdict.")


# ---------------------------------------------------------------------------
# tab 3: failed stocks
# ---------------------------------------------------------------------------

with tab_grave:
    if not world.n_dead:
        st.info("This world has no failed stocks. Add some with the sidebar slider, or load a database that has them.")
    else:
        g_n = st.number_input("Stocks held (N)", 1, max(world.n_stocks, 2), min(config.DEFAULT_TOP_N, world.n_stocks), key="g_n")
        table, handle = graveyard(world_key, int(g_n))
        st.write("Every failed stock, and how the blend (half value, half momentum) and pure value treated it. "
                 "These runs use ALL stocks, so the failed ones are eligible.")
        gdf = pd.DataFrame(table)
        st.dataframe(percent_columns(gdf, ["fell from peak"]), hide_index=True, width="stretch")
        ticker = st.selectbox("Follow one failed stock", L.failed_tickers(world))
        left, right = st.columns([3, 2])
        with left:
            st.code("\n".join(L.graveyard_story(world, handle, ticker)), language=None)
        with right:
            history = pd.DataFrame(L.price_history(world, ticker), columns=["date", "price"])
            history["date"] = pd.to_datetime(history["date"])
            st.line_chart(history.set_index("date")["price"])


# ---------------------------------------------------------------------------
# tab 4: how it works
# ---------------------------------------------------------------------------

with tab_how:
    guide = os.path.join(HERE, "STUDY_GUIDE.md")
    if not os.path.isfile(guide):
        st.info("STUDY_GUIDE.md was not found next to app.py.")
    else:
        with open(guide, encoding="utf-8") as f:
            text = f.read()
        head, *sections = text.split("\n## ")
        st.markdown(head)
        for section in sections:
            title, _, body = section.partition("\n")
            with st.expander(title):
                st.markdown(body)
