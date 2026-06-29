import os
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import psycopg
from psycopg_pool import ConnectionPool
from datetime import datetime
from databricks import sql as dbsql
from databricks.sdk import WorkspaceClient
from databricks.sdk.core import Config

HTTP_PATH = os.environ.get("SQL_WAREHOUSE_HTTP_PATH", "")
ENDPOINT = os.environ.get("MODEL_SERVING_ENDPOINT", "raghavan-trading-signal-predictor")
VS_ENDPOINT = os.environ.get("VS_ENDPOINT", "raghavan-trading-signals-vs")
VS_INDEX = os.environ.get("VS_INDEX_NAME", "raghavan_trading_signals.gold.regime_vector_index")
CATALOG = os.environ.get("CATALOG", "raghavan_trading_signals")
LAKEBASE_HOST = os.environ.get("LAKEBASE_HOST", "")             # Lakebase endpoint hostname
LAKEBASE_ENDPOINT = os.environ.get("LAKEBASE_ENDPOINT", "")     # projects/.../endpoints/...

# The deployed App authenticates as its own service principal via OAuth — no PAT, no DATABRICKS_TOKEN.
# Config() auto-reads DATABRICKS_CLIENT_ID / DATABRICKS_CLIENT_SECRET + host injected by Apps.
cfg = Config()
w = WorkspaceClient(config=cfg)

st.set_page_config(page_title="Trading Signal Generator", page_icon="📈", layout="wide")


# ---- Delta / SQL warehouse access (OLAP) ----
@st.cache_resource
def conn():
    # cfg.host includes the https:// scheme; server_hostname wants the bare host.
    hostname = cfg.host.replace("https://", "").replace("http://", "").rstrip("/")
    return dbsql.connect(server_hostname=hostname, http_path=HTTP_PATH,
                         credentials_provider=lambda: cfg.authenticate)


def q(sql):
    c = conn().cursor()
    c.execute(sql)
    cols = [d[0] for d in c.description]
    rows = c.fetchall()
    c.close()
    return pd.DataFrame(rows, columns=cols)


# ---- Lakebase / Postgres access (OLTP), OAuth token auto-rotates per new connection ----
@st.cache_resource
def pg_pool():
    class OAuthConn(psycopg.Connection):
        @classmethod
        def connect(cls, conninfo="", **kwargs):
            kwargs["password"] = w.postgres.generate_database_credential(
                endpoint=LAKEBASE_ENDPOINT).token
            return super().connect(conninfo, **kwargs)

    user = os.environ["DATABRICKS_CLIENT_ID"]   # the app SP is its own Postgres role
    return ConnectionPool(
        conninfo=f"host={LAKEBASE_HOST} dbname=databricks_postgres user={user} sslmode=require",
        connection_class=OAuthConn, min_size=1, max_size=4, open=True)


def pg_q(sql):
    with pg_pool().connection() as c:
        with c.cursor() as cur:
            cur.execute(sql)
            cols = [d[0] for d in cur.description]
            return pd.DataFrame(cur.fetchall(), columns=cols)


def predict(features_dict):
    try:
        resp = w.serving_endpoints.query(name=ENDPOINT, dataframe_records=[features_dict])
        return {"predictions": resp.predictions}
    except Exception as e:
        return {"error": str(e)}


st.title("Trading Signal Generator")
st.caption("Algorithmic trading signals powered by Databricks AI")

with st.sidebar:
    page = st.radio("Page", ["Today's Signals", "Market Regime Radar", "Live Prediction",
                             "Portfolio Tracker", "Model Health"])
    if st.button("Refresh"):
        st.cache_data.clear(); st.rerun()
    st.caption(f"Refreshed {datetime.now():%Y-%m-%d %H:%M}")


# ---------- 1. Today's Signals ----------
if page == "Today's Signals":
    st.header("Today's Trading Signals")

    @st.cache_data(ttl=300)
    def load():
        return q(f"SELECT * FROM {CATALOG}.gold.v_daily_signals ORDER BY signal DESC, volume_ratio DESC")

    df = load()
    if df.empty:
        st.warning("No signals yet — run the daily pipeline (notebook 11).")
    else:
        b = (df.signal == "BUY").sum(); s = (df.signal == "SELL").sum()
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Total", len(df)); c2.metric("BUY", int(b))
        c3.metric("SELL", int(s)); c4.metric("Buy %", f"{b/len(df)*100:.1f}%")
        left, right = st.columns([1, 2])
        left.plotly_chart(px.pie(df, names="signal", color="signal",
                          color_discrete_map={"BUY": "#00C853", "SELL": "#FF1744"}),
                          use_container_width=True)
        fig = px.scatter(df, x="rsi_14", y="volume_ratio", color="signal",
                         color_discrete_map={"BUY": "#00C853", "SELL": "#FF1744"},
                         hover_data=["symbol", "close_price"])
        fig.add_vline(x=30, line_dash="dash"); fig.add_vline(x=70, line_dash="dash")
        right.plotly_chart(fig, use_container_width=True)
        st.dataframe(df, use_container_width=True)


# ---------- 2. Market Regime Radar (AI Search) ----------
elif page == "Market Regime Radar":
    st.header("Market Regime Radar")
    st.caption("AI Search finds the historical weeks most similar to now — and what happened next.")

    @st.cache_data(ttl=600)
    def regimes():
        return q(f"""SELECT week_start, avg_market_return, weekly_vix,
                     pct_stocks_oversold, pct_macd_bullish, avg_volatility
                     FROM {CATALOG}.gold.weekly_market_regimes ORDER BY week_start""")

    @st.cache_data(ttl=600)
    def similar():
        from databricks.ai_search.client import AISearchClient
        vsc = AISearchClient()   # OAuth via the app's service principal
        idx = vsc.get_index(endpoint_name=VS_ENDPOINT, index_name=VS_INDEX)
        latest = q(f"""SELECT week_start, regime_vector FROM {CATALOG}.gold.weekly_market_regimes
                       ORDER BY week_start DESC LIMIT 1""")
        wid = latest.iloc[0]["week_start"]
        vec = list(latest.iloc[0]["regime_vector"])
        res = idx.similarity_search(query_vector=vec, num_results=6,
                  columns=["week_start", "weekly_vix", "next_1w_return", "next_1w_positive"])
        rows = [r for r in res["result"]["data_array"] if str(r[0]) != str(wid)][:5]
        return wid, pd.DataFrame(rows, columns=["week_start", "weekly_vix",
                                                "next_1w_return", "next_1w_positive"])

    rg = regimes()
    if rg.empty:
        st.warning("No regime data.")
    else:
        cur = rg.iloc[-1]
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Avg Return", f"{cur.avg_market_return:.4f}")
        c2.metric("VIX", f"{cur.weekly_vix:.1f}")
        c3.metric("% Oversold", f"{cur.pct_stocks_oversold*100:.1f}%")
        c4.metric("% MACD Bullish", f"{cur.pct_macd_bullish*100:.1f}%")
        fig = px.line(rg, x="week_start", y="weekly_vix", title="Weekly VIX")
        fig.add_hline(y=20, line_dash="dash"); fig.add_hline(y=30, line_dash="dash")
        st.plotly_chart(fig, use_container_width=True)
        try:
            wid, sim = similar()
            st.subheader(f"5 weeks most similar to {wid}")
            if not sim.empty:
                up = (sim.next_1w_positive == 1).sum()
                st.info(f"In **{up}/{len(sim)}** similar weeks the market rose the following week.")
                st.dataframe(sim, use_container_width=True)
        except Exception as e:
            st.warning(f"AI Search unavailable: {e}")


# ---------- 3. Live Prediction ----------
elif page == "Live Prediction":
    st.header("Live Prediction")

    @st.cache_data(ttl=300)
    def stocks():
        return q(f"SELECT DISTINCT symbol FROM {CATALOG}.gold.daily_features ORDER BY symbol")

    sym = st.selectbox("Stock", stocks()["symbol"].tolist())
    if st.button("Get Prediction", type="primary"):
        feat = q(f"""SELECT * FROM {CATALOG}.gold.daily_features
                     WHERE symbol = '{sym}' ORDER BY trade_date DESC LIMIT 1""")
        if feat.empty:
            st.error("No features.")
        else:
            row = feat.iloc[0]
            c1, c2, c3 = st.columns(3)
            c1.metric("Close", f"${row['close']:.2f}")
            c2.metric("RSI", f"{row['rsi_14']:.1f}")
            c3.metric("Vol 20d", f"{row['volatility_20d']:.4f}")
            meta = {"symbol", "trade_date", "open", "high", "low", "close", "adj_close",
                    "volume", "dividends", "stock_splits", "prev_close", "bronze_ingested_at",
                    "bronze_source_file", "next_day_return", "next_day_direction"}
            fdict = {k: (float(v) if v is not None else 0.0)
                     for k, v in row.to_dict().items() if k not in meta}
            res = predict(fdict)
            if "error" in res:
                st.error(res["error"])
            else:
                pred = res.get("predictions", [None])[0]
                sig = "BUY" if pred == 1 else "SELL"
                st.markdown(f"### Prediction: :{'green' if sig=='BUY' else 'red'}[{sig}]")


# ---------- 4. Portfolio Tracker ----------
elif page == "Portfolio Tracker":
    st.header("Portfolio Tracker")

    # These tables live in Lakebase (Postgres), so query via pg_q — NOT the SQL warehouse.
    # Postgres tables are in the `public` schema, so no catalog/schema prefix.
    @st.cache_data(ttl=60)
    def positions():
        return pg_q("""SELECT symbol, quantity, avg_entry_price, current_price, unrealized_pnl,
                       position_type FROM portfolio_positions WHERE is_open = TRUE""")

    @st.cache_data(ttl=60)
    def alerts():
        return pg_q("""SELECT alert_type, symbol, message, severity FROM alerts
                       WHERE is_acknowledged = FALSE ORDER BY created_at DESC""")

    a = alerts()
    for _, r in a[a.severity == "CRITICAL"].iterrows():
        st.error(f"**{r.alert_type}**: {r.message}")
    for _, r in a[a.severity == "WARNING"].iterrows():
        st.warning(f"**{r.alert_type}**: {r.message}")

    p = positions()
    if p.empty:
        st.info("No open positions — run batch scoring + Lakebase seeding.")
    else:
        st.metric("Open Positions", len(p))
        st.dataframe(p, use_container_width=True)
        p["exposure"] = p.quantity * p.avg_entry_price
        st.plotly_chart(px.bar(p.sort_values("exposure"), x="exposure", y="symbol",
                        orientation="h", title="Exposure by Stock"), use_container_width=True)


# ---------- 5. Model Health ----------
elif page == "Model Health":
    st.header("Model Health")

    @st.cache_data(ttl=300)
    def acc():
        return q(f"SELECT * FROM {CATALOG}.gold.v_accuracy_summary ORDER BY week")

    @st.cache_data(ttl=300)
    def pnl():
        return q(f"SELECT * FROM {CATALOG}.gold.v_portfolio_simulation ORDER BY trade_date")

    a = acc()
    if a.empty:
        st.info("Needs 2+ days of signals to compute outcomes.")
    else:
        overall = a.correct_signals.sum() / a.total_signals.sum() * 100
        st.metric("Overall Accuracy", f"{overall:.1f}%")
        fig = px.bar(a, x="week", y="accuracy_pct", title="Weekly Accuracy")
        fig.add_hline(y=50, line_dash="dash", annotation_text="Random (50%)")
        fig.add_hline(y=55, line_dash="dash", annotation_text="Target (55%)")
        st.plotly_chart(fig, use_container_width=True)
    pn = pnl()
    if not pn.empty:
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=pn.trade_date, y=pn.cumulative_strategy_pct, name="Strategy"))
        fig.add_trace(go.Scatter(x=pn.trade_date, y=pn.cumulative_benchmark_pct, name="Buy & Hold"))
        st.plotly_chart(fig, use_container_width=True)

st.divider()
st.caption("Unity Catalog · Autoloader · DLT · Spark · AutoML · LSTM · MLflow · Model Serving · "
           "AI Search · Iceberg · Lakeflow · Lakebase · SQL Warehouse · Genie · Databricks Apps")