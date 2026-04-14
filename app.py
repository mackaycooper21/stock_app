# app.py
# -------------------------------------------------------
# Stock Analysis Dashboard
# Run with: uv run streamlit run app.py
# -------------------------------------------------------

import math
from datetime import date, timedelta

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import yfinance as yf
from scipy import stats

# -- Page configuration ----------------------------------
st.set_page_config(page_title="Stock Analyzer", layout="wide")
st.title("Stock Analysis Dashboard")
st.caption("Analyze historical stock price performance, risk, and return metrics.")

# -- Sidebar: user inputs --------------------------------
st.sidebar.header("Settings")

ticker = st.sidebar.text_input("Stock Ticker", value="AAPL").upper().strip()
default_start = date.today() - timedelta(days=365)

start_date = st.sidebar.date_input(
    "Start Date",
    value=default_start,
    min_value=date(1970, 1, 1),
)

end_date = st.sidebar.date_input(
    "End Date",
    value=date.today(),
    min_value=date(1970, 1, 1),
)

if start_date >= end_date:
    st.sidebar.error("Start date must be before end date.")
    st.stop()

ma_window = st.sidebar.slider(
    "Moving Average Window (days)",
    min_value=5,
    max_value=200,
    value=50,
    step=5,
)

risk_free_rate = (
    st.sidebar.number_input(
        "Risk-Free Rate (%)",
        min_value=0.0,
        max_value=20.0,
        value=4.5,
        step=0.1,
    )
    / 100
)

vol_window = st.sidebar.slider(
    "Rolling Volatility Window (days)",
    min_value=10,
    max_value=120,
    value=30,
    step=5,
)


# -- Data download ----------------------------------------
@st.cache_data(show_spinner="Fetching data...", ttl=3600)
def load_data(ticker_symbol: str, start: date, end: date) -> pd.DataFrame:
    """Download daily stock data from Yahoo Finance."""
    df = yf.download(ticker_symbol, start=start, end=end, progress=False, auto_adjust=False)
    return df


def format_percent(value: float) -> str:
    """Safely format a decimal as a percentage."""
    if pd.isna(value) or np.isinf(value):
        return "N/A"
    return f"{value:.2%}"


def format_number(value: float, decimals: int = 2, prefix: str = "") -> str:
    """Safely format a numeric value."""
    if pd.isna(value) or np.isinf(value):
        return "N/A"
    return f"{prefix}{value:,.{decimals}f}"


# -- Main logic -------------------------------------------
if not ticker:
    st.info("Enter a stock ticker in the sidebar to get started.")
    st.stop()

try:
    df = load_data(ticker, start_date, end_date)
except Exception as e:
    st.error(f"Failed to download data for {ticker}: {e}")
    st.stop()

if df.empty:
    st.error(f"No data found for {ticker}. Check the ticker symbol and try again.")
    st.stop()

if isinstance(df.columns, pd.MultiIndex):
    df.columns = df.columns.get_level_values(0)

required_cols = {"Open", "High", "Low", "Close", "Volume"}
missing_cols = required_cols - set(df.columns)

if missing_cols:
    st.error(f"Missing expected columns: {', '.join(sorted(missing_cols))}")
    st.stop()

df = df.copy()
df = df.sort_index()

# -- Derived columns --------------------------------------
df["Daily Return"] = df["Close"].pct_change()
df[f"{ma_window}-Day MA"] = df["Close"].rolling(window=ma_window, min_periods=ma_window).mean()
df["Cumulative Return"] = (1 + df["Daily Return"]).cumprod() - 1
df["Rolling Volatility"] = (
    df["Daily Return"].rolling(window=vol_window, min_periods=vol_window).std() * math.sqrt(252)
)

returns_clean = df["Daily Return"].dropna()

if returns_clean.empty:
    st.error("Not enough data to calculate returns. Try expanding the date range.")
    st.stop()

if ma_window > len(df):
    st.warning(
        f"The selected {ma_window}-day moving average window is longer than the available "
        f"data ({len(df)} trading days), so the moving average line may not appear."
    )

if vol_window > len(df):
    st.warning(
        f"The selected {vol_window}-day volatility window is longer than the available "
        f"data ({len(df)} trading days), so the rolling volatility line may not appear."
    )

# -- Key metrics ------------------------------------------
latest_close = float(df["Close"].iloc[-1])
total_return = float(df["Cumulative Return"].iloc[-1])
avg_daily_ret = float(returns_clean.mean())
daily_volatility = float(returns_clean.std())

ann_return = avg_daily_ret * 252
ann_volatility = daily_volatility * math.sqrt(252)
sharpe = (ann_return - risk_free_rate) / ann_volatility if ann_volatility > 0 else np.nan

skewness = float(returns_clean.skew())
kurtosis = float(returns_clean.kurtosis())
max_close = float(df["Close"].max())
min_close = float(df["Close"].min())

# Optional extra metrics
trading_days = len(df)
latest_volume = float(df["Volume"].iloc[-1])
avg_volume = float(df["Volume"].mean())

# -- Key metrics display ----------------------------------
st.subheader(f"{ticker} — Key Metrics")

col1, col2, col3, col4 = st.columns(4)
col1.metric("Latest Close", format_number(latest_close, 2, "$"))
col2.metric("Total Return", format_percent(total_return))
col3.metric("Annualized Return", format_percent(ann_return))
col4.metric("Sharpe Ratio", format_number(sharpe, 2))

col5, col6, col7, col8 = st.columns(4)
col5.metric("Annualized Volatility", format_percent(ann_volatility))
col6.metric("Skewness", format_number(skewness, 2))
col7.metric("Excess Kurtosis", format_number(kurtosis, 2))
col8.metric("Avg Daily Return", format_percent(avg_daily_ret))

col9, col10, col11, col12 = st.columns(4)
col9.metric("Period High", format_number(max_close, 2, "$"))
col10.metric("Period Low", format_number(min_close, 2, "$"))
col11.metric("Latest Volume", format_number(latest_volume, 0))
col12.metric("Avg Volume", format_number(avg_volume, 0))

st.caption(f"Trading days in selected period: {trading_days}")

st.divider()

# -- Price & Moving Average chart -------------------------
st.subheader("Price & Moving Average")

fig_price = go.Figure()

fig_price.add_trace(
    go.Scatter(
        x=df.index,
        y=df["Close"],
        mode="lines",
        name="Close Price",
        line=dict(width=2),
    )
)

fig_price.add_trace(
    go.Scatter(
        x=df.index,
        y=df[f"{ma_window}-Day MA"],
        mode="lines",
        name=f"{ma_window}-Day MA",
        line=dict(width=2, dash="dash"),
    )
)

fig_price.update_layout(
    yaxis_title="Price (USD)",
    xaxis_title="Date",
    template="plotly_white",
    height=450,
    legend_title="Series",
    margin=dict(l=20, r=20, t=40, b=20),
)

st.plotly_chart(fig_price, use_container_width=True)

# -- Volume chart -----------------------------------------
st.subheader("Daily Trading Volume")

fig_vol = go.Figure()
fig_vol.add_trace(
    go.Bar(
        x=df.index,
        y=df["Volume"],
        name="Volume",
        opacity=0.75,
    )
)

fig_vol.update_layout(
    yaxis_title="Shares Traded",
    xaxis_title="Date",
    template="plotly_white",
    height=350,
    margin=dict(l=20, r=20, t=40, b=20),
)

st.plotly_chart(fig_vol, use_container_width=True)

# -- Daily returns distribution ---------------------------
st.subheader("Distribution of Daily Returns")

fig_hist = go.Figure()
fig_hist.add_trace(
    go.Histogram(
        x=returns_clean,
        nbinsx=60,
        opacity=0.75,
        name="Daily Returns",
        histnorm="probability density",
    )
)

mu = float(returns_clean.mean())
sigma = float(returns_clean.std())

if sigma > 0:
    x_range = np.linspace(float(returns_clean.min()), float(returns_clean.max()), 200)
    fig_hist.add_trace(
        go.Scatter(
            x=x_range,
            y=stats.norm.pdf(x_range, mu, sigma),
            mode="lines",
            name="Normal Distribution",
            line=dict(width=2),
        )
    )

fig_hist.update_layout(
    xaxis_title="Daily Return",
    yaxis_title="Density",
    template="plotly_white",
    height=350,
    margin=dict(l=20, r=20, t=40, b=20),
)

st.plotly_chart(fig_hist, use_container_width=True)

# Jarque-Bera test
if len(returns_clean) >= 2:
    jb_stat, jb_pvalue = stats.jarque_bera(returns_clean)
    interpretation = (
        "Fail to reject normality (p > 0.05)"
        if jb_pvalue > 0.05
        else "Reject normality (p ≤ 0.05)"
    )
    st.caption(
        f"Jarque-Bera test: statistic = {jb_stat:.2f}, p-value = {jb_pvalue:.4f} — {interpretation}"
    )

# -- Cumulative return chart ------------------------------
st.subheader("Cumulative Return Over Time")

fig_cum = go.Figure()
fig_cum.add_trace(
    go.Scatter(
        x=df.index,
        y=df["Cumulative Return"],
        mode="lines",
        name="Cumulative Return",
        fill="tozeroy",
        line=dict(width=2),
    )
)

fig_cum.update_layout(
    yaxis_title="Cumulative Return",
    yaxis_tickformat=".0%",
    xaxis_title="Date",
    template="plotly_white",
    height=400,
    margin=dict(l=20, r=20, t=40, b=20),
)

st.plotly_chart(fig_cum, use_container_width=True)

# -- Rolling volatility chart -----------------------------
st.subheader("Rolling Annualized Volatility")

fig_roll_vol = go.Figure()
fig_roll_vol.add_trace(
    go.Scatter(
        x=df.index,
        y=df["Rolling Volatility"],
        mode="lines",
        name=f"{vol_window}-Day Rolling Volatility",
        line=dict(width=2),
    )
)

fig_roll_vol.update_layout(
    yaxis_title="Annualized Volatility",
    yaxis_tickformat=".0%",
    xaxis_title="Date",
    template="plotly_white",
    height=400,
    margin=dict(l=20, r=20, t=40, b=20),
)

st.plotly_chart(fig_roll_vol, use_container_width=True)

# -- Raw data ---------------------------------------------
with st.expander("View Raw Data"):
    st.dataframe(df.tail(60), use_container_width=True)