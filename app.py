# app.py
# -------------------------------------------------------
# FDA I Project - Stock Comparison and Analysis Application
# Run with: streamlit run app.py
# -------------------------------------------------------

import math
from datetime import date, timedelta

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import yfinance as yf
from plotly.subplots import make_subplots
from scipy import stats

TRADING_DAYS = 252
BENCHMARK = "^GSPC"

st.set_page_config(page_title="Stock Comparison and Analysis App", layout="wide")
st.title("Stock Comparison and Analysis App")
st.caption("Compare stocks, benchmark performance, return distributions, and diversification effects.")


# -----------------------------
# Helper functions
# -----------------------------
def clean_ticker_list(text: str) -> list[str]:
    raw = [x.strip().upper() for x in text.replace("\n", ",").split(",")]
    tickers = [x for x in raw if x]
    seen = []
    for t in tickers:
        if t not in seen:
            seen.append(t)
    return seen


def fmt_pct(x):
    if pd.isna(x) or np.isinf(x):
        return "N/A"
    return f"{x:.2%}"


def fmt_num(x, decimals=2, prefix=""):
    if pd.isna(x) or np.isinf(x):
        return "N/A"
    return f"{prefix}{x:,.{decimals}f}"


@st.cache_data(ttl=3600, show_spinner=False)
def download_data(tickers, start_date, end_date):
    """
    Downloads adjusted close data for selected tickers + benchmark.
    Returns:
        price_df, valid_tickers, invalid_tickers
    """
    all_symbols = tickers + [BENCHMARK]
    data = {}
    invalid = []

    for symbol in all_symbols:
        try:
            df = yf.download(
                symbol,
                start=start_date,
                end=end_date + timedelta(days=1),
                progress=False,
                auto_adjust=False,
                threads=False,
            )

            if df.empty:
                invalid.append(symbol)
                continue

            col_name = "Adj Close" if "Adj Close" in df.columns else "Close"
            if col_name not in df.columns:
                invalid.append(symbol)
                continue

            series = df[col_name].copy()
            series.name = symbol
            data[symbol] = series

        except Exception:
            invalid.append(symbol)

    if not data:
        return pd.DataFrame(), [], invalid

    price_df = pd.concat(data.values(), axis=1).sort_index()
    valid = [c for c in price_df.columns if c != BENCHMARK]

    return price_df, valid, invalid


@st.cache_data(ttl=3600, show_spinner=False)
def align_and_clean_prices(price_df, user_tickers):
    """
    Truncate to overlapping range and drop user tickers with >5% missing values.
    Benchmark is handled separately and aligned after user ticker cleanup.
    """
    warnings = []

    stock_prices = price_df[user_tickers].copy()

    missing_pct = stock_prices.isna().mean()
    drop_tickers = missing_pct[missing_pct > 0.05].index.tolist()

    if drop_tickers:
        stock_prices = stock_prices.drop(columns=drop_tickers)
        warnings.append(
            f"Dropped ticker(s) with more than 5% missing values: {', '.join(drop_tickers)}"
        )

    if stock_prices.shape[1] < 2:
        return pd.DataFrame(), [], warnings

    stock_prices = stock_prices.dropna(how="any")
    if stock_prices.empty:
        return pd.DataFrame(), [], warnings

    overlap_start = stock_prices.index.min().date()
    overlap_end = stock_prices.index.max().date()

    warnings.append(
        f"Using overlapping date range across selected stocks: {overlap_start} to {overlap_end}"
    )

    final_cols = list(stock_prices.columns)

    if BENCHMARK in price_df.columns:
        benchmark = price_df[[BENCHMARK]].copy()
        benchmark = benchmark.loc[stock_prices.index].dropna()
        common_index = stock_prices.index.intersection(benchmark.index)
        stock_prices = stock_prices.loc[common_index]
        benchmark = benchmark.loc[common_index]
        aligned_prices = pd.concat([stock_prices, benchmark], axis=1)
    else:
        aligned_prices = stock_prices.copy()
        warnings.append("S&P 500 benchmark could not be downloaded.")

    return aligned_prices, final_cols, warnings


@st.cache_data(ttl=3600, show_spinner=False)
def compute_returns_and_stats(prices, selected_tickers):
    returns = prices.pct_change().dropna()

    stats_rows = {}
    for col in prices.columns:
        r = returns[col].dropna()
        stats_rows[col] = {
            "Annualized Mean Return": r.mean() * TRADING_DAYS,
            "Annualized Volatility": r.std() * math.sqrt(TRADING_DAYS),
            "Skewness": r.skew(),
            "Kurtosis": r.kurtosis(),
            "Min Daily Return": r.min(),
            "Max Daily Return": r.max(),
        }

    summary_df = pd.DataFrame(stats_rows).T

    ew_returns = returns[selected_tickers].mean(axis=1)
    wealth_df = (1 + returns).cumprod() * 10000
    wealth_df["Equal-Weight Portfolio"] = (1 + ew_returns).cumprod() * 10000

    return returns, summary_df, ew_returns, wealth_df


@st.cache_data(ttl=3600, show_spinner=False)
def compute_rolling_volatility(returns, window):
    return returns.rolling(window).std() * math.sqrt(TRADING_DAYS)


@st.cache_data(ttl=3600, show_spinner=False)
def compute_rolling_correlation(returns, stock_a, stock_b, window):
    return returns[stock_a].rolling(window).corr(returns[stock_b])


def portfolio_metrics_for_weight(returns_ab, w):
    a = returns_ab.iloc[:, 0]
    b = returns_ab.iloc[:, 1]

    mean_a = a.mean() * TRADING_DAYS
    mean_b = b.mean() * TRADING_DAYS

    cov_annual = returns_ab.cov() * TRADING_DAYS
    var_a = cov_annual.iloc[0, 0]
    var_b = cov_annual.iloc[1, 1]
    cov_ab = cov_annual.iloc[0, 1]

    port_return = w * mean_a + (1 - w) * mean_b
    port_var = (w**2) * var_a + ((1 - w) ** 2) * var_b + 2 * w * (1 - w) * cov_ab
    port_vol = math.sqrt(max(port_var, 0))

    return port_return, port_vol


# -----------------------------
# Sidebar
# -----------------------------
st.sidebar.header("Inputs")

ticker_text = st.sidebar.text_area(
    "Enter 2 to 5 stock tickers (comma-separated)",
    value="AAPL, MSFT, NVDA",
    height=100,
)

default_end = date.today()
default_start = default_end - timedelta(days=365 * 3)

start_date = st.sidebar.date_input(
    "Start Date",
    value=default_start,
    min_value=date(1970, 1, 1),
)

end_date = st.sidebar.date_input(
    "End Date",
    value=default_end,
    min_value=date(1970, 1, 1),
)

vol_window = st.sidebar.selectbox("Rolling Volatility Window", [30, 60, 90], index=0)
corr_window = st.sidebar.selectbox("Rolling Correlation Window", [30, 60, 90], index=0)

with st.sidebar.expander("About / Methodology"):
    st.markdown(
        """
        - Uses **adjusted close prices** from Yahoo Finance via `yfinance`
        - Uses **simple arithmetic returns**: `pct_change()`
        - Annualized return = mean daily return × **252**
        - Annualized volatility = daily std. dev. × **sqrt(252)**
        - Cumulative wealth uses `(1 + r).cumprod()`
        - Benchmark = **S&P 500 (`^GSPC`)**
        """
    )

run_app = st.sidebar.button("Run Analysis", use_container_width=True)


# -----------------------------
# Validation and loading
# -----------------------------
if not run_app:
    st.info("Enter 2 to 5 tickers, choose dates, and click **Run Analysis**.")
    st.stop()

tickers = clean_ticker_list(ticker_text)

if len(tickers) < 2 or len(tickers) > 5:
    st.error("Please enter between 2 and 5 ticker symbols.")
    st.stop()

if start_date >= end_date:
    st.error("Start date must be before end date.")
    st.stop()

if (end_date - start_date).days < 365:
    st.error("Please select a date range of at least 1 year.")
    st.stop()

with st.spinner("Downloading and preparing data..."):
    prices_raw, valid_downloads, invalid_tickers = download_data(tickers, start_date, end_date)

if invalid_tickers:
    invalid_user_tickers = [t for t in invalid_tickers if t != BENCHMARK]
    if invalid_user_tickers:
        st.error(f"These ticker(s) failed to download or had insufficient data: {', '.join(invalid_user_tickers)}")

valid_user_tickers = [t for t in valid_downloads if t in tickers]

if len(valid_user_tickers) < 2:
    st.error("At least 2 valid stock tickers are required to continue.")
    st.stop()

prices, selected_tickers, clean_warnings = align_and_clean_prices(prices_raw, valid_user_tickers)

if prices.empty or len(selected_tickers) < 2:
    st.error("After cleaning/alignment, fewer than 2 usable tickers remained. Try different symbols or dates.")
    st.stop()

for msg in clean_warnings:
    st.warning(msg)

with st.spinner("Calculating returns and analytics..."):
    returns, summary_df, ew_returns, wealth_df = compute_returns_and_stats(prices, selected_tickers)

benchmark_available = BENCHMARK in prices.columns


# -----------------------------
# Tabs
# -----------------------------
tab1, tab2, tab3, tab4 = st.tabs(
    [
        "Price & Return Analysis",
        "Risk & Distribution",
        "Correlation & Diversification",
        "Raw Data",
    ]
)

# =========================================================
# TAB 1
# =========================================================
with tab1:
    st.subheader("Adjusted Close Price Chart")

    chart_options = selected_tickers.copy()
    if benchmark_available:
        chart_options += [BENCHMARK]

    visible_series = st.multiselect(
        "Select series to display",
        options=chart_options,
        default=chart_options,
    )

    if visible_series:
        fig_price = go.Figure()
        for col in visible_series:
            fig_price.add_trace(
                go.Scatter(
                    x=prices.index,
                    y=prices[col],
                    mode="lines",
                    name=col,
                )
            )
        fig_price.update_layout(
            title="Adjusted Close Prices",
            xaxis_title="Date",
            yaxis_title="Price",
            template="plotly_white",
            height=500,
        )
        st.plotly_chart(fig_price, use_container_width=True)
    else:
        st.warning("Select at least one series to display.")

    st.subheader("Summary Statistics")
    st.dataframe(
        summary_df.style.format(
            {
                "Annualized Mean Return": "{:.2%}",
                "Annualized Volatility": "{:.2%}",
                "Skewness": "{:.2f}",
                "Kurtosis": "{:.2f}",
                "Min Daily Return": "{:.2%}",
                "Max Daily Return": "{:.2%}",
            }
        ),
        use_container_width=True,
    )

    st.subheader("Cumulative Wealth Index")
    wealth_options = selected_tickers.copy()
    if benchmark_available:
        wealth_options += [BENCHMARK]
    wealth_options += ["Equal-Weight Portfolio"]

    fig_wealth = go.Figure()
    for col in wealth_options:
        fig_wealth.add_trace(
            go.Scatter(
                x=wealth_df.index,
                y=wealth_df[col],
                mode="lines",
                name=col,
            )
        )

    fig_wealth.update_layout(
        title="Growth of $10,000 Investment",
        xaxis_title="Date",
        yaxis_title="Portfolio Value ($)",
        template="plotly_white",
        height=500,
    )
    st.plotly_chart(fig_wealth, use_container_width=True)


# =========================================================
# TAB 2
# =========================================================
with tab2:
    st.subheader("Rolling Annualized Volatility")

    roll_vol = compute_rolling_volatility(returns[selected_tickers], vol_window)

    fig_vol = go.Figure()
    for col in selected_tickers:
        fig_vol.add_trace(
            go.Scatter(
                x=roll_vol.index,
                y=roll_vol[col],
                mode="lines",
                name=col,
            )
        )

    fig_vol.update_layout(
        title=f"{vol_window}-Day Rolling Annualized Volatility",
        xaxis_title="Date",
        yaxis_title="Annualized Volatility",
        template="plotly_white",
        height=500,
    )
    st.plotly_chart(fig_vol, use_container_width=True)

    st.subheader("Distribution Analysis")
    chosen_stock = st.selectbox("Choose a stock", selected_tickers)

    dist_view = st.radio(
        "Choose distribution view",
        ["Histogram + Fitted Normal", "Q-Q Plot"],
        horizontal=True,
    )

    chosen_returns = returns[chosen_stock].dropna()

    jb_stat, jb_p = stats.jarque_bera(chosen_returns)
    normality_msg = "Rejects normality (p < 0.05)" if jb_p < 0.05 else "Fails to reject normality (p >= 0.05)"

    st.markdown(
        f"**Jarque-Bera test for {chosen_stock}:** statistic = `{jb_stat:.2f}`, p-value = `{jb_p:.4f}` - {normality_msg}"
    )

    if dist_view == "Histogram + Fitted Normal":
        mu, sigma = stats.norm.fit(chosen_returns)

        fig_hist = go.Figure()
        fig_hist.add_trace(
            go.Histogram(
                x=chosen_returns,
                nbinsx=60,
                histnorm="probability density",
                name="Daily Returns",
                opacity=0.75,
            )
        )

        x_vals = np.linspace(chosen_returns.min(), chosen_returns.max(), 200)
        y_vals = stats.norm.pdf(x_vals, mu, sigma)

        fig_hist.add_trace(
            go.Scatter(
                x=x_vals,
                y=y_vals,
                mode="lines",
                name="Fitted Normal",
            )
        )

        fig_hist.update_layout(
            title=f"{chosen_stock} Daily Return Distribution",
            xaxis_title="Daily Return",
            yaxis_title="Density",
            template="plotly_white",
            height=500,
        )
        st.plotly_chart(fig_hist, use_container_width=True)

    else:
        qq = stats.probplot(chosen_returns, dist="norm")
        theo = qq[0][0]
        ordered = qq[0][1]
        slope = qq[1][0]
        intercept = qq[1][1]

        line_x = np.array([theo.min(), theo.max()])
        line_y = slope * line_x + intercept

        fig_qq = go.Figure()
        fig_qq.add_trace(
            go.Scatter(
                x=theo,
                y=ordered,
                mode="markers",
                name="Observed Quantiles",
            )
        )
        fig_qq.add_trace(
            go.Scatter(
                x=line_x,
                y=line_y,
                mode="lines",
                name="Reference Line",
            )
        )

        fig_qq.update_layout(
            title=f"{chosen_stock} Q-Q Plot vs Normal Distribution",
            xaxis_title="Theoretical Quantiles",
            yaxis_title="Sample Quantiles",
            template="plotly_white",
            height=500,
        )
        st.plotly_chart(fig_qq, use_container_width=True)

    st.subheader("Box Plot of Daily Returns")

    fig_box = go.Figure()
    for col in selected_tickers:
        fig_box.add_trace(
            go.Box(
                y=returns[col].dropna(),
                name=col,
                boxmean=True,
            )
        )

    fig_box.update_layout(
        title="Daily Return Distributions by Stock",
        xaxis_title="Stock",
        yaxis_title="Daily Return",
        template="plotly_white",
        height=500,
    )
    st.plotly_chart(fig_box, use_container_width=True)


# =========================================================
# TAB 3
# =========================================================
with tab3:
    st.subheader("Correlation Heatmap")

    corr_matrix = returns[selected_tickers].corr()

    heatmap = go.Figure(
        data=go.Heatmap(
            z=corr_matrix.values,
            x=corr_matrix.columns,
            y=corr_matrix.index,
            zmid=0,
            text=np.round(corr_matrix.values, 2),
            texttemplate="%{text}",
            colorscale="RdBu",
            reversescale=True,
        )
    )
    heatmap.update_layout(
        title="Pairwise Correlation Matrix of Daily Returns",
        xaxis_title="Stock",
        yaxis_title="Stock",
        template="plotly_white",
        height=500,
    )
    st.plotly_chart(heatmap, use_container_width=True)

    st.subheader("Scatter Plot of Daily Returns")
    col_a, col_b = st.columns(2)
    with col_a:
        scatter_a = st.selectbox("Select Stock A", selected_tickers, key="scatter_a")
    with col_b:
        scatter_b = st.selectbox(
            "Select Stock B",
            [t for t in selected_tickers if t != scatter_a],
            key="scatter_b",
        )

    scatter_df = returns[[scatter_a, scatter_b]].dropna()

    fig_scatter = go.Figure()
    fig_scatter.add_trace(
        go.Scatter(
            x=scatter_df[scatter_a],
            y=scatter_df[scatter_b],
            mode="markers",
            name=f"{scatter_a} vs {scatter_b}",
        )
    )
    fig_scatter.update_layout(
        title=f"Daily Returns: {scatter_a} vs {scatter_b}",
        xaxis_title=f"{scatter_a} Daily Return",
        yaxis_title=f"{scatter_b} Daily Return",
        template="plotly_white",
        height=500,
    )
    st.plotly_chart(fig_scatter, use_container_width=True)

    st.subheader("Rolling Correlation")
    col_c, col_d = st.columns(2)
    with col_c:
        corr_a = st.selectbox("Rolling Corr Stock A", selected_tickers, key="corr_a")
    with col_d:
        corr_b = st.selectbox(
            "Rolling Corr Stock B",
            [t for t in selected_tickers if t != corr_a],
            key="corr_b",
        )

    roll_corr = compute_rolling_correlation(returns[[corr_a, corr_b]], corr_a, corr_b, corr_window)

    fig_rc = go.Figure()
    fig_rc.add_trace(
        go.Scatter(
            x=roll_corr.index,
            y=roll_corr,
            mode="lines",
            name="Rolling Correlation",
        )
    )
    fig_rc.update_layout(
        title=f"{corr_window}-Day Rolling Correlation: {corr_a} vs {corr_b}",
        xaxis_title="Date",
        yaxis_title="Correlation",
        template="plotly_white",
        height=500,
    )
    st.plotly_chart(fig_rc, use_container_width=True)

    st.subheader("Two-Asset Portfolio Explorer")

    col_e, col_f = st.columns(2)
    with col_e:
        port_a = st.selectbox("Portfolio Stock A", selected_tickers, key="port_a")
    with col_f:
        port_b = st.selectbox(
            "Portfolio Stock B",
            [t for t in selected_tickers if t != port_a],
            key="port_b",
        )

    weight_a = st.slider(
        f"Weight on {port_a}",
        min_value=0,
        max_value=100,
        value=50,
        step=1,
    ) / 100

    pair_returns = returns[[port_a, port_b]].dropna()

    current_ret, current_vol = portfolio_metrics_for_weight(pair_returns, weight_a)

    m1, m2 = st.columns(2)
    m1.metric("Portfolio Annualized Return", fmt_pct(current_ret))
    m2.metric("Portfolio Annualized Volatility", fmt_pct(current_vol))

    weights = np.linspace(0, 1, 101)
    curve_vol = []
    curve_ret = []

    for w in weights:
        r, v = portfolio_metrics_for_weight(pair_returns, w)
        curve_ret.append(r)
        curve_vol.append(v)

    fig_curve = go.Figure()
    fig_curve.add_trace(
        go.Scatter(
            x=weights,
            y=curve_vol,
            mode="lines",
            name="Portfolio Volatility Curve",
        )
    )
    fig_curve.add_trace(
        go.Scatter(
            x=[weight_a],
            y=[current_vol],
            mode="markers",
            name="Current Weight",
            marker=dict(size=10),
        )
    )
    fig_curve.update_layout(
        title=f"Two-Asset Portfolio Volatility: {port_a} / {port_b}",
        xaxis_title=f"Weight on {port_a}",
        yaxis_title="Annualized Portfolio Volatility",
        template="plotly_white",
        height=500,
    )
    st.plotly_chart(fig_curve, use_container_width=True)

    st.info(
        "This curve shows the diversification effect. When two stocks are not perfectly positively correlated, "
        "combining them can produce a portfolio with lower volatility than either stock by itself. "
        "The lower the correlation, the stronger this effect tends to be."
    )


# =========================================================
# TAB 4
# =========================================================
with tab4:
    st.subheader("Adjusted Close Price Data")
    st.dataframe(prices, use_container_width=True)

    st.subheader("Daily Returns")
    st.dataframe(returns, use_container_width=True)
    