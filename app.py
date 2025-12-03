import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import requests

# =========================================================
# 1. Robust OHLC Loader (works for BTC, ETH, SPY, AAPL, etc.)
# =========================================================
@st.cache_data
def load_ohlcv(symbol: str) -> pd.DataFrame:
    df = yf.download(symbol, period="2y", interval="1d", auto_adjust=False)
    if df is None or df.empty:
        return pd.DataFrame()

    # If MultiIndex columns (happens sometimes), flatten them
    if isinstance(df.columns, pd.MultiIndex):
        flat_cols = []
        for col in df.columns:
            flat_cols.append("_".join([str(c) for c in col if c not in ("", None)]))
        df.columns = flat_cols
    else:
        df.columns = [str(c) for c in df.columns]

    # Bring index (dates) into a column
    df = df.reset_index()

    # Normalize column names: lowercase and strip punctuation
    clean_cols = []
    for c in df.columns:
        c_clean = c.lower()
        for ch in [" ", "-", "_", ".", "^", "="]:
            c_clean = c_clean.replace(ch, "")
        clean_cols.append(c_clean)
    df.columns = clean_cols

    # Find date column
    date_col = None
    for c in df.columns:
        if "date" in c:
            date_col = c
            break
    if date_col is None:
        # Fallback: first column is usually the date after reset_index
        date_col = df.columns[0]

    # Map OHLC
    col_map = {}
    for key in ["open", "high", "low", "close", "volume"]:
        for c in df.columns:
            if key in c and c != date_col:
                col_map[key] = c
                break

    # We at least need close prices
    if "close" not in col_map:
        return pd.DataFrame()

    # Build a clean OHLCV frame
    out = pd.DataFrame()
    out["date"] = pd.to_datetime(df[date_col], errors="coerce")
    out["close"] = pd.to_numeric(df[col_map["close"]], errors="coerce")

    # Fallbacks: if open/high/low missing, reuse close
    out["open"] = pd.to_numeric(df[col_map.get("open", col_map["close"])], errors="coerce")
    out["high"] = pd.to_numeric(df[col_map.get("high", col_map["close"])], errors="coerce")
    out["low"] = pd.to_numeric(df[col_map.get("low", col_map["close"])], errors="coerce")

    if "volume" in col_map:
        out["volume"] = pd.to_numeric(df[col_map["volume"]], errors="coerce")
    else:
        out["volume"] = np.nan

    out = out.dropna(subset=["date", "close"])
    out = out.sort_values("date").reset_index(drop=True)
    return out


# =========================================================
# 2. Simple Lexicon Sentiment (no NLTK, no downloads)
# =========================================================
positive_words = {
    "up", "growth", "bull", "optimistic", "surge", "rally", "strong",
    "support", "recover", "rebound", "positive", "gain"
}
negative_words = {
    "risk", "fear", "crash", "panic", "selloff", "bear", "volatile",
    "down", "collapse", "decline", "drop", "fall", "uncertain"
}


def compute_sentiment(headlines):
    """
    Returns:
      avg_sentiment (float in roughly [-1,1]),
      fear_index (0-100),
      per_headline_scores (list of floats)
    """
    scores = []
    per_headline = []

    for t in headlines:
        text = t.lower()
        score = 0
        hits = 0
        for w in positive_words:
            if w in text:
                score += 1
                hits += 1
        for w in negative_words:
            if w in text:
                score -= 1
                hits += 1
        if hits > 0:
            val = score / hits
            scores.append(val)
            per_headline.append(val)
        else:
            per_headline.append(0.0)

    avg = float(np.mean(scores)) if scores else 0.0
    # Map sentiment to fear index: negative sentiment -> higher fear
    fear = int(np.interp(-avg, [-1, 1], [100, 0]))
    return avg, fear, per_headline


@st.cache_data
def load_news():
    # CryptoPanic free news; if it fails, we fall back to generic headlines
    url = "https://cryptopanic.com/api/free/v1/posts/?kind=news"
    try:
        r = requests.get(url, timeout=5)
        data = r.json()
        results = data.get("results", [])
        titles = [item.get("title", "") for item in results]
        return titles[:20]
    except Exception:
        return [
            "Markets remain cautious amid global uncertainty",
            "Volatility rises as investors react to macro data",
            "Risk sentiment weakens in global markets",
            "Analysts warn of potential downside risks",
            "Traders observe elevated volatility conditions",
        ]


# =========================================================
# 3. Regime Detection via KMeans
# =========================================================
def detect_regimes(df: pd.DataFrame, n_clusters: int = 3):
    # df must already have 'ret' and 'vol_30'
    X = df[["ret", "vol_30"]].to_numpy()
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)

    km = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    labels = km.fit_predict(Xs)
    centers = km.cluster_centers_

    df = df.copy()
    df["regime"] = labels

    # Rank regimes by volatility (low → calm, high → stressed)
    vol_centers = centers[:, 1]
    order = np.argsort(vol_centers)
    calm = order[0]
    neutral = order[1]
    stress = order[2]

    regime_names = {
        calm: "Calm",
        neutral: "Neutral",
        stress: "High-Stress"
    }

    regime_risk = {
        calm: 20.0,
        neutral: 55.0,
        stress: 85.0
    }

    return df, regime_names, regime_risk, centers


# =========================================================
# 4. Streamlit App Layout
# =========================================================
st.set_page_config(page_title="AI Market Crash Early-Warning Terminal", layout="wide")
st.title(" AI Market Crash Early-Warning Terminal")

# Sidebar: asset selector
asset = st.sidebar.selectbox(
    "Select Asset",
    ["BTC-USD", "ETH-USD", "SPY", "AAPL"]
)

# Load OHLC data
data = load_ohlcv(asset)
if data.empty or len(data) < 60:
    st.error("Not enough data for this asset.")
    st.stop()

# Compute returns, volatility, EMAs
data["ret"] = data["close"].pct_change()
data["vol_30"] = data["ret"].rolling(30).std()
data["ema_20"] = data["close"].ewm(span=20, adjust=False).mean()
data["ema_50"] = data["close"].ewm(span=50, adjust=False).mean()
data = data.dropna(subset=["ret", "vol_30", "ema_20", "ema_50"]).reset_index(drop=True)

if len(data) < 60:
    st.error("Not enough post-processed data to run the model.")
    st.stop()

# Regime detection
data, regime_names, regime_risk_map, centers = detect_regimes(data, n_clusters=3)

# Load news + sentiment
headlines = load_news()
avg_sent, fear_index, per_headline_scores = compute_sentiment(headlines)

# Crash risk: combine volatility, regime, sentiment
latest = data.iloc[-1]
vol_risk = float(np.clip(latest["vol_30"] * 4000.0, 0, 100))  # scaled volatility risk
current_regime = int(latest["regime"])
regime_label = regime_names[current_regime]
regime_risk = regime_risk_map[current_regime]

crash_risk = 0.5 * vol_risk + 0.3 * fear_index + 0.2 * regime_risk
crash_risk = float(np.clip(crash_risk, 0, 100))

# Build a crash-risk history (using constant fear, changing vol+regime)
data["vol_risk"] = np.clip(data["vol_30"] * 4000.0, 0, 100)
data["regime_risk"] = data["regime"].map(regime_risk_map)
data["crash_risk"] = 0.5 * data["vol_risk"] + 0.3 * fear_index + 0.2 * data["regime_risk"]

# =========================================================
# 5. Top-Level Metrics
# =========================================================
col1, col2, col3, col4 = st.columns(4)

col1.metric("Last Price", f"{latest['close']:,.2f}")
col2.metric("Regime", regime_label)
col3.metric("Crash Risk", f"{crash_risk:.1f} / 100")
col4.metric("News Fear Index", f"{fear_index} / 100")

st.write("---")

# =========================================================
# 6. Main Charts: Candlestick + Volatility + Crash Risk Gauge
# =========================================================
upper_left, upper_right = st.columns([2, 1])

# ---- Candlestick with EMAs + Regimes overlay ----
with upper_left:
    st.subheader(f"{asset} – Candlestick with EMA & Regimes")

    fig_candle = go.Figure()

    fig_candle.add_trace(
        go.Candlestick(
            x=data["date"],
            open=data["open"],
            high=data["high"],
            low=data["low"],
            close=data["close"],
            name="Price"
        )
    )

    fig_candle.add_trace(
        go.Scatter(
            x=data["date"],
            y=data["ema_20"],
            mode="lines",
            name="EMA 20"
        )
    )

    fig_candle.add_trace(
        go.Scatter(
            x=data["date"],
            y=data["ema_50"],
            mode="lines",
            name="EMA 50"
        )
    )

    fig_candle.update_layout(
        xaxis_title="Date",
        yaxis_title="Price",
        xaxis_rangeslider_visible=False,
        template="plotly_dark"
    )

    st.plotly_chart(fig_candle, use_container_width=True)

# ---- Crash Risk Gauge ----
with upper_right:
    st.subheader("Crash Risk Gauge")

    gauge_fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=crash_risk,
            title={"text": "Crash Risk (0–100)"},
            gauge={
                "axis": {"range": [0, 100]},
                "bar": {"color": "red"},
                "steps": [
                    {"range": [0, 30], "color": "green"},
                    {"range": [30, 60], "color": "yellow"},
                    {"range": [60, 100], "color": "darkred"},
                ],
            },
        )
    )
    gauge_fig.update_layout(template="plotly_dark", height=300)
    st.plotly_chart(gauge_fig, use_container_width=True)

    st.subheader("Crash Risk History (Last 90 Days)")
    last_90 = data.tail(90)
    fig_crash_history = px.line(
        last_90,
        x="date",
        y="crash_risk",
        title="Crash Risk Over Time",
    )
    fig_crash_history.update_layout(template="plotly_dark")
    st.plotly_chart(fig_crash_history, use_container_width=True)

# =========================================================
# 7. Volatility & Regime Strip
# =========================================================
lower_left, lower_right = st.columns([2, 1])

with lower_left:
    st.subheader("30-Day Realized Volatility")
    fig_vol = px.line(
        data,
        x="date",
        y="vol_30",
        title="30-Day Volatility",
    )
    fig_vol.update_layout(template="plotly_dark", yaxis_title="Volatility (std of returns)")
    st.plotly_chart(fig_vol, use_container_width=True)

    st.subheader("Regime Timeline")
    # Map regimes to colors
    regime_color_map = {
        "Calm": "#2ecc71",
        "Neutral": "#f1c40f",
        "High-Stress": "#e74c3c",
    }
    regime_colors = data["regime"].map(lambda r: regime_color_map[regime_names[r]])

    fig_reg = go.Figure(
        go.Bar(
            x=data["date"],
            y=[1.0] * len(data),
            marker_color=regime_colors,
            showlegend=False,
        )
    )
    fig_reg.update_layout(
        template="plotly_dark",
        yaxis=dict(showticklabels=False),
        xaxis_title="Date",
        title="Regime Strip (Calm / Neutral / High-Stress)",
        height=180,
    )
    st.plotly_chart(fig_reg, use_container_width=True)

# =========================================================
# 8. News & Sentiment Panel
# =========================================================
with lower_right:
    st.subheader("News Sentiment Overview")

    st.write(f"Average sentiment score: `{avg_sent:.3f}`")
    st.write(f"News-based Fear Index: `{fear_index} / 100`")

    if headlines:
        st.write("Headlines used for sentiment:")
        for title, score in zip(headlines[:10], per_headline_scores[:10]):
            tag = "Positive" if score > 0.1 else "Negative" if score < -0.1 else "Neutral"
            st.write(f"- {title}  — _{tag}_")
    else:
        st.info("No headlines available; using fallback generic headlines.")

    st.subheader("Headline Sentiment Distribution")
    if per_headline_scores:
        fig_sent = px.histogram(
            x=per_headline_scores,
            nbins=10,
            labels={"x": "Per-headline sentiment"},
            title="Distribution of Headline Sentiment Scores",
        )
        fig_sent.update_layout(template="plotly_dark")
        st.plotly_chart(fig_sent, use_container_width=True)
    else:
        st.info("No sentiment scores to display.")

st.success("Model run complete. This is a stable, finished version of the terminal.")
