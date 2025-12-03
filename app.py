import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import requests
import plotly.express as px
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# -------------------------------------------------------
# PAGE CONFIG (Dark Quant Style)
# -------------------------------------------------------
st.set_page_config(page_title="Market Regime & Crash Risk Monitor", layout="wide")

st.markdown(
    """
    <style>
    body { background-color: #0b0c10; }
    .main { background-color: #0b0c10; color: #eaeaea; }
    h1, h2, h3, h4, h5, h6 { color: #66fcf1 !important; }
    .stMetric { background-color: #1f2833 !important; border-radius: 8px; padding: 8px; }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("Market Regime & Crash Risk Monitor")
st.write("Institutional-grade dashboard combining volatility, regimes, and sentiment-based risk.")

# -------------------------------------------------------
# ASSET SELECTOR
# -------------------------------------------------------
ASSETS = {
    "Bitcoin (BTC-USD)": "BTC-USD",
    "Ethereum (ETH-USD)": "ETH-USD",
    "S&P 500 Index (^GSPC)": "^GSPC",
    "Nasdaq 100 Index (^NDX)": "^NDX",
    "Gold Futures (GC=F)": "GC=F",
}

asset_name = st.sidebar.selectbox("Select Asset", list(ASSETS.keys()))
symbol = ASSETS[asset_name]

# -------------------------------------------------------
# UNIVERSAL CLEANER FOR ALL PRICE DATA
# -------------------------------------------------------
@st.cache_data
def load_price_data(symbol):
    df = yf.download(symbol, period="3y", interval="1d", auto_adjust=False)

    if df is None or df.empty:
        return pd.DataFrame()

    df = df.reset_index()  # Ensure Date column exists

    # --- UNIVERSAL COLUMN NORMALIZATION & SYMBOL REMOVAL ---
    df.columns = (
        df.columns
        .str.lower()
        .str.replace(" ", "")
        .str.replace("-", "")
        .str.replace("_", "")
        .str.replace(".", "")
        .str.replace("^", "")
        .str.replace("=", "")
    )

    # Debug print
    st.write("Columns returned:", list(df.columns))

    # --- UNIVERSAL CLOSE DETECTOR ---
    close_candidates = [c for c in df.columns if "close" in c]

    if len(close_candidates) == 0:
        # fallback if close not found
        num_cols = [c for c in df.columns if np.issubdtype(df[c].dtype, np.number)]
        if "open" in df.columns:
            close_candidates = ["open"]
        elif len(num_cols) > 0:
            close_candidates = [num_cols[0]]
        else:
            st.error("No usable price columns found.")
            return pd.DataFrame()

    close_col = close_candidates[0]
    df["ClosePrice"] = df[close_col].astype(float)

    # Ensure Date column
    if "date" not in df.columns:
        df["date"] = df.index
    df["date"] = pd.to_datetime(df["date"], errors="coerce")

    df = df.dropna(subset=["date", "ClosePrice"]).sort_values("date").reset_index(drop=True)

    # Features
    df["Returns"] = df["ClosePrice"].pct_change()
    df["Volatility_30d"] = df["Returns"].rolling(30).std()

    return df.dropna().reset_index(drop=True)


df = load_price_data(symbol)
if df.empty:
    st.error("No valid price data found.")
    st.stop()

# -------------------------------------------------------
# SIMPLE NLP SENTIMENT (No NLTK)
# -------------------------------------------------------
@st.cache_data
def load_headlines():
    try:
        url = "https://cryptopanic.com/api/v1/posts/?auth_token=bbf69ca77e536fa8d3&public=true"
        r = requests.get(url, timeout=5).json()
        if "results" in r:
            return [item.get("title", "") for item in r["results"]][:20]
    except:
        pass
    return [
        "Markets remain cautious amid global uncertainty",
        "Volatility rises as investors react to macro data",
        "Risk sentiment weakens across global markets",
        "Analysts warn of potential downside risks",
        "Traders observe elevated volatility conditions",
    ]


def simple_sentiment_lexicon(text_list):
    pos_words = {"bull", "optimistic", "growth", "upside", "rebound", "strong"}
    neg_words = {"crash", "panic", "fear", "risk", "bear", "selloff", "volatility", "uncertain"}

    scores = []
    for t in text_list:
        txt = t.lower()
        score, hits = 0, 0
        for w in pos_words:
            if w in txt:
                score += 1; hits += 1
        for w in neg_words:
            if w in txt:
                score -= 1; hits += 1
        if hits > 0:
            scores.append(score / hits)

    return np.mean(scores) if scores else 0


headlines = load_headlines()
avg_sentiment = simple_sentiment_lexicon(headlines)
sentiment_fear = max(0, -avg_sentiment * 100)  # ↗ fear = negative sentiment

# -------------------------------------------------------
# MARKET REGIME DETECTION
# -------------------------------------------------------
def detect_regimes(df):
    features = df[["Returns", "Volatility_30d"]]
    scaler = StandardScaler()
    X = scaler.fit_transform(features)

    kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
    labels = kmeans.fit_predict(X)

    df["Regime"] = labels

    centers = pd.DataFrame(kmeans.cluster_centers_, columns=["ret", "vol"])
    centers["stress"] = centers["vol"] - centers["ret"]  # Higher vol, lower return = more stress
    order = centers.sort_values("stress", ascending=False).index.tolist()

    names = {order[0]: "High-Stress", order[1]: "Neutral", order[2]: "Calm"}
    risk =  {order[0]: 85.0,        order[1]: 55.0,      order[2]: 25.0}

    return df, names, risk


df, regime_names, regime_risk_map = detect_regimes(df)

# -------------------------------------------------------
# CRASH RISK SCORE
# -------------------------------------------------------
latest_vol = df["Volatility_30d"].iloc[-1]
vol_risk = min(100, latest_vol * 4000)

latest_reg = int(df["Regime"].iloc[-1])
reg_label = regime_names.get(latest_reg, "Neutral")
reg_risk  = regime_risk_map.get(latest_reg, 55)

crash_risk = 0.5 * vol_risk + 0.3 * sentiment_fear + 0.2 * reg_risk
crash_risk = float(np.clip(crash_risk, 0, 100))

# -------------------------------------------------------
# METRICS DISPLAY
# -------------------------------------------------------
c1, c2, c3, c4 = st.columns(4)
c1.metric("Price", f"{df['ClosePrice'].iloc[-1]:,.2f}")
c2.metric("Daily Return", f"{df['Returns'].iloc[-1]*100:,.2f} %")
c3.metric("30D Vol", f"{df['Volatility_30d'].iloc[-1]*100:,.2f} %")
c4.metric("Crash Risk", f"{crash_risk:,.1f} / 100")

st.caption(f"**Regime:** {reg_label} | **Sentiment Fear:** {sentiment_fear:,.1f}")

# -------------------------------------------------------
# PRICE + REGIME CHART
# -------------------------------------------------------
fig_price = px.line(df, x="date", y="ClosePrice", title="Price with Market Regimes")
fig_reg = px.scatter(df, x="date", y="ClosePrice", color=df["Regime"].map(regime_names))
for trace in fig_reg.data:
    fig_price.add_trace(trace)
st.plotly_chart(fig_price, use_container_width=True)

# -------------------------------------------------------
# VOLATILITY
# -------------------------------------------------------
fig_vol = px.line(df, x="date", y="Volatility_30d", title="30D Realized Volatility")
st.plotly_chart(fig_vol, use_container_width=True)

# -------------------------------------------------------
# RETURNS
# -------------------------------------------------------
fig_ret = px.line(df, x="date", y="Returns", title="Daily Returns")
st.plotly_chart(fig_ret, use_container_width=True)

# -------------------------------------------------------
# SENTIMENT PANEL
# -------------------------------------------------------
st.subheader("News Sentiment")
for h in headlines[:10]:
    st.write("- " + h)

st.write(f"**Average Sentiment:** {avg_sentiment:.3f}")
st.write(f"**Sentiment Fear Score:** {sentiment_fear:.1f} / 100")

st.success("Dashboard loaded successfully.")
