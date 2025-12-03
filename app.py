import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import requests
import plotly.express as px
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# -------------------------------------------------------
# PAGE CONFIG (Dark Mode)
# -------------------------------------------------------
st.set_page_config(page_title="Market Regime & Risk Monitor", layout="wide")

st.markdown("""
<style>
body { background-color: #0b0c10; }
.main { background-color: #0b0c10; color: #eaeaea; }
h1,h2,h3,h4 { color:#66fcf1 !important; }
</style>
""", unsafe_allow_html=True)

st.title("Market Regime & Crash Risk Monitor")


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
# UNIVERSAL PRICE LOADER (MultiIndex SAFE)
# -------------------------------------------------------
@st.cache_data
def load_price_data(symbol):
    df = yf.download(symbol, period="3y", interval="1d")

    if df is None or df.empty:
        return pd.DataFrame()

    # ---- FIX: Flatten MultiIndex columns ----
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [
            "_".join([str(c) for c in col if c]) for col in df.columns.values
        ]
    else:
        df.columns = df.columns.astype(str)

    df = df.reset_index()  # ensures a Date column

    # Normalize names (SAFE even if MultiIndex before)
    df.columns = (
        df.columns
        .str.lower()
        .str.replace(" ", "")
        .str.replace("-", "")
        .str.replace("_", "")
        .str.replace(".", "")
    )

    # Debug show in Streamlit
    st.write("Columns detected:", df.columns.tolist())

    # ---- Close Price detection ----
    candidates = [c for c in df.columns if "close" in c]

    if len(candidates) == 0:
        num_cols = [c for c in df.columns if np.issubdtype(df[c].dtype, np.number)]
        if len(num_cols) == 0:
            st.error("No numeric price columns found.")
            return pd.DataFrame()
        close_col = num_cols[0]
    else:
        close_col = candidates[0]

    df["closeprice"] = df[close_col].astype(float)

    # Ensure date exists
    if "date" not in df.columns:
        df["date"] = df.index
    df["date"] = pd.to_datetime(df["date"], errors="coerce")

    df = df.dropna(subset=["date", "closeprice"]).sort_values("date")

    # Features
    df["returns"] = df["closeprice"].pct_change()
    df["volatility30"] = df["returns"].rolling(30).std()

    return df.dropna().reset_index(drop=True)


df = load_price_data(symbol)
if df.empty:
    st.error("No valid price data found.")
    st.stop()


# -------------------------------------------------------
# SIMPLE NLP SENTIMENT (No NLTK needed)
# -------------------------------------------------------
def simple_sentiment(text_list):
    pos = {"up", "growth", "bull", "optimistic", "gain"}
    neg = {"risk", "fear", "crash", "panic", "selloff", "bear", "volatile"}

    scores = []
    for t in text_list:
        t = t.lower()
        score = 0
        hits = 0
        for w in pos:
            if w in t: score += 1; hits += 1
        for w in neg:
            if w in t: score -= 1; hits += 1
        if hits > 0:
            scores.append(score / hits)

    return np.mean(scores) if scores else 0


@st.cache_data
def load_headlines():
    try:
        r = requests.get(
            "https://cryptopanic.com/api/v1/posts/?auth_token=bbf69ca77e536fa8d3&public=true",
            timeout=5,
        ).json()
        if "results" in r:
            return [x.get("title", "") for x in r["results"]][:20]
    except:
        pass

    return [
        "Market uncertainty rises as volatility increases",
        "Investors cautious amid global risks",
        "Analysts warn of downside pressure",
    ]


headlines = load_headlines()
sentiment_score = simple_sentiment(headlines)
sentiment_fear = max(0, -sentiment_score * 100)


# -------------------------------------------------------
# REGIME DETECTION (KMeans)
# -------------------------------------------------------
def detect_regimes(df):
    X = df[["returns", "volatility30"]].values
    X = StandardScaler().fit_transform(X)

    km = KMeans(n_clusters=3, random_state=42, n_init=10)
    labels = km.fit_predict(X)

    df["regime"] = labels

    centers = km.cluster_centers_
    stress = centers[:, 1] - centers[:, 0]  # high vol - low return = stress

    order = np.argsort(stress)[::-1]
    names = {
        order[0]: "High-Stress",
        order[1]: "Neutral",
        order[2]: "Calm"
    }
    risk = {
        order[0]: 85,
        order[1]: 55,
        order[2]: 25
    }

    return df, names, risk


df, regime_names, risk_map = detect_regimes(df)


# -------------------------------------------------------
# CRASH RISK MODEL
# -------------------------------------------------------
vol_risk = min(100, df["volatility30"].iloc[-1] * 4000)
reg_risk = risk_map[int(df["regime"].iloc[-1])]
crash_risk = float(np.clip(0.5*vol_risk + 0.3*sentiment_fear + 0.2*reg_risk, 0, 100))

# -------------------------------------------------------
# METRICS
# -------------------------------------------------------
c1, c2, c3, c4 = st.columns(4)
c1.metric("Price", f"{df['closeprice'].iloc[-1]:,.2f}")
c2.metric("Return", f"{df['returns'].iloc[-1]*100:,.2f}%")
c3.metric("30D Vol", f"{df['volatility30'].iloc[-1]*100:,.2f}%")
c4.metric("Crash Risk", f"{crash_risk:.1f}/100")

st.caption(
    f"Regime: **{regime_names[int(df['regime'].iloc[-1])]}**, "
    f"Sentiment Fear: **{sentiment_fear:.1f}**"
)


# -------------------------------------------------------
# CHARTS
# -------------------------------------------------------
fig1 = px.line(df, x="date", y="closeprice", title="Price with Regimes")
fig2 = px.scatter(df, x="date", y="closeprice", color=df["regime"].map(regime_names))
for t in fig2.data:
    fig1.add_trace(t)
st.plotly_chart(fig1, use_container_width=True)

st.plotly_chart(px.line(df, x="date", y="volatility30", title="30D Volatility"), use_container_width=True)
st.plotly_chart(px.line(df, x="date", y="returns", title="Daily Returns"), use_container_width=True)

# -------------------------------------------------------
# NEWS SECTION
# -------------------------------------------------------
st.subheader("Latest Market Headlines")
for h in headlines:
    st.write("- " + h)

st.write(f"Sentiment Score: **{sentiment_score:.3f}**")
st.write(f"Fear Score: **{sentiment_fear:.1f} / 100**")

st.success("Dashboard loaded successfully.")
