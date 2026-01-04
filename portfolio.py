import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import os
from datetime import datetime, timedelta
from sklearn.linear_model import LinearRegression
from streamlit_autorefresh import st_autorefresh


AVAILABLE_ASSETS = [
    "BTC-USD", "ETH-USD", "SOL-USD", "BNB-USD", "XRP-USD", "ADA-USD", "DOGE-USD", "AVAX-USD", "DOT-USD", "MATIC-USD",
    "AC.PA", "ACA.PA", "AI.PA", "AIR.PA", "ALO.PA", "ATO.PA", "BN.PA", "BNP.PA", "CA.PA", "CAP.PA", 
    "CS.PA", "DG.PA", "DSY.PA", "EL.PA", "EN.PA", "ENGI.PA", "ERF.PA", "GLE.PA", "HO.PA", "KER.PA", 
    "LR.PA", "MC.PA", "ML.PA", "ORA.PA", "OR.PA", "PUB.PA", "RI.PA", "RNO.PA", "SAF.PA", "SAN.PA", 
    "SGO.PA", "STLAP.PA", "SU.PA", "SW.PA", "TEP.PA", "TTE.PA", "URW.PA", "VIE.PA", "VIV.PA",
    "AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "TSLA", "NFLX", "AMD", "INTC", 
    "JPM", "BAC", "WMT", "PG", "JNJ", "XOM", "CVX", "KO", "PEP", "MCD", "DIS", "NKE",
    "^FCHI", "^GSPC", "^IXIC", "GC=F", "CL=F", "EURUSD=X"
]
AVAILABLE_ASSETS.sort()


def get_current_price(ticker):
    try:
        df = yf.download(ticker, period="1d", interval="1m", progress=False)
        if df.empty: df = yf.download(ticker, period="5d", interval="1d", progress=False)
        if df.empty: return None
        if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.get_level_values(0)
        price = df['Close'].iloc[-1]
        if isinstance(price, pd.Series): price = price.iloc[0]
        return float(price)
    except: return None


def optimize_sma_params(df):
    best_ret = -np.inf
    best_params = (20, 50)
    for s in range(10, 50, 10):
        for l in range(50, 160, 20):
            if s >= l: continue
            d = df.copy()
            d['S'] = d['Close'].rolling(window=s).mean()
            d['L'] = d['Close'].rolling(window=l).mean()
            d['Sig'] = np.where(d['S'] > d['L'], 1, 0)
            ret = d['Close'].pct_change().shift(-1) * d['Sig']
            total_ret = (1 + ret.fillna(0)).cumprod().iloc[-1]
            if total_ret > best_ret:
                best_ret = total_ret
                best_params = (s, l)
    return best_params, best_ret

def module_market_analysis():
        from app import (
        fetch_data as _fetch_data,
        strategy_sma as _strategy_sma,
        strategy_rsi as _strategy_rsi,
        get_advanced_metrics as _get_advanced_metrics,
        get_ml_forecast as _get_ml_forecast,
    )
st.header("Market Analysis (Single Asset)")
ticker = st.sidebar.selectbox("Rechercher un Actif", AVAILABLE_ASSETS, index=AVAILABLE_ASSETS.index("ENGI.PA") if "ENGI.PA" in AVAILABLE_ASSETS else 0)
    
    # Ajout de 1d, 5d, 1mo pour voir le court terme
period = st.sidebar.selectbox("Périodicité", ["1d", "5d", "1mo", "6mo", "1y", "2y", "5y", "max"])
    
strat_choice = st.sidebar.radio("Stratégie", ["SMA Crossover", "RSI Momentum"])

data = _fetch_data(ticker, period)
if data is not None:
        st.sidebar.markdown("---")
        st.sidebar.subheader("Auto-Optimisation")
        default_s, default_l = 20, 50
        if strat_choice == "SMA Crossover":
            if st.sidebar.button("Trouver Meilleurs Paramètres"):
                with st.spinner("Analyse..."):
                    best_p, best_r = optimize_sma_params(data)
                    st.sidebar.success(f"Top: {best_p[0]}/{best_p[1]} (Gain: {(best_r-1)*100:.1f}%)")
                    default_s, default_l = best_p
            p1 = st.sidebar.slider("SMA Courte", 5, 50, default_s)
            p2 = st.sidebar.slider("SMA Longue", 51, 200, default_l)
        else:
            p1 = st.sidebar.slider("RSI Période", 7, 30, 14)
            p2 = None

        cur_price = get_current_price(ticker)
        if cur_price:
            prev = float(data['Close'].iloc[-1])
            var = ((cur_price - prev)/prev)*100
            st.metric(f"Prix Actuel ({ticker})", f"{cur_price:.2f} €/$", f"{var:+.2f} %")

        processed = _strategy_sma(data, p1, p2) if strat_choice == "SMA Crossover" else strategy_rsi(data, p1)
        metrics = _get_advanced_metrics(processed)
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=data.index, y=data['Close'], name="Prix Réel", line=dict(color='gray', width=1)))
        init_p = float(data['Close'].iloc[0])
        fig.add_trace(go.Scatter(x=data.index, y=metrics["cum_strat"] * init_p, name="Stratégie", line=dict(color='#00CCFF', width=2)))
        
        if st.sidebar.checkbox("Prédiction ML"):
            dates, preds, err = _get_ml_forecast(data)
            fig.add_trace(go.Scatter(x=dates, y=preds, name="Forecast", line=dict(color='orange')))
            fig.add_trace(go.Scatter(x=list(dates)+list(dates)[::-1], y=list(preds+err)+list(preds-err)[::-1], fill='toself', fillcolor='rgba(255,165,0,0.2)', line=dict(color='rgba(0,0,0,0)')))

        st.plotly_chart(fig, use_container_width=True)
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Sharpe", metrics["sharpe"])
        c2.metric("Max DD", metrics["max_dd"])
        c3.metric("Win Rate", metrics["win_rate"])
        c4.metric("Total Ret", metrics["total_ret"])
else: st.error("Données indisponibles (Marché fermé ou Ticker invalide).")
